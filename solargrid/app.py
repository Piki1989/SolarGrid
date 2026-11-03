"""Mapa: wyszukiwanie adresu / klik w budynek -> wykryte panele i szacowana moc.

    SOLARGRID_CKPT=runs/v1/best.pt .venv/bin/uvicorn solargrid.app:app --port 8000
    (z laptopa: ssh -L 8000:localhost:8000 mpikus@thebrain.kis.agh.edu.pl)

Lokalnie na Macu (GPU przez MPS, ortofoto z WMS GUGiK, bez arkuszy na dysku):
    SOLARGRID_CKPT=runs/all_v1/best.pt SOLARGRID_ORTHO=wms .venv/bin/uvicorn solargrid.app:app --port 8001
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import geopandas as gpd
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pyproj import Transformer
from shapely.geometry import box

from . import correct, estimate, gugik, ortho
from .predict import Segmenter

STATIC = Path(__file__).parent / "static"
DETECTIONS = ortho.DATA / "detections" / "pv.gpkg"

app = FastAPI(title="SolarGrid")


# Goście przez tunel Cloudflare (cloudflared dokłada nagłówek cf-connecting-ip): hasło
# SOLARGRID_GUEST_PASSWORD (Basic auth, dowolny login) i tylko mapa z wyceną; ocena i galerie
# etykiet zostają dla nas (localhost / tunel SSH). Bez ustawionego hasła tunel nie wpuszcza nikogo.
GUEST_BLOCKED = ("/ocena", "/api/ocena", "/review", "/popraw", "/api/popraw")


@app.middleware("http")
async def guest_gate(request: Request, call_next):
    if "cf-connecting-ip" not in request.headers:
        return await call_next(request)
    if request.url.path.startswith(GUEST_BLOCKED):
        return Response("Niedostępne", status_code=403)
    password = os.environ.get("SOLARGRID_GUEST_PASSWORD", "")
    auth = request.headers.get("authorization", "")
    try:
        given = base64.b64decode(auth.removeprefix("Basic ")).decode().partition(":")[2]
    except ValueError:
        given = ""
    if not password or not secrets.compare_digest(given, password):
        return Response("Podaj hasło", status_code=401, headers={"WWW-Authenticate": 'Basic realm="SolarGrid"'})
    return await call_next(request)


@app.exception_handler(requests.RequestException)
def gugik_down(request, exc):
    return JSONResponse({"detail": "Usługa GUGiK nie odpowiada, spróbuj za chwilę."}, status_code=502)

(ortho.DATA / "labels").mkdir(parents=True, exist_ok=True)
# Galerie z solargrid.review: /review/<zbiór>/review/index.html
app.mount("/review", StaticFiles(directory=ortho.DATA / "labels", html=True), name="review")
_to_2180 = Transformer.from_crs(4326, 2180, always_xy=True).transform


def device() -> str:
    """SOLARGRID_DEVICE albo pierwsza karta CUDA, a na Macu GPU Apple (MPS)."""
    import torch
    return os.environ.get("SOLARGRID_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "mps")


@lru_cache(maxsize=1)
def segmenter() -> Segmenter:
    ckpt = os.environ.get("SOLARGRID_CKPT")
    if not ckpt:
        raise HTTPException(503, "Brak modelu: ustaw SOLARGRID_CKPT")
    return Segmenter(Path(ckpt), device())


@lru_cache(maxsize=1)
def sam():
    """SAM 2 do dopasowania obrysu budynku do dachu (align.py); SOLARGRID_ALIGN=0 wyłącza."""
    if os.environ.get("SOLARGRID_ALIGN", "1") == "0":
        return None
    from .labels import load_sam
    return load_sam(device())


@lru_cache(maxsize=1)
def detections() -> gpd.GeoDataFrame | None:
    return gpd.read_file(DETECTIONS).to_crs(4326) if DETECTIONS.exists() else None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/estimate")
def api_estimate(q: str = Query(..., min_length=3)):
    return estimate.estimate_address(segmenter(), q, sam())


@app.get("/api/estimate_xy")
def api_estimate_xy(lon: float, lat: float):
    x, y = _to_2180(lon, lat)
    found = gugik.building_at(x, y)
    if not found:
        return {"status": "brak_budynku"}
    bid, building = found
    return {"budynek_id": bid, "budynek": estimate.to_geojson(building),
            "powierzchnia_budynku_m2": round(building.area, 1),
            **estimate.estimate_building(segmenter(), building, sam=sam())}


RATINGS = ortho.DATA / "labels"


def rating_dir(name: str) -> Path:
    if not name.startswith("ocena_") or "/" in name or not (RATINGS / name / "items.json").exists():
        raise HTTPException(404, f"Brak zestawu {name}")
    return RATINGS / name


@app.get("/ocena")
def ocena():
    return FileResponse(STATIC / "ocena.html")


@app.get("/api/ocena/items")
def ocena_items(set: str = "ocena_v1"):
    """Domy do oceny bez nazw modeli (ślepe porównanie) + dotychczasowe głosy."""
    d = rating_dir(set)
    items = json.loads((d / "items.json").read_text())
    votes = {}
    path = d / "ratings.jsonl"
    for line in path.open() if path.exists() else []:
        r = json.loads(line)
        votes[r["id"]] = r["vote"]
    return [{"id": it["id"], "vote": votes.get(it["id"])} for it in items]


class Vote(BaseModel):
    set: str = "ocena_v1"
    id: str
    vote: Literal["a", "b", "oba_dobre", "oba_zle"]


@app.post("/api/ocena")
def ocena_vote(v: Vote):
    with (rating_dir(v.set) / "ratings.jsonl").open("a") as f:
        f.write(json.dumps({"id": v.id, "vote": v.vote, "t": time.time()}) + "\n")
    return {"ok": True}


def correct_dir(name: str) -> Path:
    if not name.startswith("popraw_") or "/" in name or not (RATINGS / name / "items.json").exists():
        raise HTTPException(404, f"Brak zestawu {name}")
    return RATINGS / name


@app.get("/popraw")
def popraw():
    return FileResponse(STATIC / "popraw.html")


@app.get("/popraw/siatka")
def popraw_siatka():
    return FileResponse(STATIC / "siatka.html")


@app.get("/api/popraw/items")
def popraw_items(set: str):
    d = correct_dir(set)
    last = correct.log(d)
    keep = ("id", "kind", "split", "start", "score", "data_nalotu", "auto_m2", "budynki")
    return [{**{k: it[k] for k in keep}, "prio_kwp": it.get("prio_kwp", 0),
             "status": last.get(it["id"], {}).get("status")}
            for it in json.loads((d / "items.json").read_text())]


def _mask_png(path: Path) -> Response:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return Response(m.tobytes(), media_type="application/octet-stream",
                    headers={"X-Shape": f"{m.shape[0]},{m.shape[1]}", "X-Source": path.parent.name})


@app.get("/api/popraw/mask")
def popraw_mask(set: str, id: str, source: Literal["saved", "auto"] = "saved"):
    """Maska jako surowe bajty (H×W, 0/1/255): zapisana poprawka, a gdy jej nie ma — propozycja modelu."""
    d = correct_dir(set)
    saved, auto = d / "masks" / f"{id}.png", d / "auto" / f"{id}.png"
    if not auto.exists():
        raise HTTPException(404, "Brak kafelka")
    return _mask_png(saved if source == "saved" and saved.exists() else auto)


@app.post("/api/popraw/save")
async def popraw_save(request: Request, set: str, id: str,
                      status: Literal["zapisz", "brak_pv", "niepewne", "akceptuj", "do_poprawy"],
                      h: int = 0, w: int = 0, start: str = "auto", sec: float = 0, who: str = "", via: str = "edytor"):
    """zapisz -> ok (propozycja bez zmian) / poprawione / brak_pv (pusta maska).
    Z widoku siatki (bez przesyłania maski): akceptuj -> ok (kopia propozycji), brak_pv z pustym
    body, do_poprawy -> oznaczone jako złe, czeka na edytor (bez maski, jak niepewne)."""
    d = correct_dir(set)
    auto_path = d / "auto" / f"{id}.png"
    if not auto_path.exists():
        raise HTTPException(404, "Brak kafelka")
    mask_path = d / "masks" / f"{id}.png"
    body = await request.body()
    if status == "akceptuj":
        m = cv2.imread(str(auto_path), cv2.IMREAD_GRAYSCALE)
        status = "ok" if m.any() else "brak_pv"
        cv2.imwrite(str(mask_path), m)
    elif status == "brak_pv" and not body:
        cv2.imwrite(str(mask_path), np.zeros_like(cv2.imread(str(auto_path), cv2.IMREAD_GRAYSCALE)))
    elif status in ("zapisz", "brak_pv"):
        m = np.frombuffer(body, np.uint8)
        if m.size != h * w or not np.isin(m, (0, 1, 255)).all():
            raise HTTPException(400, "Zła maska")
        m = m.reshape(h, w)
        if status == "zapisz":
            if not m.any():
                status = "brak_pv"
            elif start == "auto" and np.array_equal(m, cv2.imread(str(auto_path), cv2.IMREAD_GRAYSCALE)):
                status = "ok"
            else:
                status = "poprawione"
        cv2.imwrite(str(mask_path), m)
    elif mask_path.exists():  # niepewne / do poprawy nie trafiają do treningu
        mask_path.unlink()
    with (d / "log.jsonl").open("a") as f:
        f.write(json.dumps({"id": id, "status": status, "sec": round(sec, 1), "who": who[:40],
                            "via": via[:10], "t": time.time()}) + "\n")
    return {"ok": True, "status": status}


class SamQuery(BaseModel):
    set: str
    id: str
    points: list[tuple[float, float, int]]  # (x, y, 1 = obiekt / 0 = tło) w pikselach kafelka


@app.post("/api/popraw/sam")
def popraw_sam(q: SamQuery):
    """Maska jednego obiektu z SAM 2 po kliknięciu (narzędzie „SAM” na stronie /popraw)."""
    d = correct_dir(q.set)
    img = cv2.imread(str(d / "images" / f"{q.id}.jpg"))
    if img is None or not q.points:
        raise HTTPException(404, "Brak kafelka")
    pred = sam()
    if pred is None:
        raise HTTPException(503, "SAM wyłączony (SOLARGRID_ALIGN=0)")
    pts = np.array([p[:2] for p in q.points], np.float32)
    lab = np.array([p[2] for p in q.points], np.int32)
    with estimate._sam_lock:
        # Kolejne kliknięcia w ten sam kafelek nie liczą obrazu SAM od nowa
        # (estimate_building zeruje sam_tile, bo podmienia obraz w predyktorze).
        if estimate.sam_tile != (q.set, q.id):
            pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            estimate.sam_tile = (q.set, q.id)
        masks, scores, _ = pred.predict(point_coords=pts, point_labels=lab, multimask_output=len(pts) == 1)
    # Przy jednym kliknięciu SAM daje 3 propozycje: bierzemy najmniejszą z sensownym wynikiem,
    # bo pojedynczy klik w panel to zwykle panel/rząd paneli, a nie cały dach.
    order = np.argsort([m.sum() for m in masks])
    best = next((i for i in order if scores[i] >= 0.8 * scores.max()), int(np.argmax(scores)))
    m = masks[best].astype(np.uint8)
    return Response(m.tobytes(), media_type="application/octet-stream",
                    headers={"X-Shape": f"{m.shape[0]},{m.shape[1]}"})


@app.get("/api/detections")
def api_detections(bbox: str):
    """Wykrycia z przebiegu po całym regionie (GeoJSON) w prostokącie lon/lat."""
    gdf = detections()
    if gdf is None:
        return {"type": "FeatureCollection", "features": []}
    minx, miny, maxx, maxy = map(float, bbox.split(","))
    sub = gdf[gdf.intersects(box(minx, miny, maxx, maxy))].head(5000)
    return sub.__geo_interface__
