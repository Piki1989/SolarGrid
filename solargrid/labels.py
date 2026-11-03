"""Etykiety PV z OSM doprecyzowane przez SAM 2.

Dla każdego kafelka (obiekt OSM w środku) wszystkie obiekty PV z OSM w kadrze stają się
podpowiedziami dla SAM 2: poligon → prostokąt, punkt → punkt. Wynikowa maska jest sumą
masek, które przeszły filtry jakości. Kafelki: JPG + PNG etykiety (0 tło, 1 PV, 255 ignoruj)
+ records.jsonl.

    python -m solargrid.labels --out ~/datasets/malopolska_pv/labels/osm_sam --gpu 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio.features
import torch
from pyproj import Transformer
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree
from tqdm import tqdm

from . import buildings, ortho

OSM_PATH = ortho.DATA / "meta" / "osm_pv_geom.json"
TILE_PX = 1024
TILE_RES = 0.05  # m/px — kafelek 51.2 m
MAX_OBJ_M = 40.0  # większe obiekty (farmy) pomijamy w tej rundzie
ROOF_AREA_M2 = (1.5, 600.0)  # sensowna powierzchnia maski z promptu punktowego
POLY_SHIFT_M = 3.0  # tolerowane przesunięcie poligonów OSM względem ortofoto
POLY_BOX_PAD_M = 1.0  # większy margines promptu łapie cień i otoczenie
IGNORE_POLY_M, IGNORE_POINT_M = 3.0, 8.0  # strefa "ignoruj" wokół odrzuconych obiektów
IGNORE = 255
ROOF_MAX_FRAC = 0.5  # maks. pole etykiety z punktu / pole obrysu budynku
VERIFY_MIN_PROB = 0.5  # średnie p(PV) z modelu w masce SAM dla punktów edytowanych po nalocie
GRID_MIN_RECT = 0.85  # minimalna prostokątność maski składowej w grid_union


def load_osm() -> list[dict]:
    tr = Transformer.from_crs(4326, 2180, always_xy=True)
    out = []
    for el in json.loads(OSM_PATH.read_text())["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "node":
            geom = Point(*tr.transform(el["lon"], el["lat"]))
        elif el["type"] == "way" and len(el.get("geometry", [])) >= 4:
            geom = Polygon([tr.transform(p["lon"], p["lat"]) for p in el["geometry"]]).buffer(0)
        else:
            continue
        if geom.is_empty:
            continue
        place = tags.get("generator:place") or tags.get("location") or ""
        out.append({"id": f"{el['type'][0]}{el['id']}", "geom": geom, "roof": place == "roof",
                    "power": tags.get("power", ""), "edited": el.get("timestamp", "")[:10]})
    return out


def load_sam(device: str):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)


def shifted_iou(a: np.ndarray, b: np.ndarray, max_shift_px: int, step: int = 4) -> float:
    """Najlepsze IoU masek przy przesunięciu b o ±max_shift_px (liczone w skali 1/step)."""
    a, b = a[::step, ::step], b[::step, ::step]
    s = max_shift_px // step
    best = 0.0
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            bs = np.roll(b, (dy, dx), axis=(0, 1))
            best = max(best, (a & bs).sum() / max((a | bs).sum(), 1))
    return float(best)


def grid_union(predictor, g: Polygon, ref: np.ndarray, transform, spacing_m: float = 2.0) -> np.ndarray:
    """Suma masek SAM z punktów siatki wewnątrz poligonu. Maska składowa musi leżeć w
    poszerzonym poligonie i być wyraźnie mniejsza od niego (inaczej to trawnik/dach)."""
    h, w = ref.shape
    minx, miny, maxx, maxy = g.bounds
    pts = [(x, y) for x in np.arange(minx + spacing_m / 2, maxx, spacing_m)
           for y in np.arange(miny + spacing_m / 2, maxy, spacing_m) if g.contains(Point(x, y))]
    out = np.zeros((h, w), bool)
    if not pts:
        return out
    allowed = rasterio.features.geometry_mask([g.buffer(POLY_SHIFT_M)], out_shape=(h, w),
                                              transform=transform, invert=True)
    coords = np.array([[to_px(transform, x, y)] for x, y in pts])
    masks, scores, _ = predictor.predict(point_coords=coords, point_labels=np.ones((len(pts), 1)),
                                         multimask_output=True)
    masks = masks.reshape(len(pts), -1, h, w).astype(bool)
    scores = np.asarray(scores).reshape(len(pts), -1)
    min_px = 1.5 / abs(transform.a * transform.e)
    for cand, sc in zip(masks, scores):
        best, best_rect = None, GRID_MIN_RECT
        for mk, s in zip(cand, sc):
            area = mk.sum()
            if s < 0.8 or area < min_px or area > 0.6 * ref.sum() or (mk & allowed).sum() < 0.85 * area:
                continue
            r = rectangularity(mk)
            if r >= best_rect:
                best, best_rect = mk, r
        if best is not None:
            out |= best
    return out


def rectangularity(mask: np.ndarray) -> float:
    """Pole maski / pole minimalnego obróconego prostokąta. Rzędy paneli ~0.9+, cienie i trawa mniej."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cs) != 1:
        return 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(cs[0])
    return float(mask.sum() / max(rw * rh, 1))


def to_px(transform, x, y):
    r, c = rasterio.transform.rowcol(transform, x, y)
    return float(c), float(r)


def segment_tile(predictor, rgb, transform, objs, flight_date: str,
                 prob: np.ndarray | None = None, verify_all: bool = False) -> tuple[np.ndarray, list[dict]]:
    """Etykieta kafelka (0 tło, 1 PV, 255 ignoruj) i raport per obiekt.

    Obiekt z OSM, dla którego SAM nie dał wiarygodnej maski, to nadal znana instalacja —
    jego otoczenie oznaczamy jako ignoruj, żeby model nie uczył się, że PV to tło.
    Punkt edytowany w OSM po nalocie mógł nie istnieć na zdjęciu (SAM i tak zaznaczy
    fragment dachu). Data edycji to nie data montażu (w Krakowie >50% punktów edytowano
    po nalocie), więc gdy jest mapa prawdopodobieństwa z modelu (prob), przyjmujemy
    maskę SAM, jeśli model też widzi tam PV; inaczej tylko ignoruj.
    Poligon weryfikuje się kształtem, więc zostaje.

    verify_all (runda 2, prob z modelu dostrojonego na Małopolsce): weryfikujemy każdy
    obiekt. SAM z punktu na płaskim dachu zaznacza często cały dach, a poligony farm
    sprzed budowy dają maski na trawie — wtedy PV = część maski SAM, którą widzi model,
    reszta maski to ignoruj.
    """
    predictor.set_image(rgb)
    px_area = abs(transform.a * transform.e)
    h, w = rgb.shape[:2]
    full = np.zeros((h, w), bool)
    unsure = np.zeros((h, w), bool)
    rejected = []
    report = []

    def reject(o, **info):
        rejected.append(o["geom"].buffer(IGNORE_POINT_M if isinstance(o["geom"], Point) else IGNORE_POLY_M))
        report.append({"id": o["id"], "kind": "point" if isinstance(o["geom"], Point) else "poly",
                       "ok": False, **info})

    for o in objs:
        g = o["geom"]
        if isinstance(g, Point):
            after = o["edited"] > flight_date
            if after and prob is None:
                reject(o, why="po_nalocie")
                continue
            pt = np.array([to_px(transform, g.x, g.y)])
            masks, scores, _ = predictor.predict(point_coords=pt, point_labels=np.array([1]),
                                                 multimask_output=True)
            areas = masks.reshape(len(masks), -1).sum(1) * px_area
            # SAM z punktu często zaznacza cały dach. Instalacja to zwykle część jednej
            # połaci, więc maska nie może przekraczać ROOF_MAX_FRAC obrysu budynku.
            cap = ROOF_AREA_M2[1]
            roofs = buildings.in_bounds(g.buffer(1.0).bounds)
            roofs = roofs[roofs.intersects(g.buffer(1.0))]
            if len(roofs):
                cap = min(cap, ROOF_MAX_FRAC * roofs.area.max())
            ok = (areas >= ROOF_AREA_M2[0]) & (areas <= cap)
            if not ok.any():
                reject(o, why="area")
                continue
            i = int(np.argmax(np.where(ok, scores, -1)))
            m, score, iou = masks[i].astype(bool), float(scores[i]), None
            if after and not verify_all:
                p_in = float(prob[m].mean())
                if p_in < VERIFY_MIN_PROB:
                    reject(o, why="po_nalocie", prob=round(p_in, 3))
                    continue
        else:
            # Poligony OSM bywają przesunięte o 1–3 m względem ortofoto: prompt z lekko
            # poszerzonego prostokąta, a zgodność liczona z tolerancją przesunięcia.
            minx, miny, maxx, maxy = g.buffer(POLY_BOX_PAD_M).bounds
            x0, y1 = to_px(transform, minx, miny)
            x1, y0 = to_px(transform, maxx, maxy)
            masks, scores, _ = predictor.predict(box=np.array([x0, y0, x1, y1]), multimask_output=True)
            ref = rasterio.features.geometry_mask([g], out_shape=(h, w), transform=transform, invert=True)
            ious = [shifted_iou(mk.astype(bool), ref, round(POLY_SHIFT_M / transform.a)) for mk in masks]
            i = int(np.argmax(ious))
            m, score, iou = masks[i].astype(bool), float(scores[i]), ious[i]
            ratio = m.sum() / max(ref.sum(), 1)
            if iou < 0.6 or not 0.5 <= ratio <= 2.0:
                # Jeden poligon OSM obejmuje często kilka osobnych rzędów paneli.
                m = grid_union(predictor, g, ref, transform)
                ratio = m.sum() / max(ref.sum(), 1)
                if not 0.3 <= ratio <= 1.2:
                    reject(o, why="iou", iou=round(iou, 3), ratio=round(float(ratio), 2))
                    continue
                score, iou = 1.0, shifted_iou(m, ref, round(POLY_SHIFT_M / transform.a))
                # Siatka mogła pominąć część rzędów — reszta poligonu to "ignoruj", nie tło.
                rejected.append(g.buffer(IGNORE_POLY_M))
        if score < 0.75:
            reject(o, why="score", score=round(score, 3))
            continue
        if verify_all:
            p_in = float(prob[m].mean())
            if p_in < VERIFY_MIN_PROB:
                pv = m & (prob > 0.5)
                if pv.sum() * px_area < ROOF_AREA_M2[0]:
                    reject(o, why="model", prob=round(p_in, 3))
                    continue
                unsure |= m & ~pv
                m = pv
        full |= m
        report.append({"id": o["id"], "kind": "point" if iou is None else "poly", "ok": True,
                       "score": round(score, 3), "iou": None if iou is None else round(iou, 3),
                       "area_m2": round(float(m.sum() * px_area), 1)})

    label = full.astype(np.uint8)
    label[unsure & ~full] = IGNORE
    if rejected:
        ignore = rasterio.features.geometry_mask(rejected, out_shape=(h, w), transform=transform, invert=True)
        label[ignore & ~full] = IGNORE
    return label, report


_NEG: dict = {}  # stan dla procesów potomnych (fork) w make_negatives


def _neg_tile(job):
    """Jeden kandydat na kafelek tła (w procesie potomnym). None gdy brak zdjęcia/pokrycia."""
    cand, bounds = job
    st = _NEG
    res, out = st["res"], st["out"]
    # Bez dopełniania z gorszych nalotów — rozmyte kafelki uczą złych rzeczy.
    crop = ortho.crop_bounds(bounds, res=res, max_pixel=0.07 if res < 0.1 else 0.25)
    if crop is None or crop[2]["pokrycie"] < 0.95:
        return None
    rgb, transform, meta = crop
    label = np.zeros(rgb.shape[:2], np.uint8)
    if st["verifier"] is not None:
        sure = (st["verifier"].prob(rgb, res, tta=False) > 0.5).astype(np.uint8)
        k = max(3, round(2.0 / res) | 1)  # + 1 m wokół
        label[cv2.dilate(sure, np.ones((k, k), np.uint8)) > 0] = IGNORE
    label[buildings.roof_mask(bounds, transform, label.shape, pad_m=2.0)] = IGNORE
    tree = st["pv_tree"]
    pv = [tree.geometries[i].buffer(10.0) for i in tree.query(box(*bounds), predicate="intersects")]
    if pv:  # znane PV (np. naziemne) — też ignoruj
        label[rasterio.features.geometry_mask(pv, out_shape=label.shape, transform=transform,
                                              invert=True)] = IGNORE
    tile_id = f"neg_{st['seed']}_{cand:07d}"
    cv2.imwrite(str(out / "images" / f"{tile_id}.jpg"), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out / "masks" / f"{tile_id}.png"), label)
    return {"tile": tile_id, "bounds": bounds, "res": res, **meta,
            "ignore_frac": round(float((label == IGNORE).mean()), 3)}


def make_negatives(out: Path, n: int, res: float, seed: int = 0, verifier=None, roofs_min_m2: float = 0.0,
                   workers: int = 1):
    """Losowe kafelki z etykietą 'tło' poza budynkami i 'ignoruj' na dachach.

    Na dachu może być nieoznaczona instalacja (w Małopolsce PV ma ~10–15% domów), więc
    dachów nie uczymy jako tła. Uczymy za to, że pola, rowy, drogi, plandeki itp. to nie PV.
    Z verifier: miejsca, gdzie model jest pewny PV (farmy i budynki spoza OSM), też ignoruj.
    roofs_min_m2 > 0: kafelki wycentrowane na losowych budynkach co najmniej tej wielkości
    (bloki, hale) — etykietę dachów robi potem solargrid.pseudo z nauczyciela.
    workers > 1: wycinanie (CPU: dekodowanie JPEG z GeoTIFF + reprojekcja) w puli procesów;
    z verifier (GPU) zawsze 1.
    """
    import multiprocessing as mp

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = ortho.index()
    src_px = idx["piksel"] <= 0.07 if res < 0.1 else idx["piksel"] == 0.25
    sheets = idx[idx["path"].notna() & src_px]
    bld = buildings._gdf()  # wczytane przed fork — procesy potomne dzielą pamięć
    if roofs_min_m2:
        bld = bld[bld.area >= roofs_min_m2]
        bld = bld.iloc[bld.sindex.query(sheets.union_all(), predicate="intersects")]
        print(f"budynki ≥{roofs_min_m2:.0f} m² w pobranych arkuszach: {len(bld)}")
    half = TILE_PX * res / 2
    _NEG.update(res=res, out=out, seed=seed, verifier=verifier,
                pv_tree=STRtree([o["geom"] for o in load_osm()]))

    def jobs():
        cand = 0
        while True:
            if roofs_min_m2:
                c = bld.geometry.iloc[rng.integers(len(bld))].centroid
                cx, cy = c.x + rng.uniform(-half / 2, half / 2), c.y + rng.uniform(-half / 2, half / 2)
            else:
                minx, miny, maxx, maxy = sheets.geometry.iloc[rng.integers(len(sheets))].bounds
                cx, cy = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            yield cand, (cx - half, cy - half, cx + half, cy + half)
            cand += 1

    workers = 1 if verifier is not None else workers
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    results = pool.imap(_neg_tile, jobs(), chunksize=4) if pool else map(_neg_tile, jobs())
    written = 0
    with (out / "records.jsonl").open("a") as rec, tqdm(total=n) as bar:
        for r in results:
            if r is None:
                continue
            if written >= n:  # nadmiar z kafelków w locie — usuń, żeby nie trafił do treningu
                for sub, ext in (("images", "jpg"), ("masks", "png")):
                    (out / sub / f"{r['tile']}.{ext}").unlink(missing_ok=True)
                continue
            rec.write(json.dumps(r) + "\n")
            written += 1
            bar.update()
            if written == n and pool:
                pool.terminate()  # imap może jeszcze oddać kilka gotowych wyników
                break
    if pool:
        pool.join()
        # Procesy przerwane w trakcie zapisu mogły zostawić pliki bez wpisu w records.
        keep = {json.loads(line)["tile"] for line in (out / "records.jsonl").open()}
        for sub in ("images", "masks"):
            for f in (out / sub).iterdir():
                if f.stem.startswith(f"neg_{seed}_") and f.stem not in keep:
                    f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ortho.DATA / "labels" / "osm_sam")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--negatives", type=int, default=0, help="zamiast OSM: N losowych kafelków tła")
    ap.add_argument("--neg-res", type=float, default=0.05)
    ap.add_argument("--roofs-min-m2", type=float, default=0.0, help="negatywy wycentrowane na dużych budynkach")
    ap.add_argument("--workers", type=int, default=1, help="procesy CPU do wycinania negatywów")
    ap.add_argument("--verify", type=Path, help="checkpoint modelu do weryfikacji punktów edytowanych po nalocie")
    ap.add_argument("--verify-all", action="store_true", help="weryfikuj modelem każdy obiekt (runda 2)")
    args = ap.parse_args()
    verifier = None
    if args.verify:
        from .predict import Segmenter
        verifier = Segmenter(args.verify, f"cuda:{args.gpu}")
    if args.negatives:
        ortho.build_index()
        make_negatives(args.out, args.negatives, args.neg_res, verifier=verifier if args.verify_all else None,
                       roofs_min_m2=args.roofs_min_m2, workers=args.workers)
        return
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    done = set()
    rec_path = args.out / "records.jsonl"
    if rec_path.exists():
        done = {json.loads(line)["tile"] for line in rec_path.open()}

    ortho.build_index()
    objs = [o for o in load_osm() if max(o["geom"].bounds[2] - o["geom"].bounds[0],
                                         o["geom"].bounds[3] - o["geom"].bounds[1]) <= MAX_OBJ_M]
    covered = ortho.index()
    covered = covered[covered["path"].notna() & (covered["piksel"] <= 0.07)]
    cov_tree = STRtree(list(covered.geometry))
    objs = [o for o in objs if len(cov_tree.query(o["geom"].centroid, predicate="intersects"))]
    tree = STRtree([o["geom"] for o in objs])
    print(f"obiekty OSM w pobranych arkuszach ≤7 cm: {len(objs)}")

    predictor = load_sam(f"cuda:{args.gpu}")
    claimed: set[str] = set()  # obiekt już widoczny w środku innego kafelka
    half = TILE_PX * TILE_RES / 2
    n = 0
    with rec_path.open("a") as rec, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for o in tqdm(objs):
            if o["id"] in claimed:
                continue
            c = o["geom"].centroid
            bounds = (c.x - half, c.y - half, c.x + half, c.y + half)
            tile_id = o["id"]
            inner = box(*bounds).buffer(-half / 4)
            in_tile = [objs[i] for i in tree.query(box(*bounds), predicate="intersects")]
            claimed.update(x["id"] for x in in_tile if inner.contains(x["geom"].centroid))
            if tile_id in done:
                continue
            crop = ortho.crop_bounds(bounds, res=TILE_RES, max_pixel=0.07)
            if crop is None or crop[2]["pokrycie"] < 0.95:
                continue
            rgb, transform, meta = crop
            prob = None
            if verifier and (args.verify_all or any(
                    isinstance(x["geom"], Point) and x["edited"] > meta["data_nalotu"] for x in in_tile)):
                prob = verifier.prob(rgb, TILE_RES, tta=True)
            mask, report = segment_tile(predictor, rgb, transform, in_tile, meta["data_nalotu"], prob,
                                        verify_all=args.verify_all)
            cv2.imwrite(str(args.out / "images" / f"{tile_id}.jpg"), rgb[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(args.out / "masks" / f"{tile_id}.png"), mask)
            rec.write(json.dumps({"tile": tile_id, "bounds": bounds, "res": TILE_RES, **meta,
                                  "pv_frac": round(float((mask == 1).mean()), 4), "objects": report}) + "\n")
            rec.flush()
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"zapisano {n} kafelków -> {args.out}")


if __name__ == "__main__":
    main()
