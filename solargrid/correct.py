"""Zbiór do ręcznej poprawy masek PV (strona /popraw w app.py).

    python -m solargrid.correct plan --domain fine --random 3000 \
        --exclude ~/datasets/malopolska_pv/labels/popraw_v1 --out ~/datasets/malopolska_pv/labels/popraw_v2
    python -m solargrid.correct build --models all_v1 all_v2 --out ~/datasets/malopolska_pv/labels/popraw_v2 \
        --shard 0 --nshards 16 --device cuda:0   # … po jednym procesie na shard (download/popraw_v2.sh)
    python -m solargrid.correct finalize ~/datasets/malopolska_pv/labels/popraw_v2 --test 500 --scratch 150
    python -m solargrid.correct summary ~/datasets/malopolska_pv/labels/popraw_v2

Kafelki leżą na stałej siatce 51,2 m (EPSG:2180), więc się nie nakładają: kafelek testowy
nigdy nie dzieli pikseli z treningowym (między zestawami pilnuje tego --exclude). Dwie domeny:
fine (arkusze ≤7 cm, kafelek 1024 px po 5 cm) i coarse (tylko 25 cm, kafelek 205 px natywnie)
— osobne katalogi, bo solargrid.train bierze jedno gsd na źródło. W kafelku zaznaczamy całe PV, nie tylko na jednym
domu. Kandydaci: komórki z obiektami PV z OSM lub z detekcji całego regionu
(detections/pv.gpkg) oraz losowe komórki z budynkami (fałszywe wykrycia).

Katalog to od razu źródło dla solargrid.train (images/ + masks/, maska 0 tło / 1 PV /
255 ignoruj). masks/ zawiera tylko kafelki zatwierdzone przez człowieka; propozycja modelu
leży w auto/. Kolejność w items.json: najpierw test (losowo, żeby był bezstronny), potem
trening od kafelków, na których modele są najmniej pewne lub się nie zgadzają.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.prepared import prep

from . import buildings, labels, ortho
from .predict import Segmenter

CELL_M = 51.2
IGNORE = 255
DETECTIONS = ortho.DATA / "detections" / "pv.gpkg"


def cell_of(x: float, y: float) -> tuple[int, int]:
    return int(x // CELL_M), int(y // CELL_M)


def cell_bounds(c: tuple[int, int]) -> tuple[float, float, float, float]:
    return c[0] * CELL_M, c[1] * CELL_M, (c[0] + 1) * CELL_M, (c[1] + 1) * CELL_M


DOMAINS = {  # max piksel arkusza, rozdzielczość kafelka (25 cm natywnie: trening ma widzieć to co detekcja)
    "fine": (0.07, 0.05),
    "coarse": (0.25, 0.25),
}
BIG_KWP = 50.0


def covered(domain: str):
    """Obszar z arkuszami danej domeny (coarse = tylko tam, gdzie nie ma ≤7 cm)."""
    idx = ortho.index()
    idx = idx[idx["path"].notna()]
    fine = idx[idx["piksel"] <= 0.07].union_all()
    return fine if domain == "fine" else idx.union_all().difference(fine)


def building_kwp(det: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Budynki z wykrytym PV i sumaryczną mocą wykryć (punkt wykrycia wewnątrz obrysu)."""
    bld = buildings._gdf()
    pts = det[["geometry", "moc_kwp"]].assign(geometry=det.geometry.representative_point())
    j = gpd.sjoin(pts, bld[["geometry"]], predicate="within")
    kwp = j.groupby("index_right")["moc_kwp"].sum()
    return bld.loc[kwp.index, ["geometry"]].assign(kwp=kwp.values)


def candidates(domain: str, n_random: int, exclude: set, seed: int = 0) -> list[tuple[str, tuple[int, int], float]]:
    """Komórki siatki: (rodzaj, komórka, priorytet = moc największego budynku z PV w komórce [kWp]).

    PV z detekcji całego regionu (detections/pv.gpkg), a bez niej z OSM. Komórki dużych
    obiektów (≥ BIG_KWP) bierzemy w całości, także fragmenty bez wykrycia. Losowe komórki
    z budynkami bez wykrytego PV łapią przeoczenia i dają negatywy."""
    rng = random.Random(seed)
    cov = covered(domain)
    prio = collections.defaultdict(float)
    if DETECTIONS.exists():
        det = gpd.read_file(DETECTIONS).to_crs(2180)
        det = det[(det["piksel_m"] <= 0.07) == (domain == "fine")]
        kind = "detekcja"
        bk = building_kwp(det)
        for g, kwp in zip(bk.geometry, bk["kwp"]):
            x0, y0, x1, y1 = g.bounds
            for c in [(i, j) for i in range(int(x0 // CELL_M), int(x1 // CELL_M) + 1)
                      for j in range(int(y0 // CELL_M), int(y1 // CELL_M) + 1)]:
                if g.intersects(box(*cell_bounds(c))):
                    prio[c] = max(prio[c], kwp)
        pts = det.geometry.representative_point()
    else:
        kind, pts = "osm_pv", [o["geom"].centroid for o in labels.load_osm()]
    pv_cells = {cell_of(p.x, p.y) for p in pts} | {c for c, v in prio.items() if v >= BIG_KWP}
    inside = prep(cov)
    pv_cells = sorted(c for c in pv_cells - exclude if inside.intersects(box(*cell_bounds(c))))
    out = [(kind, c, prio.get(c, 0.0)) for c in pv_cells]
    taken = set(pv_cells) | exclude
    bld = buildings._gdf()
    bld = bld.iloc[bld.sindex.query(cov, predicate="intersects")]
    bld = bld[bld.area >= 30]
    n = 0
    for i in rng.sample(range(len(bld)), min(len(bld), n_random * 3)):
        if n >= n_random:
            break
        p = bld.geometry.iloc[i].centroid
        c = cell_of(p.x, p.y)
        if c not in taken:
            taken.add(c)
            out.append(("losowy", c, 0.0))
            n += 1
    return out


def score(probs: list[np.ndarray]) -> float:
    """Ile pracy prawdopodobnie wymaga kafelek (0–2): udział niepewnych pikseli + niezgodność modeli."""
    mean = np.mean(probs, axis=0)
    pv = (mean > 0.5).sum()
    unsure = ((mean > 0.2) & (mean < 0.8)).sum()
    masks = [p > 0.5 for p in probs]
    dis = 0.0
    for a in masks[1:]:
        union = (a | masks[0]).sum()
        dis = max(dis, 1 - (a & masks[0]).sum() / union if union else 0.0)
    return round(float(unsure / max(pv + unsure, 100) + dis), 4)


def cell_ids(out: Path) -> set:
    items = json.loads((out / "items.json").read_text())
    return {tuple(map(int, it["id"][1:].split("_"))) for it in items}


def plan(domain: str, n_random: int, exclude: list[Path], out: Path):
    """Kandydaci raz dla wszystkich części -> out/plan.json."""
    out.mkdir(parents=True, exist_ok=True)
    skip = set().union(*[cell_ids(e) for e in exclude])
    cands = candidates(domain, n_random, skip)
    (out / "plan.json").write_text(json.dumps({"domain": domain, "cells": cands}))
    print(f"{domain}: {len(cands)} komórek ({sum(p >= BIG_KWP for _, _, p in cands)} dużych obiektów) -> {out}")


def build(models: list[str], out: Path, device: str, shard: int = 0, nshards: int = 1):
    """Kafelki jednej części (shard) planu -> out/parts/items_<shard>.json; potem finalize."""
    for d in ("images", "auto", "masks", "parts"):
        (out / d).mkdir(parents=True, exist_ok=True)
    p = json.loads((out / "plan.json").read_text())
    max_pixel, res = DOMAINS[p["domain"]]
    todo = [(kind, tuple(c), prio) for kind, c, prio in p["cells"]][shard::nshards]
    segs = [Segmenter(Path("runs") / m / "best.pt", device) for m in models]
    items = []
    for k, (kind, c, prio) in enumerate(todo):
        bounds = cell_bounds(c)
        crop = ortho.crop_bounds(bounds, res=res, max_pixel=max_pixel)
        if crop is None or crop[2]["pokrycie"] < 0.98:
            continue
        rgb, tr, meta = crop
        probs = [s.prob(rgb, res) for s in segs]
        auto = (probs[0] > 0.5).astype(np.uint8)  # propozycja = główny model; reszta tylko do niepewności
        tile = f"c{c[0]}_{c[1]}"
        cv2.imwrite(str(out / "images" / f"{tile}.jpg"), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "auto" / f"{tile}.png"), auto)
        bld = [[[(x - bounds[0]) / res, (bounds[3] - y) / res] for x, y in g.exterior.coords]
               for g in buildings.in_bounds(bounds) if g.geom_type == "Polygon"]
        items.append({"id": tile, "kind": kind, "bounds": bounds, "res": res, **meta,
                      "auto_m2": round(float(auto.sum() * res * res), 1), "score": score(probs),
                      "prio_kwp": round(prio, 1), "models": models, "budynki": bld})
        if k % 500 == 0:
            print(f"[{shard}] {k}/{len(todo)}", flush=True)
    (out / "parts" / f"items_{shard}.json").write_text(json.dumps(items))
    print(f"[{shard}] {len(items)} kafelków")


def finalize(out: Path, n_test: int, n_scratch: int):
    """Scala części: test losowy (część od zera, bez podpowiedzi modelu — żeby zmierzyć, ile
    poprawianie zawyża zgodność z modelem), trening od dużych obiektów, potem od niepewnych."""
    items = [it for p in sorted((out / "parts").glob("items_*.json")) for it in json.loads(p.read_text())]
    items.sort(key=lambda it: it["id"])
    rng = random.Random(2)
    rng.shuffle(items)
    for k, it in enumerate(items):
        it["split"] = "test" if k < n_test else "train"
        it["start"] = "empty" if k < n_scratch else "auto"
    test = items[:n_test]
    train = sorted(items[n_test:], key=lambda it: (-(it.get("prio_kwp", 0) >= BIG_KWP), -it.get("prio_kwp", 0), -it["score"]))
    (out / "items.json").write_text(json.dumps(test + train))
    big = sum(it.get("prio_kwp", 0) >= BIG_KWP for it in train)
    print(f"{len(items)} kafelków ({len(test)} test, {n_scratch} od zera, {big} dużych obiektów na początku) -> {out}")


def log(out: Path) -> dict[str, dict]:
    """Ostatni wpis dla każdego kafelka."""
    path = out / "log.jsonl"
    last = {}
    for line in path.open() if path.exists() else []:
        r = json.loads(line)
        last[r["id"]] = r
    return last


def summary(out: Path):
    items = json.loads((out / "items.json").read_text())
    last = log(out)
    by = collections.defaultdict(collections.Counter)
    secs = collections.defaultdict(list)
    for it in items:
        r = last.get(it["id"])
        status = r["status"] if r else "czeka"
        by[it["split"]][status] += 1
        if r:
            secs[status].append(r.get("sec", 0))
    for split, c in by.items():
        print(f"[{split}] " + ", ".join(f"{k}: {v}" for k, v in c.most_common()))
    for status, s in secs.items():
        print(f"  {status}: mediana {np.median(s):.0f} s na kafelek")
    # Zgodność propozycji modelu z poprawioną maską (tylko kafelki startujące od propozycji).
    ious = []
    for it in items:
        r = last.get(it["id"])
        if not r or r["status"] not in ("ok", "poprawione") or it["start"] != "auto":
            continue
        a = cv2.imread(str(out / "auto" / f"{it['id']}.png"), cv2.IMREAD_GRAYSCALE) == 1
        m = cv2.imread(str(out / "masks" / f"{it['id']}.png"), cv2.IMREAD_GRAYSCALE)
        valid = m != IGNORE
        union = ((a | (m == 1)) & valid).sum()
        if union:
            ious.append((a & (m == 1) & valid).sum() / union)
    if ious:
        print(f"IoU propozycji modelu vs poprawka: średnio {np.mean(ious):.3f} na {len(ious)} kafelkach z PV")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--domain", choices=DOMAINS, default="fine", help="fine: arkusze ≤7 cm, coarse: tylko 25 cm")
    pl.add_argument("--random", type=int, default=1000, help="losowe komórki z budynkami")
    pl.add_argument("--exclude", type=Path, nargs="*", default=[], help="zestawy, których komórek nie powtarzamy")
    pl.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--models", nargs="+", required=True, help="pierwszy = propozycja; kolejne = niezgodność")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--shard", type=int, default=0)
    b.add_argument("--nshards", type=int, default=1)
    f = sub.add_parser("finalize")
    f.add_argument("out", type=Path)
    f.add_argument("--test", type=int, default=500)
    f.add_argument("--scratch", type=int, default=150, help="ile kafelków testu oznaczać od zera")
    s = sub.add_parser("summary")
    s.add_argument("out", type=Path)
    args = ap.parse_args()
    if args.cmd == "plan":
        plan(args.domain, args.random, args.exclude, args.out)
    elif args.cmd == "build":
        build(args.models, args.out, args.device, args.shard, args.nshards)
    elif args.cmd == "finalize":
        finalize(args.out, args.test, args.scratch)
    else:
        summary(args.out)


if __name__ == "__main__":
    main()
