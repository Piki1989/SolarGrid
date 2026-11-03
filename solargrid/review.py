"""Galeria kontroli jakości etykiet: nakładki maski na zdjęcie + statusy obiektów OSM.

    python -m solargrid.review ~/datasets/malopolska_pv/labels/osm_sam
    -> <katalog>/review/index.html (serwowane przez app pod /review/<nazwa>/)
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np

THUMB = 384


def overlay(img: np.ndarray, label: np.ndarray) -> np.ndarray:
    out = img.copy()
    pv, ign = label == 1, label == 255
    out[pv] = (0.45 * out[pv] + 0.55 * np.array([48, 59, 255])).astype(np.uint8)  # BGR czerwony
    out[ign] = (0.6 * out[ign] + 0.4 * np.array([0, 200, 255])).astype(np.uint8)  # żółty
    cs, _ = cv2.findContours(pv.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args()
    out = args.root / "review"
    out.mkdir(exist_ok=True)
    recs = [json.loads(line) for line in (args.root / "records.jsonl").open()][: args.limit]
    cards = []
    for r in recs:
        t = r["tile"]
        img = cv2.imread(str(args.root / "images" / f"{t}.jpg"))
        lab = cv2.imread(str(args.root / "masks" / f"{t}.png"), cv2.IMREAD_GRAYSCALE)
        both = np.hstack([cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA),
                          cv2.resize(overlay(img, lab), (THUMB, THUMB), interpolation=cv2.INTER_AREA)])
        cv2.imwrite(str(out / f"{t}.jpg"), both, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ok = sum(o["ok"] for o in r["objects"])
        bad = [f'{o["id"]}:{o.get("why")}' for o in r["objects"] if not o["ok"]]
        cards.append(f'<figure data-bad="{int(bool(bad))}"><img loading="lazy" src="{t}.jpg">'
                     f'<figcaption><b>{t}</b> · {r["data_nalotu"]} · OK {ok}/{len(r["objects"])}'
                     f'{" · " + html.escape(", ".join(bad)) if bad else ""}</figcaption></figure>')
    (out / "index.html").write_text(f"""<!doctype html><html lang="pl"><meta charset="utf-8">
<title>Kontrola etykiet</title><style>
body{{font:13px system-ui;margin:16px;background:#111;color:#ddd}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(780px,1fr));gap:12px}}
figure{{margin:0;background:#222;padding:6px;border-radius:6px}} img{{width:100%}}
figure[data-bad="1"] figcaption{{color:#fc6}}
</style><h1>{html.escape(str(args.root.name))}: {len(recs)} kafelków</h1>
<p>Czerwony = PV (etykieta), żółty = ignoruj (OSM zna PV, SAM nie dał pewnej maski).</p>
<main>{''.join(cards)}</main></html>""")
    print(f"{len(recs)} kafelków -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
