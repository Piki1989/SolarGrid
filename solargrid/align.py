"""Dopasowanie obrysu budynku (EGiB/OSM) do dachu na ortofoto.

Obrys to ściany przy gruncie, a ortofoto nie jest „true ortho”: dach wysokiego budynku jest
przesunięty radialnie o wysokość × tg kąta widzenia (do kilku m), a okapy wystają ~0.3–0.8 m.
Szukamy przesunięcia i poszerzenia obrysu, przy którym jego krawędź leży na najsilniejszych
krawędziach zdjęcia (gradient jasności, prostopadły do krawędzi obrysu). Same krawędzie potrafią
przeciągnąć obrys na sąsiedni budynek albo krawędź cienia, więc kilku najlepszych kandydatów
sprawdza SAM 2: podpowiedź = prostokąt kandydata, wygrywa ten, którego obrys najlepiej pokrywa
się z maską dachu od SAM.
"""
from __future__ import annotations

import cv2
import numpy as np
import rasterio.features
from shapely import affinity

MAX_SHIFT_M = 4.0
EAVES_M = (0.0, 0.4, 0.8)  # poszerzenie obrysu o okap
SHIFT_PENALTY = 0.03  # na metr: przy równych krawędziach wolimy mniejsze przesunięcie
MIN_GAIN = 1.10  # wynik musi być o 10% lepszy niż obrys bez zmian, inaczej zostawiamy EGiB
SAMPLE_PX = 1.0  # co ile pikseli próbkujemy krawędź
TOP_K = 6  # kandydaci do sprawdzenia przez SAM
MIN_SEP_M = 0.75  # kandydaci różnią się przesunięciem co najmniej o tyle
MIN_SAM_IOU = 0.6  # gorsze dopasowanie do maski SAM = zostaje obrys EGiB


def _edges(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    s = np.percentile(np.hypot(gx, gy), 99) + 1e-6
    return gx / s, gy / s


def _boundary(geom, inv) -> tuple[np.ndarray, np.ndarray]:
    """Punkty na krawędzi obrysu (piksele) i normalne do krawędzi."""
    pts, nrm = [], []
    for p in getattr(geom, "geoms", [geom]):
        xy = np.array([inv * c for c in p.exterior.coords])
        for a, b in zip(xy[:-1], xy[1:]):
            d = b - a
            n = max(int(np.hypot(*d) / SAMPLE_PX), 1)
            t = (np.arange(n) + 0.5) / n
            pts.append(a + t[:, None] * d)
            nrm.append(np.repeat([[-d[1], d[0]] / (np.hypot(*d) + 1e-9)], n, axis=0))
    return np.concatenate(pts), np.concatenate(nrm)


def _scores(gx, gy, pts, nrm, shifts) -> np.ndarray:
    """Średnia |gradient · normalna| na krawędzi dla każdego przesunięcia (piksele)."""
    h, w = gx.shape
    out = np.empty(len(shifts))
    for i, (sx, sy) in enumerate(shifts):
        x = np.clip(np.round(pts[:, 0] + sx).astype(int), 0, w - 1)
        y = np.clip(np.round(pts[:, 1] + sy).astype(int), 0, h - 1)
        out[i] = np.abs(gx[y, x] * nrm[:, 0] + gy[y, x] * nrm[:, 1]).mean()
    return out


def _candidates(gx, gy, building, tr, max_shift_m):
    """Najlepsze (wynik, sx, sy, okap) z przeszukiwania krawędzi, różniące się przesunięciem."""
    px = abs(tr.a)
    inv = ~tr
    r = int(max_shift_m / px)
    coarse = max(1, int(0.25 / px))  # siatka co 25 cm, potem doprecyzowanie co piksel
    base = _scores(gx, gy, *_boundary(building, inv), [(0, 0)])[0]
    found = []
    for eave in EAVES_M:
        g = building.buffer(eave, join_style="mitre") if eave else building
        pts, nrm = _boundary(g, inv)
        grid = np.array([(sx, sy) for sx in range(-r, r + 1, coarse) for sy in range(-r, r + 1, coarse)])
        sc = _scores(gx, gy, pts, nrm, grid) - SHIFT_PENALTY * np.hypot(*grid.T) * px * base
        picked = []
        for k in np.argsort(-sc):
            if len(picked) == TOP_K:
                break
            if all(np.hypot(*(grid[k] - grid[j])) * px >= MIN_SEP_M for j in picked):
                picked.append(k)
        for k in picked:
            cx, cy = grid[k]
            fine = np.array([(cx + dx, cy + dy) for dx in range(-coarse, coarse + 1) for dy in range(-coarse, coarse + 1)])
            sf = _scores(gx, gy, pts, nrm, fine) - SHIFT_PENALTY * np.hypot(*fine.T) * px * base
            j = int(sf.argmax())
            found.append((float(sf[j]), int(fine[j][0]), int(fine[j][1]), eave))
    found.sort(reverse=True)
    return base, found[:TOP_K]


def _place(building, tr, sx, sy, eave):
    g = building.buffer(eave, join_style="mitre") if eave else building
    return affinity.translate(g, sx * tr.a, sy * tr.e)  # piksele -> metry (tr.e < 0)


def _sam_iou(predictor, geom, tr, shape) -> float:
    poly = rasterio.features.geometry_mask([geom], out_shape=shape, transform=tr, invert=True)
    ys, xs = np.nonzero(poly)
    if not len(xs):
        return 0.0
    masks, _, _ = predictor.predict(box=np.array([xs.min(), ys.min(), xs.max(), ys.max()]), multimask_output=False)
    m = masks[0] > 0
    return float((m & poly).sum() / max((m | poly).sum(), 1))


def align(rgb: np.ndarray, tr, building, predictor=None, max_shift_m: float = MAX_SHIFT_M):
    """Zwraca (obrys dopasowany do dachu, info). Przy braku wyraźnego dopasowania — obrys EGiB.

    predictor: SAM2ImagePredictor (labels.load_sam) — bez niego tylko krawędzie.
    """
    gx, gy = _edges(rgb)
    base, cands = _candidates(gx, gy, building, tr, max_shift_m)
    cands = [c for c in cands if c[0] >= MIN_GAIN * base]
    iou = None
    if predictor is not None and cands:
        predictor.set_image(rgb)
        ious = [_sam_iou(predictor, _place(building, tr, *c[1:]), tr, rgb.shape[:2]) for c in cands]
        k = int(np.argmax(ious))
        iou0 = _sam_iou(predictor, building, tr, rgb.shape[:2])
        if ious[k] < max(MIN_SAM_IOU, iou0):
            cands = []
        else:
            cands, iou = [cands[k]], ious[k]
    if not cands:
        return building, {"dopasowano": False, "przesuniecie_m": [0.0, 0.0], "okap_m": 0.0}
    score, sx, sy, eave = cands[0]
    info = {"dopasowano": True, "przesuniecie_m": [round(sx * tr.a, 2), round(sy * tr.e, 2)],
            "okap_m": eave, "wynik": round(score / max(base, 1e-6), 2)}
    if iou is not None:
        info["sam_iou"] = round(iou, 2)
    return _place(building, tr, sx, sy, eave), info
