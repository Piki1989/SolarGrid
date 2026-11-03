"""Galeria: czerwony = obrys EGiB, niebieski = dopasowany do dachu."""
import sys, time, cv2, numpy as np
from solargrid import ortho, rating, align, labels
sam = labels.load_sam("cuda:0") if "--sam" in sys.argv else None
n = int(sys.argv[1])
tiles = []
for kind, b in rating.sample_buildings(100)[:n]:
    c = ortho.crop(b, buffer_m=8)
    if c is None: continue
    rgb, tr, meta = c
    t = time.time(); g, info = align.align(rgb, tr, b, sam); dt = time.time() - t
    inv = ~tr
    img = rgb[..., ::-1].copy()
    for geom, col in ((b, (48, 59, 255)), (g, (255, 132, 10))):
        for p in getattr(geom, "geoms", [geom]):
            cv2.polylines(img, [np.array([inv * xy for xy in p.exterior.coords]).round().astype(np.int32)], True, col, max(1, img.shape[0] // 250))
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
    cv2.putText(img, f"{info['przesuniecie_m']} o{info['okap_m']} s{info.get('sam_iou')} {'OK' if info['dopasowano'] else '--'} {dt:.1f}s", (4, 16), 0, 0.42, (255, 255, 255), 1)
    tiles.append(img); print(info, f"{dt:.1f}s", flush=True)
while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite("/tmp/align_gallery.jpg", np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
