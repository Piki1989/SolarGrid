# SolarGrid — wykrywanie PV na dachach (Małopolska)

Kod lokalnie, dane i trening na `thebrain.kis.agh.edu.pl` (`~/solargrid`, dane w `~/datasets`).
Synchronizacja kodu: `./sync.sh` (rsync bez danych, `uv.lock` trzymany lokalnie).

## Dane
- Ortofoto GUGiK: `download/fetch_ortho.sh` (tmux `ortho`) -> `~/datasets/malopolska_pv/ortho/<rok>_<px>_RGB/`
- Metadane (skorowidze WFS, granice PRG, OSM PV): `~/datasets/malopolska_pv/meta/`
- BDAPPV (Francja, maski PV): `~/datasets/bdappv/bdappv/{google,ign}`

## Potok
| krok | polecenie |
|---|---|
| etykiety z OSM + SAM 2 | `python -m solargrid.labels --gpu 0` |
| podgląd etykiet | `python -m solargrid.review ~/datasets/malopolska_pv/labels/osm_sam` |
| trening (2× GPU) | `torchrun --nproc_per_node 2 -m solargrid.train --src <katalog>:<gsd> ... --out runs/<nazwa>` |
| moc dla adresu | `python -m solargrid.estimate "Kraków, Floriańska 3" --ckpt runs/<nazwa>/best.pt` |
| cały region | `python -m solargrid.detect --ckpt ... --shard 0 --nshards 2` (+ `--merge`) |
| mapa | `SOLARGRID_CKPT=runs/<nazwa>/best.pt uvicorn solargrid.app:app --port 8000` |

Mapa z laptopa: `ssh -N -L 8000:localhost:8000 mpikus@thebrain.kis.agh.edu.pl` -> http://localhost:8000

## Uwagi
- NCCL na thebrain: wtyczka RoCE powoduje SIGSEGV — `train.py` ustawia `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=lo`.
- NMPT z WCS jako `image/tiff` jest kwantowany do 1 m — używamy `image/x-aaigrid`.
