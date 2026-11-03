#!/usr/bin/env bash
# Zbiór do ręcznej poprawy masek (30.10.2026): wszystkie komórki z PV z OSM w arkuszach ≤7 cm
# + losowe budynki. Czeka na all_v2 (runda 8b, 60 epok na wszystkich danych); propozycje =
# średnia all_v2 i all_v1, ich niezgodność ustawia kolejkę treningową.
#   setsid nohup bash download/popraw_v1.sh > runs/popraw_v1.log 2>&1 < /dev/null &
set -euo pipefail
cd ~/solargrid
echo "[$(date)] czekam na all_v2"
while pgrep -f "download/round8b.sh" > /dev/null; do sleep 300; done
[ -s runs/all_v2/best.pt ]
echo "[$(date)] buduję popraw_v1"
.venv/bin/python -m solargrid.correct build --models all_v2 all_v1 --osm 100000 --random 1500 \
  --test 500 --scratch 150 --out ~/datasets/malopolska_pv/labels/popraw_v1 --device cuda:1 2>&1 | grep -v -i warn
echo "[$(date)] popraw_v1 gotowy"
