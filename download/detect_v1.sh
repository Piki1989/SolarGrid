#!/usr/bin/env bash
# Detekcja PV w całej Małopolsce modelem all_v1 (30.10.2026).
# Wąskie gardło to CPU (czytanie arkuszy, maski dachów, wielokąty; ~1 rdzeń na proces),
# a nie GPU, więc 16 shardów naprzemiennie na obu kartach (~4,5 GB RAM każdy).
# Wznawialne: ponowne uruchomienie pomija zapisane paczki (przy tym samym NSH).
#   setsid nohup bash download/detect_v1.sh > runs/detect_v1.log 2>&1 < /dev/null &
set -euo pipefail
cd ~/solargrid
CKPT=${1:-runs/all_v1/best.pt}
NSH=16
echo "[$(date)] start $CKPT, $NSH shardów"
for s in $(seq 0 $((NSH - 1))); do
  OMP_NUM_THREADS=4 .venv/bin/python -m solargrid.detect --ckpt $CKPT --shard $s --nshards $NSH \
    --device cuda:$((s % 2)) > runs/detect_v1_shard$s.log 2>&1 &
done
wait
.venv/bin/python -m solargrid.detect --merge
echo "[$(date)] gotowe"
