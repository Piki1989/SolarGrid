#!/usr/bin/env bash
# Zbiory do weryfikacji wszystkich wykryć z detekcji regionu (31.10.2026), po planach z
# `correct plan` (fine -> popraw_v2, coarse/25 cm -> popraw_v2_25). Propozycja = all_v1,
# niezgodność z all_v2 ustawia kolejkę; duże obiekty (≥50 kWp) na początku.
# Odczyt i maski idą na CPU, więc 16 procesów naprzemiennie na obu kartach.
#   setsid nohup bash download/popraw_v2.sh > runs/popraw_v2.log 2>&1 < /dev/null &
set -euo pipefail
cd ~/solargrid
L=~/datasets/malopolska_pv/labels
NSH=16
for spec in "popraw_v2 500 150" "popraw_v2_25 300 100"; do
  read -r name test scratch <<< "$spec"
  echo "[$(date)] $name"
  for s in $(seq 0 $((NSH - 1))); do
    [ -s $L/$name/parts/items_$s.json ] && continue  # wznawianie
    OMP_NUM_THREADS=4 .venv/bin/python -m solargrid.correct build --models all_v1 all_v2 --out $L/$name \
      --shard $s --nshards $NSH --device cuda:$((s % 2)) > runs/${name}_shard$s.log 2>&1 &
  done
  wait
  .venv/bin/python -m solargrid.correct finalize $L/$name --test $test --scratch $scratch
done
echo "[$(date)] gotowe"
