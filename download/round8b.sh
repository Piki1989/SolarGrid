#!/usr/bin/env bash
# Runda 8b: końcówka rundy 8 z dłuższym treningiem (60 zamiast 20 epok, polecenie 30.10.2026:
# „epok nie żałuj”). all_v1 rósł do samego końca cosinusa (0,798 → 0,805 w epokach 14–17),
# best.pt i tak bierze najlepszą epokę z walidacji. Podmienia round8.sh przed treningiem:
# czeka na trwające solargrid.pseudo, dorabia brakujące etykiety nauczyciela i trenuje.
#   setsid nohup bash download/round8b.sh > runs/round8b.log 2>&1 < /dev/null &
set -euo pipefail
cd ~/solargrid
PY=.venv/bin/python
LBL=~/datasets/malopolska_pv/labels
BD=~/datasets/bdappv/bdappv
T=runs/bdappv_v1/best.pt

done_pseudo() { [ -s $2/records.jsonl ] && [ $(wc -l < $2/records.jsonl) -eq $(wc -l < $1/records.jsonl) ]; }

echo "[$(date)] czekam na trwające etykiety nauczyciela"
while pgrep -f "solargrid.pseudo" > /dev/null; do sleep 30; done
for s in roofs_025 neg_025; do
  done_pseudo $LBL/$s $LBL/${s}_t || \
    $PY -m solargrid.pseudo --teacher $T --src $LBL/$s --out $LBL/${s}_t --gpu 0 --no-osm-ignore
done
done_pseudo $LBL/osm_sam_r8 $LBL/teacher_r8

echo "[$(date)] trening all_v2 (60 epok)"
.venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
  --init $T \
  --src $BD/google:0.1 --src $BD/ign:0.2 \
  --src $LBL/teacher_r8:0.05 --src $LBL/roofs_005_t:0.05 --src $LBL/neg_005_t:0.05 \
  --src $LBL/roofs_025_t:0.25 --src $LBL/neg_025_t:0.25 \
  --out runs/all_v2 --epochs 60 --lr 1e-4 --bs 8 --workers 12 --val-frac 0.1

echo "[$(date)] ocena all_v2 vs all_v1"
$PY -m solargrid.rating build --models all_v1 all_v2 --n 100 \
  --out $LBL/ocena_v4 2>&1 | grep -v -i warn | tail -1
echo "[$(date)] runda 8b gotowa: runs/all_v2/best.pt"
