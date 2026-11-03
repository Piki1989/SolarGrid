#!/usr/bin/env bash
# Runda 1 (bez nadzoru), tylko dane z Małopolski. Model startuje z pretreningu BDAPPV.
# Uruchamiać w tmux z ~/solargrid:  tmux new -d -s round1 "bash download/round1.sh 2>&1 | tee -a runs/round1.log"
set -euo pipefail
cd ~/solargrid
PY=.venv/bin/python
L=download/lists
LBL=~/datasets/malopolska_pv/labels

wait_for() {  # wait_for <argumenty solargrid.ready>
  until $PY -m solargrid.ready "$@"; do sleep 600; done
}
wait_train() { while tmux has-session -t train 2>/dev/null; do sleep 300; done; }

wait_train
# Listy 2025 5 cm są posortowane wg liczby PV z OSM: pierwsze 78 arkuszy = 89% instalacji.
echo "[$(date)] czekam na 2026 3 cm + 78 arkuszy 2025 5 cm z największą liczbą PV"
wait_for $L/2026_03cm_RGB.txt --min 0.98
wait_for $L/2025_05cm_RGB.txt --first 78 --min 0.95

echo "[$(date)] etykiety OSM + SAM 2"
$PY -m solargrid.labels --out $LBL/osm_sam --gpu 0
$PY -m solargrid.review $LBL/osm_sam
$PY -m solargrid.labels --negatives 3000 --neg-res 0.05 --out $LBL/neg_005

echo "[$(date)] dostrajanie na Małopolsce"
.venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
  --init runs/bdappv_v1/best.pt \
  --src $LBL/osm_sam:0.05 --src $LBL/neg_005:0.05 \
  --out runs/mp_v1 --epochs 30 --lr 1e-4 --bs 8 --workers 12 --val-frac 0.15
echo "[$(date)] runda 1 gotowa: runs/mp_v1/best.pt"
