#!/usr/bin/env bash
# Runda 1b: etykiety z weryfikacją punktów edytowanych w OSM po nalocie (model BDAPPV),
# negatywy równolegle na drugim GPU, potem dostrajanie na Małopolsce.
#   tmux new -d -s round1 "bash download/round1b.sh 2>&1 | tee -a runs/round1.log"
set -euo pipefail
cd ~/solargrid
PY=.venv/bin/python
LBL=~/datasets/malopolska_pv/labels

echo "[$(date)] etykiety OSM + SAM 2 + weryfikacja"
CUDA_VISIBLE_DEVICES=1 $PY -m solargrid.labels --negatives 3000 --neg-res 0.05 --out $LBL/neg_005 &
NEG=$!
$PY -m solargrid.labels --out $LBL/osm_sam_v2 --gpu 0 --verify runs/bdappv_v1/best.pt
$PY -m solargrid.review $LBL/osm_sam_v2
wait $NEG

echo "[$(date)] dostrajanie na Małopolsce"
.venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
  --init runs/bdappv_v1/best.pt \
  --src $LBL/osm_sam_v2:0.05 --src $LBL/neg_005:0.05 \
  --out runs/mp_v1 --epochs 30 --lr 1e-4 --bs 8 --workers 12 --val-frac 0.15
echo "[$(date)] runda 1 gotowa: runs/mp_v1/best.pt"
