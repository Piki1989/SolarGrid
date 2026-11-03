#!/usr/bin/env bash
# Runda 3: jak runda 2, ale etykiety z punktów ograniczone do połowy obrysu budynku
# (ROOF_MAX_FRAC) — mp_v2 nauczył się zaznaczać całe dachy. Negatywy z rundy 2.
#   tmux new -d -s round3 "bash download/round3.sh 2>&1 | tee -a runs/round3.log"
set -euo pipefail
cd ~/solargrid
PY=.venv/bin/python
LBL=~/datasets/malopolska_pv/labels

echo "[$(date)] etykiety OSM + SAM 2 + limit obrysu + weryfikacja mp_v1"
$PY -m solargrid.labels --out $LBL/osm_sam_r3 --gpu 0 --verify runs/mp_v1/best.pt --verify-all
$PY -m solargrid.review $LBL/osm_sam_r3

echo "[$(date)] trening mp_v3"
.venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
  --init runs/bdappv_v1/best.pt \
  --src $LBL/osm_sam_r3:0.05 --src $LBL/neg_005_r2:0.05 \
  --out runs/mp_v3 --epochs 30 --lr 1e-4 --bs 8 --workers 12 --val-frac 0.15
echo "[$(date)] runda 3 gotowa: runs/mp_v3/best.pt"
