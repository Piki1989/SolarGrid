#!/usr/bin/env bash
# Runda 2: etykiety i negatywy weryfikowane modelem mp_v1 (dostrojonym na Małopolsce),
# trening od pretreningu BDAPPV na czystszych etykietach.
#   tmux new -d -s round2 "bash download/round2.sh 2>&1 | tee -a runs/round2.log"
set -euo pipefail
cd ~/solargrid
PY=.venv/bin/python
LBL=~/datasets/malopolska_pv/labels
VER=runs/mp_v1/best.pt

echo "[$(date)] etykiety OSM + SAM 2 + weryfikacja mp_v1"
CUDA_VISIBLE_DEVICES=1 $PY -m solargrid.labels --negatives 3000 --neg-res 0.05 --out $LBL/neg_005_r2 \
  --verify $VER --verify-all --gpu 0 &
NEG=$!
$PY -m solargrid.labels --out $LBL/osm_sam_r2 --gpu 0 --verify $VER --verify-all
$PY -m solargrid.review $LBL/osm_sam_r2
wait $NEG

echo "[$(date)] trening mp_v2"
.venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
  --init runs/bdappv_v1/best.pt \
  --src $LBL/osm_sam_r2:0.05 --src $LBL/neg_005_r2:0.05 \
  --out runs/mp_v2 --epochs 30 --lr 1e-4 --bs 8 --workers 12 --val-frac 0.15
echo "[$(date)] runda 2 gotowa: runs/mp_v2/best.pt"
