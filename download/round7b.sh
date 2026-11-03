#!/usr/bin/env bash
# Runda 7b: dokończenie all_large. 29.10 ok. 14:48 padł serwer tmux (bez restartu maszyny)
# i zabił trening po 15/20 epokach. train.py nie zapisuje stanu optymalizatora, więc
# ostatnie 5 epok = dostrojenie z last.pt z lr ~ tym, na którym przerwał (1e-5, cosinus do 0).
#   setsid nohup bash download/round7b.sh > runs/round7b.log 2>&1 &
set -euo pipefail
cd ~/solargrid
LBL=~/datasets/malopolska_pv/labels
BD=~/datasets/bdappv/bdappv
echo "[$(date)] dokończenie all_large"
.venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
  --encoder tu-convnext_large.fb_in22k_ft_in1k --init runs/all_large/last.pt \
  --src $BD/google:0.1 --src $BD/ign:0.2 \
  --src $LBL/teacher_r5:0.05 --src $LBL/roofs_005_t:0.05 --src $LBL/neg_005_t:0.05 \
  --out runs/all_large_b --epochs 5 --lr 1e-5 --bs 8 --workers 12 --val-frac 0.1
echo "[$(date)] ocena all_large_b vs all_v1"
.venv/bin/python -m solargrid.rating build --models all_v1 all_large_b --n 100 \
  --out $LBL/ocena_v3 2>&1 | grep -v -i warn | tail -1
echo "[$(date)] runda 7b gotowa"
