#!/usr/bin/env bash
# Runda 6: jeden model na wszystkich danych. Pełny BDAPPV + Małopolska z etykietami
# nauczyciela (bdappv_v1), w tym kafelki na dużych budynkach, gdzie dach bez paneli = tło
# (wcześniejsze modele mp_v* zalewały duże dachy plamą — ocena 100 domów, 29.10.2026).
#   tmux new -d -s round6 "bash download/round6.sh 2>&1 | tee -a runs/round6.log"
set -euo pipefail
cd ~/solargrid
PY=.venv/bin/python
LBL=~/datasets/malopolska_pv/labels
BD=~/datasets/bdappv/bdappv
T=runs/bdappv_v1/best.pt

done_pseudo() { [ -s $2/records.jsonl ] && [ $(wc -l < $2/records.jsonl) -eq $(wc -l < $1/records.jsonl) ]; }

echo "[$(date)] kafelki na dużych budynkach"
[ -s $LBL/roofs_005/records.jsonl ] || \
  $PY -m solargrid.labels --negatives 3000 --neg-res 0.05 --roofs-min-m2 300 --workers 128 --out $LBL/roofs_005
echo "[$(date)] etykiety nauczyciela: duże dachy i losowe tło"
done_pseudo $LBL/roofs_005 $LBL/roofs_005_t || \
  $PY -m solargrid.pseudo --teacher $T --src $LBL/roofs_005 --out $LBL/roofs_005_t --gpu 0 --no-osm-ignore
done_pseudo $LBL/neg_005_r2 $LBL/neg_005_t || \
  $PY -m solargrid.pseudo --teacher $T --src $LBL/neg_005_r2 --out $LBL/neg_005_t --gpu 0 --no-osm-ignore

echo "[$(date)] trening all_v1"
.venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
  --init $T \
  --src $BD/google:0.1 --src $BD/ign:0.2 \
  --src $LBL/teacher_r5:0.05 --src $LBL/roofs_005_t:0.05 --src $LBL/neg_005_t:0.05 \
  --out runs/all_v1 --epochs 20 --lr 1e-4 --bs 8 --workers 12 --val-frac 0.1
echo "[$(date)] runda 6 gotowa: runs/all_v1/best.pt"
