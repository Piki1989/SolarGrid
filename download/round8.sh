#!/usr/bin/env bash
# Runda 8: trening na wszystkich danych po zakończeniu pobierania (polecenie 29.10.2026).
# Przepis all_v1 (ConvNeXt-Base wygrał walidację z Large: 0,804 vs 0,790), ale dane z całej
# pobranej Małopolski: nowe kafelki OSM z rocznika 2024 5 cm oraz kafelki 25 cm (dachy dużych
# budynków + losowe tło z całego regionu) z etykietami nauczyciela.
#   setsid nohup bash download/round8.sh > runs/round8.log 2>&1 < /dev/null &
set -euo pipefail
cd ~/solargrid
PY=.venv/bin/python
LBL=~/datasets/malopolska_pv/labels
BD=~/datasets/bdappv/bdappv
ORTHO=~/datasets/malopolska_pv/ortho
T=runs/bdappv_v1/best.pt

echo "[$(date)] czekam na koniec pobierania"
while pgrep -f download/fetch_ortho.sh > /dev/null; do sleep 300; done
# Druga przebieżka dociąga pliki, na których wget poddał się po 10 próbach.
echo "[$(date)] dociąganie brakujących plików"
JOBS=12 bash download/fetch_ortho.sh $ORTHO download/lists/*_RGB.txt || true

done_pseudo() { [ -s $2/records.jsonl ] && [ $(wc -l < $2/records.jsonl) -eq $(wc -l < $1/records.jsonl) ]; }

echo "[$(date)] kafelki OSM + SAM 2 (r3 + nowe arkusze)"
# Kopia r3 (twarde dowiązania), labels wznawia i dorabia tylko kafelki z nowych arkuszy.
[ -d $LBL/osm_sam_r8 ] || cp -al $LBL/osm_sam_r3 $LBL/osm_sam_r8
$PY -m solargrid.labels --out $LBL/osm_sam_r8 --gpu 0 --verify runs/mp_v1/best.pt --verify-all
$PY -m solargrid.review $LBL/osm_sam_r8

echo "[$(date)] kafelki 25 cm: duże dachy i losowe tło"
[ -s $LBL/roofs_025/records.jsonl ] || \
  $PY -m solargrid.labels --negatives 3000 --neg-res 0.25 --roofs-min-m2 300 --workers 128 --out $LBL/roofs_025
[ -s $LBL/neg_025/records.jsonl ] || \
  $PY -m solargrid.labels --negatives 3000 --neg-res 0.25 --workers 128 --out $LBL/neg_025

echo "[$(date)] etykiety nauczyciela"
done_pseudo $LBL/osm_sam_r8 $LBL/teacher_r8 || \
  $PY -m solargrid.pseudo --teacher $T --src $LBL/osm_sam_r8 --out $LBL/teacher_r8 --gpu 0 --no-osm-ignore
for s in roofs_025 neg_025; do
  done_pseudo $LBL/$s $LBL/${s}_t || \
    $PY -m solargrid.pseudo --teacher $T --src $LBL/$s --out $LBL/${s}_t --gpu 0 --no-osm-ignore
done

echo "[$(date)] trening all_v2"
.venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
  --init $T \
  --src $BD/google:0.1 --src $BD/ign:0.2 \
  --src $LBL/teacher_r8:0.05 --src $LBL/roofs_005_t:0.05 --src $LBL/neg_005_t:0.05 \
  --src $LBL/roofs_025_t:0.25 --src $LBL/neg_025_t:0.25 \
  --out runs/all_v2 --epochs 20 --lr 1e-4 --bs 8 --workers 12 --val-frac 0.1

echo "[$(date)] ocena all_v2 vs all_v1"
$PY -m solargrid.rating build --models all_v1 all_v2 --n 100 \
  --out $LBL/ocena_v4 2>&1 | grep -v -i warn | tail -1
echo "[$(date)] runda 8 gotowa: runs/all_v2/best.pt"
