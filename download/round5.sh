#!/usr/bin/env bash
# Runda 5: pseudoetykiety z modelu BDAPPV (precyzyjne kształty paneli) na kafelkach
# Różnica względem rundy 4: bez stref ignoruj wokół obiektów OSM (--no-osm-ignore).
# z Małopolski + część BDAPPV w treningu, żeby model nie zaczął rysować całych połaci.
#   tmux new -d -s round5 "bash download/round5.sh 2>&1 | tee -a runs/round5.log"
set -euo pipefail
cd ~/solargrid
PY=.venv/bin/python
LBL=~/datasets/malopolska_pv/labels
BD=~/datasets/bdappv/bdappv

echo "[$(date)] pseudoetykiety nauczyciela bdappv_v1"
[ -s $LBL/teacher_r5/records.jsonl ] && [ $(wc -l < $LBL/teacher_r5/records.jsonl) -eq $(wc -l < $LBL/osm_sam_r3/records.jsonl) ] || \
$PY -m solargrid.pseudo --teacher runs/bdappv_v1/best.pt --src $LBL/osm_sam_r3 --out $LBL/teacher_r5 --gpu 0 --no-osm-ignore

# Podzbiór BDAPPV (dowiązania), żeby nie zdominował ~5 tys. kafelków z Małopolski.
for s in google:2500 ign:1500; do
  name=${s%%:*}; n=${s##*:}; dst=$BD/subset_$name
  if [ ! -d $dst ] || [ $(ls $dst/masks | wc -l) -ne $n ]; then
    rm -rf $dst; mkdir -p $dst/images $dst/masks
    # shuf -n zamiast | head: head ucina potok i przy pipefail skrypt kończy się błędem.
    ls $BD/$name/masks | shuf -n $n --random-source=<(yes) | while read m; do
      ln -s $BD/$name/masks/$m $dst/masks/$m
      ln -s $BD/$name/images/${m%.png}.png $dst/images/${m%.png}.png
    done
  fi
done

echo "[$(date)] trening mp_v5"
.venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
  --init runs/bdappv_v1/best.pt \
  --src $LBL/teacher_r5:0.05 --src $LBL/neg_005_r2:0.05 \
  --src $BD/subset_google:0.1 --src $BD/subset_ign:0.2 \
  --out runs/mp_v5 --epochs 30 --lr 1e-4 --bs 8 --workers 12 --val-frac 0.15
echo "[$(date)] runda 4 gotowa: runs/mp_v5/best.pt"
