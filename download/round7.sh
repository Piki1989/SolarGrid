#!/usr/bin/env bash
# Runda 7: większy enkoder (ConvNeXt-Large, 196 M parametrów zamiast 88 M w Base).
# A) pretrening na BDAPPV (ten sam podział co bdappv_v1, val-frac 0.05, porównywalne IoU 0.868),
# B) dostrojenie na wszystkich danych jak all_v1 (runda 6).
#   tmux new -d -s round7 "bash download/round7.sh 2>&1 | tee -a runs/round7.log"
set -euo pipefail
cd ~/solargrid
PY=.venv/bin/python
LBL=~/datasets/malopolska_pv/labels
BD=~/datasets/bdappv/bdappv
ENC=tu-convnext_large.fb_in22k_ft_in1k

echo "[$(date)] czekam na rundę 6 i budowę oceny"
while tmux has-session -t round6 2>/dev/null || pgrep -f solargrid.rating >/dev/null; do sleep 60; done

echo "[$(date)] test pamięci GPU"
$PY - <<PYEOF
import torch, segmentation_models_pytorch as smp
m = smp.UnetPlusPlus(encoder_name="$ENC", encoder_weights=None, classes=1).cuda().to(memory_format=torch.channels_last)
x = torch.randn(8, 3, 512, 512, device="cuda").to(memory_format=torch.channels_last)
with torch.autocast("cuda", dtype=torch.bfloat16):
    m(x).float().mean().backward()
print(f"bs 8: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
PYEOF

if [ ! -s runs/bdappv_large/best.pt ] || [ $(wc -l < runs/bdappv_large/log.jsonl) -lt 25 ]; then
  echo "[$(date)] A: pretrening bdappv_large"
  .venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
    --encoder $ENC \
    --src $BD/google:0.1 --src $BD/ign:0.2 \
    --out runs/bdappv_large --epochs 25 --lr 2e-4 --bs 8 --workers 12 --val-frac 0.05
fi

echo "[$(date)] B: dostrojenie all_large"
.venv/bin/torchrun --nproc_per_node 2 -m solargrid.train \
  --encoder $ENC --init runs/bdappv_large/best.pt \
  --src $BD/google:0.1 --src $BD/ign:0.2 \
  --src $LBL/teacher_r5:0.05 --src $LBL/roofs_005_t:0.05 --src $LBL/neg_005_t:0.05 \
  --out runs/all_large --epochs 20 --lr 7e-5 --bs 8 --workers 12 --val-frac 0.1
echo "[$(date)] runda 7 gotowa: runs/bdappv_large, runs/all_large"
