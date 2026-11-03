#!/usr/bin/env bash
# Restart aplikacji na serwerze poza tmux (29.10 tmux padł i zabrał aplikację).
#   bash download/app.sh [checkpoint]      domyślnie runs/all_v1/best.pt
# pkill i uruchomienie w osobnym skrypcie: wzorzec w linii poleceń ssh zabijał własną sesję.
set -euo pipefail
cd ~/solargrid
CKPT=${1:-runs/all_v1/best.pt}
pkill -f "bin/uvicorn solargrid.app" || true
sleep 2
SOLARGRID_CKPT=$CKPT SOLARGRID_DEVICE=cuda:1 \
  setsid nohup .venv/bin/uvicorn solargrid.app:app --host 127.0.0.1 --port 8000 > /tmp/app.log 2>&1 < /dev/null &
sleep 8
tail -1 /tmp/app.log
