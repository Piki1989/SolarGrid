#!/usr/bin/env bash
# Synchronizuje kod lokalny -> serwer katedralny. Tylko kod: --delete działa wyłącznie
# w obrębie wysyłanych ścieżek, katalogi wynikowe na serwerze (runs/, .venv) są nietykalne.
set -euo pipefail
cd "$(dirname "$0")"
rsync -az --delete --exclude __pycache__ solargrid/ mpikus@thebrain.kis.agh.edu.pl:solargrid/solargrid/
rsync -az --delete download/ mpikus@thebrain.kis.agh.edu.pl:solargrid/download/
rsync -az pyproject.toml uv.lock README.md sync.sh mpikus@thebrain.kis.agh.edu.pl:solargrid/
