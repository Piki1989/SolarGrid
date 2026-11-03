#!/usr/bin/env bash
# Pobiera ortofotomapy GUGiK dla Małopolski wg list w lists/*.txt.
# Użycie: ./fetch_ortho.sh [DEST] [LISTA...]   (domyślnie wszystkie listy RGB)
# Wznawialne: wget -c, pliki już pobrane są pomijane.
set -euo pipefail
DEST=${1:-$HOME/datasets/ortho}; shift || true
HERE=$(cd "$(dirname "$0")" && pwd)
LISTS=("$@"); [ ${#LISTS[@]} -eq 0 ] && LISTS=("$HERE"/lists/*_RGB.txt)
JOBS=${JOBS:-8}
for L in "${LISTS[@]}"; do
  name=$(basename "$L" .txt); out="$DEST/$name"; mkdir -p "$out"
  echo "== $name ($(wc -l < "$L") plików) -> $out"
  xargs -P "$JOBS" -n 1 wget -q -c --tries=10 --waitretry=30 --timeout=120 -P "$out" < "$L" \
    || echo "!! część plików z $name nie pobrana — uruchom ponownie" >&2
  echo "   gotowe: $(ls "$out" | wc -l) plików, $(du -sh "$out" | cut -f1)"
done
