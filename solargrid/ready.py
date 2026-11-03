"""Czy lista pobierania jest kompletna? Exit 0 gdy ≥ --min arkuszy z listy jest na dysku w całości.

    python -m solargrid.ready download/lists/2025_05cm_RGB.txt --min 0.98 [--first 78]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import ortho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lists", nargs="+", type=Path)
    ap.add_argument("--min", type=float, default=0.98)
    ap.add_argument("--first", type=int, default=0, help="sprawdzaj tylko pierwsze N pozycji listy")
    args = ap.parse_args()
    idx = ortho.build_index()
    complete = set(idx.loc[idx["path"].notna(), "plik"])
    ok = True
    for lst in args.lists:
        names = [u.rsplit("/", 1)[1] for u in lst.read_text().split()]
        if args.first:
            names = names[: args.first]
        frac = sum(n in complete for n in names) / max(len(names), 1)
        print(f"{lst.stem}: {frac:.1%}")
        ok &= frac >= args.min
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
