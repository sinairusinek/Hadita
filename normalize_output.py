#!/usr/bin/env python3
"""Put a run's stored output into canonical form before it reaches an RA.

The scorer folds variant codepoints at COMPARISON time (encoding_fold +
normalize_for_compare), so scores were never affected by spelling. But the JSON
and the PAGE XML we push to Transkribus keep whatever the model emitted, and an
RA then sees Arabic commas where the GT convention is ASCII, bare tatweel where
the GT writes "-", and so on. Measured on the 2026-09-09 pages 12-20 run: 100 of
1521 non-empty cells (6.6%) were not in canonical form.

This rewrites the JSON and regenerates the XML from it, applying exactly the
folds the scorer already applies -- so it can never change a score, only what a
human reads.

Multi-row cells are the one thing NOT silently fixed. A newline inside a cell
means the model merged two register rows into one; joining them with a space
would hide a real segmentation failure. They are marked with a leading "⚠ " so
the RA can see them, and reported on stderr.

  python normalize_output.py --tag prod-iter4 --pages 12 13 14 15 16 17 18 19 20
  python normalize_output.py --tag prod-iter4 --pages 12 --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_g3v6_local as base
from digit_norm import LEFT_COLS, encoding_fold, normalize_for_compare

OUT = Path("exp2608")
XML_SRC = Path("Transkribus upload/final3")
FLAG = "⚠ "


def canonical(v: str) -> tuple[str, bool]:
    """Return (canonical text, is_multirow)."""
    if not v or not v.strip():
        return "", False
    multi = "\n" in v.strip()
    # Fold each physical line on its own so a merged cell keeps its structure.
    parts = [normalize_for_compare(encoding_fold(p)) for p in v.split("\n")]
    parts = [p for p in parts if p]
    text = " / ".join(parts)
    if multi:
        text = FLAG + text
    return text, multi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pages", nargs="+", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tot = changed = merged = 0
    for p in args.pages:
        jp = OUT / f"Hadita_{p}_{args.tag}.json"
        if not jp.exists():
            print(f"  ! p{p}: {jp} missing")
            continue
        grid = json.load(open(jp, encoding="utf-8"))
        n_ch = n_mg = 0
        for r in grid:
            for c in list(r):
                v = r.get(c, "")
                if not v or not v.strip():
                    continue
                tot += 1
                new, multi = canonical(v)
                if multi:
                    n_mg += 1
                    print(f"    p{p} MERGED ROWS in {c}: {v!r} -> {new!r}")
                if new != v:
                    n_ch += 1
                r[c] = new
        changed += n_ch
        merged += n_mg
        print(f"  p{p}: {n_ch} cells normalized, {n_mg} merged-row cells flagged")
        if args.dry_run:
            continue
        jp.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
        xml_p = XML_SRC / f"Hadita_{p}.xml"
        patched, _, _ = base.patch_xml(
            xml_p.read_text(encoding="utf-8"), grid, LEFT_COLS)
        (OUT / f"Hadita_{p}_{args.tag}.xml").write_text(patched, encoding="utf-8")

    print(f"\n{changed} of {tot} non-empty cells normalized; "
          f"{merged} merged-row cells flagged with {FLAG!r}"
          + ("  (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
