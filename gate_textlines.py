#!/usr/bin/env python3
"""gate_textlines.py — drop TextLines from cells that contain no handwriting.

PyLaia transcribes every TextLine it is handed. Our XML put one in *every*
cell, so on page 11 Hadid02 returned text for 416 of 532 cells that are blank
paper — 78% invented. That would dominate the Phase 4 agreement states and hand
the RA hundreds of fabricated cells per page to delete.

The TableCell (and therefore the table structure, the column tags and the RA's
ability to type into any cell) is kept. Only the TextLine goes, so recognition
has nothing to invent into.

Threshold calibration on the GT pages 3/4/5 (1976 cells, 957 with GT text):
ink per cell is bimodal — cells with GT text have a median of 222 handwriting
pixels, blank cells a median of 0, with 756 of 1019 at exactly zero. Every
threshold between 8 and 25 behaves the same, so 12 is used. It keeps 1092
lines, loses the line from 11.8% of GT-text cells and still keeps one for 24%
of blank cells. The 97 GT-text cells with *zero* ink are irreducible: some are
ditto/dash conventions with nothing to read, and some may indicate GT text
mapped to the wrong cell — worth investigating separately, but no ink threshold
can rescue them.

Operates in place on `Transkribus upload/final2/*.xml`; the geometry is left
untouched, so this never re-runs detection and cannot shift columns.

Usage:
  python gate_textlines.py --dry-run
  python gate_textlines.py --pages 11
  python gate_textlines.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

FINAL2_DIR = ROOT / "Transkribus upload" / "final2"
REPORT_TSV = ROOT / "gate_textlines.tsv"
INK_MIN_PX = 12
LUMA_OFFSET = 45

# patch_baselines re-serialises with ElementTree, so Coords come back as
# `<Coords points="…"></Coords>` rather than self-closing — accept both forms.
CELL_RX = re.compile(
    r'<TableCell id="cell_r(\d+)_c(\d+)"(?P<attrs>[^>]*)>\s*'
    r'<Coords points="(?P<pts>[^"]+)"\s*(?:/>|></Coords>)'
    r'(?P<rest>.*?)</TableCell>', re.S)
LINE_RX = re.compile(r"\s*<TextLine\b.*?</TextLine>", re.S)


def hand_mask(img: np.ndarray) -> np.ndarray:
    """Handwriting pixels: ink minus printed rules and large dark areas."""
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 3)
    ink = (gray < float(np.median(gray)) - LUMA_OFFSET).astype(np.uint8)
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (61, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, 61)))
    big = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    drop = cv2.dilate(cv2.bitwise_or(cv2.bitwise_or(h, v), big), np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(cv2.bitwise_and(ink, cv2.bitwise_not(drop)),
                            cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def gate_page(page: int, threshold: int, dry_run: bool) -> dict | None:
    xml_path = FINAL2_DIR / f"Hadita_{page}.xml"
    jpeg = FINAL2_DIR / f"Hadita_{page}.jpeg"
    if not xml_path.exists() or not jpeg.exists():
        return None
    img = cv2.imread(str(jpeg))
    if img is None:
        return None
    mask = hand_mask(img)
    xml = xml_path.read_text(encoding="utf-8")

    stats = {"page": page, "cells": 0, "kept": 0, "dropped": 0, "already_none": 0,
             "dropped_with_text": 0}

    def repl(m: re.Match) -> str:
        stats["cells"] += 1
        poly = np.array([[int(a) for a in p.split(",")] for p in m.group("pts").split()],
                        np.int32)
        cell_mask = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(cell_mask, [poly], 1)
        n_ink = int((mask & cell_mask).sum())
        rest = m.group("rest")
        if not LINE_RX.search(rest):
            stats["already_none"] += 1
            return m.group(0)
        if n_ink >= threshold:
            stats["kept"] += 1
            return m.group(0)
        stats["dropped"] += 1
        if any(u.strip() for u in re.findall(r"<Unicode>(.*?)</Unicode>", rest, re.S)):
            stats["dropped_with_text"] += 1
        gated = LINE_RX.sub("", rest)
        return (f'<TableCell id="cell_r{m.group(1)}_c{m.group(2)}"{m.group("attrs")}>\n'
                f'        <Coords points="{m.group("pts")}"></Coords>{gated}</TableCell>')

    out = CELL_RX.sub(repl, xml)
    if not dry_run:
        xml_path.write_text(out, encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+")
    ap.add_argument("--threshold", type=int, default=INK_MIN_PX)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pages = sorted(args.pages) if args.pages else sorted(
        int(p.stem.split("_")[1]) for p in FINAL2_DIR.glob("Hadita_*.xml"))

    rows = []
    for i, page in enumerate(pages, 1):
        s = gate_page(page, args.threshold, args.dry_run)
        if s is None:
            print(f"[{i}/{len(pages)}] page {page:>3}  SKIP (missing xml/jpeg)")
            continue
        rows.append(s)
        pct = 100 * s["dropped"] / s["cells"] if s["cells"] else 0
        print(f"[{i}/{len(pages)}] page {page:>3}  {s['cells']:>4} cells  "
              f"kept {s['kept']:>4}  dropped {s['dropped']:>4} ({pct:4.1f}%)"
              + (f"  [dry-run]" if args.dry_run else ""), flush=True)

    if rows and not args.dry_run:
        cols = ["page", "cells", "kept", "dropped", "dropped_with_text", "already_none"]
        with open(REPORT_TSV, "w", encoding="utf-8") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    tc = sum(r["cells"] for r in rows)
    tk = sum(r["kept"] for r in rows)
    td = sum(r["dropped"] for r in rows)
    tw = sum(r["dropped_with_text"] for r in rows)
    print(f"\n{len(rows)} pages: {tc} cells, kept {tk} lines, dropped {td} "
          f"({100*td/tc:.1f}%)")
    print(f"  of the dropped, {tw} carried local text (GT/Approach-M seed, "
          "cleared before pushing anyway)")
    if not args.dry_run:
        print(f"  report → {REPORT_TSV.name}")


if __name__ == "__main__":
    main()
