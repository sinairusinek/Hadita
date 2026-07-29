#!/usr/bin/env python3
"""score_col_alignment.py — how well do final2/ column boundaries sit on the
printed rules?

Column identification is the weakest part of the segmentation, so measure it
rather than guess: for every column boundary in every horizontal band, look for
a printed vertical line within ±TOL px of where the XML puts the boundary. The
score is the fraction of (boundary × band) pairs that find one.

This is a *ranking*, not a verdict — a boundary can be right and still score
low where the printed rule has faded. Use it to send the RA to the worst pages
first, and to catch pages whose grid is genuinely off.

Outputs col_alignment.tsv (page, score, worst columns) sorted worst-first.

Usage:
  python score_col_alignment.py
  python score_col_alignment.py --pages 3 4 21
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from scipy.signal import find_peaks  # noqa: E402

from segment_unified import LEFT_COLS  # noqa: E402

NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
FINAL2_DIR = ROOT / "Transkribus upload" / "final2"
OUT_TSV = ROOT / "col_alignment.tsv"
TOL = 8       # px: how close a printed line must be to count as a match
N_BANDS = 8

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def line_peaks(strip: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5)
    v = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))
    proj = np.sum(v, axis=0).astype(float)
    pk, _ = find_peaks(proj, height=proj.mean() + 0.3 * proj.std(),
                       distance=max(20, strip.shape[1] // 40))
    return np.asarray(pk, dtype=float)


def score_page(page: int) -> dict | None:
    xml = FINAL2_DIR / f"Hadita_{page}.xml"
    jpeg = FINAL2_DIR / f"Hadita_{page}.jpeg"
    if not xml.exists() or not jpeg.exists():
        return None
    img = cv2.imread(str(jpeg))
    root = ET.parse(xml).getroot()

    # Boundary x at each row's mid-height, read back out of the written XML.
    edges: dict[int, list[tuple[float, float]]] = {}
    for cell in root.iter(f"{NS}TableCell"):
        c = int(cell.attrib["col"])
        pts = [tuple(int(v) for v in p.split(","))
               for p in cell.find(f"{NS}Coords").attrib["points"].split()]
        xs = sorted(p[0] for p in pts)
        ys = [p[1] for p in pts]
        ymid = (min(ys) + max(ys)) / 2
        edges.setdefault(c, []).append((ymid, float(xs[0])))
        edges.setdefault(c + 1, []).append((ymid, float(xs[-1])))

    h = img.shape[0]
    band_h = h // N_BANDS
    peaks = [line_peaks(img[b * band_h:(h if b == N_BANDS - 1 else (b + 1) * band_h)])
             for b in range(N_BANDS)]

    hits, total = 0, 0
    per_col: dict[int, list[int]] = {}
    for c, samples in edges.items():
        for ymid, x in samples:
            b = min(N_BANDS - 1, int(ymid // band_h))
            if peaks[b].size == 0:
                continue
            ok = int(np.min(np.abs(peaks[b] - x)) <= TOL)
            hits += ok
            total += 1
            per_col.setdefault(c, []).append(ok)
    if not total:
        return None
    col_scores = {c: sum(v) / len(v) for c, v in per_col.items() if v}
    worst = sorted(col_scores.items(), key=lambda kv: kv[1])[:3]
    return {
        "page": page,
        "score": round(hits / total, 3),
        "n_cols": max(edges) if edges else 0,
        "worst_cols": "; ".join(
            f"{c}:{LEFT_COLS[c] if c < len(LEFT_COLS) else 'edge'}={s:.2f}"
            for c, s in worst),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+")
    args = ap.parse_args()
    pages = sorted(args.pages) if args.pages else sorted(
        int(p.stem.split("_")[1]) for p in FINAL2_DIR.glob("Hadita_*.xml"))

    rows = []
    for i, p in enumerate(pages, 1):
        r = score_page(p)
        if r:
            rows.append(r)
        print(f"[{i}/{len(pages)}] page {p:>3}  "
              f"{r['score'] if r else 'n/a'}", flush=True)

    rows.sort(key=lambda r: r["score"])
    cols = ["page", "score", "n_cols", "worst_cols"]
    with open(OUT_TSV, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    scores = [r["score"] for r in rows]
    print(f"\n{len(rows)} pages scored → {OUT_TSV.name}")
    print(f"  median {np.median(scores):.3f}   worst {min(scores):.3f}   "
          f"best {max(scores):.3f}")
    for band, lo, hi in [("poor (<0.5)", 0, 0.5), ("fair (0.5-0.75)", 0.5, 0.75),
                         ("good (>=0.75)", 0.75, 1.01)]:
        sel = [r for r in rows if lo <= r["score"] < hi]
        print(f"  {band:<18} {len(sel):3}  {' '.join(str(r['page']) for r in sel[:30])}")


if __name__ == "__main__":
    main()
