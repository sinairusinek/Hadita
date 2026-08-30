#!/usr/bin/env python3
"""push_gt_to_final3.py — put the proxy GT text into the final3 geometry.

Purpose: see, in Transkribus, which GT-bearing cells the ink gate dropped a
TextLine from. The GT was transcribed against final2 rows, and final3 changed
the row count on 5 of the 6 GT pages, so rows are aligned by CONTENT
(Needleman-Wunsch over each row's non-empty cell values), never by index.

  python push_gt_to_final3.py --pages 3 4 5 6 9 10 --dry-run
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import run_g3v6_local as base  # noqa: E402
from digit_norm import LEFT_COLS  # noqa: E402
from score_g3_vs_gt import load_trx_xml  # noqa: E402

FINAL3 = ROOT / "Transkribus upload" / "final3"
OUT = ROOT / "exp2608"
GT_FMT = "g3_results/Hadita_{n}_Transkribus_latest.xml"


def sig(row: dict) -> str:
    return " ".join(str(row.get(c, "")).strip() for c in LEFT_COLS).strip()


def align(gt: list[dict], n_grid: int) -> list[dict]:
    """Needleman-Wunsch align GT rows onto n_grid slots by row signature."""
    grid_sigs = [""] * n_grid
    gs = [sig(r) for r in gt]
    n, m = len(gs), n_grid
    # score: reward placing a non-empty GT row; gaps cost 1
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] - 1
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] - 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = D[i - 1][j - 1] + (1.0 if gs[i - 1] else 0.2)
            D[i][j] = max(match, D[i - 1][j] - 1, D[i][j - 1] - 1)
    out = [{c: "" for c in LEFT_COLS} for _ in range(n_grid)]
    i, j = n, m
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j - 1] + (1.0 if gs[i - 1] else 0.2):
            out[j - 1] = {c: str(gt[i - 1].get(c, "") or "") for c in LEFT_COLS}
            i -= 1
            j -= 1
        elif D[i][j] == D[i - 1][j] - 1:
            i -= 1
        else:
            j -= 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+", default=[3, 4, 5, 6, 9, 10])
    ap.add_argument("--tag", default="gt-f3")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for p in args.pages:
        gt = load_trx_xml(GT_FMT.format(n=p))
        xml = (FINAL3 / f"Hadita_{p}.xml").read_text(encoding="utf-8")
        n_grid = base.xml_grid_size(xml)[0]
        grid = align(gt, n_grid)
        placed = sum(1 for r in grid if any(v.strip() for v in r.values()))
        patched, _, _ = base.patch_xml(xml, grid, LEFT_COLS)
        (OUT / f"Hadita_{p}_{args.tag}.xml").write_text(patched, encoding="utf-8")
        print(f"page {p}: {len(gt)} GT rows -> {n_grid} grid slots, {placed} placed")


if __name__ == "__main__":
    main()
