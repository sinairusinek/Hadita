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
import re
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


def align(gt: list[dict], n_grid: int, offset: int = None) -> list[dict]:
    """Place GT rows onto n_grid slots, anchored at the TOP.

    The GT was transcribed against final2 rows and final3 changed the row count
    on 5 of the 6 GT pages, so the two are not index-identical. But an earlier
    Needleman-Wunsch pass over row *signatures* was worse than useless: with
    nothing to match on it drifted the whole page down (p9 started at row 9,
    p10 at row 7), which then looked like a catastrophic ink-gate failure
    (a bogus 88%/74% miss rate) when it was purely my alignment.

    Both grids start at the same printed header line, so row 0 is row 0. Extra
    grid slots belong at the BOTTOM, where final3 recovered rows. Index-align
    from the top and stop; do not try to be clever without evidence to align on.
    """
    out = [{c: "" for c in LEFT_COLS} for _ in range(n_grid)]
    off = offset if offset is not None else 0
    for i, row in enumerate(gt):
        j = i + off
        if 0 <= j < n_grid:
            out[j] = {c: str(row.get(c, "") or "") for c in LEFT_COLS}
    return out


def best_offset(gt: list[dict], serial_rows: set[int]) -> int:
    """Align on Serial_No, which is unique per row and present on every entry.

    Index alignment alone is wrong where the GT carries a CONTINUATION row at
    the top -- the tail of an entry begun on the previous page, holding only
    money-column values with no serial or date (p4, p5). The grid starts at the
    first *written* line, so the GT sits one row lower. Serial numbers give an
    unambiguous anchor: p4 matches 34/34 at -1, p5 29/29 at -1, the rest 0.
    """
    gt_ser = [i for i, r in enumerate(gt) if str(r.get("Serial_No", "") or "").strip()]
    if not gt_ser or not serial_rows:
        return 0
    best, hits = 0, -1
    for off in range(-3, 4):
        h = sum(1 for i in gt_ser if (i + off) in serial_rows)
        if h > hits:
            best, hits = off, h
    return best


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
        ser = {int(m.group(1)) for m in re.finditer(r'id="line_cell_r(\d+)_c0"', xml)}
        off = best_offset(gt, ser)
        grid = align(gt, n_grid, off)
        placed = sum(1 for r in grid if any(v.strip() for v in r.values()))
        patched, _, _ = base.patch_xml(xml, grid, LEFT_COLS)
        (OUT / f"Hadita_{p}_{args.tag}.xml").write_text(patched, encoding="utf-8")
        print(f"page {p}: {len(gt)} GT rows -> {n_grid} grid slots, {placed} placed (offset {off:+d})")


if __name__ == "__main__":
    main()
