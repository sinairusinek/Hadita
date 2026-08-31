"""Score exp2608 outputs against proxy GT, with optional ink-occupancy row alignment.

Imports the frozen metric from score_g3_vs_gt (never modifies it).

  python score_exp2608.py --tags g3flashpreview g37flash g31pro
  python score_exp2608.py --tags g37flash --ink      # E2: ink-aligned rows
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_g3_vs_gt import (LEFT_COLS, classify, load_gt, load_trx_xml,
                            ra_cost, _normalize)

OUT_DIR = Path("exp2608")
GT_XML = "g3_results/Hadita_{p}_Transkribus_latest.xml"
PAGES = [3, 4, 5, 6, 9, 10]


def load_reference(page: int) -> list[dict]:
    """Page 3 has typed GT; others use the RA-corrected Transkribus proxy."""
    if page == 3:
        rows = load_gt(3)
        if rows:
            return rows
    p = Path(GT_XML.format(p=page))
    if not p.exists():
        return []
    return [r for r in load_trx_xml(str(p)) if any(v.strip() for v in r.values())]


def load_pred(page: int, tag: str) -> list[dict] | None:
    p = OUT_DIR / f"Hadita_{page}_{tag}.json"
    if not p.exists():
        return None
    rows = json.load(open(p, encoding="utf-8"))
    rows = [{c: (r.get(c, "") or "").strip() for c in LEFT_COLS} for r in rows]
    # Grid-shaped output (grounded OCR) carries blank rows for uninked grid rows;
    # the reference is already filtered to non-empty rows, so filter to match.
    return [r for r in rows if any(v.strip() for v in r.values())]


def score_pairs(pairs) -> dict:
    counts = {k: 0 for k in ["perfect", "empty_both", "single_digit",
                             "multi_digit", "wrong", "missed_row", "phantom_row"]}
    for kind, g, p in pairs:
        if kind == "missed":
            counts["missed_row"] += 1
            continue
        if kind == "phantom":
            counts["phantom_row"] += 1
            continue
        for c in LEFT_COLS:
            counts[classify(g.get(c, ""), p.get(c, ""))] += 1
    return counts


def pairs_by_offset(gt: list[dict], pr: list[dict]) -> list:
    """Frozen-scorer behaviour: best constant offset in -2..+2."""
    best_off, best = 0, -1
    for off in range(-2, 3):
        s = 0
        for j in range(len(pr)):
            i = j + off
            if 0 <= i < len(gt):
                for c in LEFT_COLS:
                    gv = _normalize(gt[i][c])
                    if gv and gv == _normalize(pr[j][c]):
                        s += 1
        if s > best:
            best, best_off = s, off
    out, used_gt, used_pr = [], set(), set()
    for j in range(len(pr)):
        i = j + best_off
        if 0 <= i < len(gt):
            out.append(("matched", gt[i], pr[j]))
            used_gt.add(i)
            used_pr.add(j)
    out += [("missed", gt[i], None) for i in range(len(gt)) if i not in used_gt]
    out += [("phantom", None, pr[j]) for j in range(len(pr)) if j not in used_pr]
    return out, best_off


def occupancy(row: dict) -> tuple:
    """Binary pattern of which columns carry content."""
    return tuple(1 if _normalize(row.get(c, "")) else 0 for c in LEFT_COLS)


def pairs_by_alignment(gt: list[dict], pr: list[dict]) -> list:
    """E2: Needleman-Wunsch over rows, allowing gaps anywhere (not a constant offset).

    Similarity = agreement on non-empty cell values, plus a weaker bonus for
    matching occupancy pattern (which is what survives when OCR misreads a value
    but still sees that a cell is inked).
    """
    n, m = len(gt), len(pr)
    GAP = -3.0

    def sim(i: int, j: int) -> float:
        g, p = gt[i], pr[j]
        exact = shared = 0
        for c in LEFT_COLS:
            gv, pv = _normalize(g.get(c, "")), _normalize(p.get(c, ""))
            if gv and gv == pv:
                exact += 1
            if bool(gv) == bool(pv):
                shared += 1
        return 3.0 * exact + 0.15 * shared

    F = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        F[i][0] = F[i - 1][0] + GAP
    for j in range(1, m + 1):
        F[0][j] = F[0][j - 1] + GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            F[i][j] = max(F[i - 1][j - 1] + sim(i - 1, j - 1),
                          F[i - 1][j] + GAP, F[i][j - 1] + GAP)
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(F[i][j] - (F[i-1][j-1] + sim(i-1, j-1))) < 1e-9:
            out.append(("matched", gt[i - 1], pr[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and abs(F[i][j] - (F[i-1][j] + GAP)) < 1e-9:
            out.append(("missed", gt[i - 1], None))
            i -= 1
        else:
            out.append(("phantom", None, pr[j - 1]))
            j -= 1
    return out[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--ink", action="store_true",
                    help="E2: NW row alignment instead of constant offset")
    ap.add_argument("--pages", nargs="+", type=int, default=PAGES)
    args = ap.parse_args()

    mode = "NW-aligned" if args.ink else "offset-aligned"
    print(f"\nscoring mode: {mode}\n")
    header = f"{'tag':<18}{'page':>5}{'gt':>4}{'pred':>5}{'perf':>6}{'miss':>6}{'phan':>6}{'ks':>7}"
    grand = {}
    for tag in args.tags:
        print(header)
        print("-" * len(header))
        tot = 0
        pages_done = []
        for page in args.pages:
            gt = load_reference(page)
            pr = load_pred(page, tag)
            if not gt or pr is None:
                print(f"{tag:<18}{page:>5}   -- (missing {'GT' if not gt else 'pred'})")
                continue
            if args.ink:
                pairs = pairs_by_alignment(gt, pr)
            else:
                pairs, _ = pairs_by_offset(gt, pr)
            counts = score_pairs(pairs)
            ks = ra_cost(counts)
            tot += ks
            pages_done.append(page)
            print(f"{tag:<18}{page:>5}{len(gt):>4}{len(pr):>5}"
                  f"{counts['perfect']:>6}{counts['missed_row']:>6}"
                  f"{counts['phantom_row']:>6}{ks:>7}")
        grand[tag] = (tot, pages_done)
        print(f"{'TOTAL':<18}{'':>5}{'':>4}{'':>5}{'':>6}{'':>6}{'':>6}{tot:>7}\n")

    if len(grand) > 1:
        print("summary (lower ks = better):")
        for tag, (tot, pgs) in sorted(grand.items(), key=lambda x: x[1][0]):
            print(f"  {tag:<20} {tot:>6} ks over pages {pgs}")


if __name__ == "__main__":
    main()
