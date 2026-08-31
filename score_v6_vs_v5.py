"""Score v6 (run_g3v6_local.py output) vs v5 (Colab Hadita_{N}_G3.json) vs proxy GT
(Transkribus latest digit-normalized XML) for pages 3, 4, 5, 6, 9, 10.

Reuses classify() / _normalize() / score() / ra_cost() / load_g3() / load_trx_xml()
from score_g3_vs_gt.py so the methodology matches earlier reports exactly.
"""
import json
import sys
from pathlib import Path

from digit_norm import LEFT_COLS
from score_g3_vs_gt import (
    classify, _normalize, load_g3, load_trx_xml, filter_nonempty_rows,
    align_by_position, score, ra_cost,
)

PAGES = [3, 4, 5, 6, 9, 10]

V5_DIR = Path("g3_results")
V6_DIR = Path(".")
GT_DIR = Path("g3_results")


def _short(counts: dict) -> str:
    parts = []
    for k in ("perfect", "single_digit", "multi_digit", "wrong", "missed_row", "phantom_row"):
        parts.append(f"{k[0].upper() if k!='perfect' else 'P'}={counts.get(k,0):>3}")
    parts.append(f"ks={ra_cost(counts):>3}")
    return " ".join(parts)


def main():
    print(f"{'page':>4}  {'src':>3}  {'rows':>4}  perfect   single   multi   wrong   miss   phantom   RA_ks  Δks_vs_v5")
    print("-" * 100)

    summary = []
    for page in PAGES:
        gt_path = GT_DIR / f"Hadita_{page}_Transkribus_latest.xml"
        v5_path = V5_DIR / f"Hadita_{page}_G3.json"
        v6_path = V6_DIR / f"Hadita_{page}_G3v6.json"
        if not gt_path.exists():
            print(f"  page {page}: missing proxy GT {gt_path}; skipping")
            continue
        gt_raw = load_trx_xml(str(gt_path))
        gt = filter_nonempty_rows(gt_raw)

        v5_ks, v6_ks = None, None
        if v5_path.exists():
            v5_rows = load_g3(str(v5_path))
            counts5, _, _, _ = score(gt, v5_rows)
            v5_ks = ra_cost(counts5)
            print(f"  {page:>4}  v5   {len(v5_rows):>4}  "
                  f"{counts5['perfect']:>6}   {counts5['single_digit']:>5}   "
                  f"{counts5['multi_digit']:>4}   {counts5['wrong']:>4}   "
                  f"{counts5['missed_row']:>3}   {counts5['phantom_row']:>5}   "
                  f"{v5_ks:>4}")
        else:
            print(f"  {page:>4}  v5    --  (no v5 file)")

        if v6_path.exists():
            v6_rows = load_g3(str(v6_path))
            counts6, _, _, _ = score(gt, v6_rows)
            v6_ks = ra_cost(counts6)
            delta = f"{v6_ks - v5_ks:+d}" if v5_ks is not None else "n/a"
            print(f"  {page:>4}  v6   {len(v6_rows):>4}  "
                  f"{counts6['perfect']:>6}   {counts6['single_digit']:>5}   "
                  f"{counts6['multi_digit']:>4}   {counts6['wrong']:>4}   "
                  f"{counts6['missed_row']:>3}   {counts6['phantom_row']:>5}   "
                  f"{v6_ks:>4}   {delta}")
        else:
            print(f"  {page:>4}  v6    --  (no v6 file)")

        summary.append((page, v5_ks, v6_ks, len(gt)))
        print()

    print("=" * 100)
    print(f"Summary across {len(summary)} pages (RA_cost = keystrokes; lower = better):")
    print(f"  {'page':>4}  {'gt_rows':>7}  {'v5_ks':>6}  {'v6_ks':>6}  {'Δ (v6-v5)':>9}")
    total_v5, total_v6 = 0, 0
    n_v5, n_v6 = 0, 0
    for page, v5, v6, gt_rows in summary:
        v5s = f"{v5:>6}" if v5 is not None else "    --"
        v6s = f"{v6:>6}" if v6 is not None else "    --"
        delta = f"{v6-v5:+d}" if (v5 is not None and v6 is not None) else "n/a"
        print(f"  {page:>4}  {gt_rows:>7}  {v5s}  {v6s}  {delta:>9}")
        if v5 is not None: total_v5 += v5; n_v5 += 1
        if v6 is not None: total_v6 += v6; n_v6 += 1
    print()
    print(f"  v5 total (over {n_v5} pages): {total_v5} keystrokes")
    print(f"  v6 total (over {n_v6} pages): {total_v6} keystrokes")
    pages_both = [p for p, v5, v6, _ in summary if v5 is not None and v6 is not None]
    common5 = sum(v5 for p, v5, v6, _ in summary if v5 is not None and v6 is not None)
    common6 = sum(v6 for p, v5, v6, _ in summary if v5 is not None and v6 is not None)
    print(f"  on {len(pages_both)} pages with BOTH: v5={common5}, v6={common6}, "
          f"delta={common6-common5:+d} keystrokes "
          f"({100*(common6-common5)/common5:+.1f}%)")


if __name__ == "__main__":
    main()
