#!/usr/bin/env python3
"""score_hadid02_final2.py — Phase 2 scoring: hadid02 on undamaged images.

Reuses the frozen methodology in `score_g3_vs_gt.py` (cell classes
perfect/single_digit/multi_digit/wrong/missed_row/phantom_row and the keystroke
RA-cost model) without touching it — that file's `main()` is the v5 LOW-vs-
MEDIUM comparison and must keep producing the same numbers (house rule 1).

What this measures: previous Hadid inference ran inside Transkribus on the
*dewarped* images, which Phase 0 showed to be damaged on every page. These runs
used the final2 pages — undamaged deskewed images with curved cell polygons and
TextLines only where there is ink. If Hadid improves, the rebuild paid for
itself on HTR quality alone, independent of the RA-correction argument.

GT per page: `ground_truth.tsv` where verified (page 3), otherwise the
RA-corrected Transkribus transcript kept in `g3_results/Hadita_{N}_
Transkribus_latest.xml` — proxy GT, not gold, per project_proxy_gt_pages.

Usage:
  python score_hadid02_final2.py
  python score_hadid02_final2.py --pages 3 9
"""
from __future__ import annotations

import argparse
from pathlib import Path

from score_g3_vs_gt import (  # noqa: E402  frozen methodology, imported not edited
    filter_nonempty_rows, load_gt, load_trx_xml, ra_cost, report, score,
)

ROOT = Path(__file__).parent
GT_PAGES = [3, 4, 5, 6, 9, 10]
VERIFIED_GT = {3}          # pages present in ground_truth.tsv
PRED_FMT = "g3_results/Hadita_{n}_Hadid02_final2.xml"
PROXY_FMT = "g3_results/Hadita_{n}_Transkribus_latest.xml"


def load_reference(page: int) -> tuple[list[dict], str]:
    if page in VERIFIED_GT:
        rows = load_gt(page)
        if rows:
            return rows, "ground_truth.tsv (verified)"
    p = ROOT / PROXY_FMT.format(n=page)
    if not p.exists():
        return [], "none"
    return filter_nonempty_rows(load_trx_xml(str(p))), f"{p.name} (RA-corrected proxy)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+", default=GT_PAGES)
    args = ap.parse_args()

    totals: dict[str, int] = {}
    per_page = []
    for page in args.pages:
        pred_path = ROOT / PRED_FMT.format(n=page)
        if not pred_path.exists():
            print(f"page {page}: no hadid02 prediction ({pred_path.name}) — skipping")
            continue
        ref, ref_label = load_reference(page)
        if not ref:
            print(f"page {page}: no reference transcript — skipping")
            continue
        pred = filter_nonempty_rows(load_trx_xml(str(pred_path)))

        print("\n" + "=" * 80)
        print(f"PAGE {page} — hadid02 on final2   (reference: {ref_label})")
        print(f"  reference rows: {len(ref)}   prediction rows (non-empty): {len(pred)}")
        counts, per_col, errs, _ = score(ref, pred)
        report(f"PAGE {page} — hadid02 / final2", counts, per_col, errs, 0.0)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        per_page.append((page, counts))

    if not per_page:
        return
    print("\n" + "=" * 80)
    print("COMBINED — hadid02 on final2")
    comparable = sum(v for k, v in totals.items() if k != "empty_both")
    good = totals.get("perfect", 0)
    print(f"  pages scored     : {len(per_page)}")
    print(f"  comparable cells : {comparable}")
    if comparable:
        print(f"  perfect          : {good} ({100*good/comparable:.1f}%)")
        for k in ("single_digit", "multi_digit", "wrong", "missed_row", "phantom_row"):
            v = totals.get(k, 0)
            print(f"  {k:<17}: {v} ({100*v/comparable:.1f}%)")
    ks = ra_cost(totals)
    print(f"  RA cost          : {ks:.0f} keystrokes "
          f"({ks/len(per_page):.0f} per page)")
    print("\n  per page:")
    for page, counts in per_page:
        c = sum(v for k, v in counts.items() if k != "empty_both")
        print(f"    page {page:>3}: {counts.get('perfect',0):>4}/{c:<4} perfect "
              f"({100*counts.get('perfect',0)/c if c else 0:5.1f}%)  "
              f"missed_row={counts.get('missed_row',0):>3} "
              f"phantom_row={counts.get('phantom_row',0):>3}  "
              f"RA={ra_cost(counts):.0f}ks")


if __name__ == "__main__":
    main()
