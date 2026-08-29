"""E6: score the agreement layer's states against proxy GT.

For each proxy-GT page:
  - load the agreement TSV (som + hadid02, grid-indexed)
  - NW-align the som rows to GT rows (same alignment as score_exp2608 --ink),
    keeping track of which grid row each aligned pred row came from
  - inside each agreement state, count cells where som == GT (frozen _normalize)

The Phase-4 gate (PLAN_GT_pipeline_2026-07.md): MATCH-cell error rate <= 1%.

Also emits exp2608/e6_reader_tasks.tsv: every MISMATCH / ONLY_* cell in an
aligned row, with grid coordinates + all three values, for the in-session
third reader (Claude) to adjudicate from image crops.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from score_g3_vs_gt import LEFT_COLS, _normalize
from score_exp2608 import load_reference, pairs_by_alignment

OUT = Path("exp2608")
PAGES = [3, 4, 5, 6, 9, 10]


def load_som_grid(page: int) -> list[dict]:
    rows = json.load(open(OUT / f"Hadita_{page}_som-g37-flash-rep.json",
                          encoding="utf-8"))
    return [{c: (r.get(c, "") or "").strip() for c in LEFT_COLS} for r in rows]


def load_agreement(page: int) -> dict[tuple[int, str], dict]:
    out = {}
    with open(OUT / f"agreement_{page}.tsv", encoding="utf-8") as f:
        for rec in csv.DictReader(f, delimiter="\t"):
            out[(int(rec["row"]), rec["col"])] = rec
    return out


def main() -> None:
    tot = Counter()
    reader_tasks = []
    print(f"{'page':>4} {'aligned':>8} {'state':<14} {'cells':>6} {'errors':>7} {'err%':>7}")
    for page in PAGES:
        gt = load_reference(page)
        grid = load_som_grid(page)
        agree = load_agreement(page)

        nonempty = [(i, r) for i, r in enumerate(grid)
                    if any(v.strip() for v in r.values())]
        pred_rows = [r for _, r in nonempty]
        pairs = pairs_by_alignment(gt, pred_rows)

        # map each matched pred row (by identity) back to its grid index
        grid_idx_of = {id(r): i for i, r in nonempty}

        page_counts: dict[str, Counter] = {}
        n_aligned = 0
        for kind, g, p in pairs:
            if kind != "matched":
                continue
            n_aligned += 1
            gi = grid_idx_of[id(p)]
            for col in LEFT_COLS:
                rec = agree.get((gi, col))
                if rec is None:
                    continue
                state = rec["state"]
                gv = _normalize(g.get(col, ""))
                sv = _normalize(p.get(col, ""))
                if state == "BOTH_EMPTY" and not gv:
                    continue  # true negatives, uninteresting
                c = page_counts.setdefault(state, Counter())
                c["cells"] += 1
                if gv != sv:
                    c["errors"] += 1
                    c[f"err_{'gt_empty' if not gv else 'som_empty' if not sv else 'diff'}"] += 1
                if state == "MISMATCH" or state.startswith("ONLY_"):
                    reader_tasks.append({
                        "page": page, "grid_row": gi, "col": col,
                        "col_idx": LEFT_COLS.index(col), "state": state,
                        "som": rec.get("som", ""), "hadid02": rec.get("hadid02", ""),
                        "gt": g.get(col, ""),
                    })
        for state, c in sorted(page_counts.items()):
            rate = 100.0 * c["errors"] / c["cells"] if c["cells"] else 0.0
            print(f"{page:>4} {n_aligned:>5}/{len(gt):<2} {state:<14} "
                  f"{c['cells']:>6} {c['errors']:>7} {rate:>6.1f}%")
            tot[(state, "cells")] += c["cells"]
            tot[(state, "errors")] += c["errors"]

    print("\n=== TOTALS over 6 pages (aligned rows only) ===")
    states = sorted({s for s, _ in tot})
    for s in states:
        cells, errs = tot[(s, "cells")], tot[(s, "errors")]
        rate = 100.0 * errs / cells if cells else 0.0
        print(f"  {s:<14} {cells:>6} cells {errs:>6} errors {rate:>6.2f}%")

    m_cells, m_errs = tot[("MATCH", "cells")], tot[("MATCH", "errors")]
    gate = 100.0 * m_errs / m_cells if m_cells else 0.0
    print(f"\nPhase-4 gate: MATCH-cell error rate = {gate:.2f}% "
          f"({'PASS' if gate <= 1.0 else 'FAIL'}, threshold 1%)")

    task_path = OUT / "e6_reader_tasks.tsv"
    with open(task_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=[
            "page", "grid_row", "col", "col_idx", "state", "som", "hadid02", "gt"])
        w.writeheader()
        w.writerows(reader_tasks)
    print(f"wrote {task_path} ({len(reader_tasks)} cells for the third reader)")


if __name__ == "__main__":
    main()
