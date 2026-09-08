#!/usr/bin/env python3
"""Does Mistral's per-cell confidence predict per-cell error?

This is the whole point of the Mistral pilot. The 2026-06-26 handoff set the
adoption bar as Spearman rho > 0.4 between confidence and error; that bar is
kept here rather than restated, so the decision is the one already agreed.

Calibration matters more than keystrokes for us. Every reader in the stack, the
incumbent included, hands back a transcription with no usable uncertainty, so a
human has to re-check everything. A confidence that ranks cells by
wrongness-probability lets ~2/3 cells be routed automatically, which is the open
recommendation in feedback_23_confusion_unfixable.

  python score_mistral_calibration.py
  python score_mistral_calibration.py --tag mistral-ocr --pages 3 4

Reports, in order of decision value:
  1. Spearman rho (confidence vs error) + the >0.4 gate
  2. AUC: P(a random wrong cell scores below a random right cell)
  3. Decile table: is the bottom decile actually where the errors live?
  4. Review-budget curve: reviewing the N% least-confident catches what share?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from digit_norm import LEFT_COLS, encoding_fold  # noqa: E402
from score_g3_vs_gt import _normalize  # noqa: E402
from score_exp2608 import load_reference, load_pred, pairs_by_offset, PAGES  # noqa: E402

OUT_DIR = ROOT / "exp2608"


def _rank(xs: list[float]) -> list[float]:
    """Average ranks, so ties do not distort rho (confidences tie a lot)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def auc(conf_wrong: list[float], conf_right: list[float]) -> float:
    """P(random wrong cell ranks below a random right cell). 0.5 = useless."""
    if not conf_wrong or not conf_right:
        return float("nan")
    wins = ties = 0
    for w in conf_wrong:
        for r in conf_right:
            if w < r:
                wins += 1
            elif w == r:
                ties += 1
    return (wins + 0.5 * ties) / (len(conf_wrong) * len(conf_right))


def collect(tag: str, pages: list[int], tolerant: bool) -> list[dict]:
    """Join per-cell confidence to per-cell correctness against the GT."""
    out = []
    for page in pages:
        cf = OUT_DIR / f"Hadita_{page}_{tag}.conf.json"
        if not cf.exists():
            continue
        conf_by_rc = {(r["row"], r["col"]): r for r in json.load(open(cf, encoding="utf-8"))}
        gt, pr = load_reference(page), load_pred(page, tag)
        if not gt or pr is None:
            continue
        pairs, off = pairs_by_offset(gt, pr)
        # pairs_by_offset shifts prediction rows onto GT rows; the confidence file
        # is keyed by PREDICTION row, so undo the offset when looking a cell up.
        for kind, g, p in pairs:
            if kind != "matched":
                continue
            for ci, col in enumerate(LEFT_COLS):
                gv, pv = g.get(col, ""), p.get(col, "")
                if tolerant:
                    gv, pv = encoding_fold(gv), encoding_fold(pv)
                gv, pv = _normalize(gv), _normalize(pv)
                if not gv and not pv:
                    continue                      # both blank: nothing to judge
                rec = None
                for (rr, cc), v in conf_by_rc.items():
                    if cc == ci and _normalize(encoding_fold(v["text"])) == pv and pv:
                        rec = v
                        break
                if rec is None or rec.get("confidence") is None:
                    continue
                out.append({"conf": float(rec["confidence"]),
                            "wrong": 0 if gv == pv else 1,
                            "col": col, "gt": gv, "pred": pv})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="mistral-ocr")
    ap.add_argument("--pages", nargs="+", type=int, default=PAGES)
    ap.add_argument("--tolerant", action="store_true", default=True,
                    help="fold equivalent codepoints before judging correctness")
    ap.add_argument("--gate", type=float, default=0.4,
                    help="Spearman rho adoption bar from the 2026-06-26 handoff")
    args = ap.parse_args()

    rows = collect(args.tag, args.pages, args.tolerant)
    if not rows:
        sys.exit(f"no confidence data for tag {args.tag!r}. Run run_mistral_ocr.py "
                 f"first (it writes exp2608/Hadita_N_{args.tag}.conf.json).")

    n = len(rows)
    wrong = [r for r in rows if r["wrong"]]
    conf_w = [r["conf"] for r in wrong]
    conf_r = [r["conf"] for r in rows if not r["wrong"]]
    rho = spearman([r["conf"] for r in rows], [float(r["wrong"]) for r in rows])
    a = auc(conf_w, conf_r)

    print(f"\n{'='*66}\nMistral confidence calibration — tag {args.tag!r}, "
          f"pages {args.pages}\n{'='*66}")
    print(f"  cells with confidence + GT : {n}")
    print(f"  wrong                      : {len(wrong)} ({len(wrong)/n:.1%})")
    if conf_w and conf_r:
        print(f"  mean confidence  correct   : {sum(conf_r)/len(conf_r):.4f}")
        print(f"  mean confidence  wrong     : {sum(conf_w)/len(conf_w):.4f}")

    print(f"\n  Spearman rho (conf vs error) : {rho:+.3f}   "
          f"[gate: rho < -{args.gate} means confidence predicts error]")
    print(f"  AUC (wrong ranks below right): {a:.3f}   [0.5 = useless]")

    # Sign note: error is coded 1=wrong, so a USEFUL confidence gives NEGATIVE rho.
    useful = (rho == rho) and rho <= -args.gate
    print(f"\n  VERDICT: {'PASSES' if useful else 'FAILS'} the "
          f"rho>{args.gate} bar from the 2026-06-26 handoff")

    print("\n  Confidence deciles (low confidence first):")
    print(f"    {'decile':<8}{'cells':>7}{'wrong':>7}{'err rate':>10}{'mean conf':>11}")
    srt = sorted(rows, key=lambda r: r["conf"])
    for d in range(10):
        lo, hi = d * n // 10, (d + 1) * n // 10
        chunk = srt[lo:hi]
        if not chunk:
            continue
        w = sum(c["wrong"] for c in chunk)
        print(f"    {d+1:<8}{len(chunk):>7}{w:>7}{w/len(chunk):>9.1%}"
              f"{sum(c['conf'] for c in chunk)/len(chunk):>11.4f}")

    print("\n  Review-budget curve (review the least-confident X% of cells):")
    print(f"    {'budget':<9}{'cells':>7}{'errors caught':>15}{'of all errors':>15}")
    tot_err = len(wrong)
    for pct in (5, 10, 20, 30, 50):
        k = max(1, n * pct // 100)
        caught = sum(c["wrong"] for c in srt[:k])
        share = caught / tot_err if tot_err else float("nan")
        print(f"    {str(pct)+'%':<9}{k:>7}{caught:>15}{share:>14.1%}")
    print(f"\n  (random review of {n and 20}% would catch ~20% of errors; "
          f"beat that or the score is noise)\n")


if __name__ == "__main__":
    main()
