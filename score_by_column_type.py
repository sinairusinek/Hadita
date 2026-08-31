"""E8 scorer: split accuracy by column type, and measure empty-cell hallucination.

Tests the provenance-derived prediction for Baseer__Nakba: strong on Arabic-word
columns, weak/hallucinatory on Arabic-Indic numeral columns and on empty cells.

Also reports the specific numeral confusions the talk cares about (2/3, 3/4, 4/6, 6/8).

  python score_by_column_type.py --tags som-g37-flash baseer
"""
from __future__ import annotations

import argparse
from collections import Counter

from score_exp2608 import load_reference, load_pred, pairs_by_alignment
from score_g3_vs_gt import LEFT_COLS, classify, _normalize

PAGES = [3, 4, 5, 6, 9, 10]

# Column taxonomy. "Nature_of_Entry" is the one genuinely word-bearing column;
# the rest of the register is numerals, check marks and dashes.
WORD_COLS = {"Nature_of_Entry"}
NUMERAL_COLS = set(LEFT_COLS) - WORD_COLS - {"Remarks"}

AR_DIGITS = set("٠١٢٣٤٥٦٧٨٩")
CONFUSION_PAIRS = [("٢", "٣"), ("٣", "٤"), ("٤", "٦"), ("٦", "٨")]


def is_numeric(s: str) -> bool:
    s = _normalize(s)
    return bool(s) and any(ch in AR_DIGITS for ch in s)


def digit_confusions(gt: str, pr: str) -> list[tuple[str, str]]:
    """Same-length digit strings differing in one position -> that substitution."""
    g, p = _normalize(gt), _normalize(pr)
    if not g or not p or len(g) != len(p):
        return []
    return [(a, b) for a, b in zip(g, p) if a != b and a in AR_DIGITS and b in AR_DIGITS]


def analyse(tag: str, pages: list[int]) -> dict:
    st = {
        "num_total": 0, "num_perfect": 0,
        "word_total": 0, "word_perfect": 0,
        "empty_gt_filled_pred": 0, "empty_gt_total": 0,
        "nonempty_gt_empty_pred": 0,
        "confusions": Counter(), "all_subs": Counter(),
        "pages": [],
    }
    for page in pages:
        gt = load_reference(page)
        pr = load_pred(page, tag)
        if not gt or pr is None:
            continue
        st["pages"].append(page)
        for kind, g, p in pairs_by_alignment(gt, pr):
            if kind != "matched":
                continue
            for c in LEFT_COLS:
                if c == "Remarks":
                    continue
                gv, pv = _normalize(g.get(c, "")), _normalize(p.get(c, ""))
                if not gv:
                    st["empty_gt_total"] += 1
                    if pv:
                        st["empty_gt_filled_pred"] += 1
                    continue
                if not pv:
                    st["nonempty_gt_empty_pred"] += 1
                bucket = "word" if c in WORD_COLS else "num"
                st[f"{bucket}_total"] += 1
                if classify(gv, pv) == "perfect":
                    st[f"{bucket}_perfect"] += 1
                for a, b in digit_confusions(gv, pv):
                    st["all_subs"][(a, b)] += 1
                    key = tuple(sorted((a, b)))
                    if key in [tuple(sorted(x)) for x in CONFUSION_PAIRS]:
                        st["confusions"][key] += 1
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--pages", nargs="+", type=int, default=PAGES)
    args = ap.parse_args()

    rows = []
    for tag in args.tags:
        st = analyse(tag, args.pages)
        if not st["pages"]:
            print(f"{tag}: no data")
            continue
        nt, np_ = st["num_total"], st["num_perfect"]
        wt, wp = st["word_total"], st["word_perfect"]
        et, ef = st["empty_gt_total"], st["empty_gt_filled_pred"]
        rows.append((tag, st, nt, np_, wt, wp, et, ef))

    print(f"\n{'tag':<20}{'numeral cells':>16}{'word cells':>14}"
          f"{'empty-GT filled':>18}{'GT-text missed':>16}")
    print("-" * 84)
    for tag, st, nt, np_, wt, wp, et, ef in rows:
        na = f"{np_}/{nt} ({100*np_/nt:.1f}%)" if nt else "n/a"
        wa = f"{wp}/{wt} ({100*wp/wt:.0f}%)" if wt else "n/a"
        ha = f"{ef}/{et} ({100*ef/et:.1f}%)" if et else "n/a"
        print(f"{tag:<20}{na:>16}{wa:>14}{ha:>18}"
              f"{st['nonempty_gt_empty_pred']:>16}")

    print("\nTargeted numeral confusions (GT digit vs predicted digit, same-length cells):")
    for tag, st, *_ in rows:
        tot = sum(st["confusions"].values())
        detail = "  ".join(f"{a}/{b}:{n}" for (a, b), n in
                           sorted(st["confusions"].items(), key=lambda x: -x[1]))
        print(f"  {tag:<20} {tot:>3} in the four target pairs   {detail}")
    print("\nTop digit substitutions overall (GT→pred):")
    for tag, st, *_ in rows:
        top = "  ".join(f"{a}→{b}:{n}" for (a, b), n in st["all_subs"].most_common(6))
        print(f"  {tag:<20} {top}")
    print(f"\npages: {rows[0][1]['pages'] if rows else '-'}")


if __name__ == "__main__":
    main()
