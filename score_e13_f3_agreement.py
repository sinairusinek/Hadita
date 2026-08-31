"""E13: redo the E9 agreement tiering on final3 geometry.

E9 (final2) found Gemini x Baseer/NAKBA unusable as an auto-accept gate:
cells where som and baseer agreed but hadid02 did not were wrong 20.95% of
the time. Half the final2 corpus had defective grids (E12), so that number
may have been measuring bad cell boundaries rather than bad reading.

final3 removes that confound: som, baseer, kraken and GT are all built on the
same lattice, so cells compare directly by (row, col) with no NW alignment.

Tiers, exactly as E9 defined them, on cells where som has content:
  both        som == baseer  and  som == kraken
  baseer_only som == baseer  and  som != kraken
  kraken_only som != baseer  and  som == kraken
  neither     som != baseer  and  som != kraken

Error rate inside a tier = fraction of its cells where som != GT. A tier is
usable for auto-accept if that rate is <= 1% (the Phase-4 gate).

Remarks is app metadata, not a register column, and is excluded throughout.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from digit_norm import LEFT_COLS  # noqa: E402
from score_g3_vs_gt import _normalize  # noqa: E402

OUT = Path("exp2608")
PAGES = [3, 4, 5, 6, 9, 10]   # 9,10 GT recovered from doc 15829823 (recover_gt_9_10.py)
COLS = [c for c in LEFT_COLS if c != "Remarks"]

# som is the primary reader; baseer and kraken are the two candidate partners.
LAYERS = {
    "gt": "gt-f3",
    "som": "som-f3v2",
    "baseer": "baseer-f3",
    "kraken": "kraken-f3c",
}


def load(page: int, suffix: str) -> list[dict] | None:
    p = OUT / f"Hadita_{page}_{suffix}.json"
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))


def load_gt_xml(page: int) -> list[dict] | None:
    """GT ships as PAGE XML; read cells from line_cell_r{r}_c{c} TextLines."""
    import re
    import xml.etree.ElementTree as ET

    p = OUT / f"Hadita_{page}_gt-f3.xml"
    if not p.exists():
        return None
    ns = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
    root = ET.parse(p).getroot()
    cells: dict[tuple[int, int], str] = {}
    maxr = -1
    for tl in root.iter(f"{{{ns['p']}}}TextLine"):
        m = re.match(r"line_cell_r(\d+)_c(\d+)", tl.get("id", ""))
        if not m:
            continue
        r, c = int(m.group(1)), int(m.group(2))
        uni = tl.find(f".//{{{ns['p']}}}Unicode")
        cells[(r, c)] = (uni.text or "").strip() if uni is not None else ""
        maxr = max(maxr, r)
    if maxr < 0:
        return None
    return [
        {c: cells.get((r, i), "") for i, c in enumerate(LEFT_COLS)}
        for r in range(maxr + 1)
    ]


def main() -> None:
    tiers = Counter()
    errs = Counter()
    per_page = []
    disagree_examples = []

    for page in PAGES:
        gt = load_gt_xml(page)
        som = load(page, LAYERS["som"])
        bas = load(page, LAYERS["baseer"])
        krk = load(page, LAYERS["kraken"])
        if not all(x is not None for x in (gt, som, bas, krk)):
            print(f"page {page}: MISSING a layer, skipped")
            continue

        n = min(len(gt), len(som), len(bas), len(krk))
        pt, pe = Counter(), Counter()
        for r in range(n):
            for c in COLS:
                s = _normalize((som[r].get(c) or "").strip())
                if not s:
                    continue  # gate is about accepting content, not blanks
                g = _normalize((gt[r].get(c) or "").strip())
                b = _normalize((bas[r].get(c) or "").strip())
                k = _normalize((krk[r].get(c) or "").strip())
                tier = ("both" if (s == b and s == k) else
                        "baseer_only" if s == b else
                        "kraken_only" if s == k else "neither")
                pt[tier] += 1
                if s != g:
                    pe[tier] += 1
                    if tier == "baseer_only" and len(disagree_examples) < 12:
                        disagree_examples.append(
                            (page, r, c, f"som={s!r} gt={g!r} baseer={b!r}"))
        tiers.update(pt)
        errs.update(pe)
        per_page.append((page, n, pt, pe))

    print("\n=== E13: agreement tiers on final3 (6 proxy-GT pages) ===")
    print("primary=som-f3v2  partners=baseer-f3, kraken-f3c  ref=gt-f3\n")
    print(f"{'page':>5} {'rows':>5}  " + "  ".join(f"{t:>16}" for t in
          ("both", "baseer_only", "kraken_only", "neither")))
    for page, n, pt, pe in per_page:
        cells = "  ".join(
            f"{pe[t]:>3}/{pt[t]:<4}{100*pe[t]/pt[t] if pt[t] else 0:>7.1f}%"
            for t in ("both", "baseer_only", "kraken_only", "neither"))
        print(f"{page:>5} {n:>5}  {cells}")

    print(f"\n{'TIER':<14} {'cells':>7} {'errors':>7} {'err%':>8}  verdict")
    for t in ("both", "baseer_only", "kraken_only", "neither"):
        n, e = tiers[t], errs[t]
        pct = 100 * e / n if n else 0.0
        verdict = ("PASS (<=1%)" if pct <= 1.0 and n else
                   "FAIL" if n else "-")
        print(f"{t:<14} {n:>7} {e:>7} {pct:>7.2f}%  {verdict}")

    tot = sum(tiers.values())
    acc = tiers["both"] + tiers["kraken_only"]
    acc_e = errs["both"] + errs["kraken_only"]
    print(f"\ntotal content cells: {tot}")
    if tot:
        print(f"auto-accept if gate = som==kraken (both+kraken_only): "
              f"{acc} cells = {100*acc/tot:.1f}%, err {100*acc_e/acc if acc else 0:.2f}%")
        ba = tiers["both"] + tiers["baseer_only"]
        ba_e = errs["both"] + errs["baseer_only"]
        print(f"auto-accept if gate = som==baseer (both+baseer_only): "
              f"{ba} cells = {100*ba/tot:.1f}%, err {100*ba_e/ba if ba else 0:.2f}%")

    if disagree_examples:
        print("\nsample baseer_only errors (som+baseer agree, both wrong):")
        for page, r, c, s in disagree_examples:
            print(f"  p{page} r{r:<3} {c[:34]:<34} {s}")


if __name__ == "__main__":
    main()
