#!/usr/bin/env python3
"""audit_final3.py — flag pages in a built upload folder that need human eyes.

Deliberately NOT a pass/fail score. Row COUNT cannot tell a correct grid from
one that is half a row out of phase (a build once scored 34/34 rows while every
band edge bisected the handwriting), so this only sorts pages into "looks like
the others" and "look at this one", and every flag names what to look at.

Checks, per page:
  cols      19 columns emitted (a miscount mislabels every cell to its right)
  rows      row count within the corpus's own IQR, not against a fixed 34
  ink_edge  fraction of band EDGES that cross handwriting: the phase check.
            High means the lattice is cutting through the writing.
  ink_out   handwriting ink below the last band: rows the grid failed to reach.
"""
import csv, re, sys, xml.etree.ElementTree as ET
from typing import Optional
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from gate_textlines import hand_mask  # noqa: E402

EDGE_BAND = 6      # px each side of an edge counted as "on the edge"
EDGE_FLAG = 0.25   # fraction of edges crossing ink before we flag phase
OUT_FLAG  = 4000   # handwriting px below the last band before we flag a cut


def page_stats(xml_path: Path) -> Optional[dict]:
    jpeg = xml_path.with_suffix(".jpeg")
    if not jpeg.exists():
        return None
    t = ET.parse(xml_path); ns = {"p": t.getroot().tag.split("}")[0][1:]}
    pg = t.getroot().find("p:Page", ns)
    bands, cols = set(), set()
    for tl in pg.findall(".//p:TextLine", ns):
        m = re.match(r"line_cell_r(\d+)_c(\d+)", tl.get("id") or "")
        if not m:
            continue
        cols.add(int(m.group(2)))
        ys = [y for _, y in (tuple(map(int, q.split(","))) for q in
                             tl.find("p:Coords", ns).get("points").split())]
        bands.add((min(ys), max(ys)))
    if not bands:
        return None
    bands = sorted(bands)
    img = cv2.imread(str(jpeg))
    mask = hand_mask(img) > 0
    h, w = mask.shape
    edges = sorted({b[0] for b in bands} | {b[1] for b in bands})
    hits = 0
    for y in edges:
        lo, hi = max(0, y - EDGE_BAND), min(h, y + EDGE_BAND)
        if mask[lo:hi, :].sum() > 0.02 * (hi - lo) * w:
            hits += 1
    last = bands[-1][1]
    return {"page": int(xml_path.stem.split("_")[1]),
            "n_cols": len(cols), "n_rows": len(bands),
            "top": bands[0][0], "bottom": last,
            "edge_ink": round(hits / max(1, len(edges)), 3),
            "ink_below": int(mask[min(last + 5, h):, :].sum())}


def main() -> None:
    d = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "Transkribus upload/final3")
    rows = []
    for f in sorted(d.glob("Hadita_*.xml"), key=lambda p: int(p.stem.split("_")[1])):
        s = page_stats(f)
        if s:
            rows.append(s)
    if not rows:
        print("no pages found"); return
    counts = np.array([r["n_rows"] for r in rows])
    q1, q3 = np.percentile(counts, [25, 75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    for r in rows:
        f = []
        if r["n_cols"] != 19:                f.append(f"COLS={r['n_cols']}")
        if not lo <= r["n_rows"] <= hi:      f.append(f"ROWS={r['n_rows']}")
        if r["edge_ink"] > EDGE_FLAG:        f.append(f"EDGE_INK={r['edge_ink']}")
        if r["ink_below"] > OUT_FLAG:        f.append(f"INK_BELOW={r['ink_below']}")
        r["flags"] = " ".join(f)
    with open(ROOT / "audit_final3.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if r["flags"]]
    print(f"{len(rows)} pages; rows median {int(np.median(counts))} "
          f"(IQR {int(q1)}-{int(q3)}, flag outside {lo:.0f}-{hi:.0f})")
    print(f"edge_ink median {np.median([r['edge_ink'] for r in rows]):.3f}")
    print(f"\n{len(bad)} flagged:")
    for r in bad:
        print(f"  p{r['page']:>3}  rows={r['n_rows']:>2} cols={r['n_cols']} "
              f"edge_ink={r['edge_ink']:.2f}  {r['flags']}")
    print("\n-> audit_final3.tsv")


if __name__ == "__main__":
    main()
