"""Per-cell ink measurement for the talk's ink-gate histogram (slide 11).

gate_textlines.py reports per-page totals only. The slide needs the underlying
distribution: how many pixels of handwriting sit under each cell polygon, split
by whether the cell carries text. Reuses the gate's own masks so the histogram
and the shipped threshold describe the same measurement.
"""
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from gate_textlines import CELL_RX, hand_mask, chroma_mask  # noqa: E402

# The final3 XMLs have already been through the gate, so their cells carry no
# text to split the distribution by. The six proxy-GT pages do: RA-corrected
# transcripts against the same final3 geometry.
IMG_DIR = ROOT / "Transkribus upload" / "final3"
GT_DIR = ROOT / "exp2608"
GT_PAGES = [3, 4, 5, 6, 9, 10]
OUT = Path(__file__).parent / "data" / "ink_per_cell.tsv"


def measure(page: int) -> list[tuple]:
    xml_path = GT_DIR / f"Hadita_{page}_gt-f3.xml"
    jpeg = IMG_DIR / f"Hadita_{page}.jpeg"
    if not xml_path.exists() or not jpeg.exists():
        return []
    img = cv2.imread(str(jpeg))
    if img is None:
        return []
    mask, cmask = hand_mask(img), chroma_mask(img)
    rows = []
    for m in CELL_RX.finditer(xml_path.read_text(encoding="utf-8")):
        poly = np.array([[int(a) for a in p.split(",")] for p in m.group("pts").split()],
                        np.int32)
        cell = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(cell, [poly], 1)
        text = " ".join(t.strip() for t in
                        re.findall(r"<Unicode>(.*?)</Unicode>", m.group("rest"), re.S))
        rows.append((page, int(m.group(1)), int(m.group(2)),
                     int((mask & cell).sum()), int((cmask & cell).sum()),
                     1 if text.strip() else 0))
    return rows


def main() -> None:
    pages = GT_PAGES
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["page", "row", "col", "ink_px", "chroma_px", "has_text"])
        for i, page in enumerate(pages, 1):
            rows = measure(page)
            w.writerows(rows)
            print(f"[{i}/{len(pages)}] page {page}: {len(rows)} cells", flush=True)


if __name__ == "__main__":
    main()
