"""Ink-gate a Hadid02 recognition pull: blank any cell whose final2 polygon has
fewer than INK_MIN_PX handwriting pixels. Writes exp2608/Hadita_{N}_hadid02_gated.json
for use as an agreement_layer source.

  python gate_hadid_pull.py 11 12 ... 20
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from gate_textlines import hand_mask, INK_MIN_PX
from agreement_layer import load_source
from digit_norm import LEFT_COLS

FINAL2 = Path("Transkribus upload/final2")
CELL_COORD_RE = re.compile(
    r'<TableCell\s+id="cell_r(\d+)_c(\d+)"[^>]*>.*?<Coords points="([^"]+)"',
    re.DOTALL)


def cell_ink(page: int) -> dict[tuple[int, int], int]:
    img = cv2.imread(str(FINAL2 / f"Hadita_{page}.jpeg"))
    mask = hand_mask(img)
    out = {}
    xml = (FINAL2 / f"Hadita_{page}.xml").read_text(encoding="utf-8")
    for m in CELL_COORD_RE.finditer(xml):
        r, c = int(m.group(1)), int(m.group(2))
        pts = np.array([[int(v) for v in p.split(",")] for p in m.group(3).split()])
        cell_mask = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(cell_mask, [pts], 255)
        out[(r, c)] = int(cv2.countNonZero(cv2.bitwise_and(mask, cell_mask)))
    return out


def main() -> None:
    for page in [int(a) for a in sys.argv[1:]]:
        rows = load_source(Path(f"g3_results/Hadita_{page}_Hadid02_final2.xml"))
        ink = cell_ink(page)
        kept = dropped = 0
        for r, row in enumerate(rows):
            for ci, col in enumerate(LEFT_COLS):
                if row.get(col, "").strip():
                    if ink.get((r, ci), 0) < INK_MIN_PX:
                        row[col] = ""
                        dropped += 1
                    else:
                        kept += 1
        out = Path(f"exp2608/Hadita_{page}_hadid02_gated.json")
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"page {page}: kept {kept}, dropped {dropped} inkless cells -> {out}")


if __name__ == "__main__":
    main()
