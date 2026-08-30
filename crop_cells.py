"""Extract per-cell (or per-row) crops from the final2 corpus by XML polygon geometry.

Shared primitive for line-level recognition models (E8). The polygons are curved
quads following the page bow, so a crop is the polygon's bounding rect with the
outside-polygon area whitened, then optionally deskewed to a rectangle.

  python crop_cells.py 9 --out-dir crops --inked-only
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from gate_textlines import hand_mask

FINAL2 = Path("Transkribus upload/final2")
CELL_RX = re.compile(
    r'<TableCell id="cell_r(\d+)_c(\d+)"[^>]*>\s*<Coords points="([^"]+)"')
INK_MIN_PX = 12


def parse_cells(xml_path: Path) -> dict[tuple[int, int], np.ndarray]:
    xml = xml_path.read_text(encoding="utf-8")
    return {(int(m.group(1)), int(m.group(2))):
            np.array([[int(a) for a in p.split(",")] for p in m.group(3).split()],
                     dtype=np.int32)
            for m in CELL_RX.finditer(xml)}


def cell_ink(mask: np.ndarray, poly: np.ndarray) -> int:
    cm = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(cm, [poly], 1)
    return int((mask & cm).sum())


def crop_cell(img: np.ndarray, poly: np.ndarray, pad: int = 6,
              whiten_outside: bool = True) -> np.ndarray:
    """Bounding-rect crop of a (possibly curved) cell quad."""
    x, y, w, h = cv2.boundingRect(poly)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    sub = img[y0:y1, x0:x1].copy()
    if whiten_outside:
        m = np.zeros(sub.shape[:2], np.uint8)
        cv2.fillPoly(m, [poly - [x0, y0]], 1)
        m = cv2.dilate(m, np.ones((pad * 2 + 1,) * 2, np.uint8))
        sub[m == 0] = 255
    return sub


def crop_page(page: int, out_dir: Path, inked_only: bool, pad: int,
              src_dir: Path = None,
              min_side: int = 32) -> list[dict]:
    src = src_dir or FINAL2
    img_p, xml_p = src / f"Hadita_{page}.jpeg", src / f"Hadita_{page}.xml"
    if not img_p.exists() or not xml_p.exists():
        print(f"  ! page {page}: missing inputs")
        return []
    img = cv2.imread(str(img_p))
    cells = parse_cells(xml_p)
    mask = hand_mask(img) if inked_only else None

    page_dir = out_dir / f"page{page}"
    page_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for (r, c), poly in sorted(cells.items()):
        if mask is not None and cell_ink(mask, poly) < INK_MIN_PX:
            continue
        sub = crop_cell(img, poly, pad=pad)
        if sub.shape[0] < min_side or sub.shape[1] < min_side:
            continue
        name = f"r{r:02d}_c{c:02d}.png"
        cv2.imwrite(str(page_dir / name), sub)
        index.append({"page": page, "row": r, "col": c, "file": str(page_dir / name),
                      "w": int(sub.shape[1]), "h": int(sub.shape[0])})
    (page_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"  page {page}: {len(index)} crops -> {page_dir}")
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--out-dir", default="crops")
    ap.add_argument("--inked-only", action="store_true",
                    help="skip cells with no handwriting (recommended)")
    ap.add_argument("--pad", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out_dir)
    total = 0
    for p in args.pages:
        total += len(crop_page(p, out, args.inked_only, args.pad))
    print(f"total crops: {total}")


if __name__ == "__main__":
    main()
