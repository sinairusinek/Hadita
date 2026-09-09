"""E18: stamp BOTH row and column indices onto a page (2-D set-of-marks).

make_som_pages.py marks rows only. Columns were left to the model, which had to
map the prompt's 19-name list onto the page by counting cells right-to-left --
exactly the counting task set-of-marks was introduced to remove. On page 11 that
produced a 4-5 column drift in the tail money columns (Total_Tax vs
Net_Assessment), and it is the same defect as the earlier `som-f3v2x` "the check
mark moved one column".

This adds a column number above each column plus a vertical rule at each column's
left edge, so the model READS a column number instead of counting to it.

  python make_som_grid.py 11 --dir "Transkribus upload/final3" --out som_grid

Design notes:
  * Column labels go in a TOP margin, in BLUE, to stay distinguishable from the
    red row numbers. Row marks keep their existing colour and geometry so this
    is a pure addition -- anything that regresses is attributable to the columns.
  * The vertical rules are drawn at the same faint yellow as the row separators
    and are 1px thinner, so they read as grid, not as ink.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

SRC = Path("Transkribus upload/final3")
OUT = Path("som_grid")

CELL_RX = re.compile(
    r'<TableCell id="cell_r(\d+)_c(\d+)"[^>]*>\s*<Coords points="([^"]+)"')


def bands(xml_path: Path):
    """Return (row_bands, col_bands) as index -> (lo, hi) on each axis."""
    xml = xml_path.read_text(encoding="utf-8")
    rows: dict[int, list] = {}
    cols: dict[int, list] = {}
    for m in CELL_RX.finditer(xml):
        r, c = int(m.group(1)), int(m.group(2))
        pts = np.array([[int(a) for a in p.split(",")] for p in m.group(3).split()])
        rows.setdefault(r, []).append(pts)
        cols.setdefault(c, []).append(pts)
    rb = {r: (int(np.vstack(p)[:, 1].min()), int(np.vstack(p)[:, 1].max()))
          for r, p in rows.items()}
    cb = {c: (int(np.vstack(p)[:, 0].min()), int(np.vstack(p)[:, 0].max()))
          for c, p in cols.items()}
    return rb, cb


def stamp(page: int, margin: int = 190, top: int = 150) -> Path | None:
    img_p = SRC / f"Hadita_{page}.jpeg"
    xml_p = SRC / f"Hadita_{page}.xml"
    if not img_p.exists() or not xml_p.exists():
        print(f"  ! page {page}: missing inputs")
        return None
    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]
    rb, cb = bands(xml_p)

    canvas = np.full((h + top, w + 2 * margin, 3), 255, np.uint8)
    canvas[top:top + h, margin:margin + w] = img

    # --- rows: unchanged from make_som_pages.py, shifted down by `top` ---
    for r, (y0, y1) in sorted(rb.items()):
        yc = (y0 + y1) // 2 + top
        label = str(r)
        for x_anchor in (margin - 20, margin + w + 20):
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
            x = x_anchor - tw if x_anchor < margin else x_anchor
            cv2.putText(canvas, label, (x, yc + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.line(canvas, (margin, y0 + top), (margin + w, y0 + top), (0, 200, 255), 2)
    if rb:
        last = max(rb)
        cv2.line(canvas, (margin, rb[last][1] + top),
                 (margin + w, rb[last][1] + top), (0, 200, 255), 2)

    # --- columns: the new part ---
    for c, (x0, x1) in sorted(cb.items()):
        xc = (x0 + x1) // 2 + margin
        label = str(c)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
        cv2.putText(canvas, label, (xc - tw // 2, top - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4, cv2.LINE_AA)
        # tick joining the label to its column, then a full-height rule
        cv2.line(canvas, (xc, top - 22), (xc, top - 4), (255, 0, 0), 3)
        cv2.line(canvas, (x0 + margin, top), (x0 + margin, top + h), (0, 200, 255), 1)
    if cb:
        last = max(cb)
        cv2.line(canvas, (cb[last][1] + margin, top),
                 (cb[last][1] + margin, top + h), (0, 200, 255), 1)

    OUT.mkdir(parents=True, exist_ok=True)
    out_p = OUT / f"Hadita_{page}_som.jpg"
    cv2.imwrite(str(out_p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  page {page}: {len(rb)} rows + {len(cb)} cols stamped -> {out_p}")
    return out_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    global SRC, OUT
    if args.dir:
        SRC = Path(args.dir)
    if args.out:
        OUT = Path(args.out)
    for p in args.pages:
        stamp(p)


if __name__ == "__main__":
    main()
