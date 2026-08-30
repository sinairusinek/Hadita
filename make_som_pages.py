"""E3 helper: stamp row indices onto a page image (Set-of-Marks prompting).

Uses the final2 XML cell polygons to find each grid row's y-band, then draws the row
number in the left and right margins plus a faint separator line. The model is then
asked to key its output to the PRINTED row numbers instead of counting rows itself.

  python make_som_pages.py 9 10          -> som/Hadita_{N}_som.jpg
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

FINAL2 = Path("Transkribus upload/final2")
# --dir/--out let the SoM stamp run against a later geometry generation
# (final3) without copying this file; the final2 defaults are unchanged.
SRC = FINAL2
OUT = Path("som")

CELL_RX = re.compile(
    r'<TableCell id="cell_r(\d+)_c(\d+)"[^>]*>\s*<Coords points="([^"]+)"')


def row_bands(xml_path: Path) -> dict[int, tuple[int, int, int, int]]:
    """row index -> (y_top, y_bottom, x_left, x_right) across all its cells."""
    xml = xml_path.read_text(encoding="utf-8")
    acc: dict[int, list] = {}
    for m in CELL_RX.finditer(xml):
        r = int(m.group(1))
        pts = np.array([[int(a) for a in p.split(",")] for p in m.group(3).split()])
        acc.setdefault(r, []).append(pts)
    out = {}
    for r, polys in acc.items():
        allp = np.vstack(polys)
        out[r] = (int(allp[:, 1].min()), int(allp[:, 1].max()),
                  int(allp[:, 0].min()), int(allp[:, 0].max()))
    return out


def stamp(page: int, margin: int = 190) -> Path | None:
    img_p = SRC / f"Hadita_{page}.jpeg"
    xml_p = SRC / f"Hadita_{page}.xml"
    if not img_p.exists() or not xml_p.exists():
        print(f"  ! page {page}: missing inputs")
        return None
    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]
    bands = row_bands(xml_p)

    canvas = np.full((h, w + 2 * margin, 3), 255, np.uint8)
    canvas[:, margin:margin + w] = img

    for r, (y0, y1, x0, x1) in sorted(bands.items()):
        yc = (y0 + y1) // 2
        label = str(r)
        for x_anchor in (margin - 20, margin + w + 20):
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
            x = x_anchor - tw if x_anchor < margin else x_anchor
            cv2.putText(canvas, label, (x, yc + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4, cv2.LINE_AA)
        # faint full-width separator at the row's top edge
        cv2.line(canvas, (margin, y0), (margin + w, y0), (0, 200, 255), 2)
    # close the last band
    if bands:
        last = max(bands)
        cv2.line(canvas, (margin, bands[last][1]),
                 (margin + w, bands[last][1]), (0, 200, 255), 2)

    OUT.mkdir(exist_ok=True)
    out_p = OUT / f"Hadita_{page}_som.jpg"
    cv2.imwrite(str(out_p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  page {page}: {len(bands)} rows stamped -> {out_p}")
    return out_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--dir", help="source upload folder (default final2)")
    ap.add_argument("--out", help="output folder (default som/)")
    args = ap.parse_args()

    global SRC, OUT
    if args.dir:
        SRC = Path(args.dir)
    if args.out:
        OUT = Path(args.out)
        OUT.mkdir(parents=True, exist_ok=True)
    for p in args.pages:
        stamp(p)


if __name__ == "__main__":
    main()
