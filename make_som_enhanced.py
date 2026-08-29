"""E11 helper: pencil-enhanced Set-of-Marks pages for faint-ink pages.

Background-division normalization (img / large-Gaussian-blur) flattens the paper and
amplifies faint pencil strokes, then a mild gamma darkens them; red ink survives
because the division is per-channel. The enhanced page is stamped with the same SoM
row marks and written to som/Hadita_{N}_som_enh.jpg (originals untouched).

  python make_som_enhanced.py 15 17 18
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import make_som_pages as msp


def enhance(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32) + 1.0
    bg = cv2.GaussianBlur(f, (0, 0), 25)
    norm = np.clip(f / (bg * 1.02), 0, 1.0)  # paper -> ~1.0, ink -> <1
    out = 255.0 * norm ** 3.0                # gamma>1 amplifies faint strokes
    return np.clip(out, 0, 255).astype(np.uint8)


def stamp_enhanced(page: int, margin: int = 190) -> Path:
    img = cv2.imread(str(msp.FINAL2 / f"Hadita_{page}.jpeg"))
    img = enhance(img)
    h, w = img.shape[:2]
    bands = msp.row_bands(msp.FINAL2 / f"Hadita_{page}.xml")
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
        cv2.line(canvas, (margin, y0), (margin + w, y0), (0, 200, 255), 2)
    if bands:
        last = max(bands)
        cv2.line(canvas, (margin, bands[last][1]),
                 (margin + w, bands[last][1]), (0, 200, 255), 2)
    out_p = msp.OUT / f"Hadita_{page}_som_enh.jpg"
    cv2.imwrite(str(out_p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"page {page}: enhanced+stamped -> {out_p}")
    return out_p


if __name__ == "__main__":
    for p in [int(a) for a in sys.argv[1:]]:
        stamp_enhanced(p)
