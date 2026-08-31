#!/usr/bin/env python3
"""make_kraken_gt.py — build a ketos training set from the proxy GT.

One PNG + one .gt.txt per non-empty GT cell (kraken's `path` format). Cells are
cropped from the final3 geometry, so the crop a model trains on is exactly the
crop it will be asked to read.

Pages 9 and 10 are held out by default: they are the sparse pages, and with only
six GT pages a random split would put nearly all the training signal on the four
dense ones anyway. Holding out whole pages also avoids leaking a page's own hand
into its evaluation.

  python make_kraken_gt.py --out kraken_gt
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from digit_norm import LEFT_COLS  # noqa: E402
from score_g3_vs_gt import load_trx_xml  # noqa: E402
from push_gt_to_final3 import best_offset  # noqa: E402

D = ROOT / "Transkribus upload" / "final3"
GT_FMT = "g3_results/Hadita_{n}_Transkribus_latest.xml"


def cell_polys(page: int) -> dict[tuple[int, int], np.ndarray]:
    t = ET.parse(D / f"Hadita_{page}.xml")
    out = {}
    for tc in t.getroot().iter():
        if not tc.tag.endswith("TableCell"):
            continue
        r, c = tc.get("row"), tc.get("col")
        co = [e for e in tc.iter() if e.tag.endswith("Coords")]
        if r is None or not co:
            continue
        out[(int(r), int(c))] = np.array(
            [tuple(map(int, q.split(","))) for q in co[0].get("points").split()], np.int32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="kraken_gt")
    ap.add_argument("--train-pages", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--val-pages", type=int, nargs="+", default=[9, 10])
    ap.add_argument("--pad", type=int, default=6)
    args = ap.parse_args()

    for split, pages in (("train", args.train_pages), ("val", args.val_pages)):
        d = ROOT / args.out / split
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for page in pages:
            gt = load_trx_xml(GT_FMT.format(n=page))
            polys = cell_polys(page)
            # Serial-anchored offset: on p4/p5 the GT carries a leading
            # carry-forward row that the grid does not, so index alignment
            # pairs every crop with the WRONG row's transcription.
            xml = (D / f"Hadita_{page}.xml").read_text(encoding="utf-8")
            ser = {int(m.group(1))
                   for m in re.finditer(r'id="line_cell_r(\d+)_c0"', xml)}
            off = best_offset(gt, ser)
            img = cv2.imread(str(D / f"Hadita_{page}.jpeg"))
            for r, row in enumerate(gt):
                for ci, col in enumerate(LEFT_COLS):
                    txt = str(row.get(col, "") or "").strip()
                    rr = r + off
                    if not txt or (rr, ci) not in polys:
                        continue
                    p = polys[(rr, ci)]
                    x0, y0 = max(0, p[:, 0].min() - args.pad), max(0, p[:, 1].min() - args.pad)
                    x1, y1 = min(img.shape[1], p[:, 0].max() + args.pad), min(img.shape[0], p[:, 1].max() + args.pad)
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    stem = d / f"p{page}_r{rr:02d}_c{ci:02d}"
                    cv2.imwrite(str(stem.with_suffix(".png")), img[y0:y1, x0:x1])
                    stem.with_suffix(".gt.txt").write_text(txt, encoding="utf-8")
                    n += 1
        print(f"{split}: {n} samples -> {d}")


if __name__ == "__main__":
    main()
