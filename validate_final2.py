#!/usr/bin/env python3
"""validate_final2.py — structural checks on Transkribus upload/final2/.

Catches the failure modes that have cost us Transkribus uploads or silent
mis-tagging before, one check per known incident:

  imageFilename    must name the paired file exactly, or Transkribus rescales
                   every coordinate (house rule 8)
  image size       imageWidth/Height must match the actual JPEG
  bounds           no cell corner outside the image
  geometry         no zero-area or self-crossing quad; rows must not overlap
  columns          19 columns, tagged positionally from LEFT_COLS, with
                   Net_Assessment_Mils last — 59 pages in final/ had 18
  escaping         raw <, > or & inside <Unicode> makes Transkribus 500
                   (house rule 4)
  baselines        one per TextLine, inside its cell's y-range

Usage:
  python validate_final2.py                 # everything in final2/
  python validate_final2.py --dir "Transkribus upload/final"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from segment_unified import LEFT_COLS  # noqa: E402

NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
DEFAULT_DIR = ROOT / "Transkribus upload" / "final2"


def _pts(s: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in p.split(",")) for p in s.split()]


def _area(poly: list[tuple[int, int]]) -> float:
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n))) / 2


def check_page(xml_path: Path) -> list[str]:
    problems: list[str] = []
    raw = xml_path.read_text(encoding="utf-8")

    # Escaping: look at the literal text between the Unicode tags.
    for body in re.findall(r"<Unicode>(.*?)</Unicode>", raw, re.S):
        stripped = re.sub(r"&(amp|lt|gt|quot|apos|#\d+);", "", body)
        if any(ch in stripped for ch in "<>&"):
            problems.append(f"unescaped markup character in <Unicode>: {body[:40]!r}")
            break

    root = ET.fromstring(raw)
    page = root.find(f"{NS}Page")
    if page is None:
        return problems + ["no <Page> element"]

    img_name = page.attrib.get("imageFilename", "")
    if img_name != f"{xml_path.stem}.jpeg":
        problems.append(f"imageFilename={img_name!r} does not pair with {xml_path.name}")
    jpeg = xml_path.with_suffix(".jpeg")
    if not jpeg.exists():
        problems.append("paired jpeg missing")
    else:
        import cv2
        im = cv2.imread(str(jpeg))
        w, h = int(page.attrib["imageWidth"]), int(page.attrib["imageHeight"])
        if im is None:
            problems.append("paired jpeg unreadable")
        elif (im.shape[1], im.shape[0]) != (w, h):
            problems.append(f"declared size {w}x{h} != actual {im.shape[1]}x{im.shape[0]}")

    W, H = int(page.attrib["imageWidth"]), int(page.attrib["imageHeight"])
    table = page.find(f"{NS}TableRegion")
    if table is None:
        return problems + ["no <TableRegion>"]

    n_cols = int(table.attrib["columns"])
    if n_cols != len(LEFT_COLS):
        problems.append(f"{n_cols} columns, expected {len(LEFT_COLS)}")

    row_y: dict[int, tuple[int, int]] = {}
    tags_by_col: dict[int, set[str]] = {}
    for cell in table.iter(f"{NS}TableCell"):
        r, c = int(cell.attrib["row"]), int(cell.attrib["col"])
        poly = _pts(cell.find(f"{NS}Coords").attrib["points"])
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        if min(xs) < 0 or min(ys) < 0 or max(xs) > W or max(ys) > H:
            problems.append(f"cell r{r}c{c} outside the image")
        if _area(poly) < 100:
            problems.append(f"cell r{r}c{c} is degenerate (area {_area(poly):.0f})")
        prev = row_y.setdefault(r, (min(ys), max(ys)))
        row_y[r] = (min(prev[0], min(ys)), max(prev[1], max(ys)))
        m = re.search(r"structure \{type:([^;]+);\}", cell.attrib.get("custom", ""))
        if m:
            tags_by_col.setdefault(c, set()).add(m.group(1))

        for tl in cell.iter(f"{NS}TextLine"):
            bl = tl.find(f"{NS}Baseline")
            if bl is None:
                problems.append(f"cell r{r}c{c} has no baseline")
                continue
            by = [p[1] for p in _pts(bl.attrib["points"])]
            if not all(min(ys) <= y <= max(ys) for y in by):
                problems.append(f"cell r{r}c{c} baseline outside its cell")

    for c, tags in sorted(tags_by_col.items()):
        expected = LEFT_COLS[c] if c < len(LEFT_COLS) else None
        if len(tags) > 1:
            problems.append(f"column {c} has mixed tags {sorted(tags)}")
        elif expected and next(iter(tags)) != expected:
            problems.append(f"column {c} tagged {next(iter(tags))!r}, expected {expected!r}")

    for r in sorted(row_y)[:-1]:
        if r + 1 in row_y and row_y[r][1] > row_y[r + 1][1]:
            problems.append(f"row {r} extends below row {r+1}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    d = Path(args.dir)

    xmls = sorted(d.glob("Hadita_*.xml"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Validating {len(xmls)} page(s) in {d}\n")
    bad = 0
    for x in xmls:
        try:
            problems = check_page(x)
        except Exception as exc:
            problems = [f"parse failure: {exc}"]
        if problems:
            bad += 1
            print(f"{x.name}:")
            for p in problems[:6]:
                print(f"    {p}")
            if len(problems) > 6:
                print(f"    … and {len(problems) - 6} more")
    print(f"\n{len(xmls) - bad}/{len(xmls)} pages clean")


if __name__ == "__main__":
    main()
