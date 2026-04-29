"""Patch <Baseline> y-coordinates in existing PAGE XMLs without touching geometry.

Reads the XMLs in 'Transkribus upload/{original,western arabic transliteration}/'
for the requested page(s), recomputes every <Baseline> from its TextLine's
<Coords> y-range using the target frac, and writes the file back in-place.

  Cell baselines          frac = 0.90  (10 % above the cell bottom)
  taxpayer_name baseline  frac = 0.90
  taxpayer_index baseline frac = 0.97  (nearly flush with bottom)

X-coordinates of each baseline are taken from the *existing* baseline so that
the column-aligned x values written by write_page_xml are preserved exactly.
No dewarp, no OCR, no geometry changes.

Usage:
    python patch_baselines.py              # patches pages 10 and 50
    python patch_baselines.py --page 10
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import xml.etree.ElementTree as ET

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
ET.register_namespace("", NS)


CELL_FRAC = 0.90
STRIP_FRACS = {
    "taxpayer_name": 0.90,
    "taxpayer_index": 0.97,
}

UPLOAD_DIR = Path(__file__).parent / "Transkribus upload"
VARIANTS = ["original", "western arabic transliteration"]


def _parse_pts(points_str: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in p.split(",")) for p in points_str.split()]


def _y_range(coords_elem) -> tuple[int, int]:
    pts = _parse_pts(coords_elem.attrib["points"])
    ys = [p[1] for p in pts]
    return min(ys), max(ys)


def _bl_x_range(baseline_elem) -> tuple[int, int]:
    pts = _parse_pts(baseline_elem.attrib["points"])
    xs = [p[0] for p in pts]
    return min(xs), max(xs)


def patch_xml(xml_path: Path) -> int:
    """Patch baselines in one XML file; return number of baselines changed."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tag = lambda name: f"{{{NS}}}{name}"
    changed = 0

    # --- TableCell baselines ---
    for cell in root.iter(tag("TableCell")):
        for tl in cell.iter(tag("TextLine")):
            coords_el = tl.find(tag("Coords"))
            baseline_el = tl.find(tag("Baseline"))
            if coords_el is None or baseline_el is None:
                continue
            y0, y1 = _y_range(coords_el)
            x0, x1 = _bl_x_range(baseline_el)
            yb = round(y0 + CELL_FRAC * (y1 - y0))
            new_pts = f"{x0},{yb} {x1},{yb}"
            if baseline_el.attrib["points"] != new_pts:
                baseline_el.attrib["points"] = new_pts
                changed += 1

    # --- TextRegion baselines (taxpayer_name / taxpayer_index) ---
    for tr in root.iter(tag("TextRegion")):
        region_id = tr.attrib.get("id", "")
        frac = STRIP_FRACS.get(region_id)
        if frac is None:
            continue
        for tl in tr.iter(tag("TextLine")):
            coords_el = tl.find(tag("Coords"))
            baseline_el = tl.find(tag("Baseline"))
            if coords_el is None or baseline_el is None:
                continue
            y0, y1 = _y_range(coords_el)
            x0, x1 = _bl_x_range(baseline_el)
            yb = round(y0 + frac * (y1 - y0))
            new_pts = f"{x0},{yb} {x1},{yb}"
            if baseline_el.attrib["points"] != new_pts:
                baseline_el.attrib["points"] = new_pts
                changed += 1

    if changed:
        # ET.write produces <?xml ...?> header; preserve the PAGE XML declaration
        # by using short_empty_elements=False for readability.
        tree.write(xml_path, encoding="unicode", xml_declaration=True,
                   short_empty_elements=False)
        # ET emits ns0: prefixes when the default namespace wasn't found.
        # Re-read and strip "ns0:" so Transkribus stays happy.
        raw = xml_path.read_text(encoding="utf-8")
        raw = re.sub(r'\bns0:', '', raw)
        raw = re.sub(r'\bxmlns:ns0=', 'xmlns=', raw)
        xml_path.write_text(raw, encoding="utf-8")

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=int, action="append", dest="pages",
                        help="page number to patch (can be repeated; default: 10 50)")
    args = parser.parse_args()
    pages = args.pages or [10, 50]

    for page in pages:
        for variant in VARIANTS:
            xml_path = UPLOAD_DIR / variant / f"Hadita_{page}.xml"
            if not xml_path.exists():
                print(f"  SKIP (not found): {xml_path}")
                continue
            n = patch_xml(xml_path)
            status = f"{n} baselines updated" if n else "no changes"
            print(f"  page {page} [{variant}]: {status}")


if __name__ == "__main__":
    main()
