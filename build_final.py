"""Build production-ready PAGE XML + dewarped JPEG pairs into Transkribus upload/final/.

Uses the finalised pipeline:
  - x_left_col_mode = geometric  (drops spurious Serial_No boundary)
  - eastern Arabic digits
  - baselines lowered via patch_baselines (cells frac=0.90, index frac=0.97)
  - standard filenames: Hadita_{N}.jpeg / Hadita_{N}.xml

Discovers all available original scans automatically. Skips pages already
present in final/ and pages that fail (empty/non-table pages).

Usage:
    python build_final.py              # all available pages not yet done
    python build_final.py --page 3
    python build_final.py --redo       # re-process even already-done pages
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import cv2

ROOT      = Path(__file__).parent
FINAL_DIR = ROOT / "Transkribus upload" / "final"
ORIG_GLOB = "000nvrj-432316TAX 1-85_page-{:04d}.jpg"


def all_available_pages() -> list[int]:
    """Return sorted list of page numbers that have an original scan."""
    pages = []
    for p in ROOT.glob("000nvrj-432316TAX 1-85_page-*.jpg"):
        try:
            n = int(p.stem.split("-")[-1])
            pages.append(n)
        except ValueError:
            pass
    return sorted(pages)


def ensure_page_config(page: int) -> None:
    """Add a default PAGE_CONFIG entry for page if not already present."""
    from segment_unified import PAGE_CONFIG, _page_cfg
    if page not in PAGE_CONFIG:
        PAGE_CONFIG[page] = _page_cfg(page)


def build_page(page: int) -> str:
    """Process one page. Returns a short status string."""
    from dewarp import process_page as run_dewarp, PROCESSED_DIR, H_META, H_HEADER, ROW_PITCH
    from segment_unified import write_page_xml, load_text_rows, recognize_top_strip, PAGE_CONFIG, W2E
    from patch_baselines import patch_xml

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    def _to_eastern(t: str) -> str:
        return t.translate(W2E)

    result = run_dewarp(page, from_cache=False, x_left_col_mode="geometric")

    jpeg_src = PROCESSED_DIR / f"Hadita-{page}Processed.jpg"
    jpeg_dst = FINAL_DIR / f"Hadita_{page}.jpeg"
    shutil.copyfile(jpeg_src, jpeg_dst)

    col_ranges = result["col_ranges"]
    n_rows     = result["n_rows"]
    row_ranges = [(i * ROW_PITCH, (i + 1) * ROW_PITCH) for i in range(n_rows)]
    page_w     = result["out_w"]
    page_h     = result["out_h"]

    canvas    = cv2.imread(str(jpeg_dst))
    text_rows = load_text_rows(page, PAGE_CONFIG[page])
    top_strip = recognize_top_strip(canvas, H_META, page)

    xml_out = FINAL_DIR / f"Hadita_{page}.xml"
    write_page_xml(
        col_ranges, row_ranges,
        y_offset=H_HEADER,
        page_w=page_w, page_h=page_h,
        image_filename=jpeg_dst.name,
        out_path=xml_out,
        text_rows=text_rows, bands=None,
        text_fn=_to_eastern,
        col_tags=True,
        row_baseline_y=None,
        top_strip=top_strip,
    )
    patch_xml(xml_out)
    return f"{page_w}×{page_h}, {n_rows}r × {len(col_ranges)-1}c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=int, action="append", dest="pages")
    parser.add_argument("--redo", action="store_true",
                        help="re-process pages already present in final/")
    args = parser.parse_args()

    if args.pages:
        pages = sorted(args.pages)
    else:
        pages = all_available_pages()
        if not args.redo:
            done = {int(p.stem.split("_")[1])
                    for p in FINAL_DIR.glob("Hadita_*.jpeg") if p.stem.split("_")[1].isdigit()}
            pages = [p for p in pages if p not in done]

    for page in pages:
        ensure_page_config(page)

    total = len(pages)
    skipped, failed = [], []

    print(f"Processing {total} pages → {FINAL_DIR}/\n")
    for i, page in enumerate(pages, 1):
        print(f"[{i}/{total}] page {page} ...", end=" ", flush=True)
        try:
            status = build_page(page)
            print(f"OK  ({status})")
        except Exception as e:
            short = str(e).split("\n")[0]
            print(f"SKIP  ({short})")
            failed.append((page, short))
            # Remove any partial output
            for ext in (".jpeg", ".xml"):
                f = FINAL_DIR / f"Hadita_{page}{ext}"
                if f.exists():
                    f.unlink()

    print(f"\nDone. {total - len(failed)} pages written to {FINAL_DIR}/")
    if failed:
        print(f"Skipped {len(failed)} pages:")
        for page, reason in failed:
            print(f"  page {page}: {reason}")


if __name__ == "__main__":
    main()
