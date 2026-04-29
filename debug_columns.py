"""Generate debug PAGE XMLs + overlay PNGs for all x_left_col_mode options.

Outputs to debug/ with a letter suffix per mode so you can compare column
placement side-by-side without touching the production Transkribus upload folder.

Legend (also written to debug/legend.txt):
  A = none       — raw morphological detection, no correction
  B = geometric  — heuristic: pitch-ratio fix for Serial_No width
  C = gemini     — Gemini 2.5 Flash vision (uses cache if available)
  D = claude     — Claude Sonnet 4.6 vision (uses cache if available)
  E = consensus  — average of C+D if they agree within 15 px, else Gemini

Usage:
    python debug_columns.py              # pages 10 and 50
    python debug_columns.py --page 10
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).parent
DEBUG_DIR = ROOT / "debug"

MODES = [
    ("A", "none"),
    ("B", "geometric"),
    ("C", "gemini"),
    ("D", "claude"),
    ("E", "consensus"),
]

LEGEND = "\n".join(
    f"  {letter} = {mode:<12} — " + {
        "none":      "raw morphological detection, no correction",
        "geometric": "heuristic: pitch-ratio fix for Serial_No width",
        "gemini":    "Gemini 2.5 Flash vision (uses cache if available)",
        "claude":    "Claude Sonnet 4.6 vision (uses cache if available)",
        "consensus": "average of Gemini+Claude if within 15 px, else Gemini",
    }[mode]
    for letter, mode in MODES
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=int, action="append", dest="pages",
                        help="page to process (repeatable; default: 10 50)")
    args = parser.parse_args()
    pages = args.pages or [10, 50]

    # Lazy imports (heavy deps, keep startup fast for --help)
    from dewarp import (process_page as run_dewarp,
                        PROCESSED_DIR, H_META, H_HEADER, ROW_PITCH)
    from segment_unified import (write_page_xml, load_text_rows,
                                 recognize_top_strip, PAGE_CONFIG, W2E)
    from patch_baselines import patch_xml
    from render_overlay import render_page_from_paths

    def _to_eastern(text: str) -> str:
        return text.translate(W2E)

    DEBUG_DIR.mkdir(exist_ok=True)

    for page in pages:
        cfg = PAGE_CONFIG[page]
        print(f"\n{'='*60}")
        print(f"Page {page}")
        print(f"{'='*60}")

        # Load text once — same for all modes (Approach M cache)
        text_rows = load_text_rows(page, cfg)

        for letter, mode in MODES:
            print(f"  [{letter}] {mode} ...", end=" ", flush=True)

            result = run_dewarp(page, from_cache=True, x_left_col_mode=mode)

            # Copy the dewarped JPEG before the next mode overwrites it
            jpeg_src = PROCESSED_DIR / f"Hadita-{page}Processed.jpg"
            jpeg_dst = DEBUG_DIR / f"Hadita_{page}{letter}.jpeg"
            shutil.copyfile(jpeg_src, jpeg_dst)

            col_ranges = result["col_ranges"]
            n_rows     = result["n_rows"]
            row_ranges = [(i * ROW_PITCH, (i + 1) * ROW_PITCH) for i in range(n_rows)]
            page_w     = result["out_w"]
            page_h     = result["out_h"]

            canvas = cv2.imread(str(jpeg_dst))
            top_strip = recognize_top_strip(canvas, H_META, page)

            xml_out = DEBUG_DIR / f"Hadita_{page}{letter}.xml"
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

            overlay_out = DEBUG_DIR / f"Hadita_{page}{letter}_overlay.png"
            render_page_from_paths(jpeg_dst, xml_out, overlay_out)

            n_cols = len(col_ranges) - 1
            col0_w = col_ranges[1] - col_ranges[0]
            print(f"done  ({n_rows}r × {n_cols}c, Serial_No width={col0_w}px)"
                  f"  → {overlay_out.name}")

    (DEBUG_DIR / "legend.txt").write_text(
        "x_left_col_mode legend\n" + "=" * 40 + "\n" + LEGEND + "\n",
        encoding="utf-8",
    )
    print(f"\nDone. Files in {DEBUG_DIR}/")
    print(f"  legend: {DEBUG_DIR / 'legend.txt'}")


if __name__ == "__main__":
    main()
