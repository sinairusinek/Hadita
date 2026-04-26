#!/usr/bin/env python3
"""
dewarp_page3.py — Produce a fully dewarped copy of page 3.

Pipeline:
  1. deskew_page(3)            — perspective correction (existing)
  2. crop_table()               — isolate left data table
  3. detect_rows()              — Kraken row y-centers (morph fallback)
  4. detect_columns() +
     detect_columns_banded()   — per-band column x-positions (captures bow)
  5. Build dense cv2.remap:
       output-y  → source-y  via row interpolation
       output-x  → source-x  via per-band column interpolation
  6. Remap table, composite header on top, save.

Output: Hadita-3Processed.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import interp1d

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Import only from segment_unified (no Streamlit dependency).
# deskew_page() is in haditax.py which runs st.* at module level — use cache instead.
from segment_unified import (
    N_BANDS,
    crop_table,
    detect_columns,
    detect_columns_banded,
    detect_rows,
    interp_col_x,
)


def load_deskewed(page: int) -> np.ndarray:
    """Load the pre-computed deskewed image from cache."""
    for p in [
        ROOT / "images" / f"deskewed_page{page}.jpg",
        ROOT / ".ocr_cache" / f"deskewed_page{page}.png",
        ROOT / ".ocr_cache" / f"deskewed_page{page}.jpg",
    ]:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                print(f"    Loaded from {p.relative_to(ROOT)}")
                return img
    raise FileNotFoundError(f"No deskewed cache found for page {page}")

PAGE        = 3
CACHE_DIR   = ROOT / ".ocr_cache"
OUTPUT_PATH = ROOT / "Hadita-3Processed.jpg"
JPEG_Q      = 95

CFG = {"header_frac": 0.08, "table_width_frac": 0.455}


# ── Remap construction ────────────────────────────────────────────────────────

def build_remap(
    table_bgr: np.ndarray,
    rows: list[dict],
    bands: list[dict],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (map_x, map_y, out_w, out_h) for cv2.remap.

    Rows  → evenly-spaced horizontal lines in output.
    Bands → evenly-spaced vertical lines in output (per-y column x corrected).
    """
    th, tw = table_bgr.shape[:2]
    n_rows = len(rows)
    if n_rows < 2:
        raise ValueError(f"Too few rows: {n_rows}")

    row_centers = np.array([r["y_center"] for r in rows], dtype=float)
    pitch = int(np.median(np.diff(row_centers)))
    out_h = n_rows * pitch
    out_w = tw

    # ── Row interpolator: output-y → source-y ────────────────────
    # Place each output row at i*pitch; map output row boundaries to
    # source row boundaries using the detected row centers as anchors.
    out_anchors = np.arange(n_rows) * pitch + pitch / 2          # centers in output
    src_anchors  = row_centers                                     # centers in source
    # Extend to edges so the full [0, out_h] range is covered
    out_anchors = np.concatenate([[0], out_anchors, [out_h]])
    src_anchors = np.concatenate(
        [[src_anchors[0] - pitch / 2], src_anchors, [src_anchors[-1] + pitch / 2]]
    )
    row_interp = interp1d(out_anchors, src_anchors, kind="linear",
                          bounds_error=False,
                          fill_value=(src_anchors[0], src_anchors[-1]))

    # ── Column interpolators: for each boundary j, x = f(y) ──────
    n_bounds   = len(bands[0]["col_x"])
    band_ys    = np.array([b["y_center"] for b in bands], dtype=float)
    col_interps = [
        interp1d(
            band_ys,
            [b["col_x"][j] for b in bands],
            kind="linear", bounds_error=False,
            fill_value=(bands[0]["col_x"][j], bands[-1]["col_x"][j]),
        )
        for j in range(n_bounds)
    ]
    # Evenly-spaced output column boundaries
    x_left      = float(bands[0]["col_x"][0])
    x_right     = float(bands[0]["col_x"][-1])
    out_col_x   = np.linspace(x_left, x_right, n_bounds)

    # ── Dense map construction ────────────────────────────────────
    print(f"  Building remap grid {out_w}×{out_h}…")
    oy_arr = np.arange(out_h, dtype=np.float32)
    ox_arr = np.arange(out_w, dtype=np.float32)
    OY, OX = np.meshgrid(oy_arr, ox_arr, indexing="ij")          # (out_h, out_w)

    # Source y for every output row
    src_y_map = row_interp(OY).astype(np.float32)

    # Source x for every output pixel (column-interval by column-interval)
    src_x_map = np.zeros((out_h, out_w), dtype=np.float32)
    j_map     = np.clip(
        np.searchsorted(out_col_x, OX, side="right") - 1, 0, n_bounds - 2
    )

    for j in range(n_bounds - 1):
        mask     = j_map == j
        if not np.any(mask):
            continue
        src_lo   = col_interps[j    ](src_y_map).astype(np.float32)
        src_hi   = col_interps[j + 1](src_y_map).astype(np.float32)
        span_out = float(out_col_x[j + 1] - out_col_x[j])
        t        = np.clip((OX - out_col_x[j]) / span_out, 0.0, 1.0)
        src_x_map[mask] = (src_lo + t * (src_hi - src_lo))[mask]

    return src_x_map, src_y_map, out_w, out_h


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[1] Loading deskewed page {PAGE}…")
    deskewed  = load_deskewed(PAGE)
    H, W      = deskewed.shape[:2]
    print(f"    Deskewed size: {W}×{H}")

    print("[2] Cropping table…")
    table_bgr, y_offset, x_offset = crop_table(deskewed, CFG)
    th, tw    = table_bgr.shape[:2]
    print(f"    Table: {tw}×{th}  offset=(x={x_offset}, y={y_offset})")

    print("[3] Detecting rows (Kraken / morph fallback)…")
    CACHE_DIR.mkdir(exist_ok=True)
    rows = detect_rows(
        table_bgr,
        cache_path=CACHE_DIR / f"seg_page{PAGE}.json",
        use_cache=True,
        method="kraken",
    )
    print(f"    {len(rows)} rows")

    print("[4] Detecting columns (global + banded)…")
    col_ranges = detect_columns(table_bgr, table_left_x=0)
    bands      = detect_columns_banded(table_bgr, col_ranges, n_bands=N_BANDS)
    print(f"    {len(col_ranges)-1} columns, {len(bands)} usable bands")

    print("[5] Building remap…")
    map_x, map_y, out_w, out_h = build_remap(table_bgr, rows, bands)

    print("[6] Applying remap…")
    table_dewarped = cv2.remap(
        table_bgr, map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    print("[7] Compositing header + dewarped table…")
    # Header: the strip above the table, cropped to the same width as the table
    header = deskewed[0:y_offset, x_offset : x_offset + tw]
    if header.shape[1] != out_w:
        header = cv2.resize(header, (out_w, y_offset), interpolation=cv2.INTER_LINEAR)

    output = np.vstack([header, table_dewarped])
    print(f"    Output: {output.shape[1]}×{output.shape[0]}")

    print(f"[8] Saving → {OUTPUT_PATH}")
    cv2.imwrite(str(OUTPUT_PATH), output, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    print("Done.")


if __name__ == "__main__":
    main()
