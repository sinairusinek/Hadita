#!/usr/bin/env python3
"""
segment_unified.py — Unified segmentation pipeline for Hadita tax register pages.

Merges the best of two prior workflows:
  - Deskewed images (from haditax.py)
  - Kraken row detection + gap interpolation (from kraken_experiment.py)
  - Per-band column detection to capture page bow (from kraken_experiment.py)
  - Morphological row detection fallback (from haditax.py) if Kraken unavailable

Outputs PAGE XML for Transkribus upload for pages 3, 10, and 50.
Text content: GT for page 3, Approach M OCR cache for pages 10 and 50.

Usage:
  python segment_unified.py                  # process all pages
  python segment_unified.py --page 10        # single page
  python segment_unified.py --page 10 --no-cache  # re-run Kraken segmentation
  python segment_unified.py --row-method morph    # force morphological row detection
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
IMAGES_DIR   = PROJECT_DIR / "images"
CACHE_DIR    = PROJECT_DIR / ".ocr_cache"
GT_FILE      = PROJECT_DIR / "ground_truth.tsv"
SEG_MODEL    = PROJECT_DIR / "sc_100p_full_line (1).mlmodel"
REC_MODEL    = PROJECT_DIR / "gen2_sc_clean_best.mlmodel"   # Kraken recognition model (gen2)
KRAKEN_BIN   = "/opt/anaconda3/bin/kraken"
UPLOAD_DIR   = PROJECT_DIR / "Transkribus upload"

# ── Per-page config ────────────────────────────────────────────────────────────
# Tuned on page 3; adjust if pages 10/50 differ significantly after inspection.
def _page_cfg(n: int, gt: str | None = None) -> dict:
    return {"image": f"deskewed_page{n}.jpg", "transkribus_image": f"Hadita_{n}.jpeg",
            "xml_name": f"Hadita_{n}.xml", "header_frac": 0.08,
            "table_width_frac": 0.494, "table_left_x": 140, "gt_page": gt}

PAGE_CONFIG = {
    3:  _page_cfg(3,  gt="3"),
    4:  _page_cfg(4),
    5:  _page_cfg(5),
    6:  _page_cfg(6),
    7:  _page_cfg(7),
    8:  _page_cfg(8),
    9:  _page_cfg(9),
    10: _page_cfg(10),
    50: _page_cfg(50),
}

# ── Layout / detection constants ───────────────────────────────────────────────
LEFT_COLS = [
    "Serial_No", "Date",
    "Property_recorded_under_Block_No", "Property_recorded_under_Parcel_No",
    "Parcel_Cat_No", "Parcel_Area",
    "Nature_of_Entry", "New_Serial_No",
    "Reference_to_Register_of_Changes_Volume_No",
    "Reference_to_Register_of_Changes_Serial_No",
    "Tax_LP", "Tax_Mils", "Total_Tax_LP", "Total_Tax_Mils",
    "Reference_to_Register_of_Exemptions_Entry_No",
    "Reference_to_Register_of_Exemptions_Amount_LP",
    "Reference_to_Register_of_Exemptions_Amount_Mils",
    "Net_Assessment_LP", "Net_Assessment_Mils",
]
EXPECTED_COLS = len(LEFT_COLS)   # 19
DETECT_COLS   = EXPECTED_COLS + 1  # 20 (includes binding column)
N_BANDS       = 8

EASTERN = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
W2E = str.maketrans(WESTERN, EASTERN)
E2W = str.maketrans(EASTERN, WESTERN)


# ── Table crop ─────────────────────────────────────────────────────────────────

def crop_table(image: np.ndarray, cfg: dict) -> tuple[np.ndarray, int, int]:
    """Crop the left data table from the full page image.
    Returns (table_bgr, y_offset, x_offset)."""
    H, W = image.shape[:2]
    y0 = int(H * cfg["header_frac"])
    x1 = int(W * cfg["table_width_frac"])
    return image[y0:H, 0:x1], y0, 0


# ── Column detection ───────────────────────────────────────────────────────────

def _trim_vlines(v_lines: list[int], expected_cols: int,
                 left: int = 0, right: int | None = None) -> list[int]:
    if right is None:
        right = v_lines[-1] if v_lines else left
    want = expected_cols + 1
    if len(v_lines) <= want:
        return v_lines
    inner = [x for x in v_lines if left < x < right]
    ideal_step = (right - left) / expected_cols
    ideal_positions = [left + ideal_step * i for i in range(1, expected_cols)]
    chosen = []
    for ideal in ideal_positions:
        if not inner:
            break
        best = min(inner, key=lambda x: abs(x - ideal))
        chosen.append(best)
        inner = [x for x in inner if x != best]
    return sorted(set([left] + chosen + [right]))


# Empirically derived from page 3 (gold-standard alignment): the Serial_No
# column is ~2.9× wider than the median printed-column pitch. Other pages
# should match within ±50%; deviations beyond that indicate a spine/paper-edge
# artifact polluting col_ranges[0..1].
SERIAL_W_RATIO = 2.9


def fix_x_left_col_geometric(col_ranges: list[int]) -> list[int]:
    """Geometric fallback for a misdetected x_left_col (the right edge of Serial_No).

    Premise: column gaps 9..end of `col_ranges` are far from the spine and
    reliably detected. Their median pitch × SERIAL_W_RATIO predicts the
    correct Serial_No width. Two failure modes:

      A. serial_w too small (~1× pitch): col_ranges[1] is a spurious line
         (paper edge / handwriting). Drop it; the next boundary becomes the
         real Serial_No right edge. (Page 50.)
      B. serial_w too large: col_ranges[0] is too far left (spine artifact
         pulled it). Raise it to col_ranges[1] - expected_serial_w.

    Operates in *framed* coordinates. Returns a new list (possibly shorter
    by 1 in case A).
    """
    if len(col_ranges) < 12:
        return list(col_ranges)
    right_gaps = [col_ranges[i + 1] - col_ranges[i] for i in range(9, len(col_ranges) - 1)]
    if not right_gaps:
        return list(col_ranges)
    median_pitch = float(np.median(right_gaps))
    if median_pitch <= 1:
        return list(col_ranges)
    expected_serial_w = median_pitch * SERIAL_W_RATIO
    serial_w = col_ranges[1] - col_ranges[0]

    # Case A: spurious narrow first column (e.g. paper edge between spine and
    # real first printed line). Drop col_ranges[1] if a real line lives further
    # right at roughly the expected distance.
    if serial_w < 0.5 * expected_serial_w and len(col_ranges) > 3:
        next_w = col_ranges[2] - col_ranges[0]
        if 0.6 * expected_serial_w <= next_w <= 1.5 * expected_serial_w:
            log.info("Geometric x_left_col fix (drop spurious): serial_w %d → %d "
                     "(expected=%.0f, pitch=%.1f)",
                     serial_w, next_w, expected_serial_w, median_pitch)
            return [col_ranges[0]] + col_ranges[2:]

    # Case B: serial_w too large — raise col_ranges[0].
    if serial_w > 1.5 * expected_serial_w:
        new_x_left = int(round(col_ranges[1] - expected_serial_w))
        new_x_left = max(0, min(col_ranges[1] - 30, new_x_left))
        log.info("Geometric x_left_col fix (raise left): serial_w %d → %d "
                 "(expected=%.0f, pitch=%.1f)",
                 serial_w, col_ranges[1] - new_x_left, expected_serial_w, median_pitch)
        fixed = list(col_ranges)
        fixed[0] = new_x_left
        return fixed

    return list(col_ranges)


def fix_col_ranges_with_first_interior(col_ranges: list[int],
                                       x_first_interior: int) -> list[int]:
    """Force col_ranges[1] to equal x_first_interior (in framed coords).

    The right edge of the Serial_No column is the most error-prone boundary;
    when an LLM identifies it confidently, we trust it and discard any
    spurious boundaries (paper edges, handwriting, etc.) that detect_columns
    placed between col_ranges[0] and the real interior line.

    Returns a new list where col_ranges[0] is unchanged, col_ranges[1] is the
    boundary closest to x_first_interior (within tolerance) — any boundaries
    strictly between them are dropped — and the rest follow.
    """
    if len(col_ranges) < 3 or x_first_interior <= col_ranges[0]:
        return list(col_ranges)
    interior = [(i, x) for i, x in enumerate(col_ranges) if i >= 1]
    nearest_idx, nearest_x = min(interior, key=lambda p: abs(p[1] - x_first_interior))
    # Tolerance: within 1/2 of typical pitch
    if len(col_ranges) >= 12:
        right_gaps = [col_ranges[i + 1] - col_ranges[i] for i in range(9, len(col_ranges) - 1)]
        tol = 0.5 * float(np.median(right_gaps)) if right_gaps else 80.0
    else:
        tol = 80.0
    if abs(nearest_x - x_first_interior) > tol:
        log.info("LLM first-interior x=%d not near any detected boundary "
                 "(nearest=%d, tol=%.0f) — keeping original", x_first_interior, nearest_x, tol)
        return list(col_ranges)
    fixed = [col_ranges[0], col_ranges[nearest_idx]] + col_ranges[nearest_idx + 1:]
    if len(fixed) != len(col_ranges):
        log.info("LLM first-interior fix: dropped %d spurious boundaries "
                 "(col_ranges[1]: %d → %d)",
                 len(col_ranges) - len(fixed), col_ranges[1], col_ranges[nearest_idx])
    return fixed


def detect_table_frame(table_bgr: np.ndarray,
                       x_left_col_override: int | None = None) -> dict:
    """Anchor-based detection of the table's outer frame.

    Finds two strong landmarks:
      - Header-bottom horizontal line (separates printed header from data rows).
      - Long verticals that cross it; the leftmost is the leftmost printed column
        line (right edge of the serial-no column) and the rightmost is the
        page-split / binding line.

    Returns a dict with:
      header_bottom_y : int    — y of header-bottom in the cropped table image
      x_left_col      : int    — x of the leftmost printed column line
      x_right_split   : int    — x of the page-split line (right edge of table)
      x_left_frame    : int    — left edge of the canvas frame (= x_left_col − serial_w)
      serial_w        : int    — inferred width of the serial-no column
    """
    gray = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = gray.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5)

    # ── Vertical column lines (permissive opening that survives obstruction) ──
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(50, th // 25)))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    # Per-column on-fraction in the lower 60% of the table (data region)
    lower = v_mask[int(th * 0.30):, :]
    col_lower = (lower > 0).sum(axis=0)
    long_x = np.where(col_lower >= int(lower.shape[0] * 0.45))[0]
    if long_x.size == 0:
        log.warning("No long verticals; falling back to crop edges")
        line_xs: list[int] = [0, tw - 1]
    else:
        groups: list[list[int]] = [[int(long_x[0])]]
        for x in long_x[1:]:
            if x - groups[-1][-1] <= 20:
                groups[-1].append(int(x))
            else:
                groups.append([int(x)])
        line_xs = sorted(int(np.median(g)) for g in groups)
    # Leftmost printed column line: ignore crop-edge artifacts (x < 10).
    # The leftmost line should be within the leftward 25% of the crop; if no
    # strict-filter survivor lands there, retry with a lenient threshold.
    left_window_end = int(tw * 0.25)
    if x_left_col_override is not None and 10 <= x_left_col_override <= left_window_end:
        x_left_col = int(x_left_col_override)
        log.info("Left col: override → x=%d", x_left_col)
    else:
        interior_xs = [x for x in line_xs if 10 <= x <= left_window_end]
        if interior_xs:
            x_left_col = min(interior_xs)
        else:
            left_band = v_mask[int(th * 0.30):, 10:left_window_end]
            col_left = (left_band > 0).sum(axis=0)
            lb_thresh = int(left_band.shape[0] * 0.20)
            lb_long = np.where(col_left >= lb_thresh)[0]
            if lb_long.size:
                lb_g: list[list[int]] = [[int(lb_long[0])]]
                for x in lb_long[1:]:
                    if x - lb_g[-1][-1] <= 5:
                        lb_g[-1].append(int(x))
                    else:
                        lb_g.append([int(x)])
                x_left_col = 10 + int(np.median(lb_g[0]))
                log.info("Left col: lenient retry succeeded at x=%d", x_left_col)
            else:
                x_left_col = 10
                log.warning("Left col not detected; defaulting to x=%d", x_left_col)

    # Page-split line: the LEFTMOST strong vertical in the rightward 15% of the
    # crop is the binding gutter. (Anything to its right is the right page's
    # content — typically the leftmost printed column line of the next page.)
    right_window_start = int(tw * 0.88)
    right_band = v_mask[int(th * 0.30):, right_window_start:]
    col_right = (right_band > 0).sum(axis=0)
    rb_thresh = int(right_band.shape[0] * 0.20)
    rb_long = np.where(col_right >= rb_thresh)[0]
    if rb_long.size:
        # Cluster contiguous x's; take the LEFTMOST cluster's median.
        rb_groups: list[list[int]] = [[int(rb_long[0])]]
        for x in rb_long[1:]:
            if x - rb_groups[-1][-1] <= 20:
                rb_groups[-1].append(int(x))
            else:
                rb_groups.append([int(x)])
        x_right_split = right_window_start + int(np.median(rb_groups[0]))
        log.info("Page split: %d cluster(s) in right window; leftmost at x=%d",
                 len(rb_groups), x_right_split)
    else:
        x_right_split = tw - 5
        log.warning("Page split not detected; defaulting to crop edge x=%d", x_right_split)
    log.info("Long verticals (%d): leftmost x=%d, rightmost (page split) x=%d",
             len(line_xs), x_left_col, x_right_split)

    # ── Header-bottom: deepest strong horizontal in top 18% of the table ──
    # The page-top edge / outer table-frame top will be near y=0. The printed
    # column-header rows produce one or more horizontals slightly below it. We
    # want the LAST (deepest) of those — the line where data rows start.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(80, tw // 8), 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    proj_y = (h_mask > 0).sum(axis=1).astype(float)
    band_top, band_bot = 30, int(th * 0.18)  # skip page-edge; restrict to header band
    band = proj_y[band_top:band_bot].copy()
    if band.size == 0 or band.max() == 0:
        header_bottom_y = int(th * 0.08)
        log.warning("Header-bottom fallback to y=%d", header_bottom_y)
    else:
        peaks_idx, _ = find_peaks(band, height=band.max() * 0.4, distance=10)
        if peaks_idx.size == 0:
            header_bottom_y = band_top + int(np.argmax(band))
        else:
            header_bottom_y = band_top + int(peaks_idx[-1])
    log.info("Header-bottom y = %d (search band y∈[%d,%d])",
             header_bottom_y, band_top, band_bot)

    # The leftmost printed column line (x_left_col) IS the leftmost grid line.
    # The serial column lives to the right of it (between x_left_col and the
    # next printed line), not in the margin. So the frame's left edge = x_left_col.
    x_left_frame = x_left_col
    serial_w = 0

    log.info("Frame: x_left=%d (serial_w=%d), x_right=%d", x_left_frame, serial_w, x_right_split)

    return {
        "header_bottom_y": header_bottom_y,
        "x_left_col": x_left_col,
        "x_right_split": x_right_split,
        "x_left_frame": x_left_frame,
        "serial_w": serial_w,
    }


def detect_columns(table_bgr: np.ndarray, table_left_x: int,
                   expected_cols: int = EXPECTED_COLS) -> list[int]:
    """CLAHE + adaptive threshold + morphological column detection.
    Returns sorted boundary x-positions trimmed to `expected_cols` columns."""
    COL_SHIFT      = 15
    WIDE_COL_PX    = 200
    WIDE_COL_MIN_X = 1900

    gray = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = gray.shape
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm   = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    v_mask   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    v_mask   = cv2.dilate(v_mask, np.ones((1, 2), np.uint8), iterations=1)
    proj_x   = np.sum(v_mask, axis=0).astype(float)

    peaks, _ = find_peaks(proj_x,
                          height=proj_x.mean() + 0.5 * proj_x.std(),
                          distance=max(30, tw // 30))
    v_lines = peaks.tolist()

    if len(v_lines) < 3:
        log.warning("Only %d v-lines detected — using uniform grid", len(v_lines))
        col_w = (tw - table_left_x) // expected_cols
        return [table_left_x + i * col_w for i in range(expected_cols + 1)]

    if not v_lines or v_lines[0] > 30:
        v_lines = [0] + v_lines
    if v_lines[-1] < tw - 30:
        v_lines.append(tw)
    v_lines = [min(v + COL_SHIFT, tw) for v in v_lines]
    v_lines = sorted(set(v_lines))
    v_lines = _trim_vlines(v_lines, DETECT_COLS, left=0, right=tw)

    real = [x for x in v_lines if x >= table_left_x]
    if not real or real[0] > table_left_x + 30:
        real = [table_left_x] + real

    final: list[int] = [real[0]]
    for i in range(len(real) - 1):
        w = real[i + 1] - real[i]
        if w > WIDE_COL_PX and real[i] >= WIDE_COL_MIN_X:
            x_lo, x_hi = real[i] + 20, real[i + 1] - 20
            local_proj = proj_x[x_lo:x_hi]
            midpoint = (real[i] + real[i + 1]) // 2
            if local_proj.size > 10:
                low_thresh = float(np.percentile(local_proj, 20))
                cands, _ = find_peaks(local_proj, height=low_thresh, distance=10)
                if len(cands) > 0:
                    best = int(min(cands, key=lambda p: abs((p + x_lo) - midpoint)))
                    local_peak = best + x_lo
                else:
                    local_peak = midpoint
            else:
                local_peak = midpoint
            final.append(local_peak)
        final.append(real[i + 1])
    final = sorted(set(final))

    # Force exactly expected_cols columns. _trim_vlines anchors the outer
    # boundaries (left edge, right edge) and selects the expected_cols-1
    # interior boundaries closest to evenly-spaced positions.
    if len(final) - 1 != expected_cols:
        before = len(final) - 1
        final = _trim_vlines(final, expected_cols, left=final[0], right=final[-1])
        log.info("Column trim: %d → %d cols", before, len(final) - 1)

    n_cols = len(final) - 1
    if n_cols != expected_cols:
        log.warning("Column detection: %d cols (expected %d)", n_cols, expected_cols)
    log.info("Column boundaries (%d cols): %s", n_cols, final)
    return final


def detect_columns_banded(table_bgr: np.ndarray,
                           col_ranges_global: list[int],
                           n_bands: int = N_BANDS) -> list[dict]:
    """Per-band column detection to capture non-linear page bow."""
    th, tw = table_bgr.shape[:2]
    band_h = th // n_bands
    n_boundaries = len(col_ranges_global)
    bands: list[dict] = []

    for b in range(n_bands):
        y0 = b * band_h
        y1 = th if b == n_bands - 1 else (b + 1) * band_h
        y_center = (y0 + y1) // 2
        band = table_bgr[y0:y1, :]

        gray   = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm   = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(
            norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        v_mask   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
        v_mask   = cv2.dilate(v_mask, np.ones((1, 2), np.uint8), iterations=1)
        proj_x   = np.sum(v_mask, axis=0).astype(float)

        peaks, _ = find_peaks(proj_x,
                              height=proj_x.mean() + 0.3 * proj_x.std(),
                              distance=max(20, tw // 40))
        if len(peaks) < n_boundaries // 2:
            continue

        col_x: list[int] = []
        used: set[int] = set()
        for expected_x in col_ranges_global:
            candidates = [p for p in peaks if abs(p - expected_x) <= 60 and p not in used]
            if candidates:
                best = min(candidates, key=lambda p: abs(p - expected_x))
                col_x.append(int(best))
                used.add(best)
            else:
                col_x.append(expected_x)
        bands.append({"y_center": y_center, "col_x": col_x})

    if not bands:
        log.warning("No usable bands — falling back to global col_ranges")
        bands = [{"y_center": th // 2, "col_x": list(col_ranges_global)}]

    log.info("Per-band detection: %d/%d bands usable", len(bands), n_bands)
    return sorted(bands, key=lambda b: b["y_center"])


# ── Dewarp remap construction ──────────────────────────────────────────────────

def build_remap(
    table_bgr: np.ndarray,
    rows: list[dict],
    bands: list[dict],
    pheader_anchor: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    """Build cv2.remap maps that flatten the table grid.

    Rows  → evenly-spaced horizontal lines in output.
    Bands → evenly-spaced vertical lines in output (per-y column x corrected).

    If `pheader_anchor=(src_y_top, src_y_bot, out_h_top)`, the output also covers
    the printed-header band: output-y ∈ [0, out_h_top] linearly maps source-y ∈
    [src_y_top, src_y_bot]. Data rows then start at output-y = out_h_top.

    Returns (map_x, map_y, out_w, out_h, out_col_x).
    `out_col_x` is the array of column-boundary x positions in the dewarped
    output (same coordinate system as map_x), so callers can draw an overlay
    that exactly matches the dewarp's column placement.
    """
    th, tw = table_bgr.shape[:2]
    n_rows = len(rows)
    if n_rows < 2:
        raise ValueError(f"Too few rows: {n_rows}")

    row_centers = np.array([r["y_center"] for r in rows], dtype=float)
    pitch = int(np.median(np.diff(row_centers)))
    out_h_data = n_rows * pitch
    out_w = tw

    if pheader_anchor is None:
        out_h = out_h_data
        out_anchors = np.arange(n_rows) * pitch + pitch / 2
        src_anchors = row_centers
        out_anchors = np.concatenate([[0], out_anchors, [out_h]])
        src_anchors = np.concatenate(
            [[src_anchors[0] - pitch / 2], src_anchors, [src_anchors[-1] + pitch / 2]]
        )
    else:
        src_y_top, src_y_bot, out_h_top = pheader_anchor
        out_h = out_h_top + out_h_data
        data_centers_out = np.arange(n_rows) * pitch + pitch / 2 + out_h_top
        out_anchors = np.concatenate([[0], [out_h_top], data_centers_out, [out_h]])
        src_anchors = np.concatenate(
            [[src_y_top], [src_y_bot], row_centers, [row_centers[-1] + pitch / 2]]
        )
    row_interp = interp1d(
        out_anchors, src_anchors, kind="linear",
        bounds_error=False, fill_value=(src_anchors[0], src_anchors[-1]),
    )

    n_bounds = len(bands[0]["col_x"])
    band_ys = np.array([b["y_center"] for b in bands], dtype=float)
    col_interps = [
        interp1d(
            band_ys,
            [b["col_x"][j] for b in bands],
            kind="linear", bounds_error=False,
            fill_value=(bands[0]["col_x"][j], bands[-1]["col_x"][j]),
        )
        for j in range(n_bounds)
    ]
    src_col_x_ref = np.array(bands[0]["col_x"], dtype=float)
    src_span = float(src_col_x_ref[-1] - src_col_x_ref[0])
    if src_span > 0:
        out_col_x = (src_col_x_ref - src_col_x_ref[0]) / src_span * out_w
    else:
        out_col_x = np.linspace(0, out_w, n_bounds)

    oy_arr = np.arange(out_h, dtype=np.float32)
    ox_arr = np.arange(out_w, dtype=np.float32)
    OY, OX = np.meshgrid(oy_arr, ox_arr, indexing="ij")

    src_y_map = row_interp(OY).astype(np.float32)
    src_x_map = np.zeros((out_h, out_w), dtype=np.float32)
    j_map = np.clip(
        np.searchsorted(out_col_x, OX, side="right") - 1, 0, n_bounds - 2
    )
    for j in range(n_bounds - 1):
        mask = j_map == j
        if not np.any(mask):
            continue
        src_lo = col_interps[j    ](src_y_map).astype(np.float32)
        src_hi = col_interps[j + 1](src_y_map).astype(np.float32)
        span_out = float(out_col_x[j + 1] - out_col_x[j])
        t = np.clip((OX - out_col_x[j]) / span_out, 0.0, 1.0)
        src_x_map[mask] = (src_lo + t * (src_hi - src_lo))[mask]

    # Inverse map (src → out) for projecting arbitrary annotated points (e.g.
    # Kraken baselines) onto the dewarped canvas. src_anchors is monotonic
    # (rows are sorted by y_center), so a direct interp1d works for y. For x,
    # at any given src_y we know where each column boundary lives in src
    # (via col_interps) and where it lands in out (out_col_x), so linear
    # interpolation between those boundaries inverts the per-row warp.
    src_to_out_y = interp1d(
        src_anchors, out_anchors, kind="linear",
        bounds_error=False, fill_value=(out_anchors[0], out_anchors[-1]),
    )

    def src_to_out(src_x: float, src_y: float) -> tuple[float, float]:
        out_y = float(src_to_out_y(src_y))
        src_col_at_y = np.array([float(ci(src_y)) for ci in col_interps])
        out_x = float(np.interp(src_x, src_col_at_y, out_col_x))
        return out_x, out_y

    return src_x_map, src_y_map, out_w, out_h, out_col_x, src_to_out


def compute_row_baselines(
    seg_lines: list[dict],
    hb_y: int,
    src_to_out,
    framed_w: int,
    out_h: int,
    target_h: int,
    n_rows: int,
    h_header: int,
    h_meta: int,
    row_pitch: int,
) -> tuple[np.ndarray, float, int]:
    """Per-row baseline y in final-canvas coords.

    Each Kraken baseline polyline is projected to the dewarped canvas, mean-y'd,
    and bucketed into the row band that contains it. Rows with at least one real
    baseline take the mean; the remaining rows use h_header + (i+frac)*row_pitch
    where frac is calibrated from real rows' within-band offsets (default 0.7
    when nothing is available).

    Returns (row_baseline_y, frac, n_real_rows).
    """
    scale_y = target_h / out_h
    buckets: list[list[float]] = [[] for _ in range(n_rows)]
    for kline in seg_lines:
        bl = kline.get("baseline") or []
        if len(bl) < 2:
            continue
        ys = []
        for pt in bl:
            sx, sy = float(pt[0]), float(pt[1]) + hb_y
            _ox, oy = src_to_out(sx, sy)
            ys.append(oy * scale_y + h_meta)
        cy = float(np.mean(ys))
        i = int((cy - h_header) // row_pitch)
        if 0 <= i < n_rows:
            buckets[i].append(cy)

    real_y = np.full(n_rows, np.nan, dtype=np.float64)
    for i, ys in enumerate(buckets):
        if ys:
            real_y[i] = float(np.mean(ys))

    real_idx = np.where(~np.isnan(real_y))[0]
    if real_idx.size:
        fracs = (real_y[real_idx] - (h_header + real_idx * row_pitch)) / row_pitch
        frac = float(np.mean(np.clip(fracs, 0.0, 1.0)))
    else:
        frac = 0.7

    row_baseline_y = real_y.copy()
    missing = np.isnan(row_baseline_y)
    miss_idx = np.where(missing)[0]
    row_baseline_y[miss_idx] = h_header + (miss_idx + frac) * row_pitch
    return row_baseline_y, frac, int(real_idx.size)


def dewarped_grid(
    n_rows: int, n_cols: int, out_w: int, out_h: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Geometry of the uniform target grid produced by build_remap.

    Returns (row_y_boundaries, col_x_boundaries, row_pitch).
    Row boundaries length = n_rows + 1; col boundaries length = n_cols + 1.
    """
    row_pitch = out_h // n_rows
    row_y = np.arange(n_rows + 1) * row_pitch
    col_x = np.linspace(0, out_w, n_cols + 1).round().astype(int)
    return row_y, col_x, row_pitch


def interp_col_x(c_idx: int, y_table: int, bands: list[dict]) -> int:
    """Interpolate column boundary x at given y (linear between band centers)."""
    if len(bands) == 1:
        return bands[0]["col_x"][c_idx]
    ys = [b["y_center"] for b in bands]
    xs = [b["col_x"][c_idx] for b in bands]
    if y_table <= ys[0]:  return xs[0]
    if y_table >= ys[-1]: return xs[-1]
    for i in range(len(ys) - 1):
        if ys[i] <= y_table <= ys[i + 1]:
            t = (y_table - ys[i]) / (ys[i + 1] - ys[i])
            return round(xs[i] + t * (xs[i + 1] - xs[i]))
    return xs[-1]


# ── Row detection ──────────────────────────────────────────────────────────────

def detect_rows_kraken(table_bgr: np.ndarray,
                       cache_path: Path,
                       use_cache: bool = True,
                       skip_header_y: int = 200) -> list[dict]:
    """Detect rows using Kraken segmentation model.
    Returns list of row dicts with y_center, y_min, y_max."""
    lines = None
    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        cached = data.get("lines", [])
        # Schema migration: pre-baseline caches lack the `baseline` key. Force
        # regeneration so callers (e.g. baseline overlay) get the polyline data.
        if cached and "baseline" not in cached[0]:
            log.info("Cache %s lacks baselines (old schema); re-running Kraken",
                     cache_path)
        else:
            log.info("Loading cached Kraken segmentation from %s", cache_path)
            lines = cached

    if lines is None:
        with tempfile.TemporaryDirectory() as tmp:
            in_path  = Path(tmp) / "table.jpg"
            out_path = Path(tmp) / "seg.json"
            cv2.imwrite(str(in_path), table_bgr)
            cmd = [KRAKEN_BIN, "--native", "-i", str(in_path), str(out_path),
                   "segment", "--baseline", "--model", str(SEG_MODEL)]
            log.info("Running Kraken segmentation: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0 or not out_path.exists():
                raise RuntimeError(f"Kraken segmentation failed: {result.stderr[-300:]}")
            with open(out_path) as f:
                raw = json.load(f)

        lines = []
        for line in raw.get("lines", []):
            baseline = line.get("baseline", [])
            boundary = line.get("boundary", [])
            if baseline:
                ys = [pt[1] for pt in baseline]
                y_center = int(sum(ys) / len(ys))
            elif boundary:
                ys = [pt[1] for pt in boundary]
                y_center = int(sum(ys) / len(ys))
            else:
                continue
            if boundary:
                ys_b = [pt[1] for pt in boundary]
                y_min, y_max = min(ys_b), max(ys_b)
            else:
                y_min = y_max = y_center
            lines.append({
                "y_center": y_center,
                "y_min": y_min,
                "y_max": y_max,
                "baseline": baseline,
                "boundary": boundary,
            })
        lines.sort(key=lambda l: l["y_center"])
        log.info("Kraken detected %d raw line regions", len(lines))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"lines": lines}, f, indent=2)
        log.info("Cached segmentation → %s", cache_path)

    data_lines = [l for l in lines if l["y_center"] > skip_header_y]
    clusters   = _cluster_lines(data_lines)
    clusters   = interpolate_gaps(clusters)
    return clusters


def detect_rows_morph(table_bgr: np.ndarray, skip_header_y: int = 200) -> list[dict]:
    """Detect rows using CLAHE + adaptive threshold + morphological projection.
    Fallback when Kraken is unavailable. Returns same row dict format."""
    gray   = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = gray.shape
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm   = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    h_mask   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    h_mask   = cv2.dilate(h_mask, np.ones((2, 1), np.uint8), iterations=1)

    # Only project over right portion of table where ink density is lower (faint lines visible)
    strip = h_mask[:, tw * 2 // 3 : tw * 5 // 6]
    proj_y = np.sum(strip, axis=1).astype(float)

    peaks, _ = find_peaks(proj_y,
                          height=proj_y.mean() + 0.5 * proj_y.std(),
                          distance=60)
    h_lines = [int(p) for p in peaks if p > skip_header_y]

    if len(h_lines) < 3:
        log.warning("Morphological row detection: only %d lines found", len(h_lines))

    # Convert boundary lines to row center dicts (same format as Kraken output)
    rows = []
    for i in range(len(h_lines) - 1):
        y0, y1 = h_lines[i], h_lines[i + 1]
        y_center = (y0 + y1) // 2
        rows.append({"y_center": y_center, "y_min": y0, "y_max": y1})
    rows = interpolate_gaps(rows)
    log.info("Morphological row detection: %d rows from %d lines", len(rows), len(h_lines))
    return rows


def detect_rows(table_bgr: np.ndarray,
                cache_path: Path,
                use_cache: bool = True,
                method: str = "kraken",
                skip_header_y: int = 200) -> list[dict]:
    """Unified row detection. Tries Kraken first; falls back to morphological."""
    if method == "kraken":
        if not SEG_MODEL.exists():
            log.warning("Segmentation model not found at %s — using morphological fallback", SEG_MODEL)
            return detect_rows_morph(table_bgr, skip_header_y)
        try:
            return detect_rows_kraken(table_bgr, cache_path, use_cache, skip_header_y)
        except Exception as e:
            log.warning("Kraken row detection failed (%s) — using morphological fallback", e)
            return detect_rows_morph(table_bgr, skip_header_y)
    return detect_rows_morph(table_bgr, skip_header_y)


def _cluster_lines(lines: list[dict], gap_threshold: int = 60) -> list[dict]:
    if not lines:
        return []
    clusters: list[list[dict]] = []
    current = [lines[0]]
    for line in lines[1:]:
        if line["y_center"] - current[-1]["y_center"] > gap_threshold:
            clusters.append(current)
            current = [line]
        else:
            current.append(line)
    clusters.append(current)
    rows = []
    for cluster in clusters:
        rows.append({
            "y_center": sum(l["y_center"] for l in cluster) // len(cluster),
            "y_min":    min(l["y_min"] for l in cluster),
            "y_max":    max(l["y_max"] for l in cluster),
        })
    return rows


def interpolate_gaps(rows: list[dict]) -> list[dict]:
    """Insert synthetic rows in gaps spanning more than one row pitch.

    Pitch is estimated from the smaller half of gaps so it isn't inflated by
    the large outlier gaps caused by missed rows (the very thing we're trying
    to fill). A gap is split into n = round(gap / pitch) parts, inserting
    (n-1) synthetic rows whenever n ≥ 2.
    """
    if len(rows) < 2:
        return rows
    spacings = [rows[i + 1]["y_center"] - rows[i]["y_center"] for i in range(len(rows) - 1)]
    spacings_sorted = sorted(spacings)
    half = spacings_sorted[: len(spacings_sorted) // 2 + 1]
    pitch = sorted(half)[len(half) // 2]
    if pitch <= 0:
        return rows
    log.info("Row pitch estimate: %dpx (overall median: %dpx)",
             pitch, spacings_sorted[len(spacings_sorted) // 2])

    result: list[dict] = [rows[0]]
    for i in range(1, len(rows)):
        gap = rows[i]["y_center"] - rows[i - 1]["y_center"]
        n = round(gap / pitch)
        if n >= 2:
            for k in range(1, n):
                sy = rows[i - 1]["y_center"] + round(k * gap / n)
                result.append({"y_center": sy, "y_min": sy - 20, "y_max": sy + 20,
                                "synthetic": True})
                log.info("  Inserted synthetic row at y=%d (gap=%dpx, pitch=%dpx, n=%d)",
                         sy, gap, pitch, n)
        result.append(rows[i])
    return result


def rows_to_ranges(rows: list[dict], table_height: int) -> list[tuple[int, int]]:
    """Convert row center dicts to (y_top, y_bottom) boundary pairs.

    First row starts at its detected y_min (not 0) so the header area above
    the first data row is excluded rather than absorbed into row 0.
    """
    if not rows:
        return []
    centers = [r["y_center"] for r in rows]
    boundaries = [rows[0]["y_min"]]
    for i in range(len(centers) - 1):
        boundaries.append((centers[i] + centers[i + 1]) // 2)
    boundaries.append(table_height)
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(centers))]


# ── Text loading ───────────────────────────────────────────────────────────────

def load_gt_rows(page_str: str) -> list[dict]:
    """Load verified GT rows for the given page from ground_truth.tsv."""
    EASTERN_DIGITS = "٠١٢٣٤٥٦٧٨٩"
    WESTERN_DIGITS = "0123456789"
    E2W = str.maketrans(EASTERN_DIGITS, WESTERN_DIGITS)

    if not GT_FILE.exists():
        log.warning("GT file not found: %s", GT_FILE)
        return []
    rows = []
    with open(GT_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r.get("Page_Number") == page_str and r.get("Serial_No", "").strip():
                rows.append(r)
    rows.sort(key=lambda r: int(r["Serial_No"].strip().translate(E2W) or "0"))
    log.info("Loaded %d GT rows for page %s", len(rows), page_str)
    return rows


def load_approach_m_rows(page_num: int) -> list[dict]:
    """Load Approach M (Gemini) OCR cache for the given page."""
    cache_file = CACHE_DIR / f"M_page{page_num}.json"
    if not cache_file.exists():
        log.warning("Approach M cache not found: %s", cache_file)
        return []
    rows = json.loads(cache_file.read_text())
    log.info("Loaded %d Approach M rows for page %d", len(rows), page_num)
    return rows


def load_text_rows(page_num: int, cfg: dict) -> list[dict]:
    """Load GT if available for this page, otherwise Approach M."""
    if cfg["gt_page"] is not None:
        rows = load_gt_rows(cfg["gt_page"])
        if rows:
            return rows
        log.warning("GT empty for page %d — falling back to Approach M", page_num)
    return load_approach_m_rows(page_num)


# ── Kraken HTR (recognition) ───────────────────────────────────────────────────

def _kraken_recognize_image(img_bgr: np.ndarray, model_path: Path) -> str:
    """Run Kraken recognition on a single line image (no internal segmentation).

    Mirrors compare_ocr._kraken_recognize_cell but accepts a numpy BGR array.
    Returns the recognized text, or "" on failure.
    """
    if img_bgr is None or img_bgr.size == 0:
        return ""
    with tempfile.TemporaryDirectory() as tmp:
        in_path  = Path(tmp) / "line.jpg"
        out_path = Path(tmp) / "line.txt"
        cv2.imwrite(str(in_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cmd = [KRAKEN_BIN, "-i", str(in_path), str(out_path),
               "binarize", "ocr", "-s", "-m", str(model_path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return ""
        if res.returncode != 0 or not out_path.exists():
            return ""
        return out_path.read_text(encoding="utf-8").strip()


def _kraken_segment_image(img_bgr: np.ndarray) -> list[dict]:
    """Run Kraken baseline segmentation on a single image. Returns parsed lines."""
    with tempfile.TemporaryDirectory() as tmp:
        in_path  = Path(tmp) / "strip.jpg"
        out_path = Path(tmp) / "seg.json"
        cv2.imwrite(str(in_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cmd = [KRAKEN_BIN, "--native", "-i", str(in_path), str(out_path),
               "segment", "--baseline", "--model", str(SEG_MODEL)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired:
            log.warning("Kraken segment timed out on strip")
            return []
        if res.returncode != 0 or not out_path.exists():
            log.warning("Kraken segment failed on strip: %s", res.stderr[-200:])
            return []
        with open(out_path) as f:
            raw = json.load(f)
    return raw.get("lines", []) or []


def _line_bbox(line: dict, h: int, w: int, pad_y: int = 8) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) for a Kraken line in the strip's pixel coords."""
    boundary = line.get("boundary") or []
    baseline = line.get("baseline") or []
    if boundary:
        xs = [pt[0] for pt in boundary]
        ys = [pt[1] for pt in boundary]
    elif baseline:
        xs = [pt[0] for pt in baseline]
        ys_b = [pt[1] for pt in baseline]
        ys = [min(ys_b) - 30, max(ys_b) + 10]
    else:
        return 0, 0, 0, 0
    x0 = max(0, int(min(xs)) - 4)
    x1 = min(w, int(max(xs)) + 4)
    y0 = max(0, int(min(ys)) - pad_y)
    y1 = min(h, int(max(ys)) + pad_y)
    return x0, y0, x1, y1


def _trim_to_ink(crop_bgr: np.ndarray, pad_y: int = 12, pad_x: int = 16,
                 row_min_frac: float = 0.04,
                 col_min_frac: float = 0.05) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Tighten a crop to the ink-bearing band via horizontal/vertical projections.

    Y-trim: rows whose ink-pixel fraction (along the row's width) exceeds
    `row_min_frac` are kept; the band is the contiguous range from the first
    to last such row, with `pad_y` padding.

    X-trim is applied *only after* y-trim so faint full-width artifacts (page
    edge, table borders) don't dominate the column sums. Uses `col_min_frac`
    of the *trimmed* band's height as the threshold.

    No morphological opening — that erodes thin Arabic strokes. Falls back to
    the full input if no ink band is detected.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr, (0, 0, 0, 0)
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    row_frac = (binv > 0).sum(axis=1) / max(1, w)
    rows = np.where(row_frac >= row_min_frac)[0]
    if rows.size == 0:
        return crop_bgr, (0, 0, w, h)
    y0t = max(0, int(rows[0])  - pad_y)
    y1t = min(h, int(rows[-1]) + pad_y + 1)
    if y1t - y0t < 12:
        return crop_bgr, (0, 0, w, h)

    band = binv[y0t:y1t, :]
    col_frac = (band > 0).sum(axis=0) / max(1, band.shape[0])
    cols = np.where(col_frac >= col_min_frac)[0]
    if cols.size == 0:
        x0t, x1t = 0, w
    else:
        x0t = max(0, int(cols[0])  - pad_x)
        x1t = min(w, int(cols[-1]) + pad_x + 1)

    return crop_bgr[y0t:y1t, x0t:x1t].copy(), (x0t, y0t, x1t, y1t)


def recognize_top_strip(canvas_bgr: np.ndarray, h_meta: int,
                        page_num: int, debug_dir: Path | None = None,
                        y_top_frac: float = 0.25,
                        y_bot_extra: int = 0) -> dict | None:
    """Recognize the taxpayer name + index in the strip above the printed header.

    Strip layout (three equal columns over the strip's width):
      left third  = printed "Tax-payer:" label  → ignored
      middle third = handwritten taxpayer name
      right third  = handwritten index / "number #"

    The strip's vertical extent is y∈[h_meta * y_top_frac, h_meta + y_bot_extra)
    in canvas coords. The bottom overshoots H_META by `y_bot_extra` pixels to
    catch writing that bled into the printed-header band on pages whose
    metadata anchor isn't perfectly aligned. Each third's crop is then
    tight-trimmed via projection to the ink band before recognition (no
    internal segmentation).
    """
    h_meta = max(1, int(h_meta))
    H, W = canvas_bgr.shape[:2]
    y0 = int(round(h_meta * y_top_frac))
    y1 = min(H, h_meta + max(0, int(y_bot_extra)))
    if y1 - y0 < 8:
        return None
    strip = canvas_bgr[y0:y1, :].copy()
    h_s, w_s = strip.shape[:2]

    third = w_s // 3
    fields = {
        "name":  (third,     2 * third),
        "index": (2 * third, w_s),
    }

    result: dict = {}
    for key, (xa, xb) in fields.items():
        sub = strip[:, xa:xb]
        trimmed, (dx0, dy0, dx1, dy1) = _trim_to_ink(sub)
        text = _kraken_recognize_image(trimmed, REC_MODEL)
        cx0 = xa + dx0
        cy0 = y0 + dy0
        cx1 = xa + dx1
        cy1 = y0 + dy1
        coords = [(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)]
        yb_ = int(cy0 + 0.75 * (cy1 - cy0))
        baseline = [(cx0 + 5, yb_), (cx1 - 5, yb_)]
        result[key] = {"text": text, "baseline": baseline, "coords": coords,
                       "bbox": (cx0, cy0, cx1, cy1)}

    log.info("Top strip page %d: name=%r  index=%r",
             page_num, result["name"]["text"], result["index"]["text"])

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        # Show the whole strip we read (extended past h_meta) and mark the
        # h_meta boundary so the overshoot region is visible.
        ov = canvas_bgr[0:y1, :].copy()
        cv2.line(ov, (0, h_meta), (ov.shape[1], h_meta), (160, 160, 160), 1)
        for key, color in (("name", (0, 200, 0)), ("index", (0, 100, 255))):
            entry = result[key]
            x0, ya, x1, yb_ = entry["bbox"]
            cv2.rectangle(ov, (x0, ya), (x1, yb_), color, 2)
            cv2.putText(ov, f"{key}: {entry['text']}", (x0 + 6, max(15, ya + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(debug_dir / f"topstrip_page{page_num}.png"), ov)

    return result


def recognize_table_cells(canvas_bgr: np.ndarray,
                          col_ranges: list[int],
                          n_rows: int,
                          h_header: int,
                          row_pitch: int,
                          page_num: int,
                          max_workers: int = 4) -> list[dict]:
    """Run Kraken HTR on every (row × column) cell of the dewarped table.

    Returns a list of dicts shaped like load_text_rows() output: one dict per
    row, keyed by LEFT_COLS column names. Cells whose recognition yields the
    empty string remain "".
    """
    from concurrent.futures import ThreadPoolExecutor

    n_cols = len(col_ranges) - 1
    H, W = canvas_bgr.shape[:2]

    tasks: list[tuple[int, int, np.ndarray]] = []
    for r in range(n_rows):
        cy0 = h_header + r * row_pitch
        cy1 = h_header + (r + 1) * row_pitch
        cy0 = max(0, min(H, cy0))
        cy1 = max(0, min(H, cy1))
        if cy1 - cy0 < 8:
            continue
        for c in range(n_cols):
            x0 = max(0, min(W, int(col_ranges[c])))
            x1 = max(0, min(W, int(col_ranges[c + 1])))
            if x1 - x0 < 8:
                tasks.append((r, c, np.zeros((1, 1, 3), dtype=np.uint8)))
                continue
            tasks.append((r, c, canvas_bgr[cy0:cy1, x0:x1].copy()))

    log.info("HTR cells page %d: %d cells (%d rows × %d cols)",
             page_num, len(tasks), n_rows, n_cols)

    results: dict[tuple[int, int], str] = {}
    def _do(task):
        r, c, crop = task
        return (r, c), _kraken_recognize_image(crop, REC_MODEL)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for (r, c), text in pool.map(_do, tasks):
            results[(r, c)] = text

    text_rows: list[dict] = []
    for r in range(n_rows):
        row = {LEFT_COLS[c]: results.get((r, c), "") for c in range(min(n_cols, len(LEFT_COLS)))}
        text_rows.append(row)
    return text_rows


# ── PAGE XML export ────────────────────────────────────────────────────────────

def write_page_xml(col_ranges: list[int],
                   row_ranges: list[tuple[int, int]],
                   y_offset: int,
                   page_w: int, page_h: int,
                   image_filename: str,
                   out_path: Path,
                   text_rows: list[dict] | None = None,
                   bands: list[dict] | None = None,
                   text_fn=None,
                   col_tags: bool = False,
                   row_baseline_y: list[int] | None = None,
                   top_strip: dict | None = None) -> None:
    """Write a PAGE XML (2013-07-15) with TableRegion + TableCells.

    Row 0 = printed column-label header (empty). Rows 1…N = data rows.
    If bands provided, column x-positions are interpolated per row to capture bow.
    text_fn: optional transform applied to each cell text before writing.
    """
    from datetime import datetime
    now    = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    n_rows = len(row_ranges)
    n_cols = len(col_ranges) - 1
    table_h = page_h - y_offset

    def _col_x(c_idx: int, y_table: int) -> int:
        if bands:
            return interp_col_x(c_idx, y_table, bands)
        return col_ranges[c_idx]

    def _cell_pts(c0, cy0, c1, cy1) -> str:
        corners = [
            (_col_x(c0, cy0), cy0 + y_offset),
            (_col_x(c1, cy0), cy0 + y_offset),
            (_col_x(c1, cy1), cy1 + y_offset),
            (_col_x(c0, cy1), cy1 + y_offset),
        ]
        return " ".join(f"{x},{y}" for x, y in corners)

    def _baseline_pts(c0, cy0, c1, cy1, frac=0.75) -> str:
        cy_bl = round(cy0 + (cy1 - cy0) * frac)
        y_bl  = cy_bl + y_offset
        return f"{_col_x(c0, cy_bl)},{y_bl} {_col_x(c1, cy_bl)},{y_bl}"

    def _baseline_pts_canvas(c0, c1, y_canvas, cy_for_colx) -> str:
        return f"{_col_x(c0, cy_for_colx)},{y_canvas} {_col_x(c1, cy_for_colx)},{y_canvas}"

    tx0, ty0 = col_ranges[0], row_ranges[0][0] + y_offset
    tx1, ty1 = col_ranges[-1], row_ranges[-1][1] + y_offset
    table_pts = f"{tx0},{ty0} {tx1},{ty0} {tx1},{ty1} {tx0},{ty1}"

    def _xml_escape(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                 .replace('"', "&quot;"))

    cells_xml: list[str] = []
    for r_idx, (y0, y1) in enumerate(row_ranges):
        y_bl_canvas = (
            int(round(row_baseline_y[r_idx]))
            if row_baseline_y is not None and r_idx < len(row_baseline_y)
            else None
        )
        for c_idx in range(n_cols):
            cid  = f"cell_r{r_idx}_c{c_idx}"
            cpts = _cell_pts(c_idx, y0, c_idx + 1, y1)
            text = ""
            if text_rows and r_idx < len(text_rows):
                col_name = LEFT_COLS[c_idx] if c_idx < len(LEFT_COLS) else ""
                text = text_rows[r_idx].get(col_name, "").strip()
                if text and text_fn:
                    text = text_fn(text)
            if y_bl_canvas is not None:
                bl = _baseline_pts_canvas(c_idx, c_idx + 1, y_bl_canvas, (y0 + y1) // 2)
            else:
                bl = _baseline_pts(c_idx, y0, c_idx + 1, y1)
            textline_xml = (
                f'        <TextLine id="line_{cid}">\n'
                f'          <Coords points="{cpts}"/>\n'
                f'          <Baseline points="{bl}"/>\n'
                f'          <TextEquiv><Unicode>{_xml_escape(text)}</Unicode></TextEquiv>\n'
                f'        </TextLine>\n'
            )
            col_tag = LEFT_COLS[c_idx] if c_idx < len(LEFT_COLS) else f"col_{c_idx}"
            custom_attr = f' custom="structure {{type:{col_tag};}}"' if col_tags else ""
            cells_xml.append(
                f'      <TableCell id="{cid}" row="{r_idx}" col="{c_idx}" '
                f'rowSpan="1" colSpan="1"{custom_attr}>\n'
                f'        <Coords points="{cpts}"/>\n'
                + textline_xml +
                f'      </TableCell>'
            )

    text_regions_xml: list[str] = []
    if top_strip:
        for region_id, label in (("taxpayer_name", "name"),
                                  ("taxpayer_index", "index")):
            entry = top_strip.get(label)
            if not entry or not entry.get("coords"):
                continue
            coords = entry["coords"]
            baseline = entry.get("baseline") or []
            text = (entry.get("text") or "").strip()
            if text and text_fn:
                text = text_fn(text)
            coords_pts = " ".join(f"{int(x)},{int(y)}" for x, y in coords)
            if len(baseline) >= 2:
                bl_pts = " ".join(f"{int(x)},{int(y)}" for x, y in baseline)
            else:
                # Synthesize a horizontal baseline at ~75% of bbox height.
                xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
                yb = int(min(ys) + 0.75 * (max(ys) - min(ys)))
                bl_pts = f"{int(min(xs))},{yb} {int(max(xs))},{yb}"
            text_regions_xml.append(
                f'    <TextRegion id="{region_id}" type="header" '
                f'custom="structure {{type:{region_id};}}">\n'
                f'      <Coords points="{coords_pts}"/>\n'
                f'      <TextLine id="line_{region_id}">\n'
                f'        <Coords points="{coords_pts}"/>\n'
                f'        <Baseline points="{bl_pts}"/>\n'
                f'        <TextEquiv><Unicode>{_xml_escape(text)}</Unicode></TextEquiv>\n'
                f'      </TextLine>\n'
                f'    </TextRegion>'
            )

    text_regions_block = ("\n".join(text_regions_xml) + "\n") if text_regions_xml else ""

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"\n'
        '       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '       xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 '
        'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd">\n'
        '  <Metadata>\n'
        '    <Creator>segment_unified.py</Creator>\n'
        f'    <Created>{now}</Created>\n'
        f'    <LastChange>{now}</LastChange>\n'
        '  </Metadata>\n'
        f'  <Page imageFilename="{image_filename}" imageWidth="{page_w}" imageHeight="{page_h}">\n'
        + text_regions_block +
        f'    <TableRegion id="table1" rows="{n_rows}" columns="{n_cols}">\n'
        f'      <Coords points="{table_pts}"/>\n'
        + "\n".join(cells_xml) + "\n"
        '    </TableRegion>\n'
        '  </Page>\n'
        '</PcGts>\n'
    )
    out_path.write_text(xml, encoding="utf-8")
    log.info("PAGE XML → %s  (%d rows × %d cols)", out_path, n_rows, n_cols)


# ── Process one page ───────────────────────────────────────────────────────────

def process_page(page_num: int, args) -> None:
    """Run dewarp pipeline → save dewarped JPEG + write uniform-grid PAGE XML for both digit variants."""
    import shutil
    from dewarp import process_page as run_dewarp, W_OUT, H_META, H_HEADER, ROW_PITCH
    from patch_baselines import patch_xml

    cfg = PAGE_CONFIG[page_num]
    print(f"\n{'='*60}")
    print(f"Page {page_num}")
    print(f"{'='*60}")

    # 1–7. Run the full image pipeline (deskew → crop → detect → remap → normalize)
    result = run_dewarp(page_num, debug=getattr(args, "debug", False),
                        from_cache=getattr(args, "from_cache", False),
                        x_left_col_mode=getattr(args, "x_left_col_mode", "geometric"))
    n_rows = result["n_rows"]
    page_h = result["out_h"]
    page_w = result["out_w"]

    # 9. Column boundaries proportional to detected source widths; rows uniform.
    col_ranges = result.get(
        "col_ranges",
        list(np.linspace(0, W_OUT, EXPECTED_COLS + 1).round().astype(int)),
    )
    row_ranges = [(i * ROW_PITCH, (i + 1) * ROW_PITCH) for i in range(n_rows)]
    row_baseline_y = result.get("row_baseline_y_canvas")

    # 8. Load text for cells.
    #   - If --no-text: skip entirely.
    #   - If --htr-cells: run Kraken HTR on every cell of the dewarped canvas.
    #   - Else: load GT (page 3) or fall back to Approach M cache.
    canvas_bgr = cv2.imread(str(result["path"]))
    if canvas_bgr is None:
        raise RuntimeError(f"Could not read dewarped canvas: {result['path']}")

    if args.no_text:
        text_rows = None
        print("Text rows: skipped (--no-text)")
    elif getattr(args, "htr_cells", False):
        text_rows = recognize_table_cells(
            canvas_bgr, col_ranges, n_rows, H_HEADER, ROW_PITCH, page_num)
        print(f"Text rows from Kraken HTR: {len(text_rows)} ({n_rows} × {len(col_ranges)-1} cells)")
    else:
        text_rows = load_text_rows(page_num, cfg)
        print(f"Text rows loaded: {len(text_rows)} "
              f"({'GT' if cfg['gt_page'] and text_rows else 'Approach M'})")

    # 8b. Recognize the two header-strip fields (taxpayer name + index) above the table.
    top_strip = None
    if not args.no_text and getattr(args, "top_strip", True):
        top_strip = recognize_top_strip(
            canvas_bgr, H_META, page_num, debug_dir=CACHE_DIR)

    def _to_eastern(text: str) -> str: return text.translate(W2E)
    def _to_western(text: str) -> str: return text.translate(E2W)

    img_name = f"Hadita_{page_num}.jpeg"

    for sub, text_fn in [("original", _to_eastern),
                         ("western arabic transliteration", _to_western)]:
        out_dir = UPLOAD_DIR / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        # Copy dewarped JPEG into the upload folder so XML+image stay co-located.
        shutil.copyfile(result["path"], out_dir / img_name)
        # Write uniform-grid XML targeting the dewarped JPEG.
        xml_out = out_dir / f"Hadita_{page_num}.xml"
        write_page_xml(col_ranges, row_ranges,
                       y_offset=H_HEADER,
                       page_w=page_w, page_h=page_h,
                       image_filename=img_name,
                       out_path=xml_out,
                       text_rows=text_rows, bands=None, text_fn=text_fn,
                       col_tags=args.col_tags,
                       row_baseline_y=row_baseline_y,
                       top_strip=top_strip)
        patch_xml(xml_out)
        print(f"{sub:35s} → {xml_out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=int, choices=[3, 10, 50],
                        help="Process a single page (default: all pages)")
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip deskew; reuse .ocr_cache/deskewed_page{N}.{png,jpg}")
    parser.add_argument("--debug", action="store_true",
                        help="Write per-step images to processed/_debug/page{N}/")
    parser.add_argument("--col-tags", action="store_true",
                        help="Embed column names as Transkribus structural tags in TableCell custom attribute")
    parser.add_argument("--no-text", action="store_true",
                        help="Produce PAGE XML without text content")
    parser.add_argument("--htr-cells", action="store_true",
                        help="Run Kraken HTR on every table cell instead of loading text from GT/Approach M cache")
    parser.add_argument("--no-top-strip", dest="top_strip", action="store_false",
                        help="Skip taxpayer name/index recognition above the table header")
    parser.add_argument("--x-left-col-mode",
                        choices=("none", "geometric", "gemini", "claude", "consensus"),
                        default="geometric",
                        help="Strategy for fixing the right edge of the Serial_No column "
                             "(default: geometric — auto-corrects too-narrow first column)")
    parser.set_defaults(top_strip=True)
    args = parser.parse_args()

    pages = [args.page] if args.page else [3, 10, 50]
    for page_num in pages:
        process_page(page_num, args)

    print(f"\nDone. XML files written to: {UPLOAD_DIR}")


if __name__ == "__main__":
    main()
