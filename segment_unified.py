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
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent
IMAGES_DIR   = PROJECT_DIR / "images"
CACHE_DIR    = PROJECT_DIR / ".ocr_cache"
GT_FILE      = PROJECT_DIR / "ground_truth.tsv"
SEG_MODEL    = PROJECT_DIR / "sc_100p_full_line (1).mlmodel"
KRAKEN_BIN   = "/opt/anaconda3/bin/kraken"
UPLOAD_DIR   = PROJECT_DIR / "Transkribus upload"

# ── Per-page config ────────────────────────────────────────────────────────────
# Tuned on page 3; adjust if pages 10/50 differ significantly after inspection.
PAGE_CONFIG = {
    3:  {"image": "deskewed_page3.jpg",  "transkribus_image": "Hadita_3.jpeg",
         "xml_name": "Hadita_3.xml",  "header_frac": 0.08, "table_width_frac": 0.494,
         "table_left_x": 140, "gt_page": "3"},
    10: {"image": "deskewed_page10.jpg", "transkribus_image": "Hadita_10.jpeg",
         "xml_name": "Hadita_10.xml", "header_frac": 0.08, "table_width_frac": 0.494,
         "table_left_x": 140, "gt_page": None},
    50: {"image": "deskewed_page50.jpg", "transkribus_image": "Hadita_50.jpeg",
         "xml_name": "Hadita_50.xml", "header_frac": 0.08, "table_width_frac": 0.494,
         "table_left_x": 140, "gt_page": None},
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


def detect_columns(table_bgr: np.ndarray, table_left_x: int) -> list[int]:
    """CLAHE + adaptive threshold + morphological column detection.
    Returns sorted boundary x-positions trimmed to EXPECTED_COLS columns."""
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
        col_w = (tw - table_left_x) // EXPECTED_COLS
        return [table_left_x + i * col_w for i in range(EXPECTED_COLS + 1)]

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

    n_cols = len(final) - 1
    if n_cols != EXPECTED_COLS:
        log.warning("Column detection: %d cols (expected %d)", n_cols, EXPECTED_COLS)
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
    if use_cache and cache_path.exists():
        log.info("Loading cached Kraken segmentation from %s", cache_path)
        with open(cache_path) as f:
            data = json.load(f)
        lines = data["lines"]
    else:
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
            lines.append({"y_center": y_center, "y_min": y_min, "y_max": y_max})
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
                   col_tags: bool = False) -> None:
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

    tx0, ty0 = col_ranges[0], row_ranges[0][0] + y_offset
    tx1, ty1 = col_ranges[-1], row_ranges[-1][1] + y_offset
    table_pts = f"{tx0},{ty0} {tx1},{ty0} {tx1},{ty1} {tx0},{ty1}"

    cells_xml: list[str] = []
    for r_idx, (y0, y1) in enumerate(row_ranges):
        for c_idx in range(n_cols):
            cid  = f"cell_r{r_idx}_c{c_idx}"
            cpts = _cell_pts(c_idx, y0, c_idx + 1, y1)
            text = ""
            if text_rows and r_idx < len(text_rows):
                col_name = LEFT_COLS[c_idx] if c_idx < len(LEFT_COLS) else ""
                text = text_rows[r_idx].get(col_name, "").strip()
                if text and text_fn:
                    text = text_fn(text)
            textline_xml = ""
            if text:
                bl = _baseline_pts(c_idx, y0, c_idx + 1, y1)
                textline_xml = (
                    f'        <TextLine id="line_{cid}">\n'
                    f'          <Coords points="{cpts}"/>\n'
                    f'          <Baseline points="{bl}"/>\n'
                    f'          <TextEquiv><Unicode>{text}</Unicode></TextEquiv>\n'
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
    cfg        = PAGE_CONFIG[page_num]
    image_path = IMAGES_DIR / cfg["image"]

    if not image_path.exists():
        log.error("Image not found: %s", image_path)
        return

    print(f"\n{'='*60}")
    print(f"Page {page_num}  ({image_path.name})")
    print(f"{'='*60}")

    image = cv2.imread(str(image_path))
    ph, pw = image.shape[:2]

    # 1. Crop table
    table_bgr, y_offset, _ = crop_table(image, cfg)
    th, tw = table_bgr.shape[:2]
    log.info("Table crop: %d×%d px, y_offset=%d", tw, th, y_offset)

    # 2. Column detection (global + per-band)
    col_ranges = detect_columns(table_bgr, cfg["table_left_x"])
    bands      = detect_columns_banded(table_bgr, col_ranges)
    print(f"Columns: {len(col_ranges)-1} detected (expected {EXPECTED_COLS}), "
          f"{len(bands)}/{N_BANDS} bands usable")

    # 3. Row detection
    seg_cache  = CACHE_DIR / f"unified_seg_page{page_num}.json"
    skip_y     = int(th * 0.10)   # skip top 10% of table — clears the printed column-label header area
    row_dicts  = detect_rows(table_bgr, seg_cache,
                              use_cache=not args.no_cache,
                              method=args.row_method,
                              skip_header_y=skip_y)
    row_ranges = rows_to_ranges(row_dicts, th)

    synth = sum(1 for r in row_dicts if r.get("synthetic"))
    print(f"Rows: {len(row_dicts)} detected ({synth} synthetic from gap interpolation)")

    # 4. Load text for cells
    if args.no_text:
        text_rows = None
        print("Text rows: skipped (--no-text)")
    else:
        text_rows = load_text_rows(page_num, cfg)
        print(f"Text rows loaded: {len(text_rows)} "
              f"({'GT' if cfg['gt_page'] and text_rows else 'Approach M'})")

    # 5. Export PAGE XML — two variants (no header row; row 0 = serial no. 1)
    full_ranges = list(row_ranges)

    def _to_eastern(text: str) -> str:
        return text.translate(W2E)

    def _to_western(text: str) -> str:
        return text.translate(E2W)

    xml_name = cfg["xml_name"]
    img_name = cfg["transkribus_image"]

    # Original (Arabic-Indic digits)
    out_orig = UPLOAD_DIR / "original" / xml_name
    out_orig.parent.mkdir(parents=True, exist_ok=True)
    write_page_xml(col_ranges, full_ranges, y_offset, pw, ph,
                   img_name, out_orig,
                   text_rows=text_rows, bands=bands, text_fn=_to_eastern,
                   col_tags=args.col_tags)

    # Western Arabic transliteration (digits as-is)
    out_west = UPLOAD_DIR / "western arabic transliteration" / xml_name
    out_west.parent.mkdir(parents=True, exist_ok=True)
    write_page_xml(col_ranges, full_ranges, y_offset, pw, ph,
                   img_name, out_west,
                   text_rows=text_rows, bands=bands, text_fn=_to_western,
                   col_tags=args.col_tags)

    print(f"Original → {out_orig}")
    print(f"Western  → {out_west}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=int, choices=[3, 10, 50],
                        help="Process a single page (default: all pages)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached Kraken segmentation and re-run")
    parser.add_argument("--row-method", choices=["kraken", "morph"], default="kraken",
                        help="Row detection method (default: kraken, fallback: morph)")
    parser.add_argument("--col-tags", action="store_true",
                        help="Embed column names as Transkribus structural tags in TableCell custom attribute")
    parser.add_argument("--no-text", action="store_true",
                        help="Produce PAGE XML without text content")
    args = parser.parse_args()

    pages = [args.page] if args.page else [3, 10, 50]
    for page_num in pages:
        process_page(page_num, args)

    print(f"\nDone. XML files written to: {UPLOAD_DIR}")


if __name__ == "__main__":
    main()
