#!/usr/bin/env python3
"""
kraken_experiment.py — Two-model Kraken pipeline experiment on page 3.

Pipeline:
  1. Kraken segmentation model  → detect text row y-ranges on the table area
  2. OpenCV morphological detection → detect column x-ranges (vertical dividers)
  3. Crop cell[row][col] and run Kraken OCR (no-segmentation mode) on each
  4. Assemble rows → compare against ground_truth.tsv

Row segmentation method:
  - Kraken v5.3 baseline segmentation model (sc_100p_full_line.mlmodel) runs on
    the cropped table image and returns ~94 line regions.
  - Lines with y_center < 200px are discarded (column-header area).
  - Nearby lines (gap < 60px) are clustered into single row entries; the row
    y_center is the cluster mean, y_min/y_max the extremes.
  - Any gap between consecutive row centers that exceeds 1.5× the median
    inter-row spacing (~94px) triggers insertion of one synthetic row at the
    midpoint (handles rows where Kraken found no text, e.g. all-ditto rows).
  - Row y-boundaries are set at midpoints between consecutive row centers.
  Result: 35 detected rows (GT = 36; one row still may be missing at bottom).

Column detection method:
  - CLAHE contrast enhancement + adaptive threshold + morphological vertical
    kernel projection finds ~21 peaks across the full table width.
  - The first two boundaries (x < TABLE_LEFT_X=140) are dropped — that area is
    book binding / background of the opposite page, not a real column.
  - All peak x-positions are shifted +15px (COL_SHIFT) to centre on printed lines.
  - After trimming to DETECT_COLS=20 via a nearest-ideal-position algorithm, the
    one wide merged column (Exempt_LP+Mils, x≈2070–2285, ~215px) is split by
    searching for the local projection peak nearest the midpoint.
  Result: 19 columns — Serial_No (leftmost) through Net_Assessment_Mils.

Tilt correction:
  - Printed column lines are not perfectly vertical; they drift ~30px rightward
    from bottom to top over the 3645px table height (≈0.82px per 100px).
  - TILT_RATE = 30/3645. Per-row x-offset = (H/2 − y_center) × TILT_RATE,
    applied to both column boundaries when cropping each cell so the crop window
    tracks the actual printed line position.
  - Block_No and Parcel_Area columns use left_expand=25px (vs 5px default) to
    avoid clipping longer values that sit close to the left boundary.

Models:
  Segmentation : sc_100p_full_line (1).mlmodel  (4.8 MB)
  Transcription: gen2_sc_clean_best.mlmodel     (19 MB)

Usage:
  python kraken_experiment.py              # full run
  python kraken_experiment.py --seg-only   # show detected rows, no OCR
  python kraken_experiment.py --preview    # HTML column-grid preview, no OCR
  python kraken_experiment.py --page-xml   # write PAGE XML segmentation file
  python kraken_experiment.py --no-cache   # ignore cached segmentation JSON
"""

import argparse
import csv
import difflib
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
PAGE3_IMAGE = PROJECT_DIR / "images" / "deskewed_page3.jpg"
SEG_MODEL   = PROJECT_DIR / "sc_100p_full_line (1).mlmodel"
OCR_MODEL   = PROJECT_DIR / "gen2_sc_clean_best.mlmodel"
KRAKEN_BIN  = "/opt/anaconda3/bin/kraken"
GT_FILE     = PROJECT_DIR / "ground_truth.tsv"
SEG_CACHE   = PROJECT_DIR / ".ocr_cache" / "kraken_seg_page3.json"

# ── Layout constants ───────────────────────────────────────────────────────────
HEADER_HEIGHT_FRAC    = 0.08    # skip top 8% (header row with taxpayer name)
LEFT_TABLE_WIDTH_FRAC = 0.494   # left 49.4% = the data table we're OCR-ing (confirmed visually)

# x=0..TABLE_LEFT_X is book binding / background of opposite page — not a table column.
# The first real table column (Serial_No) starts at approximately x=140.
TABLE_LEFT_X = 140

# Printed column lines are not perfectly vertical — they tilt rightward going down.
# Measured: ~30 px rightward drift over the full table height (~3645 px).
# TILT_RATE = dx/dy (positive = line moves right as y increases).
# Detection averages over full height, so the detected x is the mid-height position.
# Per-row correction: x_actual(y) = x_detected + (H/2 - y) * TILT_RATE
TILT_RATE = 30 / 3645  # ≈ 0.0082 px/px

# Real table columns, left→right as they appear in the image (Serial_No leftmost).
# "Remarks" is a reviewer-only field that has no corresponding printed column.
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
EXPECTED_COLS = len(LEFT_COLS)  # 19

# Detection runs with one extra column to account for the binding area,
# which shows up as a detected peak before TABLE_LEFT_X.
DETECT_COLS = EXPECTED_COLS + 1  # 20

# Table reads left→right: Serial_No is the leftmost detected column.
COL_ORDER_RTL = False


# ── Evaluation helpers (mirrored from evaluate_page3.py) ──────────────────────
EASTERN = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
E2W = str.maketrans(EASTERN, WESTERN)
DITTO_VARIANTS = {'״', '"', '〃', "''", ',,', '"', '״'}


def normalize(val: str) -> str:
    val = val.strip().translate(E2W)
    if val in DITTO_VARIANTS or val == '"':
        val = '"'
    if val in ('--', '—', '−', '- -'):
        val = '-'
    if len(val) > 1 and val[0] == '0' and val[1:].isdigit():
        val = val.lstrip('0') or '0'
    return val


def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    ops = difflib.SequenceMatcher(None, pred, ref).get_opcodes()
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    return edits / len(ref)


def has_hebrew(text: str) -> bool:
    return any('֐' <= c <= '׿' for c in text)


# ── Step 1: Crop table area from the full page image ──────────────────────────
def crop_table(image_path: Path) -> tuple[np.ndarray, int, int]:
    """Return (table_bgr, y_offset, x_offset) of the left data table."""
    img = cv2.imread(str(image_path))
    H, W = img.shape[:2]
    y0 = int(H * HEADER_HEIGHT_FRAC)
    x1 = int(W * LEFT_TABLE_WIDTH_FRAC)
    table = img[y0:H, 0:x1]
    return table, y0, 0


# ── Step 2: Detect column dividers via morphological projection ───────────────
# (Adapted from haditax.py detect_grid_from_image — more robust on faint lines)

def _trim_vlines(v_lines: list[int], expected_cols: int,
                 left: int = 0, right: int | None = None) -> list[int]:
    """Trim over-detected vertical lines to exactly expected_cols columns."""
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


def detect_columns(table_bgr: np.ndarray) -> list[int]:
    """
    Detect column x-dividers using CLAHE + adaptive threshold + morphological
    opening. Returns sorted list of column boundary x-positions starting at
    TABLE_LEFT_X (skipping the binding area), trimmed to EXPECTED_COLS columns.
    Any column wider than WIDE_COL_PX is split at its midpoint (handles the
    printed-but-faint Exempt_LP/Mils internal divider).
    """
    COL_SHIFT        = 15    # shift detected peaks right to centre on printed line
    WIDE_COL_PX      = 200  # width threshold to detect a merged column
    WIDE_COL_MIN_X   = 1900  # only apply split heuristic right of this x
    # Serial_No is legitimately wide (~278px) but sits at x<500; Exempt merge is at x≈2070–2285.

    gray = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = gray.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    v_mask   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
    v_mask   = cv2.dilate(v_mask, np.ones((1, 2), np.uint8), iterations=1)

    proj_x = np.sum(v_mask, axis=0).astype(float)
    peaks, _ = find_peaks(
        proj_x,
        height=proj_x.mean() + 0.5 * proj_x.std(),
        distance=max(30, tw // 30),
    )
    v_lines = peaks.tolist()

    if len(v_lines) < 3:
        log.warning("Only %d v-lines detected — falling back to uniform grid", len(v_lines))
        col_w = (tw - TABLE_LEFT_X) // EXPECTED_COLS
        return [TABLE_LEFT_X + i * col_w for i in range(EXPECTED_COLS + 1)]

    # Add full-image edges, then shift right
    if not v_lines or v_lines[0] > 30:
        v_lines = [0] + v_lines
    if v_lines[-1] < tw - 30:
        v_lines.append(tw)
    v_lines = [min(v + COL_SHIFT, tw) for v in v_lines]
    v_lines = sorted(set(v_lines))

    # Trim to DETECT_COLS (20) including the binding column(s)
    v_lines = _trim_vlines(v_lines, DETECT_COLS, left=0, right=tw)

    # Drop everything before TABLE_LEFT_X (binding / opposite-page background)
    real = [x for x in v_lines if x >= TABLE_LEFT_X]
    if not real or real[0] > TABLE_LEFT_X + 30:
        real = [TABLE_LEFT_X] + real

    # Split any wide merged column in the right portion of the table.
    # Instead of using the midpoint, search for the actual (weak) peak in the
    # projection within the candidate range to find the real printed divider.
    # Only applies right of WIDE_COL_MIN_X to avoid splitting Serial_No.
    final: list[int] = [real[0]]
    for i in range(len(real) - 1):
        w = real[i + 1] - real[i]
        if w > WIDE_COL_PX and real[i] >= WIDE_COL_MIN_X:
            # Search for the best local peak within the merged column interior.
            # Use find_peaks with a low threshold (any elevation above the 20th
            # percentile) and pick the candidate nearest the midpoint so we
            # don't accidentally snap to one of the strong outer boundaries.
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
            log.info("Splitting merged column %d px at x=%d→x=%d (divider at x=%d)",
                     w, real[i], real[i + 1], local_peak)
            final.append(local_peak)
        final.append(real[i + 1])
    final = sorted(set(final))

    n_cols = len(final) - 1
    if n_cols != EXPECTED_COLS:
        log.warning("After column adjustment: %d cols detected (expected %d) — "
                    "mapping best %d", n_cols, EXPECTED_COLS, min(n_cols, EXPECTED_COLS))

    log.info("Column boundaries (%d cols): %s", n_cols, final)
    return final


# ── Step 3: Run Kraken segmentation to find text row baselines ────────────────
def run_segmentation(table_bgr: np.ndarray, cache_path: Path, use_cache: bool = True) -> list[dict]:
    """
    Run Kraken with the baseline segmentation model on the table image.
    Returns list of line dicts, each with 'y_center' and 'boundary' (xyxy).
    """
    if use_cache and cache_path.exists():
        log.info("Loading cached segmentation from %s", cache_path)
        with open(cache_path) as f:
            seg_data = json.load(f)
        return seg_data["lines"]

    with tempfile.TemporaryDirectory() as tmp:
        in_path  = Path(tmp) / "table.jpg"
        out_path = Path(tmp) / "seg.json"

        # Save table crop as JPEG
        cv2.imwrite(str(in_path), table_bgr)

        cmd = [
            KRAKEN_BIN,
            "--native",                         # JSON output
            "-i", str(in_path), str(out_path),  # input → output
            "segment",
            "--baseline",                        # neural baseline segmenter
            "--model", str(SEG_MODEL),
        ]
        log.info("Running segmentation: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            log.error("Kraken segment failed:\nSTDOUT: %s\nSTDERR: %s",
                      result.stdout, result.stderr)
            sys.exit(1)

        if not out_path.exists():
            log.error("Kraken segment produced no output file. STDERR: %s", result.stderr)
            sys.exit(1)

        with open(out_path) as f:
            raw = json.load(f)

    log.info("Segmentation raw keys: %s", list(raw.keys()))

    # Parse Kraken v5 JSON: lines have 'baseline' (list of [x,y]) and 'boundary' polygon
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

        # Bounding box from boundary polygon
        if boundary:
            xs = [pt[0] for pt in boundary]
            ys_b = [pt[1] for pt in boundary]
            y_min, y_max = min(ys_b), max(ys_b)
        else:
            y_min = y_max = y_center

        lines.append({"y_center": y_center, "y_min": y_min, "y_max": y_max})

    lines.sort(key=lambda l: l["y_center"])
    log.info("Detected %d text lines (rows)", len(lines))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"lines": lines}, f, indent=2)
    log.info("Cached segmentation to %s", cache_path)

    return lines


# ── Step 4: Cluster detected lines into table rows, then compute y-ranges ─────
def cluster_lines(lines: list[dict], gap_threshold: int = 60,
                  skip_header_y: int = 200) -> list[dict]:
    """
    Group nearby detected lines into table rows.

    Each Kraken-detected line is a single cell's text region; multiple cells in
    the same table row are spread across x but share a similar y-center.
    Lines within `gap_threshold` pixels vertically are merged into one row.

    Lines in the top `skip_header_y` pixels are discarded (column header area).
    """
    data_lines = [l for l in lines if l["y_center"] > skip_header_y]
    if not data_lines:
        return []

    clusters: list[list[dict]] = []
    current = [data_lines[0]]
    for line in data_lines[1:]:
        if line["y_center"] - current[-1]["y_center"] > gap_threshold:
            clusters.append(current)
            current = [line]
        else:
            current.append(line)
    clusters.append(current)

    rows = []
    for cluster in clusters:
        y_min    = min(l["y_min"]    for l in cluster)
        y_max    = max(l["y_max"]    for l in cluster)
        y_center = sum(l["y_center"] for l in cluster) // len(cluster)
        rows.append({"y_center": y_center, "y_min": y_min, "y_max": y_max})
    return rows


def interpolate_gaps(rows: list[dict], min_gap_factor: float = 1.5) -> list[dict]:
    """Insert a synthetic row in any gap larger than min_gap_factor × median spacing.

    Kraken segmentation misses empty-looking cells; a large gap means a row had
    no detectable text. One synthetic row per gap is inserted at the midpoint.
    """
    if len(rows) < 2:
        return rows
    spacings = [rows[i + 1]["y_center"] - rows[i]["y_center"] for i in range(len(rows) - 1)]
    median_sp = sorted(spacings)[len(spacings) // 2]
    threshold = median_sp * min_gap_factor

    result: list[dict] = [rows[0]]
    for i in range(1, len(rows)):
        gap = rows[i]["y_center"] - rows[i - 1]["y_center"]
        if gap > threshold:
            n_insert = round(gap / median_sp) - 1
            for k in range(1, n_insert + 1):
                sy = rows[i - 1]["y_center"] + round(k * gap / (n_insert + 1))
                result.append({"y_center": sy, "y_min": sy - 20, "y_max": sy + 20,
                                "synthetic": True})
                log.info("Inserted synthetic row at y=%d (gap=%.0f px, median=%.0f px)",
                         sy, gap, median_sp)
        result.append(rows[i])
    return result


def lines_to_row_ranges(rows: list[dict], table_height: int) -> list[tuple[int, int]]:
    """
    Given clustered row dicts (sorted by y_center), compute (y_top, y_bottom)
    boundaries at the midpoint between consecutive row centers.
    """
    if not rows:
        return []

    centers = [r["y_center"] for r in rows]
    boundaries = [0]
    for i in range(len(centers) - 1):
        boundaries.append((centers[i] + centers[i + 1]) // 2)
    boundaries.append(table_height)

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(centers))]


def tilt_offset(y_center: int, table_h: int) -> int:
    """X offset to apply to column boundaries at a given row y_center.

    The detected x is the average over the full column height. The actual
    printed line at y_center is shifted by (H/2 - y_center) * TILT_RATE.
    """
    return round((table_h / 2 - y_center) * TILT_RATE)


# ── Step 5–6: Batch cell OCR (single Kraken call, model loads once) ───────────
# Per-column left expansion (px): most columns need only a small buffer;
# Block_No (idx 2) and Parcel_Area (idx 5) have longer values that sit close
# to the left boundary, so they get a wider left margin.
_COL_LEFT_EXPAND = {2: 25, 5: 25}
_DEFAULT_LEFT_EXPAND = 5


def run_cell_ocr(table_bgr: np.ndarray,
                 row_ranges: list[tuple[int, int]],
                 col_ranges: list[int],
                 pad: int = 3,
                 ocr_model: Path = OCR_MODEL) -> list[list[str]]:
    """
    Crop all cells, save to a temp directory, then run one Kraken batch call
    with -I glob so the model loads only once.
    Returns 2D list results[row_idx][col_idx]; column 0 is leftmost in image.
    """
    n_rows = len(row_ranges)
    n_cols = len(col_ranges) - 1
    total  = n_rows * n_cols
    log.info("Running batch cell OCR: %d rows × %d cols = %d cells", n_rows, n_cols, total)

    results: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        valid_indices: list[int] = []

        table_h = table_bgr.shape[0]
        for r_idx, (y0, y1) in enumerate(row_ranges):
            y_center = (y0 + y1) // 2
            dx = tilt_offset(y_center, table_h)
            for c_idx in range(n_cols):
                idx = r_idx * n_cols + c_idx
                left_exp = _COL_LEFT_EXPAND.get(c_idx, _DEFAULT_LEFT_EXPAND)
                x0  = max(0, col_ranges[c_idx]     + dx - left_exp)
                x1  = min(table_bgr.shape[1], col_ranges[c_idx + 1] + dx - pad)
                y0c = max(0, y0 + pad)
                y1c = min(table_bgr.shape[0], y1   - pad)
                if x1 > x0 and y1c > y0c:
                    cv2.imwrite(str(tmp_path / f"c{idx:05d}.jpg"), table_bgr[y0c:y1c, x0:x1])
                    valid_indices.append(idx)

        log.info("Saved %d/%d non-empty cell images; running Kraken batch ...", len(valid_indices), total)

        cmd = [
            KRAKEN_BIN,
            "-I", "c*.jpg",      # relative glob; cwd=tmp_path
            "-o", ".txt",
            "ocr", "--no-segmentation", "--model", str(ocr_model),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=str(tmp_path))
        if result.returncode != 0:
            log.warning("Kraken batch OCR failed (code %d): %s", result.returncode, result.stderr[-300:])

        # Read outputs
        for idx in valid_indices:
            r_idx, c_idx = divmod(idx, n_cols)
            out_file = tmp_path / f"c{idx:05d}.txt"
            if out_file.exists():
                raw = out_file.read_text(encoding="utf-8").strip()
                results[r_idx][c_idx] = raw.replace("|", "").strip()

    return results


# ── Step 7: Map column positions to schema field names ────────────────────────
def assemble_rows(ocr_results: list[list[str]]) -> list[dict]:
    """
    Map cell[row][col_idx] → {field_name: text}.
    Columns are detected left→right; Serial_No is the leftmost column (col 0).
    """
    n_cols_detected = len(ocr_results[0]) if ocr_results else 0
    col_names = list(LEFT_COLS)  # left→right matches detected order

    if n_cols_detected != EXPECTED_COLS:
        log.warning(
            "Detected %d columns but expected %d — mapping what we have",
            n_cols_detected, EXPECTED_COLS
        )
        col_names = col_names[:n_cols_detected]

    rows = []
    for row_texts in ocr_results:
        row = {"Page_Number": "3", "Folio_Number": "1", "OCR_Method": "kraken_twostage"}
        for i, text in enumerate(row_texts):
            if i < len(col_names):
                row[col_names[i]] = text
        rows.append(row)

    return rows


# ── Step 8: Score against ground truth ────────────────────────────────────────
def score(ocr_rows: list[dict]) -> None:
    gt_rows = []
    with open(GT_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r["Page_Number"] == "3" and r.get("Serial_No", "").strip():
                gt_rows.append(r)

    print(f"\nGround truth: {len(gt_rows)} rows | OCR result: {len(ocr_rows)} rows")

    # Index GT by Serial_No
    gt_by_sno = {normalize(r["Serial_No"]): r for r in gt_rows}

    cols_to_score = [c for c in LEFT_COLS if c in (gt_rows[0] if gt_rows else {})]

    total_cells  = 0
    exact_hits   = 0
    total_cer    = 0.0
    col_exact    = {c: 0 for c in cols_to_score}
    col_total    = {c: 0 for c in cols_to_score}
    mismatches   = []
    matched_rows = 0

    for row in ocr_rows:
        sno = normalize(row.get("Serial_No", ""))
        gt = gt_by_sno.get(sno)
        if not gt:
            continue
        matched_rows += 1

        for col in cols_to_score:
            gt_val   = normalize(gt.get(col, ""))
            pred_val = normalize(row.get(col, ""))
            if not gt_val:
                continue
            if col == "Remarks" and has_hebrew(gt_val):
                continue

            total_cells += 1
            c = cer(pred_val, gt_val)
            total_cer += c
            col_total[col] += 1

            if pred_val == gt_val:
                exact_hits += 1
                col_exact[col] += 1
            elif len(mismatches) < 20:
                mismatches.append((col, gt_val, pred_val))

    if total_cells == 0:
        print("No cells matched between OCR output and ground truth.")
        print("Check that Serial_No values were OCR'd correctly (should be 1–36).")
        _show_serial_nos(ocr_rows)
        return

    exact_pct = exact_hits / total_cells * 100
    mean_cer_val  = total_cer / total_cells

    print(f"\n{'─'*60}")
    print(f"Matched rows : {matched_rows}/{len(ocr_rows)}")
    print(f"Cells scored : {total_cells}")
    print(f"Exact match  : {exact_hits}/{total_cells}  ({exact_pct:.1f}%)")
    print(f"Mean CER     : {mean_cer_val:.4f}")

    print(f"\n{'Column':<35} {'Exact':>7} {'/ Tot':>7} {'Rate':>8}")
    print("─" * 60)
    for col in cols_to_score:
        n = col_total[col]
        if n == 0:
            continue
        rate = col_exact[col] / n * 100
        marker = " ◄" if rate < 50 else ""
        print(f"  {col:<33} {col_exact[col]:>5}  / {n:<5}  {rate:>6.1f}%{marker}")

    if mismatches:
        print(f"\nSample mismatches:")
        for col, gt_v, pred_v in mismatches:
            print(f"  {col:<30} GT={gt_v!r}  PRED={pred_v!r}")


def _show_serial_nos(ocr_rows: list[dict]) -> None:
    snos = [row.get("Serial_No", "?") for row in ocr_rows[:10]]
    print(f"  First 10 Serial_No values OCR'd: {snos}")


def write_page_xml(col_ranges: list[int],
                   row_ranges: list[tuple[int, int]],
                   y_offset: int,
                   page_w: int, page_h: int,
                   out_path: Path,
                   ocr_rows: list[dict] | None = None) -> None:
    """Write a PAGE XML file (2013-07-15 schema) with a TableRegion + TableCells.

    Cell coordinates are parallelograms that account for the column tilt.
    If ocr_rows is provided, recognised text is embedded in each TextLine.
    The file can be loaded into Transkribus alongside the page image.
    """
    from datetime import datetime
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    n_rows = len(row_ranges)
    n_cols = len(col_ranges) - 1
    table_h = page_h - y_offset

    def _tilt_x(x: int, y_table: int) -> int:
        return x + round((table_h / 2 - y_table) * TILT_RATE)

    def _cell_pts(cx0: int, cy0: int, cx1: int, cy1: int) -> str:
        corners = [
            (_tilt_x(cx0, cy0), cy0 + y_offset),
            (_tilt_x(cx1, cy0), cy0 + y_offset),
            (_tilt_x(cx1, cy1), cy1 + y_offset),
            (_tilt_x(cx0, cy1), cy1 + y_offset),
        ]
        return " ".join(f"{x},{y}" for x, y in corners)

    tx0, ty0 = col_ranges[0], row_ranges[0][0] + y_offset
    tx1, ty1 = col_ranges[-1], row_ranges[-1][1] + y_offset
    table_pts = f"{tx0},{ty0} {tx1},{ty0} {tx1},{ty1} {tx0},{ty1}"

    cells_xml: list[str] = []
    for r_idx, (y0, y1) in enumerate(row_ranges):
        for c_idx in range(n_cols):
            cid = f"cell_r{r_idx}_c{c_idx}"
            cpts = _cell_pts(col_ranges[c_idx], y0, col_ranges[c_idx + 1], y1)
            text = ""
            if ocr_rows and r_idx < len(ocr_rows):
                col_name = LEFT_COLS[c_idx] if c_idx < len(LEFT_COLS) else ""
                text = ocr_rows[r_idx].get(col_name, "")
            cells_xml.append(
                f'      <TableCell id="{cid}" row="{r_idx}" col="{c_idx}" '
                f'rowSpan="1" colSpan="1">\n'
                f'        <Coords points="{cpts}"/>\n'
                f'        <TextLine id="line_{cid}">\n'
                f'          <Coords points="{cpts}"/>\n'
                f'          <TextEquiv><Unicode>{text}</Unicode></TextEquiv>\n'
                f'        </TextLine>\n'
                f'      </TableCell>'
            )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"\n'
        '       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '       xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 '
        'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd">\n'
        '  <Metadata>\n'
        '    <Creator>kraken_experiment.py</Creator>\n'
        f'    <Created>{now}</Created>\n'
        f'    <LastChange>{now}</LastChange>\n'
        '  </Metadata>\n'
        f'  <Page imageFilename="{PAGE3_IMAGE.name}" imageWidth="{page_w}" imageHeight="{page_h}">\n'
        f'    <TableRegion id="table1" rows="{n_rows}" columns="{n_cols}">\n'
        f'      <Coords points="{table_pts}"/>\n'
        + "\n".join(cells_xml) + "\n"
        '    </TableRegion>\n'
        '  </Page>\n'
        '</PcGts>\n'
    )
    out_path.write_text(xml, encoding="utf-8")
    print(f"PAGE XML written → {out_path}  ({n_rows} rows × {n_cols} cols)")


def write_column_preview(table_bgr: np.ndarray, col_ranges: list[int],
                         row_ranges: list[tuple[int, int]],
                         out_path: Path = Path("column_preview.html")) -> None:
    """Write an HTML file showing column grid overlaid on the table header + first 3 data rows."""
    import base64, io
    from PIL import Image as PILImage

    th, tw = table_bgr.shape[:2]
    preview_y1 = min(row_ranges[2][1] if len(row_ranges) >= 3 else th, th)
    strip = table_bgr[:preview_y1, :]

    # Draw tilted column lines on the strip
    # Each line runs from (x + tilt_offset(0, th), 0) to (x + tilt_offset(preview_y1, th), preview_y1)
    vis = strip.copy()
    colors = [(0, 0, 220), (0, 180, 0)]  # alternating blue-red / green
    for i, x in enumerate(col_ranges):
        x_top = x + tilt_offset(0, th)
        x_bot = x + tilt_offset(preview_y1, th)
        cv2.line(vis, (x_top, 0), (x_bot, preview_y1 - 1), colors[i % 2], 2)

    # Label columns
    for i in range(len(col_ranges) - 1):
        cx = (col_ranges[i] + col_ranges[i + 1]) // 2
        label = LEFT_COLS[i] if i < len(LEFT_COLS) else f"col{i}"
        short = label.split("_")[-1][:8]
        cv2.putText(vis, f"{i+1}:{short}", (cx - 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1, cv2.LINE_AA)

    # Encode to PNG → base64
    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    col_info = "".join(
        f"<tr><td>{i+1}</td><td>{LEFT_COLS[i] if i < len(LEFT_COLS) else '?'}</td>"
        f"<td>{col_ranges[i]}–{col_ranges[i+1]}</td>"
        f"<td>{col_ranges[i+1]-col_ranges[i]} px</td></tr>"
        for i in range(len(col_ranges) - 1)
    )
    html = f"""<!DOCTYPE html><html><body>
<h2>Column Grid Preview ({len(col_ranges)-1} cols)</h2>
<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ccc">
<h3>Column map</h3>
<table border="1" cellpadding="4"><tr><th>#</th><th>Field</th><th>x range</th><th>Width</th></tr>
{col_info}</table>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"Column preview written → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-only",  action="store_true", help="Run segmentation only, show detected rows")
    parser.add_argument("--preview",   action="store_true", help="Generate column grid HTML preview, no OCR")
    parser.add_argument("--page-xml",  action="store_true", help="Write PAGE XML segmentation file (no OCR)")
    parser.add_argument("--no-cache",  action="store_true", help="Ignore cached segmentation JSON")
    parser.add_argument("--save-csv",  metavar="FILE", default="kraken_experiment_page3.csv",
                        help="Save assembled rows to this CSV (default: kraken_experiment_page3.csv)")
    parser.add_argument("--save-cache", action="store_true",
                        help="Also write .ocr_cache/kraken_page3.json for use in haditax.py")
    parser.add_argument("--ocr-model", metavar="PATH", default=None,
                        help="Override the OCR model path (default: gen2_sc_clean_best.mlmodel)")
    args = parser.parse_args()

    if not PAGE3_IMAGE.exists():
        log.error("Page 3 image not found: %s", PAGE3_IMAGE)
        sys.exit(1)
    if not SEG_MODEL.exists():
        log.error("Segmentation model not found: %s", SEG_MODEL)
        sys.exit(1)
    # Allow runtime override of OCR model; resolve to absolute so it works
    # when Kraken batch OCR runs with cwd=tmp_path
    if args.ocr_model:
        ocr_model = Path(args.ocr_model)
        if not ocr_model.is_absolute():
            ocr_model = PROJECT_DIR / ocr_model
    else:
        ocr_model = OCR_MODEL
    if not ocr_model.exists():
        log.error("OCR model not found: %s", ocr_model)
        sys.exit(1)

    print("=" * 60)
    print("Kraken Two-Model Experiment — Page 3")
    print(f"OCR model: {ocr_model.name}")
    print("=" * 60)

    # 1. Crop table area
    log.info("Cropping table from %s", PAGE3_IMAGE)
    table_bgr, y_offset, x_offset = crop_table(PAGE3_IMAGE)
    th, tw = table_bgr.shape[:2]
    log.info("Table crop: %d × %d px (y_offset=%d)", tw, th, y_offset)

    # 2. Detect column dividers (OpenCV)
    col_ranges = detect_columns(table_bgr)
    n_cols = len(col_ranges) - 1
    print(f"\nOpenCV detected {n_cols} columns (expected {EXPECTED_COLS})")
    for i in range(n_cols):
        name = LEFT_COLS[i] if i < len(LEFT_COLS) else "?"
        print(f"  col {i+1:2d}: x={col_ranges[i]:4d}–{col_ranges[i+1]:4d}"
              f"  ({col_ranges[i+1]-col_ranges[i]:3d}px)  {name}")

    # 3. Run Kraken segmentation to find text rows
    raw_lines = run_segmentation(table_bgr, SEG_CACHE, use_cache=not args.no_cache)
    print(f"\nKraken segmentation: {len(raw_lines)} raw line regions detected")

    row_clusters = cluster_lines(raw_lines, gap_threshold=60, skip_header_y=200)
    row_clusters = interpolate_gaps(row_clusters, min_gap_factor=1.5)
    row_ranges   = lines_to_row_ranges(row_clusters, th)
    print(f"After clustering + gap interpolation: {len(row_clusters)} rows  (GT expects 36)")

    if args.seg_only:
        print("\nClustered rows (y_center, bbox):")
        for i, r in enumerate(row_clusters):
            syn = " [synthetic]" if r.get("synthetic") else ""
            print(f"  row {i+1:3d}: y={r['y_center']:4d}  [{r['y_min']:4d}–{r['y_max']:4d}]{syn}")
        return

    if args.preview:
        write_column_preview(table_bgr, col_ranges, row_ranges,
                             out_path=PROJECT_DIR / "column_preview.html")
        return

    if args.page_xml:
        img_full = cv2.imread(str(PAGE3_IMAGE))
        ph, pw = img_full.shape[:2]
        write_page_xml(col_ranges, row_ranges, y_offset, pw, ph,
                       out_path=PROJECT_DIR / "page3_segmentation.xml")
        return

    # 4. Cell-level OCR
    ocr_results = run_cell_ocr(table_bgr, row_ranges, col_ranges, ocr_model=ocr_model)

    # 5. Assemble into rows with field names
    rows = assemble_rows(ocr_results)

    # 6. Save to CSV
    out_path = PROJECT_DIR / args.save_csv
    fieldnames = ["Page_Number", "Folio_Number", "OCR_Method"] + LEFT_COLS
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {out_path}")

    # Optionally write haditax.py-compatible JSON cache
    if args.save_cache:
        import json as _json
        model_tag = ocr_model.stem.replace(" ", "_")
        cache_path = PROJECT_DIR / ".ocr_cache" / f"kraken_{model_tag}_page3.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"App cache written → {cache_path}")

    # 7. Score against ground truth
    if GT_FILE.exists():
        score(rows)
    else:
        print(f"\nGround truth file not found ({GT_FILE}); skipping scoring.")


if __name__ == "__main__":
    main()
