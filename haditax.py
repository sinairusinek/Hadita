#!/usr/bin/env python3
"""
Haditax — Streamlit ground-truth editor for Haditha tax register OCR.

Usage:
    streamlit run haditax.py
"""

import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from image_preprocess import (
    auto_table_corners,
    corners_to_overlay,
    crop_above_table,
    crop_header_strip,
    gemini_extract_schema,
    load_notebook_config,
    save_notebook_config,
)

# ── Project paths ────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
IMAGE_PATTERN = "000nvrj-432316TAX 1-85_page-{:04d}.jpg"
CACHE_DIR = PROJECT_DIR / ".ocr_cache"
GROUND_TRUTH_FILE = PROJECT_DIR / "ground_truth.tsv"
PAGE_METADATA_FILE = PROJECT_DIR / "page_metadata.tsv"
NOTEBOOK_CONFIG_FILE = PROJECT_DIR / "notebook_config.json"

GITHUB_REPO     = "sinairusinek/Hadita"
GITHUB_GT_PATH  = "ground_truth.tsv"
GITHUB_META_PATH = "page_metadata.tsv"

META_FIELDS = [
    "Tax_Payer_Arabic", "Tax_Payer_Romanized",
    "Tax_Payer_ID_Arabic", "Tax_Payer_ID_Romanized",
]

# ── Column definitions (must match compare_ocr.py) ───────────
LEFT_COLS = [
    "Serial_No", "Date",
    "Property_recorded_under_Block_No", "Property_recorded_under_Parcel_No",
    "Parcel_Cat_No", "Parcel_Area",
    "Nature_of_Entry", "New_Serial_No",
    "Reference_to_Register_of_Changes_Volume_No", "Reference_to_Register_of_Changes_Serial_No",
    "Tax_LP", "Tax_Mils", "Total_Tax_LP", "Total_Tax_Mils",
    "Reference_to_Register_of_Exemptions_Entry_No",
    "Reference_to_Register_of_Exemptions_Amount_LP",
    "Reference_to_Register_of_Exemptions_Amount_Mils",
    "Net_Assessment_LP", "Net_Assessment_Mils",
    "Remarks",
]
RIGHT_COLS = []  # right page not captured
META_COLS = ["Row_Confidence", "Red_Ink", "Disagreements"]
ALL_DATA_COLS = LEFT_COLS + META_COLS

GT_COLS = [
    "Page_Number", "Folio_Number",
    "Tax_Payer_Arabic", "Tax_Payer_Romanized",
    "Tax_Payer_ID_Arabic", "Tax_Payer_ID_Romanized",
    "Serial_No", "Date",
    "Property_recorded_under_Block_No", "Property_recorded_under_Parcel_No",
    "Parcel_Cat_No", "Parcel_Area",
    "Nature_of_Entry", "New_Serial_No",
    "Reference_to_Register_of_Changes_Volume_No", "Reference_to_Register_of_Changes_Serial_No",
    "Tax_LP", "Tax_Mils", "Total_Tax_LP", "Total_Tax_Mils",
    "\u05d4\u05e2\u05e8\u05d5\u05ea",  # Hebrew Remarks column header
    "Reference_to_Register_of_Exemptions_Entry_No",
    "Reference_to_Register_of_Exemptions_Amount_LP",
    "Reference_to_Register_of_Exemptions_Amount_Mils",
    "Net_Assessment_LP", "Net_Assessment_Mils",
    "Remarks",
    "Row_Confidence", "Red_Ink", "OCR_Method",
]

PAGE_FOLIO = {3: "1", 10: "9", 50: "49"}

# Layout fractions (from compare_ocr.py)
HEADER_HEIGHT_FRAC = 0.08
LEFT_TABLE_WIDTH_FRAC = 0.455


# ── Digit conversion ─────────────────────────────────────────

_AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_W_DIGITS  = "0123456789"
_AR_TO_W = str.maketrans(_AR_DIGITS, _W_DIGITS)
_W_TO_AR = str.maketrans(_W_DIGITS, _AR_DIGITS)

def convert_digits(text: str, mode: str) -> str:
    """Convert digits in text. mode='western' or 'arabic'."""
    if not isinstance(text, str):
        text = str(text)
    if mode == "western":
        return text.translate(_AR_TO_W)
    else:
        return text.translate(_W_TO_AR)


def convert_df_digits(df: pd.DataFrame, mode: str, skip_cols: list[str] | None = None) -> pd.DataFrame:
    """Return a copy of df with digit conversion applied to all string columns."""
    if mode == "arabic":
        return df  # canonical storage is already Eastern; only Western mode needs conversion
    df = df.copy()
    for col in df.columns:
        if skip_cols and col in skip_cols:
            continue
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].apply(lambda v: convert_digits(str(v), mode) if pd.notna(v) else v)
    return df


# ── Ditto mark expansion ─────────────────────────────────────

DITTO_CHARS = {'"', '״', '〃', '\u201c', '\u201d', '\u2033'}


def expand_dittos(rows: list[dict], cols: list[str]) -> list[dict]:
    """Return a new list of dicts with ditto marks replaced by the value from the row above."""
    prev: dict[str, str] = {c: "" for c in cols}
    result = []
    for row in rows:
        new_row = dict(row)
        for c in cols:
            val = str(row.get(c, "") or "").strip()
            if val in DITTO_CHARS:
                new_row[c] = prev[c]
            else:
                if val:
                    prev[c] = val
        result.append(new_row)
    return result


def expand_dittos_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of df with ditto marks in the given columns replaced from the row above."""
    df = df.copy()
    prev: dict[str, str] = {}
    for i in range(len(df)):
        for c in cols:
            if c not in df.columns:
                continue
            val = str(df.at[i, c]).strip() if pd.notna(df.at[i, c]) else ""
            if val in DITTO_CHARS:
                df.at[i, c] = prev.get(c, "")
            else:
                if val:
                    prev[c] = val
    return df


def _expand_date_val(val: str) -> str:
    """Prepend '١' to a 3-digit Eastern-Arabic year, or '1' to a 3-digit Western year."""
    v = val.strip()
    if len(v) == 3 and all('\u0660' <= c <= '\u0669' for c in v):
        return '\u0661' + v   # Eastern Arabic 1 + e.g. ٩٣٨ → ١٩٣٨
    if len(v) == 3 and v.isdigit():
        return '1' + v        # Western digits: 938 → 1938
    return val


def expand_dates(rows: list[dict]) -> list[dict]:
    """Return a new list of dicts with 3-digit Date values expanded to 4 digits."""
    return [{**r, "Date": _expand_date_val(str(r.get("Date", "") or ""))} for r in rows]


def expand_dates_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with 3-digit Date values expanded to 4 digits."""
    if "Date" not in df.columns:
        return df
    df = df.copy()
    df["Date"] = df["Date"].apply(lambda v: _expand_date_val(str(v) if pd.notna(v) else ""))
    return df


# ── Helpers ──────────────────────────────────────────────────

def page_image_path(page_num: int) -> Path:
    return PROJECT_DIR / IMAGE_PATTERN.format(page_num)


def load_approach_m(page_num: int) -> list[dict]:
    """Load cached Approach M (Gemini structured OCR) results."""
    cache_file = CACHE_DIR / f"M_page{page_num}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return []


def load_kraken(page_num: int) -> list[dict]:
    """Load cached Kraken two-model OCR results.

    Generated by: python kraken_experiment.py --save-cache
    Cache file:   .ocr_cache/kraken_page{N}.json
    """
    cache_file = CACHE_DIR / f"kraken_page{page_num}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return []


OCR_SOURCES = {
    "Approach M (Gemini)": load_approach_m,
    "Kraken (two-model)":  load_kraken,
}


# ── Deskew pipeline ──────────────────────────────────────────

def deskew_page(page_num: int) -> np.ndarray:
    """Streamlit-aware wrapper: load bundled/cached deskewed image, or compute via
    image_preprocess.deskew_image and cache to .ocr_cache/deskewed_page{N}.png."""
    from image_preprocess import deskew_image

    bundled   = PROJECT_DIR / "images" / f"deskewed_page{page_num}.jpg"
    cache_png = CACHE_DIR / f"deskewed_page{page_num}.png"
    cache_jpg = CACHE_DIR / f"deskewed_page{page_num}.jpg"
    if bundled.exists():
        return cv2.imread(str(bundled))
    if cache_png.exists():
        return cv2.imread(str(cache_png))
    if cache_jpg.exists():
        return cv2.imread(str(cache_jpg))

    img = cv2.imread(str(page_image_path(page_num)))
    deskewed = deskew_image(img)
    CACHE_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(cache_png), deskewed)
    return deskewed


# ── Grid detection ───────────────────────────────────────────

def _find_peaks_simple(arr, min_gap=60):
    threshold = arr.mean() * 1.3
    above = arr > threshold
    positions = []
    in_peak = False
    peak_start = 0
    for i, val in enumerate(above):
        if val and not in_peak:
            in_peak = True
            peak_start = i
        elif not val and in_peak:
            mid = (peak_start + i) // 2
            if not positions or mid - positions[-1] >= min_gap:
                positions.append(mid)
            in_peak = False
    if in_peak:
        mid = (peak_start + len(arr)) // 2
        if not positions or mid - positions[-1] >= min_gap:
            positions.append(mid)
    return positions


def _trim_vlines(v_lines: list[int], expected_cols: int, table_w: int) -> list[int]:
    """Trim over-detected vertical lines to the expected number of columns.

    Strategy: keep lines that are most evenly spread across the table width,
    always preserving leftmost and rightmost boundaries.
    """
    want = expected_cols + 1  # number of boundary lines needed
    if len(v_lines) <= want:
        return v_lines
    # Always keep first and last
    inner = v_lines[1:-1]
    # Score each inner line by how close it is to an ideal uniform position
    ideal_step = table_w / expected_cols
    keep_count = want - 2  # how many inner lines to keep
    if keep_count <= 0:
        return [v_lines[0], v_lines[-1]]
    # Pick inner lines that best match ideal grid positions
    ideal_positions = [ideal_step * i for i in range(1, expected_cols)]
    chosen = []
    for ideal in ideal_positions:
        if not inner:
            break
        best = min(inner, key=lambda x: abs(x - ideal))
        chosen.append(best)
        inner = [x for x in inner if x != best]
    result = sorted(set([v_lines[0]] + chosen + [v_lines[-1]]))
    return result


def _binarize_for_lines(gray: np.ndarray) -> np.ndarray:
    """CLAHE + adaptive threshold → inverted binary (lines = white)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 41, 5)
    return binary


def detect_grid_from_image(img_cv: np.ndarray, side: str = "left",
                           h_lines_override: list[int] | None = None):
    """Detect table grid lines using morphological opening on a deskewed image.

    Pipeline: CLAHE → Sauvola threshold → morphological open with directional
    kernels → projection histogram → peak detection.

    Returns (h_lines, v_lines, table_img) where coordinates are
    relative to table_img (i.e. after header crop).

    If h_lines_override is given, skip horizontal detection (right table
    shares row boundaries with the left).
    """
    try:
        from scipy.signal import find_peaks
        _have_scipy = True
    except ImportError:
        _have_scipy = False

    H, W = img_cv.shape[:2]

    if side == "left":
        x0, x1 = 0, int(W * LEFT_TABLE_WIDTH_FRAC)
    else:
        x0, x1 = int(W * LEFT_TABLE_WIDTH_FRAC), W

    table_img = img_cv[int(H * HEADER_HEIGHT_FRAC):, x0:x1]
    th, tw = table_img.shape[:2]
    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)

    # ── Binarize: CLAHE + Sauvola ──
    binary = _binarize_for_lines(gray)

    # ── Horizontal lines (or use override) ──
    if h_lines_override is not None:
        h_lines = h_lines_override
    else:
        # Morphological open with horizontal kernel removes handwriting,
        # keeps printed rulings. kernel_w=15 tuned on pages 3,10,50.
        h_kernel_w = 15
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
        h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

        # Dilate slightly to fill gaps in broken lines
        h_mask = cv2.dilate(h_mask, np.ones((2, 1), np.uint8), iterations=1)

        # Projection histogram along rows
        proj_y = np.sum(h_mask, axis=1).astype(float)

        if _have_scipy:
            peaks_h, _ = find_peaks(proj_y,
                                    height=proj_y.mean() + 0.5 * proj_y.std(),
                                    distance=60)
            raw_h = peaks_h.tolist()
        else:
            raw_h = _find_peaks_simple(proj_y, min_gap=60)

        # Skip the header separator (first ~5% of table height)
        min_data_y = int(th * 0.05)
        h_lines = [p for p in raw_h if p > min_data_y]

    # ── Vertical lines ──
    v_kernel_h = 15
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Dilate slightly to fill gaps
    v_mask = cv2.dilate(v_mask, np.ones((1, 2), np.uint8), iterations=1)

    proj_x = np.sum(v_mask, axis=0).astype(float)

    if _have_scipy:
        v_peaks, _ = find_peaks(proj_x,
                                height=proj_x.mean() + 0.5 * proj_x.std(),
                                distance=max(30, tw // 30))
        v_lines = v_peaks.tolist()
    else:
        v_lines = _find_peaks_simple(proj_x, min_gap=max(30, tw // 30))

    expected_cols = len(LEFT_COLS) if side == "left" else len(RIGHT_COLS)
    if len(v_lines) < 3:
        col_w = tw // expected_cols
        v_lines = [i * col_w for i in range(expected_cols + 1)]

    if not v_lines or v_lines[0] > 30:
        v_lines = [0] + v_lines
    if v_lines[-1] < tw - 30:
        v_lines.append(tw)

    # Shift column boundaries right to avoid clipping text
    COL_SHIFT = 15
    v_lines = [min(v + COL_SHIFT, tw) for v in v_lines]
    v_lines = sorted(set(v_lines))

    # Trim to expected column count if over-detected
    v_lines = _trim_vlines(v_lines, expected_cols, tw)

    return sorted(set(h_lines)), v_lines, table_img


def draw_grid_overlay_deskewed(img_cv: np.ndarray,
                                h_lines: list[int], v_lines_left: list[int],
                                h_lines_right: list[int],
                                v_lines_right: list[int]) -> Image.Image:
    """Draw grid lines on a deskewed image and return a PIL Image."""
    H, W = img_cv.shape[:2]
    out = img_cv.copy()
    header_y = int(H * HEADER_HEIGHT_FRAC)
    left_x1 = int(W * LEFT_TABLE_WIDTH_FRAC)

    # Horizontal lines (red) — drawn full width
    for y in h_lines:
        abs_y = y + header_y
        if 0 <= abs_y < H:
            cv2.line(out, (0, abs_y), (W, abs_y), (0, 0, 220), 2)

    # Left table vertical lines (blue)
    for v in v_lines_left:
        if 0 <= v < W:
            cv2.line(out, (v, header_y), (v, H), (220, 100, 0), 1)

    # Right table vertical lines (green)
    for v in v_lines_right:
        x = left_x1 + v
        if 0 <= x < W:
            cv2.line(out, (x, header_y), (x, H), (0, 180, 0), 1)

    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


# ── Cell crop helpers ────────────────────────────────────────

def crop_column_strip(table_img, v_lines, col_idx):
    """Crop a single column as a PIL image."""
    if col_idx >= len(v_lines) - 1:
        return None
    x0, x1 = v_lines[col_idx], v_lines[col_idx + 1]
    col_cv = table_img[:, x0:x1]
    return Image.fromarray(cv2.cvtColor(col_cv, cv2.COLOR_BGR2RGB))


def crop_row_strip(table_img, h_lines, row_idx):
    """Crop a single row as a PIL image."""
    if row_idx >= len(h_lines) - 1:
        return None
    y0, y1 = h_lines[row_idx], h_lines[row_idx + 1]
    row_cv = table_img[y0:y1, :]
    return Image.fromarray(cv2.cvtColor(row_cv, cv2.COLOR_BGR2RGB))


def crop_cell_base64(tbl_img, h_lines: list[int], v_lines: list[int],
                     row_idx: int, col_idx: int, pad: int = 5) -> str:
    """Crop one cell and return a base64 PNG data URI.

    Adds `pad` pixels on each side to capture letters that extend past borders.
    """
    if row_idx >= len(h_lines) - 1 or col_idx >= len(v_lines) - 1:
        return ""
    th, tw = tbl_img.shape[:2]
    y0 = max(0, h_lines[row_idx] - pad)
    y1 = min(th, h_lines[row_idx + 1] + pad)
    x0 = max(0, v_lines[col_idx] - pad)
    x1 = min(tw, v_lines[col_idx + 1] + pad)
    cell = tbl_img[y0:y1, x0:x1]
    if cell.size == 0:
        return ""
    ok, buf = cv2.imencode(".png", cell)
    if not ok:
        return ""
    import base64
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/png;base64,{b64}"


# ── Ground truth I/O ─────────────────────────────────────────

def load_existing_gt() -> list[dict]:
    """Load existing ground_truth.tsv."""
    if not GROUND_TRUTH_FILE.exists():
        return []
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save_ground_truth(all_gt_rows: list[dict]):
    """Write all GT rows back to ground_truth.tsv."""
    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8", newline="\r\n") as f:
        writer = csv.DictWriter(f, fieldnames=GT_COLS, delimiter="\t",
                                extrasaction="ignore", lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(all_gt_rows)


def load_page_metadata() -> dict[int, dict]:
    """Load page_metadata.tsv → {page_num: {field: value}}.

    For any page not present in the TSV, falls back to the Gemini cache file
    (.ocr_cache/meta_pageN.json) if it exists.
    """
    result = {}
    if PAGE_METADATA_FILE.exists():
        with open(PAGE_METADATA_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                try:
                    p = int(row["Page_Number"])
                except (KeyError, ValueError):
                    continue
                result[p] = {k: row.get(k, "") for k in META_FIELDS}

    # Seed any missing pages from Gemini cache files
    for p in PAGES:
        if p not in result:
            cache_file = CACHE_DIR / f"meta_page{p}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    result[p] = {k: data.get(k, "") for k in META_FIELDS}
                except Exception:
                    pass

    return result


def save_page_metadata(meta: dict[int, dict]):
    """Write {page_num: {field: value}} to page_metadata.tsv."""
    fieldnames = ["Page_Number", "Folio_Number"] + META_FIELDS
    with open(PAGE_METADATA_FILE, "w", encoding="utf-8", newline="\r\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore", lineterminator="\r\n")
        writer.writeheader()
        for p in sorted(meta):
            row = {"Page_Number": str(p), "Folio_Number": PAGE_FOLIO.get(p, "")}
            row.update(meta[p])
            writer.writerow(row)


# ── GitHub API save ──────────────────────────────────────────

def _github_put(content_str: str, path: str, commit_message: str) -> tuple[bool, str]:
    """PUT a file to GitHub via Contents API. Returns (ok, error_msg)."""
    import base64
    import json as _json
    import urllib.request
    import urllib.error

    try:
        token = st.secrets["GITHUB_TOKEN"]
    except Exception:
        return False, "GITHUB_TOKEN not found in Streamlit secrets."

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }

    # Fetch current SHA (required for updates; None for new files)
    sha = None
    req_get = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req_get) as resp:
            sha = _json.loads(resp.read())["sha"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return False, f"GitHub API error reading file: HTTP {e.code}"

    payload: dict = {
        "message": commit_message,
        "content": base64.b64encode(content_str.encode("utf-8")).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha

    req_put = urllib.request.Request(
        api_url,
        data=_json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req_put):
            return True, ""
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return False, f"GitHub API error saving file: HTTP {e.code} — {body[:300]}"


def _github_create_issue(title: str, body: str) -> tuple[bool, str]:
    """Create a GitHub issue. Returns (ok, error_msg)."""
    import json as _json
    import urllib.request
    import urllib.error

    try:
        token = st.secrets["GITHUB_TOKEN"]
    except Exception:
        return False, "GITHUB_TOKEN not found in Streamlit secrets."

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }
    payload = {"title": title, "body": body, "labels": ["RA report"]}
    req = urllib.request.Request(
        api_url,
        data=_json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req):
            return True, ""
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace")
        return False, f"GitHub API error creating issue: HTTP {e.code} — {body_txt[:300]}"


def _gt_tsv_string(rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=GT_COLS, delimiter="\t",
                            extrasaction="ignore", lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _meta_tsv_string(meta: dict[int, dict]) -> str:
    fieldnames = ["Page_Number", "Folio_Number"] + META_FIELDS
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, delimiter="\t",
                            extrasaction="ignore", lineterminator="\r\n")
    writer.writeheader()
    for p in sorted(meta):
        row = {"Page_Number": str(p), "Folio_Number": PAGE_FOLIO.get(p, "")}
        row.update(meta[p])
        writer.writerow(row)
    return buf.getvalue()


# ── PAGE XML export ──────────────────────────────────────────

def extract_header_metadata(page_num: int, force: bool = False) -> dict:
    """Use Gemini to extract tax payer name and ID from the page header image.

    Results are cached in .ocr_cache/meta_page{N}.json.
    Pass force=True to bypass cache and re-extract.
    """
    import io as _io
    import os
    import re

    cache_file = CACHE_DIR / f"meta_page{page_num}.json"
    if not force and cache_file.exists():
        data = json.loads(cache_file.read_text())
        # Return only if it has the expected keys
        if all(k in data for k in META_FIELDS):
            return data

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Crop the header strip (top 12% of deskewed image)
    deskewed = deskew_page(page_num)
    h = deskewed.shape[0]
    header_cv = deskewed[:int(h * 0.12), :]
    pil_header = Image.fromarray(cv2.cvtColor(header_cv, cv2.COLOR_BGR2RGB))

    prompt = """\
This is the header of a British Mandate Palestine property tax register page (Form TR/39).
The printed label "Tax-Payer" is followed by handwritten Arabic text giving the taxpayer's name,
and a handwritten number or code serving as the taxpayer's identifier.
The name may have 3–4 components. Transcribe every component exactly as written — do not
skip or merge any part of the name.

Extract exactly these four fields and return ONLY valid JSON, no markdown:
{
  "Tax_Payer_Arabic": "<taxpayer name in Arabic script as written>",
  "Tax_Payer_Romanized": "<taxpayer name romanized / transliterated to Latin>",
  "Tax_Payer_ID_Arabic": "<identifier number/code in Arabic-Indic or Eastern Arabic digits as written, or empty string if absent>",
  "Tax_Payer_ID_Romanized": "<identifier romanized to Latin digits/characters, or empty string if absent>"
}
If a field is not legible, return an empty string for it."""

    buf = _io.BytesIO()
    pil_header.save(buf, format="JPEG", quality=92)
    parts = [
        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
        types.Part.from_text(text=prompt),
    ]

    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=parts,
        config=types.GenerateContentConfig(max_output_tokens=8192),
    )
    raw = resp.text or ""
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        return {k: "" for k in META_FIELDS}

    result = json.loads(m.group())
    for k in META_FIELDS:
        result.setdefault(k, "")
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def export_page_xml(page_num: int, img_cv: np.ndarray,
                    h_left: list[int], v_left: list[int],
                    h_right: list[int], v_right: list[int],
                    ocr_rows: list[dict]) -> Path:
    """Export detected grid + OCR text as Transkribus-compatible PAGE XML.

    Returns the path to the saved XML file.
    """
    import xml.etree.ElementTree as ET
    from datetime import datetime

    H, W = img_cv.shape[:2]
    header_y = int(H * HEADER_HEIGHT_FRAC)
    left_x1 = int(W * LEFT_TABLE_WIDTH_FRAC)

    NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

    # Register namespace so ElementTree doesn't add ns0: prefixes
    ET.register_namespace("", NS)

    root = ET.Element(f"{{{NS}}}PcGts")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:schemaLocation",
             f"{NS} {NS}/pagecontent.xsd")

    meta = ET.SubElement(root, f"{{{NS}}}Metadata")
    ET.SubElement(meta, f"{{{NS}}}Creator").text = "Haditax"
    now = datetime.now().isoformat()
    ET.SubElement(meta, f"{{{NS}}}Created").text = now
    ET.SubElement(meta, f"{{{NS}}}LastChange").text = now

    img_filename = f"deskewed_page{page_num}.jpg"
    page_el = ET.SubElement(root, f"{{{NS}}}Page",
                            imageFilename=img_filename,
                            imageWidth=str(W), imageHeight=str(H))

    def coords_str(x0, y0, x1, y1):
        """Rectangle as 4-point polygon string."""
        return f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"

    cell_id = 0

    def add_table(table_el_parent, h_lines, v_lines, x_offset, cols_list, table_id):
        nonlocal cell_id
        if len(h_lines) < 2 or len(v_lines) < 2:
            return
        n_rows = len(h_lines) - 1
        n_cols = len(v_lines) - 1
        ty0 = h_lines[0] + header_y
        ty1 = h_lines[-1] + header_y
        tx0 = x_offset + v_lines[0]
        tx1 = x_offset + v_lines[-1]

        tbl = ET.SubElement(table_el_parent, f"{{{NS}}}TableRegion",
                            id=table_id,
                            rows=str(n_rows), columns=str(n_cols))
        ET.SubElement(tbl, f"{{{NS}}}Coords",
                      points=coords_str(tx0, ty0, tx1, ty1))

        for r in range(n_rows):
            for c in range(n_cols):
                cell_id += 1
                y0 = h_lines[r] + header_y
                y1 = h_lines[r + 1] + header_y
                x0 = x_offset + v_lines[c]
                x1 = x_offset + v_lines[c + 1]

                tr = ET.SubElement(tbl, f"{{{NS}}}TextRegion",
                                   id=f"cell_{cell_id}",
                                   type="paragraph")
                tr.set("custom",
                       f"readingOrder {{index:{cell_id};}} "
                       f"structure {{type:cell; row:{r}; col:{c};}}")
                ET.SubElement(tr, f"{{{NS}}}Coords",
                              points=coords_str(x0, y0, x1, y1))

                col_name = cols_list[c] if c < len(cols_list) else ""
                text = (ocr_rows[r].get(col_name, "") or "") if r < len(ocr_rows) and col_name else ""
                if text:
                    tl = ET.SubElement(tr, f"{{{NS}}}TextLine",
                                       id=f"line_{cell_id}")
                    ET.SubElement(tl, f"{{{NS}}}Coords",
                                  points=coords_str(x0 + 2, y0 + 2, x1 - 2, y1 - 2))
                    te = ET.SubElement(tl, f"{{{NS}}}TextEquiv")
                    ET.SubElement(te, f"{{{NS}}}Unicode").text = text

    add_table(page_el, h_left, v_left, 0, LEFT_COLS, "table_left")
    add_table(page_el, h_right, v_right, left_x1, RIGHT_COLS, "table_right")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    out_path = PROJECT_DIR / f"page{page_num}.xml"
    tree.write(str(out_path), encoding="unicode", xml_declaration=True)
    return out_path


# ── Correction-table fragment ────────────────────────────────
# Isolates data_editor reruns so cell commits don't scroll the
# page to the top or reset other widget state.

@st.fragment
def _render_table_editor(page_num: int, digit_mode: str,
                         expand_ditto_view: bool, display_h: int) -> None:
    all_rows  = st.session_state.get("cv_all_rows", [])
    page_meta = st.session_state.get("page_meta", {})

    st.caption("Click a cell to edit · Tab / Enter to navigate")

    page_indices = [i for i, r in enumerate(all_rows) if r["_page"] == page_num]
    page_rows    = [all_rows[i] for i in page_indices]

    df = pd.DataFrame(page_rows) if page_rows else pd.DataFrame(columns=["_page"] + ALL_COLS)
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[["_page"] + ALL_COLS].fillna("")
    df.insert(0, "#", range(1, len(df) + 1))
    for c in ["_page"] + ALL_COLS:
        df[c] = df[c].astype(str)

    display_df = convert_df_digits(df, digit_mode, skip_cols=["#", "_page"])
    if expand_ditto_view:
        display_df = expand_dittos_df(display_df, ALL_COLS)
        display_df = expand_dates_df(display_df)
        meta_now = page_meta.get(page_num, {})
        for mf in META_FIELDS:
            display_df.insert(2 + META_FIELDS.index(mf), mf, meta_now.get(mf, ""))

    col_config: dict = {
        "#":     st.column_config.NumberColumn("#",    width="small", disabled=True),
        "_page": st.column_config.TextColumn("Page",  width="small", disabled=True),
    }
    if expand_ditto_view:
        for mf in META_FIELDS:
            col_config[mf] = st.column_config.TextColumn(mf, width="medium", disabled=True)
    for c in ALL_COLS:
        col_config[c] = st.column_config.TextColumn(c, width="small")

    disabled_cols = ["#", "_page"] + (META_FIELDS if expand_ditto_view else [])
    edited_df = st.data_editor(
        display_df,
        column_config=col_config,
        use_container_width=True,
        num_rows="fixed",
        height=display_h,
        key=f"cv_editor_{page_num}_{digit_mode}_{expand_ditto_view}",
        disabled=disabled_cols,
    )

    # Sync edits back to shared session state (always store as Arabic digits).
    # Only write a cell when the user actually changed it — preserves ditto marks
    # when expand_ditto_view is on and the cell was not touched.
    for j, orig_idx in enumerate(page_indices):
        for col in ALL_COLS:
            if col not in edited_df.columns:
                continue
            displayed_val = str(display_df.at[j, col]) if j < len(display_df) else ""
            raw_edited    = edited_df.at[j, col]
            edited_val    = str(raw_edited) if pd.notna(raw_edited) else ""
            if edited_val != displayed_val:
                all_rows[orig_idx][col] = convert_digits(edited_val, "arabic")


# ═══════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING WIZARD
# ═══════════════════════════════════════════════════════════════

def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _pil_to_b64_jpeg(pil_img: Image.Image, max_w: int = 700) -> tuple[Image.Image, str]:
    """Resize to max_w, return (resized_pil, data-URI string)."""
    import base64
    w, h = pil_img.size
    if w > max_w:
        pil_img = pil_img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return pil_img, f"data:image/jpeg;base64,{b64}"


def _display_image_uri(data_uri: str, caption: str = "") -> None:
    st.markdown(
        f'<img src="{data_uri}" style="width:100%;border:1px solid #ccc;border-radius:4px">',
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def _stepper(current: str) -> None:
    stages = [("A_table", "Stage A — Table boundary"), ("B_header", "Header recognition"), ("done", "Done")]
    parts = []
    for key, label in stages:
        if key == current:
            parts.append(f"**→ {label}**")
        elif stages.index((key, label)) < [s[0] for s in stages].index(current):
            parts.append(f"~~{label}~~")
        else:
            parts.append(label)
    st.markdown("  |  ".join(parts))


def render_preprocess_view(page_num: int) -> None:
    """Image Preprocessing wizard — Stage A (table corners) + Stage B (header columns)."""
    DISPLAY_W = 700

    nb_cfg = load_notebook_config(NOTEBOOK_CONFIG_FILE)

    stage_key = f"prep_stage_{page_num}"
    if stage_key not in st.session_state:
        st.session_state[stage_key] = "A_table"
    stage = st.session_state[stage_key]

    _stepper(stage)
    st.divider()

    with st.spinner("Loading page image…"):
        _orig_path = page_image_path(page_num)
        if _orig_path.exists():
            page_img = cv2.imread(str(_orig_path))
            st.caption("Showing original (un-deskewed) image.")
        else:
            page_img = deskew_page(page_num)
            st.caption("Original image not found locally — showing deskewed version.")
    deskewed = page_img
    orig_h, orig_w = deskewed.shape[:2]
    scale = DISPLAY_W / orig_w  # display px → image px: divide by scale

    # ── Stage A — Table-area confirmation ────────────────────────
    if stage == "A_table":
        st.subheader("Stage A — Table boundary")
        st.caption("Drag the sliders to position each corner of the data table.")

        page_key = str(page_num)
        saved = nb_cfg.get("table_corners", {}).get(page_key)

        # Seed slider keys on first render (slider keys are the source of truth)
        _init = (
            [list(c) for c in saved] if saved
            else auto_table_corners(deskewed, HEADER_HEIGHT_FRAC, LEFT_TABLE_WIDTH_FRAC)
        )
        corner_labels = ["TL — Top Left", "TR — Top Right", "BR — Bottom Right", "BL — Bottom Left"]
        for i in range(4):
            xk, yk = f"prep_cx_{page_num}_{i}", f"prep_cy_{page_num}_{i}"
            if xk not in st.session_state:
                st.session_state[xk] = int(_init[i][0])
            if yk not in st.session_state:
                st.session_state[yk] = int(_init[i][1])

        def _slider_corners() -> list[list[int]]:
            return [
                [st.session_state[f"prep_cx_{page_num}_{i}"],
                 st.session_state[f"prep_cy_{page_num}_{i}"]]
                for i in range(4)
            ]

        # Image preview (always reflects current slider state)
        overlay = corners_to_overlay(deskewed, _slider_corners())
        _, data_uri = _pil_to_b64_jpeg(_bgr_to_pil(overlay), DISPLAY_W)
        _display_image_uri(data_uri, f"Page {page_num} — blue quad = table boundary")

        # Four-column slider grid — one column per corner
        cols = st.columns(4)
        for i, (label, col) in enumerate(zip(corner_labels, cols)):
            with col:
                st.markdown(f"**{label}**")
                st.slider("X (px)", 0, orig_w, key=f"prep_cx_{page_num}_{i}", step=5)
                st.slider("Y (px)", 0, orig_h, key=f"prep_cy_{page_num}_{i}", step=5)

        btn_col1, btn_col2, _ = st.columns([1, 1, 3])
        with btn_col1:
            if st.button("Reset to auto", key=f"prep_reset_{page_num}"):
                fresh = auto_table_corners(deskewed, HEADER_HEIGHT_FRAC, LEFT_TABLE_WIDTH_FRAC)
                for i in range(4):
                    st.session_state[f"prep_cx_{page_num}_{i}"] = int(fresh[i][0])
                    st.session_state[f"prep_cy_{page_num}_{i}"] = int(fresh[i][1])
                st.rerun()
        with btn_col2:
            if st.button("Save & Continue →", type="primary", key=f"prep_save_a_{page_num}"):
                nb_cfg.setdefault("table_corners", {})[page_key] = _slider_corners()
                save_notebook_config(NOTEBOOK_CONFIG_FILE, nb_cfg)
                st.session_state[stage_key] = "B_header"
                st.rerun()

    # ── Stage B — Header recognition ─────────────────────────────
    elif stage == "B_header":
        st.subheader("Stage B — Fields & column names")
        st.caption(
            "Two separate areas are sent to Gemini: the metadata band above the table "
            "(page-level fields like taxpayer, ID, etc.) and the table's own column-header rows. "
            "Edit both lists before saving."
        )

        if st.button("← Back to Stage A", key=f"prep_back_{page_num}"):
            st.session_state[stage_key] = "A_table"
            st.rerun()

        page_key = str(page_num)
        corners = nb_cfg.get("table_corners", {}).get(page_key) or auto_table_corners(
            deskewed, HEADER_HEIGHT_FRAC, LEFT_TABLE_WIDTH_FRAC
        )

        above_crop  = crop_above_table(deskewed, corners)
        header_crop = crop_header_strip(deskewed, corners)

        meta_key     = f"prep_meta_fields_{page_num}"
        col_names_key = f"prep_col_names_{page_num}"
        source_key    = f"prep_schema_source_{page_num}"
        if meta_key not in st.session_state:
            st.session_state[meta_key] = list(nb_cfg.get("metadata_fields", []))
        if col_names_key not in st.session_state:
            st.session_state[col_names_key] = list(nb_cfg.get("column_names", []))
        if source_key not in st.session_state:
            st.session_state[source_key] = "saved"

        # ── Single Gemini button ──────────────────────────────────
        btn_col, src_col = st.columns([2, 3])
        with btn_col:
            if st.button("Ask Gemini", key=f"prep_gemini_{page_num}", type="primary"):
                with st.spinner("Reading both image strips…"):
                    try:
                        meta, cols = gemini_extract_schema(above_crop, header_crop)
                        st.session_state[meta_key]     = meta
                        st.session_state[col_names_key] = cols
                        st.session_state[source_key]   = "gemini"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Gemini call failed: {exc}")
        with src_col:
            if st.session_state[source_key] == "gemini":
                st.caption("🔄 Values from Gemini — edit as needed, then Save.")
            elif nb_cfg.get("metadata_fields") or nb_cfg.get("column_names"):
                st.caption("Loaded from saved config. Click Ask Gemini to refresh.")
            else:
                st.caption("No saved config yet. Click Ask Gemini or add manually.")

        st.divider()

        # ── Section 1: page metadata fields ──────────────────────
        st.markdown("#### Page metadata fields (band above the table)")
        mc_img, mc_ed = st.columns([1, 1])
        with mc_img:
            _, above_uri = _pil_to_b64_jpeg(_bgr_to_pil(above_crop), DISPLAY_W)
            _display_image_uri(above_uri, "Metadata band above the table")
        with mc_ed:
            meta_fields: list[str] = st.session_state[meta_key]
            if meta_fields:
                st.markdown(f"**{len(meta_fields)} metadata fields** — edit as needed:")
                updated_meta = []
                for i, name in enumerate(meta_fields):
                    updated_meta.append(
                        st.text_input(f"M{i+1}.", value=name, key=f"prep_mf_{page_num}_{i}")
                    )
                ma_add, ma_del, _ = st.columns([1, 1, 4])
                with ma_add:
                    if st.button("+ Add field", key=f"prep_madd_{page_num}"):
                        st.session_state[meta_key] = updated_meta + [""]
                        st.rerun()
                with ma_del:
                    if st.button("− Remove last", key=f"prep_mdel_{page_num}", disabled=not updated_meta):
                        st.session_state[meta_key] = updated_meta[:-1]
                        st.rerun()
                meta_fields = updated_meta
            else:
                st.info("Click **Ask Gemini** above or add manually.")
                if st.button("+ Add field manually", key=f"prep_madd_manual_{page_num}"):
                    st.session_state[meta_key] = [""]
                    st.rerun()

        st.divider()

        # ── Section 2: column names ───────────────────────────────
        st.markdown("#### Table column names (header rows inside the table)")
        cc_img, cc_ed = st.columns([1, 1])
        with cc_img:
            _, header_uri = _pil_to_b64_jpeg(_bgr_to_pil(header_crop), DISPLAY_W)
            _display_image_uri(header_uri, "Column header strip (top of the table)")
        with cc_ed:
            col_names: list[str] = st.session_state[col_names_key]
            if col_names:
                st.markdown(f"**{len(col_names)} columns detected** — edit as needed:")
                updated_cols = []
                for i, name in enumerate(col_names):
                    updated_cols.append(
                        st.text_input(f"{i+1}.", value=name, key=f"prep_cn_{page_num}_{i}")
                    )
                ca_add, ca_del, _ = st.columns([1, 1, 4])
                with ca_add:
                    if st.button("+ Add column", key=f"prep_add_{page_num}"):
                        st.session_state[col_names_key] = updated_cols + [""]
                        st.rerun()
                with ca_del:
                    if st.button("− Remove last", key=f"prep_del_{page_num}", disabled=not updated_cols):
                        st.session_state[col_names_key] = updated_cols[:-1]
                        st.rerun()
                col_names = updated_cols
            else:
                st.info("Click **Ask Gemini** above or add manually.")
                if st.button("+ Add column manually", key=f"prep_add_manual_{page_num}"):
                    st.session_state[col_names_key] = [""]
                    st.rerun()

        st.divider()

        if col_names:
            if st.button("Save & Confirm ✓", type="primary", key=f"prep_save_b_{page_num}"):
                nb_cfg["metadata_fields"] = meta_fields
                nb_cfg["column_names"]    = col_names
                nb_cfg["expected_n_cols"] = len(col_names)
                save_notebook_config(NOTEBOOK_CONFIG_FILE, nb_cfg)
                st.session_state[meta_key]      = meta_fields
                st.session_state[col_names_key] = col_names
                st.session_state[source_key]    = "saved"
                st.session_state[stage_key]     = "done"
                st.rerun()

    # ── Done ──────────────────────────────────────────────────────
    elif stage == "done":
        nb_cfg = load_notebook_config(NOTEBOOK_CONFIG_FILE)
        n_cols = len(nb_cfg.get("column_names", []))
        st.success(
            f"Notebook config saved — {n_cols} columns confirmed for page {page_num}. "
            "Corners and column names are stored in `notebook_config.json`."
        )
        st.json({
            "column_names": nb_cfg.get("column_names", []),
            "table_corners_page": nb_cfg.get("table_corners", {}).get(str(page_num), []),
        })
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Edit again (Stage A)", key=f"prep_restart_{page_num}"):
                st.session_state[stage_key] = "A_table"
                st.rerun()


# ── Streamlit App ────────────────────────────────────────────

st.set_page_config(page_title="Haditax", layout="wide")

# ── User registration gate ────────────────────────────────────
try:
    _REVIEWERS = st.secrets.get("reviewers", ["Sinai"])
except Exception:
    _REVIEWERS = ["Sinai"]

if "reviewer" not in st.session_state:
    st.title("Haditax · Who are you?")
    choice = st.selectbox("Select your name to continue:", ["— pick one —"] + list(_REVIEWERS))
    if st.button("Continue", type="primary", disabled=(choice == "— pick one —")):
        st.session_state["reviewer"] = choice
        st.rerun()
    st.stop()

with st.sidebar:
    st.markdown(f"Logged in as **{st.session_state['reviewer']}**")
    if st.button("Switch user"):
        del st.session_state["reviewer"]
        st.rerun()
    st.divider()
    view_mode = st.radio(
        "View",
        ["Image Preprocessing", "Correction View"],
        key="view_mode",
    )
    st.divider()
    ocr_source = st.selectbox(
        "OCR starting point",
        list(OCR_SOURCES.keys()),
        help="Which OCR approach seeds pages that don't yet have verified ground truth.",
        key="ocr_source",
    )
    _load_ocr = OCR_SOURCES[ocr_source]

st.title("Haditax — Ground Truth Editor")

st.warning(
    "**Important for RAs:** Every time you click Save, your corrections are committed "
    "directly to GitHub. If the save button does not show a green ✅ success message — "
    "for any reason — **stop working and contact Sinai before continuing**. "
    "Corrections saved only on your computer will be permanently lost.",
)

# ── Shared page selector + view mode ─────────────────────────
ALL_COLS = LEFT_COLS
PAGES = list(PAGE_FOLIO.keys())  # [3, 10, 50]

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
with ctrl_col1:
    page_num = st.selectbox(
        "Page",
        PAGES,
        format_func=lambda p: f"Page {p}  (Folio {PAGE_FOLIO.get(p, '?')})",
        key="shared_page",
    )
with ctrl_col2:
    digit_mode = st.radio(
        "Digits",
        ["Arabic", "Western"],
        horizontal=True,
        key="digit_mode",
    ).lower()
with ctrl_col3:
    expand_ditto_view = st.checkbox(
        'Post-processing (ditto marks and metadata) in view',
        key="expand_ditto_view",
    )
# ═══════════════════════════════════════════════════════════════
# CORRECTION VIEW — all pages in one table, image synced to page
# ═══════════════════════════════════════════════════════════════
if view_mode == "Correction View":
    # Load all pages into combined session state
    if "cv_all_rows" not in st.session_state:
        # Pages that already have GT → load from ground_truth.tsv so RAs see verified data.
        # Pages without GT → fall back to Approach M OCR cache.
        gt_by_page: dict[int, list[dict]] = {}
        for row in load_existing_gt():
            try:
                p = int(row.get("Page_Number") or 0)
            except ValueError:
                continue
            if p in PAGES and row.get("Serial_No", "").strip():
                gt_by_page.setdefault(p, []).append(row)

        # Only use GT for a page when it has a substantial number of verified rows;
        # fewer rows means the data is stale/incomplete — fall back to Approach M OCR.
        MIN_GT_ROWS = 5
        gt_by_page = {p: rows for p, rows in gt_by_page.items() if len(rows) >= MIN_GT_ROWS}

        combined = []
        for p in PAGES:
            if p in gt_by_page:
                for r in gt_by_page[p]:
                    row = {c: r.get(c, "") for c in LEFT_COLS}
                    row["_page"] = p
                    combined.append(row)
            else:
                for r in _load_ocr(p):
                    row = dict(r)
                    row["_page"] = p
                    combined.append(row)
        st.session_state["cv_all_rows"] = combined

    if "page_meta" not in st.session_state:
        st.session_state["page_meta"] = load_page_metadata()
        # Ensure every known page has an entry
        for p in PAGES:
            st.session_state["page_meta"].setdefault(p, {f: "" for f in META_FIELDS})

    all_rows = st.session_state["cv_all_rows"]
    page_meta = st.session_state["page_meta"]

    # ── Page metadata section ────────────────────────────────────
    with st.expander(
        f"Page metadata — Folio {PAGE_FOLIO.get(page_num, '?')}  "
        f"(Tax payer: {page_meta[page_num].get('Tax_Payer_Arabic') or '—'})",
        expanded=not any(page_meta[page_num].values()),
    ):
        m = page_meta[page_num]
        btn_col, _ = st.columns([1, 3])
        with btn_col:
            autofill = st.button("Auto-fill from image (Gemini)",
                                 key=f"meta_autofill_{page_num}")
        if autofill:
            with st.spinner("Extracting header with Gemini…"):
                try:
                    extracted = extract_header_metadata(page_num, force=True)
                    for k, wk in [
                        ("Tax_Payer_Arabic",    f"meta_tpa_{page_num}"),
                        ("Tax_Payer_Romanized", f"meta_tpr_{page_num}"),
                        ("Tax_Payer_ID_Arabic",    f"meta_tpia_{page_num}"),
                        ("Tax_Payer_ID_Romanized", f"meta_tpir_{page_num}"),
                    ]:
                        val = extracted.get(k, "")
                        m[k] = val
                        st.session_state[wk] = val
                    st.success("Done — review and edit the fields below if needed.")
                except Exception as e:
                    st.error(f"Gemini extraction failed: {e}")

        mc1, mc2 = st.columns(2)
        with mc1:
            m["Tax_Payer_Arabic"] = st.text_input(
                "Tax Payer (Arabic)", value=m.get("Tax_Payer_Arabic", ""),
                key=f"meta_tpa_{page_num}")
            m["Tax_Payer_ID_Arabic"] = st.text_input(
                "Tax Payer ID (Arabic)", value=m.get("Tax_Payer_ID_Arabic", ""),
                key=f"meta_tpia_{page_num}")
        with mc2:
            m["Tax_Payer_Romanized"] = st.text_input(
                "Tax Payer (Romanized)", value=m.get("Tax_Payer_Romanized", ""),
                key=f"meta_tpr_{page_num}")
            m["Tax_Payer_ID_Romanized"] = st.text_input(
                "Tax Payer ID (Romanized)", value=m.get("Tax_Payer_ID_Romanized", ""),
                key=f"meta_tpir_{page_num}")

    # Load image and compute display height before splitting columns
    # so both panels can share the same height.
    with st.spinner("Loading page image..."):
        deskewed = deskew_page(page_num)
    pil_img = Image.fromarray(cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB))
    left_x = int(pil_img.width * (LEFT_TABLE_WIDTH_FRAC + 0.07))
    pil_img = pil_img.crop((0, 0, left_x, pil_img.height))
    DISPLAY_W = 700
    display_h = int(pil_img.height * DISPLAY_W / pil_img.width)

    col_img, col_tbl = st.columns([1, 1])

    with col_img:

        # ── Pan/zoom session state ───────────────────────────────
        if "img_zoom" not in st.session_state:
            st.session_state["img_zoom"] = 1.0
        if "img_pan_x" not in st.session_state:
            st.session_state["img_pan_x"] = 0.5  # fraction of image width (centre)
        if "img_pan_y" not in st.session_state:
            st.session_state["img_pan_y"] = 0.5  # fraction of image height (centre)

        zoom   = st.session_state["img_zoom"]
        pan_x  = st.session_state["img_pan_x"]
        pan_y  = st.session_state["img_pan_y"]

        # ── Navigation controls ──────────────────────────────────
        ZOOM_LEVELS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
        zoom_idx    = min(range(len(ZOOM_LEVELS)), key=lambda i: abs(ZOOM_LEVELS[i] - zoom))
        pan_step    = 0.15 / zoom  # smaller steps at higher zoom

        btn_z_out, btn_z_lbl, btn_z_in, _, btn_left, btn_up, btn_down, btn_right, _, btn_reset = st.columns(
            [1, 1, 1, 0.3, 1, 1, 1, 1, 0.3, 1]
        )
        with btn_z_out:
            if st.button("−", key="zoom_out", disabled=zoom_idx == 0):
                st.session_state["img_zoom"] = ZOOM_LEVELS[zoom_idx - 1]
                st.rerun()
        with btn_z_lbl:
            st.markdown(f"<div style='text-align:center;padding-top:6px'>{zoom:.1f}×</div>",
                        unsafe_allow_html=True)
        with btn_z_in:
            if st.button("+", key="zoom_in", disabled=zoom_idx == len(ZOOM_LEVELS) - 1):
                st.session_state["img_zoom"] = ZOOM_LEVELS[zoom_idx + 1]
                st.rerun()
        with btn_left:
            if st.button("←", key="pan_left"):
                st.session_state["img_pan_x"] = max(0.0, pan_x - pan_step)
                st.rerun()
        with btn_up:
            if st.button("↑", key="pan_up"):
                st.session_state["img_pan_y"] = max(0.0, pan_y - pan_step)
                st.rerun()
        with btn_down:
            if st.button("↓", key="pan_down"):
                st.session_state["img_pan_y"] = min(1.0, pan_y + pan_step)
                st.rerun()
        with btn_right:
            if st.button("→", key="pan_right"):
                st.session_state["img_pan_x"] = min(1.0, pan_x + pan_step)
                st.rerun()
        with btn_reset:
            if st.button("Reset", key="zoom_reset"):
                st.session_state["img_zoom"]  = 1.0
                st.session_state["img_pan_x"] = 0.5
                st.session_state["img_pan_y"] = 0.5
                st.rerun()

        # ── Compute crop and display ─────────────────────────────
        img_w, img_h = pil_img.size
        crop_w = img_w / zoom
        crop_h = img_h / zoom
        # Centre the crop on (pan_x, pan_y), clamped to image bounds
        x0 = int(max(0, min(pan_x * img_w - crop_w / 2, img_w - crop_w)))
        y0 = int(max(0, min(pan_y * img_h - crop_h / 2, img_h - crop_h)))
        x1 = int(x0 + crop_w)
        y1 = int(y0 + crop_h)

        crop = pil_img.crop((x0, y0, x1, y1)).resize(
            (DISPLAY_W, display_h), Image.LANCZOS
        )
        st.image(crop, width=DISPLAY_W)

    with col_tbl:
        _render_table_editor(page_num, digit_mode, expand_ditto_view, display_h)

# ═══════════════════════════════════════════════════════════════
# GRID VIEW — uses shared page_num from top selector (disabled)
# ═══════════════════════════════════════════════════════════════
elif view_mode == "Grid View":
    rows = _load_ocr(page_num)
    if not rows:
        st.error(
            f"No Approach M cache found for page {page_num}. "
            f"Run `compare_ocr.py --pages {page_num} --approaches M` first."
        )
        st.stop()

    edit_key = f"edit_rows_{page_num}"
    if edit_key not in st.session_state:
        st.session_state[edit_key] = [dict(r) for r in rows]
    edit_rows = st.session_state[edit_key]

    with st.spinner("Deskewing page..."):
        deskewed = deskew_page(page_num)

    h_left, v_left, tbl_left = detect_grid_from_image(deskewed, "left")
    _, v_right, tbl_right = detect_grid_from_image(deskewed, "right",
                                                   h_lines_override=h_left)
    h_right = h_left
    n_rows = max(len(h_left) - 1, len(rows))

    while len(edit_rows) < n_rows:
        edit_rows.append({c: "" for c in ALL_COLS})

    overlay = draw_grid_overlay_deskewed(deskewed, h_left, v_left, h_right, v_right)
    st.image(overlay, caption="Deskewed page with detected grid", use_container_width=True)

    img_cache_key = f"cell_imgs_{page_num}"
    if img_cache_key not in st.session_state:
        imgs = {}
        for row_idx in range(n_rows):
            for col_idx, col_name in enumerate(ALL_COLS):
                if col_idx < len(LEFT_COLS):
                    b64 = crop_cell_base64(tbl_left, h_left, v_left, row_idx, col_idx)
                else:
                    b64 = crop_cell_base64(tbl_right, h_right, v_right,
                                           row_idx, col_idx - len(LEFT_COLS))
                imgs[(row_idx, col_name)] = b64
        st.session_state[img_cache_key] = imgs

    cell_imgs = st.session_state[img_cache_key]

    df_data = []
    for row_idx in range(n_rows):
        row_dict = {"#": row_idx + 1}
        for col_name in ALL_COLS:
            row_dict[f"{col_name}_img"] = cell_imgs.get((row_idx, col_name), "")
            raw_val = edit_rows[row_idx].get(col_name, "") or ""
            row_dict[col_name] = convert_digits(raw_val, digit_mode)
        df_data.append(row_dict)

    df = pd.DataFrame(df_data)
    col_config = {"#": st.column_config.NumberColumn("#", width="small", disabled=True)}
    for col_name in ALL_COLS:
        col_config[f"{col_name}_img"] = st.column_config.ImageColumn(
            f"{col_name} (img)", width="small"
        )
        col_config[col_name] = st.column_config.TextColumn(col_name, width="small")

    display_cols = ["#"]
    for col_name in ALL_COLS:
        display_cols.append(f"{col_name}_img")
        display_cols.append(col_name)

    st.subheader("Correction Table")
    st.caption("Click a cell to edit. Tab / Enter to navigate.")

    edited_df = st.data_editor(
        df[display_cols],
        column_config=col_config,
        use_container_width=True,
        num_rows="fixed",
        height=700,
        key=f"editor_{page_num}",
    )

    for row_idx in range(n_rows):
        for col_name in ALL_COLS:
            new_val = edited_df.at[row_idx, col_name]
            edit_rows[row_idx][col_name] = new_val if pd.notna(new_val) else ""

    # Grid View buttons
    st.markdown("---")
    col_save, col_xml = st.columns(2)

    with col_save:
        if st.button("Save corrections to ground_truth.tsv", type="primary", key="gv_save"):
            existing = load_existing_gt()
            other_pages = [r for r in existing if r.get("Page_Number", "") != str(page_num)]
            folio = PAGE_FOLIO.get(page_num, "")
            new_gt_rows = []
            for row in edit_rows:
                gt_row = {c: "" for c in GT_COLS}
                gt_row["Page_Number"] = str(page_num)
                gt_row["Folio_Number"] = folio
                for col in LEFT_COLS + META_COLS:
                    if col in gt_row:
                        gt_row[col] = row.get(col, "")
                gt_row["OCR_Method"] = "ground_truth"
                new_gt_rows.append(gt_row)
            all_gt = other_pages + new_gt_rows
            all_gt.sort(key=lambda r: (int(r.get("Page_Number", 0) or 0),
                                        int(r.get("Serial_No", 0) or 0)))
            save_ground_truth(all_gt)
            st.success(f"Saved {len(new_gt_rows)} rows for page {page_num}.")

    with col_xml:
        if st.button("Export PAGE XML for Transkribus", key="gv_xml"):
            with st.spinner("Generating PAGE XML..."):
                xml_path = export_page_xml(
                    page_num, deskewed,
                    h_left, v_left, h_right, v_right, edit_rows,
                )
            st.success(f"Saved {xml_path.name}")
            st.code(str(xml_path))

# ── Correction View save + report buttons ────────────────────
if view_mode == "Correction View":
    from datetime import datetime, timezone

    st.markdown("---")
    save_btn_col, save_opt_col = st.columns([1, 2])
    with save_opt_col:
        expand_ditto_save = st.checkbox(
            'Expand ditto marks on save',
            key="expand_ditto_save",
            help='Resolves ״ to the repeated value before saving. Useful for data export but SHOULD NOT be used when saving GT for HTR training — the model must see the ditto mark, not the resolved value.',
        )
    with save_btn_col:
        if st.button("💾 Save all corrections to GitHub", type="primary", key="cv_save"):
            existing = load_existing_gt()
            save_rows = list(st.session_state.get("cv_all_rows", []))
            if expand_ditto_save:
                save_rows = expand_dittos(save_rows, LEFT_COLS)
                save_rows = expand_dates(save_rows)
            cv_pages = {r["_page"] for r in save_rows}
            other_pages = [r for r in existing
                           if int(r.get("Page_Number", 0) or 0) not in cv_pages]
            saved_meta = st.session_state.get("page_meta", {})
            new_gt_rows = []
            for row in save_rows:
                p = row["_page"]
                gt_row = {c: "" for c in GT_COLS}
                gt_row["Page_Number"] = str(p)
                gt_row["Folio_Number"] = PAGE_FOLIO.get(p, "")
                for col in LEFT_COLS + META_COLS:
                    if col in gt_row:
                        gt_row[col] = row.get(col, "")
                if expand_ditto_save:
                    for mf in META_FIELDS:
                        gt_row[mf] = saved_meta.get(p, {}).get(mf, "")
                gt_row["OCR_Method"] = "ground_truth"
                new_gt_rows.append(gt_row)
            all_gt = other_pages + new_gt_rows
            all_gt.sort(key=lambda r: (int(r.get("Page_Number", 0) or 0),
                                        int(r.get("Serial_No", 0) or 0)))

            pp_note = " (post-processing applied)" if expand_ditto_save else ""
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            pages_str = ", ".join(f"page {p}" for p in sorted(cv_pages))
            reviewer = st.session_state.get("reviewer", "unknown")
            commit_msg = f"RA save ({reviewer}): {pages_str} — {timestamp}{pp_note}"

            with st.spinner("Saving to GitHub…"):
                gt_ok,   gt_err  = _github_put(_gt_tsv_string(all_gt),
                                               GITHUB_GT_PATH, commit_msg)
                meta_ok, meta_err = _github_put(
                    _meta_tsv_string(saved_meta),
                    GITHUB_META_PATH,
                    commit_msg,
                )

            if gt_ok and meta_ok:
                # Mirror to local file so load_existing_gt() stays in sync this session
                save_ground_truth(all_gt)
                save_page_metadata(saved_meta)
                st.success(
                    f"✅ Saved {len(new_gt_rows)} rows across {len(cv_pages)} pages "
                    f"+ page metadata to GitHub{pp_note}."
                )
            else:
                errors = "\n".join(e for e in [gt_err, meta_err] if e)
                st.error(
                    f"❌ GitHub save failed — **do not continue working and contact Sinai**.\n\n"
                    f"Error details: {errors}"
                )

    # ── Report a problem ────────────────────────────────────────
    st.markdown("---")
    with st.expander("⚠️ Report a problem to Sinai"):
        report_desc = st.text_area(
            "Describe the problem (include the page number and what you were doing):",
            key="report_desc",
        )
        if st.button("Send report", key="report_send"):
            if not report_desc.strip():
                st.warning("Please describe the problem before sending.")
            else:
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                issue_title = f"RA report — page {page_num} — {timestamp}"
                issue_body = f"**Page:** {page_num} (Folio {PAGE_FOLIO.get(page_num, '?')})\n**Time:** {timestamp}\n\n{report_desc}"
                with st.spinner("Sending report…"):
                    ok, err = _github_create_issue(issue_title, issue_body)
                if ok:
                    st.success("✅ Report sent to Sinai.")
                else:
                    st.error(
                        f"❌ Could not send report automatically. "
                        f"Please email sinai.rusinek@gmail.com directly.\n\nError: {err}"
                    )

elif view_mode == "Image Preprocessing":
    render_preprocess_view(page_num)
