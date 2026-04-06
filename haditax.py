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
from streamlit_image_zoom import image_zoom
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# ── Project paths ────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
IMAGE_PATTERN = "000nvrj-432316TAX 1-85_page-{:04d}.jpg"
CACHE_DIR = PROJECT_DIR / ".ocr_cache"
GROUND_TRUTH_FILE = PROJECT_DIR / "ground_truth.tsv"
PAGE_METADATA_FILE = PROJECT_DIR / "page_metadata.tsv"

META_FIELDS = [
    "Tax_Payer_Arabic", "Tax_Payer_Romanized",
    "Tax_Payer_ID_Arabic", "Tax_Payer_ID_Romanized",
]

# ── Column definitions (must match compare_ocr.py) ───────────
LEFT_COLS = [
    "Serial_No", "Date", "Block_No", "Parcel_No", "Cat_No", "Area",
    "Nature_of_Entry", "New_Serial_No", "Volume_No", "Serial_No_Vol",
    "Tax_LP", "Tax_Mils", "Total_Tax_LP", "Total_Tax_Mils",
    "Entry_No", "Remarks",
]
RIGHT_COLS = [
    "Assessment_Year", "Amount_Assessed_LP", "Amount_Assessed_Mils",
    "Date_of_Payment", "Receipt_No",
    "Amount_Paid_LP", "Amount_Paid_Mils",
    "Balance_LP", "Balance_Mils", "Right_Side_Notes",
]
META_COLS = ["Row_Confidence", "Red_Ink", "Disagreements"]
ALL_DATA_COLS = LEFT_COLS + RIGHT_COLS + META_COLS

GT_COLS = [
    "Page_Number", "Folio_Number",
    "Tax_Payer_Arabic", "Tax_Payer_Romanized",
    "Tax_Payer_ID_Arabic", "Tax_Payer_ID_Romanized",
    "Serial_No", "Date", "Block_No", "Parcel_No", "Cat_No", "Area",
    "Nature_of_Entry", "New_Serial_No", "Volume_No", "Serial_No_Vol",
    "Tax_LP", "Tax_Mils", "Total_Tax_LP", "Total_Tax_Mils",
    "\u05d4\u05e2\u05e8\u05d5\u05ea",  # Hebrew Remarks column header
    "Entry_No", "Remarks",
    "Assessment_Year", "Amount_Assessed_LP", "Amount_Assessed_Mils",
    "Date_of_Payment", "Receipt_No",
    "Amount_Paid_LP", "Amount_Paid_Mils",
    "Balance_LP", "Balance_Mils", "Right_Side_Notes",
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
        return df  # OCR data is already in Arabic digits
    df = df.copy()
    for col in df.columns:
        if skip_cols and col in skip_cols:
            continue
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].apply(lambda v: convert_digits(str(v), mode) if pd.notna(v) else v)
    return df


# ── Helpers ──────────────────────────────────────────────────

def page_image_path(page_num: int) -> Path:
    return PROJECT_DIR / IMAGE_PATTERN.format(page_num)


def load_approach_m(page_num: int) -> list[dict]:
    """Load cached Approach M results."""
    cache_file = CACHE_DIR / f"M_page{page_num}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return []


# ── Deskew pipeline ──────────────────────────────────────────

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def deskew_page(page_num: int) -> np.ndarray:
    """Crop black borders, detect page corners, perspective-warp to a rectangle.

    Caches result to .ocr_cache/deskewed_page{N}.jpg.
    Returns the deskewed image as a BGR numpy array.
    """
    cache_path = CACHE_DIR / f"deskewed_page{page_num}.jpg"
    if cache_path.exists():
        return cv2.imread(str(cache_path))

    img = cv2.imread(str(page_image_path(page_num)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: paper (bright ~205) vs background (dark ~40)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Approximate largest contour to 4 corners
    cnt = contours[0]
    approx = None
    for eps_mult in [0.02, 0.03, 0.04, 0.05]:
        epsilon = eps_mult * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            break

    if approx is None or len(approx) != 4:
        rect = cv2.minAreaRect(cnt)
        approx = cv2.boxPoints(rect).astype(np.int32).reshape(4, 1, 2)

    src_pts = _order_corners(approx.reshape(4, 2))

    w1 = np.linalg.norm(src_pts[1] - src_pts[0])
    w2 = np.linalg.norm(src_pts[2] - src_pts[3])
    h1 = np.linalg.norm(src_pts[3] - src_pts[0])
    h2 = np.linalg.norm(src_pts[2] - src_pts[1])
    dst_w = int(max(w1, w2))
    dst_h = int(max(h1, h2))

    dst_pts = np.array([
        [0, 0], [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1], [0, dst_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    deskewed = cv2.warpPerspective(img, M, (dst_w, dst_h),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderValue=(255, 255, 255))

    CACHE_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(cache_path), deskewed, [cv2.IMWRITE_JPEG_QUALITY, 95])
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
    """Load page_metadata.tsv → {page_num: {field: value}}."""
    if not PAGE_METADATA_FILE.exists():
        return {}
    result = {}
    with open(PAGE_METADATA_FILE, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                p = int(row["Page_Number"])
            except (KeyError, ValueError):
                continue
            result[p] = {k: row.get(k, "") for k in META_FIELDS}
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


# ── PAGE XML export ──────────────────────────────────────────

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


# ── Streamlit App ────────────────────────────────────────────

st.set_page_config(page_title="Haditax", layout="wide")
st.title("Haditax — Ground Truth Editor")

# ── Shared page selector + view mode ─────────────────────────
ALL_COLS = LEFT_COLS + RIGHT_COLS
PAGES = list(PAGE_FOLIO.keys())  # [3, 10, 50]

ctrl_col1, ctrl_col2 = st.columns([2, 1])
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
view_mode = "Correction View"

# ═══════════════════════════════════════════════════════════════
# CORRECTION VIEW — all pages in one table, image synced to page
# ═══════════════════════════════════════════════════════════════
if view_mode == "Correction View":
    # Load all pages into combined session state
    if "cv_all_rows" not in st.session_state:
        combined = []
        for p in PAGES:
            page_rows = load_approach_m(p)
            for r in page_rows:
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

    col_img, col_tbl = st.columns([1, 1])

    PANEL_H = 800  # shared height for both panels in pixels

    with col_img:
        st.caption("Click to zoom · drag to pan · scroll to navigate")
        with st.spinner("Loading page image..."):
            deskewed = deskew_page(page_num)
        pil_img = Image.fromarray(cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB))
        image_zoom(pil_img, mode="dragmove", size=(700, PANEL_H), keep_aspect_ratio=True)

    with col_tbl:
        st.caption("Click a cell to edit · Tab / Enter to navigate")

        df = pd.DataFrame(all_rows)
        for c in ALL_COLS:
            if c not in df.columns:
                df[c] = ""
        df = df[["_page"] + ALL_COLS].fillna("")
        df.insert(0, "#", range(1, len(df) + 1))
        for c in ["_page"] + ALL_COLS:
            df[c] = df[c].astype(str)

        display_df = convert_df_digits(df, digit_mode, skip_cols=["#", "_page"])

        col_config = {
            "#": st.column_config.NumberColumn("#", width="small", disabled=True),
            "_page": st.column_config.TextColumn("Page", width="small", disabled=True),
        }
        for c in ALL_COLS:
            col_config[c] = st.column_config.TextColumn(c, width="small")

        edited_df = st.data_editor(
            display_df,
            column_config=col_config,
            use_container_width=True,
            num_rows="fixed",
            height=PANEL_H,
            key=f"cv_editor_{digit_mode}",
            disabled=["#", "_page"],
        )

        # Sync edits back to session state (always store as Arabic digits)
        for i in range(len(all_rows)):
            for col in ALL_COLS:
                if col in edited_df.columns:
                    val = edited_df.at[i, col]
                    s = str(val) if pd.notna(val) else ""
                    all_rows[i][col] = convert_digits(s, "arabic")

# ═══════════════════════════════════════════════════════════════
# GRID VIEW — uses shared page_num from top selector
# ═══════════════════════════════════════════════════════════════
else:
    rows = load_approach_m(page_num)
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
                for col in LEFT_COLS + RIGHT_COLS + META_COLS:
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

# ── Correction View save button ───────────────────────────────
if view_mode == "Correction View":
    st.markdown("---")
    if st.button("Save all corrections to ground_truth.tsv", type="primary", key="cv_save"):
        existing = load_existing_gt()
        cv_pages = {r["_page"] for r in st.session_state.get("cv_all_rows", [])}
        other_pages = [r for r in existing
                       if int(r.get("Page_Number", 0) or 0) not in cv_pages]
        new_gt_rows = []
        for row in st.session_state.get("cv_all_rows", []):
            p = row["_page"]
            gt_row = {c: "" for c in GT_COLS}
            gt_row["Page_Number"] = str(p)
            gt_row["Folio_Number"] = PAGE_FOLIO.get(p, "")
            for col in LEFT_COLS + RIGHT_COLS + META_COLS:
                if col in gt_row:
                    gt_row[col] = row.get(col, "")
            gt_row["OCR_Method"] = "ground_truth"
            new_gt_rows.append(gt_row)
        all_gt = other_pages + new_gt_rows
        all_gt.sort(key=lambda r: (int(r.get("Page_Number", 0) or 0),
                                    int(r.get("Serial_No", 0) or 0)))
        save_ground_truth(all_gt)
        save_page_metadata(st.session_state.get("page_meta", {}))
        st.success(f"Saved {len(new_gt_rows)} rows across {len(cv_pages)} pages + page metadata.")
