"""
image_preprocess.py — Pure image-processing helpers for the Haditax wizard view.

Functions here have no Streamlit dependency and can be imported by both
haditax.py (wizard UI) and segment_unified.py (batch pipeline).
"""

from __future__ import annotations

import io as _io
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ── Deskew (perspective warp of paper boundary) ─────────────────────────────

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def deskew_image(img: np.ndarray) -> np.ndarray:
    """Detect the paper boundary in *img* and perspective-warp it to a rectangle.

    Pure function — no caching, no Streamlit. Returns the deskewed BGR image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

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
    dst_pts = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(
        img, M, (dst_w, dst_h),
        flags=cv2.INTER_LANCZOS4,
        borderValue=(255, 255, 255),
    )


# ── Table-corner detection ──────────────────────────────────────────────────

def auto_table_corners(
    deskewed: np.ndarray,
    header_frac: float = 0.08,
    table_width_frac: float = 0.455,
) -> list[list[int]]:
    """Return four [[x,y], ...] corners (TL, TR, BR, BL) for the data table area.

    Uses the same simple fractions that crop_table() in segment_unified.py uses.
    The deskewed image is assumed to be perspective-corrected already.
    """
    H, W = deskewed.shape[:2]
    y0 = int(H * header_frac)
    x1 = int(W * table_width_frac)
    y1 = H
    x0 = 0
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]  # TL, TR, BR, BL


def corners_to_overlay(
    image: np.ndarray,
    corners: list[list[int]],
    color: tuple[int, int, int] = (0, 0, 220),
    dot_radius: int = 10,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw the quadrilateral + corner dots onto a copy of *image*. Returns copy."""
    out = image.copy()
    pts = np.array(corners, dtype=np.int32)
    n = len(pts)
    for i in range(n):
        cv2.line(out, tuple(pts[i]), tuple(pts[(i + 1) % n]), color, line_thickness)
    for pt in pts:
        cv2.circle(out, tuple(pt), dot_radius, color, -1)
    return out


# ── Header-strip crop ───────────────────────────────────────────────────────

def crop_above_table(
    deskewed: np.ndarray,
    table_corners: list[list[int]],
) -> np.ndarray:
    """Crop the page-metadata band sitting *above* the data table.

    This is the area from y=0 (top of the deskewed page) down to the top
    edge of the table quadrilateral. It typically contains the printed
    form title, tax-payer name, registration number, etc.
    Returns the cropped region as a BGR array.
    """
    top_y  = min(table_corners[0][1], table_corners[1][1])  # TL, TR y
    left_x = min(table_corners[0][0], table_corners[3][0])
    right_x = max(table_corners[1][0], table_corners[2][0])

    H, W = deskewed.shape[:2]
    return deskewed[0 : max(1, min(top_y, H)), max(0, left_x) : min(right_x, W)]


def crop_header_strip(
    deskewed: np.ndarray,
    table_corners: list[list[int]],
) -> np.ndarray:
    """Crop the printed column-header band from the *top of the data table*.

    The column headers are the first printed rows inside the table area
    (not the metadata strip above it). We take the top 15% of the table
    height starting from the table's top edge.
    Returns the cropped region as a BGR array.
    """
    top_y = min(table_corners[0][1], table_corners[1][1])  # TL, TR
    bot_y = max(table_corners[3][1], table_corners[2][1])  # BL, BR
    table_h = bot_y - top_y

    strip_y0 = top_y
    strip_y1 = top_y + max(60, int(table_h * 0.15))

    left_x = min(table_corners[0][0], table_corners[3][0])
    right_x = max(table_corners[1][0], table_corners[2][0])

    H, W = deskewed.shape[:2]
    return deskewed[max(0, strip_y0) : min(H, strip_y1), max(0, left_x) : min(W, right_x)]


# ── Gemini schema extraction (single call for both strips) ───────────────────

_SCHEMA_PROMPT = """\
You are reading two images from a British Mandate Palestine property tax register \
(Form TR/39).

IMAGE 1 — the metadata band ABOVE the main data table. It contains printed field \
labels filled in by hand for this specific page (e.g. taxpayer name, ID, village, \
district, date, etc.).

IMAGE 2 — the printed column-header rows at the TOP of the data table. The header \
may span 1, 2, or 3 printed rows with merged cells grouping related leaf columns.

Output ONLY a JSON object with exactly two keys:
  "metadata_fields": array of strings — field label names from IMAGE 1 in \
top-to-bottom, left-to-right order. Use underscores for spaces. Include only \
label names, NOT their values. E.g. "Tax_Payer", "Tax_Payer_ID", "Village", "Year".
  "column_names": array of strings — leaf column names from IMAGE 2 in \
left-to-right order. Each string concatenates ancestor tier labels with "_", e.g. \
"Reference_to_Register_of_Exemptions_Amount_Mils", "Serial_No".

Do NOT include any explanation, markdown, or extra keys. Output only the raw JSON object.\
"""


def _encode_bgr(img: np.ndarray) -> bytes:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = _io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _parse_gemini_json(raw: str) -> object:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    start = next((i for i, c in enumerate(raw) if c in "[{"), None)
    if start is None:
        raise RuntimeError(f"Gemini returned no JSON: {raw[:300]}")
    try:
        result, _ = json.JSONDecoder().raw_decode(raw, start)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned unparseable output: {raw[:300]}") from exc
    return result


def gemini_extract_schema(
    above_crop: np.ndarray,
    header_crop: np.ndarray,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> tuple[list[str], list[str]]:
    """Single Gemini call that reads both image strips and returns (metadata_fields, column_names).

    above_crop  — BGR crop of the band above the table (metadata labels).
    header_crop — BGR crop of the top of the table (column headers).
    Returns (metadata_fields, column_names) — both as flat lists of strings.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    parts = [
        types.Part.from_bytes(data=_encode_bgr(above_crop), mime_type="image/jpeg"),
        types.Part.from_bytes(data=_encode_bgr(header_crop), mime_type="image/jpeg"),
        types.Part.from_text(text=_SCHEMA_PROMPT),
    ]
    resp = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(max_output_tokens=2048),
    )
    result = _parse_gemini_json((resp.text or "").strip())

    if not isinstance(result, dict):
        raise RuntimeError(f"Expected JSON object, got: {type(result)}")
    meta = [str(s) for s in result.get("metadata_fields", [])]
    cols = [str(s) for s in result.get("column_names", [])]
    return meta, cols


# ── x_left_col detection (LLM anchor for the leftmost printed column line) ──

_X_LEFT_COL_PROMPT = """\
This image is a crop from a British Mandate Palestine tax-register scan \
(landscape, left page of a spread). The first column of the table is the \
narrow "Serial_No" column. Look at the printed vertical lines of the table \
grid; the line we want is the printed vertical line that forms the RIGHT \
edge of the Serial_No column (and the LEFT edge of the next column, "Date"). \
This is typically between 12% and 18% of the image width — well inside the \
page, not at the page/binding edge.

Return ONLY a JSON object: {{"x_first_interior": <int>}}

`x_first_interior` is the x-coordinate (pixels from the left of THIS image) \
of that printed line. Image width is {W} px. Ignore the binding shadow, the \
outer page edge, the outer-left frame line of the table, and any handwriting. \
No explanation, no markdown, no extra keys.\
"""


def _validate_x_left_col(value, image_w: int) -> Optional[int]:
    try:
        x = int(value)
    except (TypeError, ValueError):
        return None
    if not (30 <= x <= int(image_w * 0.30)):
        return None
    return x


def detect_x_left_col_gemini(
    wide_crop_bgr: np.ndarray,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> Optional[int]:
    """Ask Gemini for the x of the leftmost printed column line in wide_crop.
    Returns int x in image coords, or None if unavailable / invalid."""
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return None

    from google import genai
    from google.genai import types

    h, w = wide_crop_bgr.shape[:2]
    client = genai.Client(api_key=key)
    parts = [
        types.Part.from_bytes(data=_encode_bgr(wide_crop_bgr), mime_type="image/jpeg"),
        types.Part.from_text(text=_X_LEFT_COL_PROMPT.format(W=w)),
    ]
    try:
        cfg = types.GenerateContentConfig(
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        resp = client.models.generate_content(model=model, contents=parts, config=cfg)
        result = _parse_gemini_json((resp.text or "").strip())
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    return _validate_x_left_col(
        result.get("x_first_interior", result.get("x_left_col")), w)


def detect_x_left_col_claude(
    wide_crop_bgr: np.ndarray,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
) -> Optional[int]:
    """Cross-check via Anthropic Claude (vision). Same prompt, same return type."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    import base64
    h, w = wide_crop_bgr.shape[:2]
    img_b64 = base64.standard_b64encode(_encode_bgr(wide_crop_bgr)).decode("ascii")
    client = anthropic.Anthropic(api_key=key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": img_b64,
                    }},
                    {"type": "text", "text": _X_LEFT_COL_PROMPT.format(W=w)},
                ],
            }],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        result = _parse_gemini_json(text)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    return _validate_x_left_col(
        result.get("x_first_interior", result.get("x_left_col")), w)


def detect_x_left_col_cached(
    wide_crop_bgr: np.ndarray,
    page: int,
    cache_dir: Path,
    mode: str = "gemini",
) -> Optional[int]:
    """Cached LLM dispatch.

    mode: "gemini" | "claude" | "consensus".
    consensus = call both; if both valid and within 15px of each other, average;
                otherwise return Gemini's answer (with both written to cache for audit).
    Cache file: cache_dir/x_left_col_page{N}.json containing {"gemini":..,"claude":..,"chosen":..,"mode":..}.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"x_left_col_page{page}.json"
    cached: dict = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("mode") == mode and isinstance(cached.get("chosen"), int):
                return int(cached["chosen"])
        except Exception:
            cached = {}

    g = c = None
    if mode in ("gemini", "consensus"):
        g = detect_x_left_col_gemini(wide_crop_bgr)
    if mode in ("claude", "consensus"):
        c = detect_x_left_col_claude(wide_crop_bgr)

    if mode == "gemini":
        chosen = g
    elif mode == "claude":
        chosen = c
    else:  # consensus
        if g is not None and c is not None and abs(g - c) <= 15:
            chosen = (g + c) // 2
        else:
            chosen = g if g is not None else c

    cache_path.write_text(
        json.dumps({"mode": mode, "gemini": g, "claude": c, "chosen": chosen}, indent=2),
        encoding="utf-8",
    )
    return chosen


# ── notebook_config.json helpers ─────────────────────────────────────────────

def load_notebook_config(path: Path) -> dict:
    """Load notebook_config.json. Returns empty-ish dict if file absent."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"notebook_id": "Hadita", "column_names": [], "table_corners": {}}


def save_notebook_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
