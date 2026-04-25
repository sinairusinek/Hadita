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


# ── Gemini header recognition ────────────────────────────────────────────────

_HEADER_PROMPT = """\
This is a printed multi-tier table header from a British Mandate Palestine \
property tax register (Form TR/39). The header may span 1, 2, or 3 printed rows \
with merged cells grouping related leaf columns.

Output ONLY a JSON array of strings — the leaf column names in left-to-right order.
Each string must concatenate its ancestor tier labels with "_", for example:
  "Reference_to_Register_of_Exemptions_Amount_Mils"
  "Serial_No"
Do NOT include any explanation, markdown, or extra keys. Output only the raw JSON array.\
"""


def gemini_flatten_headers(
    header_crop: np.ndarray,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> list[str]:
    """Send *header_crop* (BGR ndarray) to Gemini and return a flat ordered list of column names.

    Raises RuntimeError if the API key is missing or the response cannot be parsed.
    Uses gemini-2.5-flash by default (printed text; Flash is fast and cheap here).
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    pil_img = Image.fromarray(cv2.cvtColor(header_crop, cv2.COLOR_BGR2RGB))
    buf = _io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)

    parts = [
        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
        types.Part.from_text(text=_HEADER_PROMPT),
    ]
    resp = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(max_output_tokens=2048),
    )
    raw = (resp.text or "").strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # Find the start of the JSON array/object and parse only that portion.
    # raw_decode stops at the end of the first valid JSON value, so trailing
    # explanatory text from Gemini doesn't cause a parse failure.
    start = next((i for i, c in enumerate(raw) if c in "[{"), None)
    if start is None:
        raise RuntimeError(f"Gemini returned no JSON: {raw[:300]}")
    try:
        result, _ = json.JSONDecoder().raw_decode(raw, start)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini returned unparseable output: {raw[:300]}") from exc

    if not isinstance(result, list):
        raise RuntimeError(f"Expected JSON array, got: {type(result)}")
    return [str(s) for s in result]


_METADATA_PROMPT = """\
This is the top portion of a British Mandate Palestine property tax register page \
(Form TR/39), above the main data table. It contains printed field labels filled in \
by hand for this specific page — things like taxpayer name, registration number, \
village/district, date, etc.

Output ONLY a JSON array of strings — the printed field label names you can read, \
in top-to-bottom, left-to-right order. Use underscores instead of spaces and keep \
the label concise, e.g. "Tax_Payer", "Tax_Payer_ID", "Village", "Year".
Do NOT include field values — only the label names.
Do NOT include any explanation, markdown, or extra keys. Output only the raw JSON array.\
"""


def gemini_extract_metadata_fields(
    above_crop: np.ndarray,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> list[str]:
    """Send the above-table metadata band to Gemini and return field label names.

    Returns a list of strings like ["Tax_Payer", "Tax_Payer_ID", ...].
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    pil_img = Image.fromarray(cv2.cvtColor(above_crop, cv2.COLOR_BGR2RGB))
    buf = _io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)

    parts = [
        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
        types.Part.from_text(text=_METADATA_PROMPT),
    ]
    resp = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(max_output_tokens=1024),
    )
    raw = (resp.text or "").strip()
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

    if not isinstance(result, list):
        raise RuntimeError(f"Expected JSON array, got: {type(result)}")
    return [str(s) for s in result]


# ── notebook_config.json helpers ─────────────────────────────────────────────

def load_notebook_config(path: Path) -> dict:
    """Load notebook_config.json. Returns empty-ish dict if file absent."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"notebook_id": "Hadita", "column_names": [], "table_corners": {}}


def save_notebook_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
