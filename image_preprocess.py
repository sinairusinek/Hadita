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


# ── notebook_config.json helpers ─────────────────────────────────────────────

def load_notebook_config(path: Path) -> dict:
    """Load notebook_config.json. Returns empty-ish dict if file absent."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"notebook_id": "Hadita", "column_names": [], "table_corners": {}}


def save_notebook_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
