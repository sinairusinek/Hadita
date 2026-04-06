#!/usr/bin/env python3
"""
compare_ocr.py — Multi-model OCR comparison for British Mandate Palestine tax registers.

Runs pages 3, 10, and 50 through 10 approaches (A–J) and produces comparison CSVs.

Approaches:
  A: Claude Opus 4.6  — full page
  B: Claude Opus 4.6  — zoomed crops
  C: Gemini 2.5 Pro   — full page
  D: Gemini 2.5 Pro   — zoomed crops
  E: Gemini 3.1 Pro   — full page
  F: Gemini 3.1 Pro   — zoomed crops
  G: Kraken + gen2_sc_clean_best — cell-level (OpenCV grid)
  H: Kraken + arabic_best        — cell-level (OpenCV grid)
  I: Kraken + gen2_sc_clean_best — column strips
  J: Kraken + arabic_best        — column strips
  K: Gemini 3 Flash   — full page   (fast, cheap)
  L: Gemini 3 Flash   — zoomed crops

Usage:
  python3 compare_ocr.py                        # run all approaches, all test pages
  python3 compare_ocr.py --pages 3 10           # specific pages
  python3 compare_ocr.py --approaches A B C     # specific approaches
  python3 compare_ocr.py --score                # score all results vs ground truth
  python3 compare_ocr.py --score --approach A   # score a single approach

Setup:
  export ANTHROPIC_API_KEY=...
  export GOOGLE_API_KEY=...
"""

import argparse
import base64
import csv
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────
# CONFIGURATION — edit these as needed
# ──────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
IMAGE_PATTERN = "000nvrj-432316TAX 1-85_page-{:04d}.jpg"
TEST_PAGES = [3, 10, 50]

# Page-to-Folio mapping (read from images; pattern is approx. Folio = Page - 2 for p3, else Page - 1)
PAGE_FOLIO = {3: "1", 10: "9", 50: "49"}

# Gemini model IDs — update to the latest available
GEMINI_25_PRO   = "gemini-2.5-pro"                  # stable GA
GEMINI_3X_PRO   = "gemini-3.1-pro-preview"          # latest, best vision/OCR
GEMINI_3_FLASH  = "gemini-3-flash-preview"           # fast, cheap, strong vision

# Kraken
KRAKEN_BIN         = "/opt/anaconda3/bin/kraken"
KRAKEN_USER_MODEL  = str(PROJECT_DIR / "gen2_sc_clean_best.mlmodel")
KRAKEN_DEFAULT_MODEL = str(
    Path.home() / "Library/Application Support/kraken/arabic_best.mlmodel"
)

# Layout fractions (of full image dimensions)
HEADER_HEIGHT_FRAC      = 0.08    # top 8% = header strip (Tax-Payer, Village, Folio)
LEFT_TABLE_WIDTH_FRAC   = 0.455   # left 45.5% = left table
BAND_COUNT              = 6       # how many horizontal bands per table side

# Output filenames
COMPARISON_TEMPLATE = "comparison_page{page}.csv"
SCORES_FILE         = "comparison_scores.csv"
CACHE_DIR           = PROJECT_DIR / ".ocr_cache"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# ALL COLUMN NAMES (canonical order)
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# OCR PROMPT
# ──────────────────────────────────────────────────────────

OCR_PROMPT_FULL = """
You are an expert palaeographer specialising in British Mandate Palestine administrative documents.
The image shows a page (or portion of a page) from a Tax Register, Form TR/39.

The page header contains:
  - "Tax-Payer" followed by an Arabic name and, immediately after the name, a handwritten
    identifier/reference number (in Eastern Arabic or Western digits)
  - "Village" followed by an Arabic village name (always الحديثة for this register)
  - "Folio No." followed by a number

The LEFT side of the page is a table with these columns (right-to-left reading order):
  Serial_No | Date | Block_No | Parcel_No | Cat_No | Area | Nature_of_Entry |
  New_Serial_No | Volume_No | Serial_No_Vol |
  Tax_LP | Tax_Mils | Total_Tax_LP | Total_Tax_Mils | Entry_No | Remarks

IMPORTANT column order note: The SECOND column (immediately after Serial_No) is "Date".
It contains a year-like number (e.g., 938) that is typically the same for every row on the page
and repeats with ditto marks. Do NOT assign this value to "Block_No". The "Block_No" column
comes THIRD (after Date). Count columns carefully from the right edge of the left table.

The RIGHT side is labelled "Assessment to Account" with columns:
  Assessment_Year | Amount_Assessed_LP | Amount_Assessed_Mils |
  Date_of_Payment | Receipt_No |
  Amount_Paid_LP | Amount_Paid_Mils |
  Balance_LP | Balance_Mils | Right_Side_Notes

Rules:
- Preserve Arabic script and Eastern Arabic numerals exactly as written (٠١٢٣٤٥٦٧٨٩).
- IMPORTANT: Carefully distinguish handwritten Eastern Arabic numerals that look similar:
  * ٢ (2) = one small angular hook/curve, compact and tight
  * ٣ (3) = two scallops/bumps, wider and more open (like a sideways "3")
  * ٤ (4) = open angle/hook facing right
  * ٦ (6) = small circle or dot
  * ٠ (0) = a dot, often smaller than ٦
- Use "" (empty string) for blank cells.
- DITTO MARKS: A cell containing a short double-tick mark (״), a single quotation-like stroke,
  or the symbol (,,) means "same value as the row above". Output the literal character '"' for these.
  Do NOT confuse ditto marks with actual numerals or letters.
  IMPORTANT: The Cat_No column typically contains a single Eastern Arabic digit (e.g., ١).
  A single vertical stroke in Cat_No is the numeral ١, NOT a ditto mark. Ditto marks have
  two strokes or a clearly curved quotation shape.
  IMPORTANT: Never output ditto ('"') for the very first data row — the first row always has
  a real value, not a ditto mark.
- Tax_LP and Tax_Mils are present whenever a tax was assessed; do not leave them blank unless
  the cell is genuinely empty in the image.
- STRIKETHROUGH: If text in a cell is crossed out / struck through, wrap it in ~~tildes~~.
  If a correction is written above or beside the deletion, output: ~~old~~ new
  If no replacement is visible, output just: ~~old~~
- Add the tag [RED] at the end of a cell value if it is written in red ink.
- Add [?] at the end if the reading is uncertain.
- NUMERAL SYSTEMS: Both Western Arabic (0-9) and Eastern Arabic (٠-٩) appear in the document.
  Output each exactly as written — do NOT convert between them. Western numerals are typically
  printed (e.g., page numbers, English annotations like "102 T.D.L 1940"); Eastern Arabic
  numerals are the handwritten entries.
- CHECKMARKS: Tick marks / checkmarks in cells → output ✓
- THOUSANDS SEPARATOR: Standardize all thousands separators to Western comma (,).
  E.g., ١٬٢٠٠ or ١,٢٠٠ → output ١,٢٠٠
- NIL / DASH: A horizontal dash or line meaning nil / zero → output -
- ENGLISH TEXT: Preserve English abbreviations and text verbatim (e.g., T.D.L, T.P.L, D.L).
- If a row is entirely empty, omit it.

Return ONLY valid JSON in this exact structure (no markdown fences, no extra text):
{
  "page_meta": {
    "folio_number": "...",
    "tax_payer_arabic": "<name in Arabic script as written>",
    "tax_payer_romanized": "<name romanized to Latin>",
    "tax_payer_id_arabic": "<identifier in Arabic-Indic/Eastern Arabic digits as written, or empty>",
    "tax_payer_id_romanized": "<identifier in Latin/Western digits, or empty>",
    "village_arabic": "<village name in Arabic>"
  },
  "rows": [
    {
      "Serial_No": "...", "Date": "...", "Block_No": "...", "Parcel_No": "...", "Cat_No": "...",
      "Area": "...", "Nature_of_Entry": "...", "New_Serial_No": "...",
      "Volume_No": "...", "Serial_No_Vol": "...",
      "Tax_LP": "...", "Tax_Mils": "...", "Total_Tax_LP": "...", "Total_Tax_Mils": "...",
      "Entry_No": "...", "Remarks": "...",
      "Assessment_Year": "...", "Amount_Assessed_LP": "...", "Amount_Assessed_Mils": "...",
      "Date_of_Payment": "...", "Receipt_No": "...",
      "Amount_Paid_LP": "...", "Amount_Paid_Mils": "...",
      "Balance_LP": "...", "Balance_Mils": "...", "Right_Side_Notes": "...",
      "Row_Confidence": "high|medium|low",
      "Red_Ink": "FALSE"
    }
  ]
}
""".strip()

OCR_PROMPT_LEFT_BAND = """
You are an expert palaeographer reading a British Mandate Palestine Tax Register (Form TR/39).
This image is a CROPPED BAND from the LEFT side of one page.
It shows a subset of rows from the left table. Columns (right-to-left reading order):
  Serial_No | Date | Block_No | Parcel_No | Cat_No | Area | Nature_of_Entry |
  New_Serial_No | Volume_No | Serial_No_Vol |
  Tax_LP | Tax_Mils | Total_Tax_LP | Total_Tax_Mils | Entry_No | Remarks

IMPORTANT column order note: The SECOND column (immediately after Serial_No) is "Date".
It contains a year-like number (e.g., 938) that is typically the same for every row on the page
and repeats with ditto marks. Do NOT assign this value to "Block_No". The "Block_No" column
comes THIRD (after Date). Count columns carefully from the right edge of the left table.

Rules:
- Preserve Arabic script and Eastern Arabic numerals exactly (٠١٢٣٤٥٦٧٨٩).
- IMPORTANT: Carefully distinguish similar handwritten numerals:
  * ٢ (2) = one small angular hook/curve, compact
  * ٣ (3) = two scallops/bumps, wider (like sideways "3")
- Empty cells → "".
- DITTO MARKS: A short double-tick (״), quotation-like stroke, or (,,) means "same as row above" → output '"'.
  The Cat_No column typically contains a single Eastern Arabic digit (e.g., ١) — a single vertical
  stroke there is ١, not a ditto mark. Never output ditto for the first visible row in any crop.
- STRIKETHROUGH: crossed-out text → ~~old~~. With correction → ~~old~~ new.
- Red ink → append [RED]. Uncertain → append [?].
- NUMERAL SYSTEMS: Western (0-9) and Eastern Arabic (٠-٩) both appear — output each as written.
- CHECKMARKS: Tick marks / checkmarks → output ✓
- THOUSANDS SEPARATOR: Standardize to Western comma (,). E.g., ١٬٢٠٠ → ١,٢٠٠
- NIL / DASH: A dash or line meaning nil/zero → output -
- ENGLISH TEXT: Preserve English abbreviations verbatim (T.D.L, T.P.L, D.L).
- Tax_LP and Tax_Mils are present whenever a tax was assessed; do not leave them blank unless
  the cell is genuinely empty in the image.
- Omit entirely empty rows.

Return ONLY valid JSON (no markdown, no extra text):
{
  "rows": [
    {"Serial_No":"...","Date":"...","Block_No":"...","Parcel_No":"...","Cat_No":"...","Area":"...",
     "Nature_of_Entry":"...","New_Serial_No":"...","Volume_No":"...","Serial_No_Vol":"...",
     "Tax_LP":"...","Tax_Mils":"...","Total_Tax_LP":"...","Total_Tax_Mils":"...","Entry_No":"...","Remarks":"...",
     "Row_Confidence":"high|medium|low","Red_Ink":"FALSE"}
  ]
}
""".strip()

OCR_PROMPT_RIGHT_BAND = """
You are an expert palaeographer reading a British Mandate Palestine Tax Register (Form TR/39).
This image is a CROPPED BAND from the RIGHT side ("Assessment to Account") of one page.
It shows a subset of rows. Columns:
  Assessment_Year | Amount_Assessed_LP | Amount_Assessed_Mils |
  Date_of_Payment | Receipt_No |
  Amount_Paid_LP | Amount_Paid_Mils |
  Balance_LP | Balance_Mils | Right_Side_Notes

Rules:
- Preserve Arabic script and Eastern Arabic numerals exactly (٠١٢٣٤٥٦٧٨٩).
- IMPORTANT: Carefully distinguish similar handwritten numerals:
  * ٢ (2) = one small angular hook/curve, compact
  * ٣ (3) = two scallops/bumps, wider (like sideways "3")
- Empty cells → "".
- DITTO MARKS: A short double-tick (״), quotation-like stroke, or (,,) means "same as row above" → output '"'.
- STRIKETHROUGH: crossed-out text → ~~old~~. With correction → ~~old~~ new.
- Red ink → append [RED]. Uncertain → append [?].
- NUMERAL SYSTEMS: Western (0-9) and Eastern Arabic (٠-٩) both appear — output each as written.
- CHECKMARKS: Tick marks / checkmarks → output ✓
- THOUSANDS SEPARATOR: Standardize to Western comma (,). E.g., ١٬٢٠٠ → ١,٢٠٠
- NIL / DASH: A dash or line meaning nil/zero → output -
- ENGLISH TEXT: Preserve English abbreviations verbatim (T.D.L, T.P.L, D.L).
- Omit entirely empty rows.

Return ONLY valid JSON (no markdown, no extra text):
{
  "rows": [
    {"Assessment_Year":"...","Amount_Assessed_LP":"...","Amount_Assessed_Mils":"...",
     "Date_of_Payment":"...","Receipt_No":"...",
     "Amount_Paid_LP":"...","Amount_Paid_Mils":"...",
     "Balance_LP":"...","Balance_Mils":"...","Right_Side_Notes":"...",
     "Row_Confidence":"high|medium|low","Red_Ink":"FALSE"}
  ]
}
""".strip()

# ── FEW-SHOT EXAMPLES (from ground truth) ──────────────────
# Representative rows showing correct transcription for tricky cases:
#   Cat_No = ١٠ (two digits, not just ١), Block_No 4-digit, large Area, Tax fields
FEW_SHOT_EXAMPLES = """

IMPORTANT — Below are correctly transcribed example rows from a similar page in this
register. Study them carefully to understand the expected output format, especially for:
  • Cat_No is typically a TWO-digit Eastern Arabic number (e.g., ١٠ not ١)
  • Block_No is always a 4-digit Eastern Arabic number (e.g., ٤١٣٢)
  • Area is an Eastern Arabic number often with a comma as thousands separator (e.g., ٣٤,٩٢٥)
  • Tax_Mils is present for most rows — do not leave blank

Example rows (JSON):
[
  {"Serial_No":"١","Date":"٩٣٨","Block_No":"٤١٣٢","Parcel_No":"٤","Cat_No":"١٠","Area":"٣٤,٩٢٥","Nature_of_Entry":"","New_Serial_No":"","Volume_No":"","Serial_No_Vol":"","Tax_LP":"","Tax_Mils":"٦٢٩","Total_Tax_LP":"","Total_Tax_Mils":"","Entry_No":"","Remarks":"","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٢","Date":"\\"","Block_No":"٤١٣٣","Parcel_No":"١","Cat_No":"١٠","Area":"٤,٧٢٩","Nature_of_Entry":"\\"","New_Serial_No":"١١٧","Volume_No":"","Serial_No_Vol":"","Tax_LP":"","Tax_Mils":"٨٥","Total_Tax_LP":"","Total_Tax_Mils":"","Entry_No":"","Remarks":"","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٣","Date":"\\"","Block_No":"\\"","Parcel_No":"٣٢","Cat_No":"١٠","Area":"١٥٩,٧٧٨","Nature_of_Entry":"\\"","New_Serial_No":"٩٢","Volume_No":"","Serial_No_Vol":"","Tax_LP":"٢","Tax_Mils":"٨٧٦","Total_Tax_LP":"","Total_Tax_Mils":"","Entry_No":"","Remarks":"","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٤","Date":"\\"","Block_No":"٤١٣٤","Parcel_No":"٢","Cat_No":"١٠","Area":"١١,٨٦٨","Nature_of_Entry":"\\"","New_Serial_No":"١٠٠","Volume_No":"","Serial_No_Vol":"","Tax_LP":"","Tax_Mils":"٢١٤","Total_Tax_LP":"","Total_Tax_Mils":"","Entry_No":"","Remarks":"","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٥","Date":"\\"","Block_No":"\\"","Parcel_No":"١٤","Cat_No":"١٠","Area":"٩,٧٤١","Nature_of_Entry":"\\"","New_Serial_No":"١٠٢","Volume_No":"T.D.L","Serial_No_Vol":"1940","Tax_LP":"","Tax_Mils":"١٧٥","Total_Tax_LP":"","Total_Tax_Mils":"","Entry_No":"","Remarks":"","Row_Confidence":"high","Red_Ink":"FALSE"}
]
"""

OCR_PROMPT_FULL_FEWSHOT = (OCR_PROMPT_FULL + "\n\n" + FEW_SHOT_EXAMPLES).strip()

# ── Multiple few-shot variants for ensemble voting ──────────
# Each variant uses a different subset of GT rows to create diverse "priors"
# so that majority voting across variants can correct uncorrelated errors.

_FS_PREAMBLE = """
IMPORTANT — Below are correctly transcribed example rows from a similar page in this
register. Study them carefully to understand the expected output format, especially for:
  • Cat_No is typically a TWO-digit Eastern Arabic number (e.g., ١٠ not ١)
  • Block_No is always a 4-digit Eastern Arabic number (e.g., ٤١٣٢)
  • Area is an Eastern Arabic number often with a comma as thousands separator (e.g., ٣٤,٩٢٥)
  • Tax_Mils is present for most rows — do not leave blank
  • A horizontal dash means nil/zero — output -

Example rows (JSON):
"""

# Variant 1 (= original M): rows 1-5 (Cat_No=10, Block 4132-4134, large Area)
_FS_ROWS_V1 = """[
  {"Serial_No":"١","Date":"٩٣٨","Block_No":"٤١٣٢","Parcel_No":"٤","Cat_No":"١٠","Area":"٣٤,٩٢٥","Nature_of_Entry":"","New_Serial_No":"","Tax_LP":"","Tax_Mils":"٦٢٩","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٢","Date":"\\"","Block_No":"٤١٣٣","Parcel_No":"١","Cat_No":"١٠","Area":"٤,٧٢٩","Nature_of_Entry":"\\"","New_Serial_No":"١١٧","Tax_LP":"","Tax_Mils":"٨٥","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٣","Date":"\\"","Block_No":"\\"","Parcel_No":"٣٢","Cat_No":"١٠","Area":"١٥٩,٧٧٨","Nature_of_Entry":"\\"","New_Serial_No":"٩٢","Tax_LP":"٢","Tax_Mils":"٨٧٦","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٤","Date":"\\"","Block_No":"٤١٣٤","Parcel_No":"٢","Cat_No":"١٠","Area":"١١,٨٦٨","Nature_of_Entry":"\\"","New_Serial_No":"١٠٠","Tax_LP":"","Tax_Mils":"٢١٤","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٥","Date":"\\"","Block_No":"\\"","Parcel_No":"١٤","Cat_No":"١٠","Area":"٩,٧٤١","Nature_of_Entry":"\\"","New_Serial_No":"١٠٢","Tax_LP":"","Tax_Mils":"١٧٥","Row_Confidence":"high","Red_Ink":"FALSE"}
]"""

# Variant 2: rows 7-11 (Cat_No=10, Block 4134-4137, medium Areas)
_FS_ROWS_V2 = """[
  {"Serial_No":"٧","Date":"\\"","Block_No":"٤١٣٤","Parcel_No":"٣٤","Cat_No":"١٠","Area":"٨,٥٤٦","Nature_of_Entry":"\\"","New_Serial_No":"١٠١","Tax_LP":"","Tax_Mils":"١٥٤","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٨","Date":"\\"","Block_No":"\\"","Parcel_No":"٤٦","Cat_No":"١٠","Area":"٢,٩٣٥","Nature_of_Entry":"\\"","New_Serial_No":"٩٤","Tax_LP":"","Tax_Mils":"٥٣","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٩","Date":"\\"","Block_No":"\\"","Parcel_No":"٤٧","Cat_No":"١٠","Area":"٢,٤١٤","Nature_of_Entry":"\\"","New_Serial_No":"٩٥","Tax_LP":"","Tax_Mils":"٤٣","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"١٠","Date":"\\"","Block_No":"\\"","Parcel_No":"٥١","Cat_No":"١٠","Area":"٩,٦٤٣","Nature_of_Entry":"\\"","New_Serial_No":"٩٦","Tax_LP":"","Tax_Mils":"١٧٤","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"١١","Date":"\\"","Block_No":"٤١٣٧","Parcel_No":"٤٢","Cat_No":"١٠","Area":"٨,٥٨٧","Nature_of_Entry":"\\"","New_Serial_No":"١٠٩","Tax_LP":"","Tax_Mils":"١٥٥","Row_Confidence":"high","Red_Ink":"FALSE"}
]"""

# Variant 3: rows 12,15,17,19,25 (Cat_No=14,13, Block 4138-4140, dash Tax, small Areas)
_FS_ROWS_V3 = """[
  {"Serial_No":"١٢","Date":"\\"","Block_No":"٤١٣٨","Parcel_No":"١","Cat_No":"١٤","Area":"٧,٨٠٠","Nature_of_Entry":"\\"","New_Serial_No":"١٠٢","Tax_LP":"","Tax_Mils":"-","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"١٥","Date":"\\"","Block_No":"٤١٣٩","Parcel_No":"٢٢","Cat_No":"١٣","Area":"٢,١٧٢","Nature_of_Entry":"\\"","New_Serial_No":"","Tax_LP":"","Tax_Mils":"١٧","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"١٧","Date":"\\"","Block_No":"\\"","Parcel_No":"٣٣","Cat_No":"١٣","Area":"١,١٩٦","Nature_of_Entry":"\\"","New_Serial_No":"","Tax_LP":"","Tax_Mils":"١٠","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"١٩","Date":"\\"","Block_No":"\\"","Parcel_No":"٣٨","Cat_No":"١٣","Area":"٢,٨٨٩","Nature_of_Entry":"\\"","New_Serial_No":"٩٧","Tax_LP":"","Tax_Mils":"٢٣","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٢٥","Date":"\\"","Block_No":"٤١٤٠","Parcel_No":"١٨","Cat_No":"١٣","Area":"٣,٧٢٩","Nature_of_Entry":"\\"","New_Serial_No":"","Tax_LP":"","Tax_Mils":"٣٠","Row_Confidence":"high","Red_Ink":"FALSE"}
]"""

# Variant 4: rows 21,23,26,29,31 (Cat_No=13,10, Block 4139-4141, small Tax, varied Areas)
_FS_ROWS_V4 = """[
  {"Serial_No":"٢١","Date":"\\"","Block_No":"٤١٣٩","Parcel_No":"٤٥","Cat_No":"١٣","Area":"٩٦٩","Nature_of_Entry":"\\"","New_Serial_No":"١٥٤","Tax_LP":"","Tax_Mils":"٨","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٢٣","Date":"\\"","Block_No":"\\"","Parcel_No":"٤٦","Cat_No":"١٣","Area":"١,٦٢٦","Nature_of_Entry":"\\"","New_Serial_No":"","Tax_LP":"","Tax_Mils":"١٣","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٢٦","Date":"\\"","Block_No":"٤١٤٠","Parcel_No":"٥٠","Cat_No":"١٣","Area":"٦٤","Nature_of_Entry":"\\"","New_Serial_No":"","Tax_LP":"","Tax_Mils":"١","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٢٩","Date":"\\"","Block_No":"٤١٤١","Parcel_No":"٧","Cat_No":"١٣","Area":"٢٠,٠٠٩","Nature_of_Entry":"\\"","New_Serial_No":"١١٣","Tax_LP":"","Tax_Mils":"١٦٠","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٣١","Date":"\\"","Block_No":"\\"","Parcel_No":"","Cat_No":"١٠","Area":"١٢,٥٦٨","Nature_of_Entry":"\\"","New_Serial_No":"١١٣","Tax_LP":"","Tax_Mils":"٢٢٦","Row_Confidence":"high","Red_Ink":"FALSE"}
]"""

# Variant 5: rows 6,13,16,27,32 (Cat_No=10,16,13, mixed Blocks, dash Tax_LP)
_FS_ROWS_V5 = """[
  {"Serial_No":"٦","Date":"\\"","Block_No":"٤١٣٤","Parcel_No":"٣٠","Cat_No":"١٠","Area":"١٩,٢٨٦","Nature_of_Entry":"\\"","New_Serial_No":"","Tax_LP":"","Tax_Mils":"٣٤٧","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"١٣","Date":"\\"","Block_No":"\\"","Parcel_No":"","Cat_No":"١٦","Area":"١٧,٠٧٠","Nature_of_Entry":"\\"","New_Serial_No":"١٠٢","Tax_LP":"","Tax_Mils":"-","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"١٦","Date":"\\"","Block_No":"\\"","Parcel_No":"","Cat_No":"١٦","Area":"٩٤٨","Nature_of_Entry":"\\"","New_Serial_No":"","Tax_LP":"-","Tax_Mils":"-","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٢٧","Date":"\\"","Block_No":"\\"","Parcel_No":"٦٥","Cat_No":"١٣","Area":"٢,٨٠١","Nature_of_Entry":"\\"","New_Serial_No":"","Tax_LP":"","Tax_Mils":"٢٢","Row_Confidence":"high","Red_Ink":"FALSE"},
  {"Serial_No":"٣٢","Date":"\\"","Block_No":"\\"","Parcel_No":"١٨","Cat_No":"١٣","Area":"٦,٥٦٨","Nature_of_Entry":"\\"","New_Serial_No":"","Tax_LP":"","Tax_Mils":"٥٣","Row_Confidence":"high","Red_Ink":"FALSE"}
]"""

FEW_SHOT_VARIANTS = [
    _FS_PREAMBLE + _FS_ROWS_V1,
    _FS_PREAMBLE + _FS_ROWS_V2,
    _FS_PREAMBLE + _FS_ROWS_V3,
    _FS_PREAMBLE + _FS_ROWS_V4,
    _FS_PREAMBLE + _FS_ROWS_V5,
]


OCR_PROMPT_HEADER = """
You are reading the header of a British Mandate Palestine Tax Register page.
Extract:
  - folio_number: the number after "Folio No."
  - tax_payer_arabic: the Arabic name after "Tax-Payer"
  - village_arabic: the Arabic name after "Village"

Return ONLY valid JSON: {"folio_number":"...","tax_payer_arabic":"...","village_arabic":"..."}
""".strip()


# ──────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ──────────────────────────────────────────────────────────

def page_image_path(page_num: int) -> Path:
    return PROJECT_DIR / IMAGE_PATTERN.format(page_num)


def pil_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def load_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# ──────────────────────────────────────────────────────────
# ZOOMED CROP STRATEGY
# ──────────────────────────────────────────────────────────

def make_zoomed_crops(img_path: Path) -> dict:
    """
    Split a full page image into:
      - header: top HEADER_HEIGHT_FRAC of page
      - left_bands: BAND_COUNT horizontal bands from left table region
      - right_bands: BAND_COUNT horizontal bands from right table region
    Returns dict with PIL images.
    """
    img = Image.open(img_path)
    W, H = img.size

    header_h   = int(H * HEADER_HEIGHT_FRAC)
    table_top  = header_h
    table_h    = H - table_top
    left_w     = int(W * LEFT_TABLE_WIDTH_FRAC)

    header = img.crop((0, 0, W, header_h))

    band_h = table_h // BAND_COUNT
    left_bands, right_bands = [], []
    for i in range(BAND_COUNT):
        y0 = table_top + i * band_h
        y1 = table_top + (i + 1) * band_h if i < BAND_COUNT - 1 else H
        left_bands.append(img.crop((0, y0, left_w, y1)))
        right_bands.append(img.crop((left_w, y0, W, y1)))

    return {"header": header, "left_bands": left_bands, "right_bands": right_bands}


# ──────────────────────────────────────────────────────────
# JSON PARSING
# ──────────────────────────────────────────────────────────

def parse_json(text: str) -> Any:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    # Strip ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object/array
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    log.warning("Could not parse JSON from response:\n%s", text[:500])
    return {}


def normalize_row(row: dict, cols: list) -> dict:
    """Ensure a row dict has all expected columns."""
    return {c: row.get(c, "") for c in cols}


# ──────────────────────────────────────────────────────────
# RATE LIMITING
# ──────────────────────────────────────────────────────────

def backoff_sleep(attempt: int, base: float = 2.0):
    t = base ** attempt
    log.info("Rate limit backoff: sleeping %.1fs", t)
    time.sleep(t)


# ──────────────────────────────────────────────────────────
# CACHE
# ──────────────────────────────────────────────────────────

CACHE_DIR.mkdir(exist_ok=True)


def cache_key(approach: str, page: int) -> Path:
    return CACHE_DIR / f"{approach}_page{page}.json"


def load_cache(approach: str, page: int) -> Optional[list]:
    p = cache_key(approach, page)
    if p.exists():
        data = json.loads(p.read_text())
        log.info("Cache hit: %s page %d (%d rows)", approach, page, len(data))
        return data
    return None


def save_cache(approach: str, page: int, rows: list):
    cache_key(approach, page).write_text(json.dumps(rows, ensure_ascii=False, indent=2))


# ──────────────────────────────────────────────────────────
# CLAUDE OPUS 4.6
# ──────────────────────────────────────────────────────────

def _claude_client():
    import anthropic
    return anthropic.Anthropic()


def _claude_ocr(client, prompt: str, b64_images: list[str], max_retries: int = 3) -> str:
    """Call Claude with one or more images and a text prompt."""
    content = []
    for b64 in b64_images:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({"type": "text", "text": prompt})

    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=16384,
                messages=[{"role": "user", "content": content}],
            )
            return resp.content[0].text
        except Exception as e:
            log.warning("Claude error (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries - 1:
                backoff_sleep(attempt + 1)
    return ""


def run_claude_full(page_num: int) -> list[dict]:
    """Approach A: Claude Opus 4.6, full page."""
    cached = load_cache("A", page_num)
    if cached is not None:
        return cached

    client = _claude_client()
    img_path = page_image_path(page_num)
    b64 = load_b64(img_path)
    raw = _claude_ocr(client, OCR_PROMPT_FULL, [b64])
    data = parse_json(raw)
    rows = [normalize_row(r, ALL_DATA_COLS) for r in data.get("rows", [])]
    save_cache("A", page_num, rows)
    return rows


def run_claude_zoomed(page_num: int) -> list[dict]:
    """Approach B: Claude Opus 4.6, zoomed crops."""
    cached = load_cache("B", page_num)
    if cached is not None:
        return cached

    client = _claude_client()
    crops = make_zoomed_crops(page_image_path(page_num))

    # Header
    h_b64 = pil_to_b64(crops["header"])
    header_raw = _claude_ocr(client, OCR_PROMPT_HEADER, [h_b64])
    header_data = parse_json(header_raw)

    # Left bands
    left_rows: list[dict] = []
    for band in crops["left_bands"]:
        b64 = pil_to_b64(band)
        raw = _claude_ocr(client, OCR_PROMPT_LEFT_BAND, [b64])
        band_data = parse_json(raw)
        left_rows.extend(band_data.get("rows", []))
        time.sleep(1)  # gentle pacing

    # Right bands
    right_rows: list[dict] = []
    for band in crops["right_bands"]:
        b64 = pil_to_b64(band)
        raw = _claude_ocr(client, OCR_PROMPT_RIGHT_BAND, [b64])
        band_data = parse_json(raw)
        right_rows.extend(band_data.get("rows", []))
        time.sleep(1)

    rows = merge_left_right(left_rows, right_rows)
    save_cache("B", page_num, rows)
    return rows


# ──────────────────────────────────────────────────────────
# GEMINI
# ──────────────────────────────────────────────────────────

def _gemini_client():
    from google import genai
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=api_key)


def _gemini_ocr(client, model_id: str, prompt: str, pil_images: list,
                max_retries: int = 3) -> str:
    from google.genai import types

    parts = []
    for img in pil_images:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=prompt))

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=parts,
                config=types.GenerateContentConfig(max_output_tokens=32768),
            )
            # resp.text can be None if finish_reason is MAX_TOKENS
            text = resp.text
            if text is None and resp.candidates:
                parts = resp.candidates[0].content.parts or []
                text = "".join(p.text for p in parts if hasattr(p, 'text') and p.text and not getattr(p, 'thought', False))
            return text or ""
        except Exception as e:
            log.warning("Gemini [%s] error (attempt %d): %s", model_id, attempt + 1, e)
            if attempt < max_retries - 1:
                backoff_sleep(attempt + 1)
    return ""


def _run_gemini_full(approach: str, model_id: str, page_num: int) -> list[dict]:
    cached = load_cache(approach, page_num)
    if cached is not None:
        return cached

    client = _gemini_client()
    img = Image.open(page_image_path(page_num))
    raw = _gemini_ocr(client, model_id, OCR_PROMPT_FULL, [img])
    data = parse_json(raw)
    rows = [normalize_row(r, ALL_DATA_COLS) for r in data.get("rows", [])]
    save_cache(approach, page_num, rows)
    return rows


def _run_gemini_zoomed(approach: str, model_id: str, page_num: int) -> list[dict]:
    cached = load_cache(approach, page_num)
    if cached is not None:
        return cached

    client = _gemini_client()
    crops = make_zoomed_crops(page_image_path(page_num))

    header_raw = _gemini_ocr(client, model_id, OCR_PROMPT_HEADER, [crops["header"]])
    _ = parse_json(header_raw)  # store if needed later

    left_rows: list[dict] = []
    for band in crops["left_bands"]:
        raw = _gemini_ocr(client, model_id, OCR_PROMPT_LEFT_BAND, [band])
        left_rows.extend(parse_json(raw).get("rows", []))
        time.sleep(1)

    right_rows: list[dict] = []
    for band in crops["right_bands"]:
        raw = _gemini_ocr(client, model_id, OCR_PROMPT_RIGHT_BAND, [band])
        right_rows.extend(parse_json(raw).get("rows", []))
        time.sleep(1)

    rows = merge_left_right(left_rows, right_rows)
    save_cache(approach, page_num, rows)
    return rows


def run_gemini25_full(page_num: int)    -> list[dict]: return _run_gemini_full("C",  GEMINI_25_PRO,  page_num)
def run_gemini25_zoomed(page_num: int)  -> list[dict]: return _run_gemini_zoomed("D", GEMINI_25_PRO,  page_num)
def run_gemini3x_full(page_num: int)    -> list[dict]: return _run_gemini_full("E",  GEMINI_3X_PRO,  page_num)
def run_gemini3x_zoomed(page_num: int)  -> list[dict]: return _run_gemini_zoomed("F", GEMINI_3X_PRO,  page_num)
def run_gemini3f_full(page_num: int)    -> list[dict]: return _run_gemini_full("K",  GEMINI_3_FLASH, page_num)
def run_gemini3f_zoomed(page_num: int)  -> list[dict]: return _run_gemini_zoomed("L", GEMINI_3_FLASH, page_num)


# ── Few-shot approaches ──

def run_gemini25_full_fewshot(page_num: int) -> list[dict]:
    """Approach M: Gemini 2.5 Pro, full page, with few-shot examples."""
    cached = load_cache("M", page_num)
    if cached is not None:
        return cached

    client = _gemini_client()
    img = Image.open(page_image_path(page_num))
    raw = _gemini_ocr(client, GEMINI_25_PRO, OCR_PROMPT_FULL_FEWSHOT, [img])
    data = parse_json(raw)
    rows = [normalize_row(r, ALL_DATA_COLS) for r in data.get("rows", [])]
    save_cache("M", page_num, rows)

    # Save page-level metadata for haditax (meta_page{N}.json)
    raw_meta = data.get("page_meta", {})
    if raw_meta:
        meta = {
            "Tax_Payer_Arabic":    raw_meta.get("tax_payer_arabic", ""),
            "Tax_Payer_Romanized": raw_meta.get("tax_payer_romanized", ""),
            "Tax_Payer_ID_Arabic":    raw_meta.get("tax_payer_id_arabic", ""),
            "Tax_Payer_ID_Romanized": raw_meta.get("tax_payer_id_romanized", ""),
        }
        (CACHE_DIR / f"meta_page{page_num}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

    return rows


# ── Column-specific cell prompts for hard columns ──

CELL_PROMPTS = {
    "Cat_No": """You are reading a SINGLE COLUMN from a British Mandate Palestine tax register.
This column is "Cat_No" (land category number).
Each cell contains a 1-2 digit Eastern Arabic number. Common values: ١٠, ١٣, ١٤, ١٦.
IMPORTANT: The digit ٠ (zero) looks like a small dot — do NOT miss it.
A cell with ١ followed by a small dot is ١٠ (ten), not just ١ (one).
Ditto marks (a short double-tick ״ or ") mean "same as row above" — output '"'.

Read each cell from top to bottom. Return ONLY a JSON array of string values, one per row.
Example: ["١٠", "\\"", "\\"", "١٣", "\\""]""",

    "Area": """You are reading a SINGLE COLUMN from a British Mandate Palestine tax register.
This column is "Area" (land area in square meters or dunams).
Each cell contains an Eastern Arabic number, often with a comma as thousands separator.
Values typically range from ١,٠٠٠ to ٢٠٠,٠٠٠. Some can be as small as ٥٠٠.
IMPORTANT: Read ALL digits carefully, especially the leading (leftmost) digit.
Common confusion: ٢ (angular hook) vs ٣ (two scallops). Count the bumps.
Ditto marks (a short double-tick ״ or ") mean "same as row above" — output '"'.

Read each cell from top to bottom. Return ONLY a JSON array of string values, one per row.
Example: ["٣٤,٩٢٥", "٤,٧٢٩", "١٥٩,٧٧٨"]""",

    "Tax_Mils": """You are reading a SINGLE COLUMN from a British Mandate Palestine tax register.
This column is "Tax_Mils" (tax amount in mils, part of LP currency).
Each cell contains an Eastern Arabic number, typically 1-3 digits. Values range from ١ to ٩٩٩.
A horizontal dash or line means nil/zero — output "-".
Do NOT leave cells blank unless genuinely empty.
Do NOT add leading zeros (output ٨٥ not ٠٨٥).
Ditto marks (a short double-tick ״ or ") mean "same as row above" — output '"'.

Read each cell from top to bottom. Return ONLY a JSON array of string values, one per row.
Example: ["٦٢٩", "٨٥", "٨٧٦", "٢١٤"]""",

    "Tax_LP": """You are reading a SINGLE COLUMN from a British Mandate Palestine tax register.
This column is "Tax_LP" (tax amount in Palestinian Pounds).
Each cell contains an Eastern Arabic number, typically 1-2 digits, or is empty.
Most cells may be empty (tax is often less than 1 LP). Do not leave blank if a value is present.
A horizontal dash or line means nil/zero — output "-".

Read each cell from top to bottom. Return ONLY a JSON array of string values, one per row.
Example: ["", "", "٢", ""]""",
}

# Columns to target with cell-level OCR (the ones M still gets wrong)
CELL_TARGET_COLS = ["Cat_No", "Area", "Tax_Mils", "Tax_LP"]


def _preprocess_cell_image(cell_img: Image.Image, upscale: int = 3) -> Image.Image:
    """Apply CLAHE contrast enhancement and upscale for better OCR."""
    import cv2
    import numpy as np

    arr = np.array(cell_img)
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    # CLAHE adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Convert back to RGB for API
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # Upscale
    h, w = rgb.shape[:2]
    upscaled = cv2.resize(rgb, (w * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4)

    return Image.fromarray(upscaled)


def _gemini_ocr_column_strip(client, strip_img: Image.Image, col_name: str) -> list[str]:
    """Send a preprocessed column strip to Gemini with column-specific prompt."""
    prompt = CELL_PROMPTS.get(col_name, "")
    if not prompt:
        return []

    raw = _gemini_ocr(client, GEMINI_25_PRO, prompt, [strip_img])
    # Parse JSON array from response
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        values = json.loads(raw)
        if isinstance(values, list):
            return [str(v) for v in values]
    except json.JSONDecodeError:
        log.warning("Could not parse column strip response for %s: %s", col_name, raw[:200])
    return []


def run_gemini25_cell_hybrid(page_num: int) -> list[dict]:
    """Approach O: Gemini 2.5 few-shot full page + cell-level OCR for hard columns.

    1. Start with approach M results (few-shot full page) as the base
    2. Detect grid and extract column strips for hard columns
    3. Preprocess (CLAHE + upscale) and send to Gemini with column-specific prompts
    4. Override base values with cell-level readings for target columns
    """
    cached = load_cache("O", page_num)
    if cached is not None:
        return cached

    # Step 1: Get base rows from approach M
    base_rows = run_gemini25_full_fewshot(page_num)
    if not base_rows:
        save_cache("O", page_num, [])
        return []

    # Step 2: Detect grid for left table
    img_path = page_image_path(page_num)
    h_lines, v_lines, table_img, _ = detect_grid(img_path, "left")

    # Map column indices — auto-detect left-side offset
    # The grid detection picks up the image edge (binding) as v_lines[0]≈0.
    # If the first column is much wider than the median, it's the binding area
    # and we need offset +1 to skip it.
    col_widths = [v_lines[i+1] - v_lines[i] for i in range(len(v_lines)-1)]
    median_w = sorted(col_widths)[len(col_widths)//2]
    grid_offset = 1 if (v_lines[0] < 30 and col_widths[0] > median_w * 2) else 0
    if grid_offset > 0:
        log.info("  Grid offset: +%d (binding edge detected, first col %dpx vs median %dpx)",
                 grid_offset, col_widths[0], median_w)
    col_indices = {col: i + grid_offset for i, col in enumerate(LEFT_COLS)}

    client = _gemini_client()

    # Step 3: For each target column, extract strip, preprocess, OCR
    col_overrides: dict[str, list[str]] = {}
    for col_name in CELL_TARGET_COLS:
        col_idx = col_indices.get(col_name)
        if col_idx is None or col_idx >= len(v_lines) - 1:
            log.warning("Column %s (idx %s) not in detected grid (%d v-lines)",
                        col_name, col_idx, len(v_lines))
            continue

        # Crop column strip
        x0, x1 = v_lines[col_idx], v_lines[col_idx + 1]
        strip_cv = table_img[:, x0:x1]
        strip_pil = Image.fromarray(cv2.cvtColor(strip_cv, cv2.COLOR_BGR2RGB))

        # Preprocess
        strip_enhanced = _preprocess_cell_image(strip_pil, upscale=2)

        # OCR
        values = _gemini_ocr_column_strip(client, strip_enhanced, col_name)
        if values:
            col_overrides[col_name] = values
            log.info("  Cell-level %s: got %d values", col_name, len(values))
        else:
            log.warning("  Cell-level %s: no values returned", col_name)

        time.sleep(1)  # rate limiting

    # Step 4: Merge — override base row values with cell-level results
    rows = [dict(r) for r in base_rows]  # deep copy
    for col_name, values in col_overrides.items():
        for i, val in enumerate(values):
            if i < len(rows):
                rows[i][col_name] = val

    save_cache("O", page_num, rows)
    return rows


def run_fewshot_ensemble(page_num: int) -> list[dict]:
    """Approach Q: Run 5 few-shot variants of Gemini 2.5 and majority-vote per cell.

    Each variant uses a different subset of GT example rows in the prompt,
    creating diverse "priors" so errors are uncorrelated across variants.
    Majority vote picks the value agreed upon by 3+ of 5 variants.
    Falls back to variant 1 (= M) on ties.
    """
    cached = load_cache("Q", page_num)
    if cached is not None:
        return cached

    client = _gemini_client()
    img = Image.open(page_image_path(page_num))

    # Run each variant (check sub-caches to avoid re-running)
    all_variant_rows: list[list[dict]] = []
    for vi, fs_text in enumerate(FEW_SHOT_VARIANTS):
        sub_key = f"Q_v{vi+1}"
        sub_cached = load_cache(sub_key, page_num)
        if sub_cached is not None:
            log.info("  Variant %d: cache hit (%d rows)", vi+1, len(sub_cached))
            all_variant_rows.append(sub_cached)
            continue

        prompt = (OCR_PROMPT_FULL + "\n\n" + fs_text).strip()
        raw = _gemini_ocr(client, GEMINI_25_PRO, prompt, [img])
        data = parse_json(raw)
        rows = [normalize_row(r, ALL_DATA_COLS) for r in data.get("rows", [])]
        save_cache(sub_key, page_num, rows)
        log.info("  Variant %d: %d rows extracted", vi+1, len(rows))
        all_variant_rows.append(rows)
        time.sleep(2)  # rate limiting between variants

    if not all_variant_rows or not all_variant_rows[0]:
        save_cache("Q", page_num, [])
        return []

    # Normalization for comparison
    _eastern = "٠١٢٣٤٥٦٧٨٩"
    _western = "0123456789"
    _e2w = str.maketrans(_eastern, _western)
    _ditto_set = {"\u05F4", "\u201C", "\u201D", ",,", '"', "\u3003"}

    def _norm(v: str) -> str:
        v = v.strip().translate(_e2w)
        if v in _ditto_set:
            v = '"'
        return v

    # Index all variants by Serial_No
    def _index_by_sno(rows):
        idx = {}
        for r in rows:
            sno = _norm(r.get("Serial_No", ""))
            if sno:
                idx[sno] = r
        return idx

    variant_indices = [_index_by_sno(rows) for rows in all_variant_rows]

    # Use variant 1 (= M) as the base row structure
    data_cols = LEFT_COLS + RIGHT_COLS
    merged = []

    for base_row in all_variant_rows[0]:
        sno = _norm(base_row.get("Serial_No", ""))
        new_row = dict(base_row)

        for col in data_cols:
            # Collect all variant values for this cell
            values = []
            raw_values = {}  # norm_val → original string
            for vi, v_idx in enumerate(variant_indices):
                row = v_idx.get(sno, {})
                raw_val = row.get(col, "")
                norm_val = _norm(raw_val)
                values.append(norm_val)
                if norm_val not in raw_values:
                    raw_values[norm_val] = raw_val

            # Majority vote: find most common value
            from collections import Counter
            counts = Counter(values)
            if counts:
                winner, count = counts.most_common(1)[0]
                if count >= 3:  # 3+ of 5 agree
                    new_row[col] = raw_values.get(winner, base_row.get(col, ""))
                # else: keep variant 1 (base) value

        merged.append(new_row)

    save_cache("Q", page_num, merged)
    return merged


def run_majority_vote_ensemble(page_num: int) -> list[dict]:
    """Approach P: Cell-level majority vote of M (Gemini 2.5 few-shot),
    C (Gemini 2.5 base), and E (Gemini 3.x).

    For each cell: if 2+ models agree, use that value. Otherwise use M (best single).
    Alignment is by Serial_No to handle different row counts.
    """
    cached = load_cache("P", page_num)
    if cached is not None:
        return cached

    rows_m = run_gemini25_full_fewshot(page_num)   # M
    rows_c = _run_gemini_full("C", GEMINI_25_PRO, page_num)  # C
    rows_e = _run_gemini_full("E", GEMINI_3X_PRO, page_num)  # E

    if not rows_m:
        save_cache("P", page_num, [])
        return []

    # Normalize function for comparison
    _eastern = "٠١٢٣٤٥٦٧٨٩"
    _western = "0123456789"
    _e2w = str.maketrans(_eastern, _western)
    ditto_set = {"\u05F4", "\u201C", "\u201D", ",,", '"', "\u3003"}

    def _norm(v: str) -> str:
        v = v.strip().translate(_e2w)
        if v in ditto_set:
            v = '"'
        return v

    # Index C and E by Serial_No for alignment
    def _index_by_sno(rows):
        idx = {}
        for r in rows:
            sno = _norm(r.get("Serial_No", ""))
            if sno:
                idx[sno] = r
        return idx

    c_idx = _index_by_sno(rows_c)
    e_idx = _index_by_sno(rows_e)

    data_cols = LEFT_COLS + RIGHT_COLS
    merged = []
    for row_m in rows_m:
        sno = _norm(row_m.get("Serial_No", ""))
        row_c = c_idx.get(sno, {})
        row_e = e_idx.get(sno, {})

        new_row = dict(row_m)  # start from M (best single model)
        for col in data_cols:
            vm = _norm(row_m.get(col, ""))
            vc = _norm(row_c.get(col, ""))
            ve = _norm(row_e.get(col, ""))

            # Majority vote: if 2+ agree on a non-empty value, use it
            if vm == vc or vm == ve:
                # M agrees with at least one other — keep M's value (already set)
                pass
            elif vc == ve and vc:
                # C and E agree but differ from M — use C/E's value
                new_row[col] = row_c.get(col, "") or row_e.get(col, "")
            # else: all three disagree — keep M (best single model)

        merged.append(new_row)

    save_cache("P", page_num, merged)
    return merged


def run_claude_full_fewshot(page_num: int) -> list[dict]:
    """Approach N: Claude Opus 4.6, full page, with few-shot examples."""
    cached = load_cache("N", page_num)
    if cached is not None:
        return cached

    client = _claude_client()
    img_path = page_image_path(page_num)
    b64 = load_b64(img_path)
    raw = _claude_ocr(client, OCR_PROMPT_FULL_FEWSHOT, [b64])
    data = parse_json(raw)
    rows = [normalize_row(r, ALL_DATA_COLS) for r in data.get("rows", [])]
    save_cache("N", page_num, rows)
    return rows


# ──────────────────────────────────────────────────────────
# ROW MERGE (left + right bands)
# ──────────────────────────────────────────────────────────

def merge_left_right(left_rows: list[dict], right_rows: list[dict]) -> list[dict]:
    """
    Align left-table rows with right-table rows by index.
    If counts differ, extend the shorter side with empty rows.
    """
    n = max(len(left_rows), len(right_rows))
    empty_left  = {c: "" for c in LEFT_COLS + META_COLS}
    empty_right = {c: "" for c in RIGHT_COLS}

    merged = []
    for i in range(n):
        l = left_rows[i]  if i < len(left_rows)  else empty_left.copy()
        r = right_rows[i] if i < len(right_rows) else empty_right.copy()
        row = {**l, **r}
        merged.append(normalize_row(row, ALL_DATA_COLS))
    return merged


# ──────────────────────────────────────────────────────────
# OPENCV TABLE DETECTION
# ──────────────────────────────────────────────────────────

def detect_grid(img_path: Path, side: str = "left") -> tuple:
    """
    Detect table grid lines using Sobel gradient peaks.

    The pre-printed form lines in these scanned documents are too faint for
    simple thresholding, so we use the Sobel Y-gradient profile: each horizontal
    line creates a peak in row-mean gradient magnitude.  Vertical columns are
    detected similarly on the X-gradient.  Falls back to a uniform grid if
    fewer than 5 lines are detected.

    Returns: (h_lines_y, v_lines_x, table_img_bgr, x_offset_in_full_image)
    """
    try:
        from scipy.signal import find_peaks as sp_find_peaks
        from scipy.ndimage import uniform_filter1d
        _have_scipy = True
    except ImportError:
        _have_scipy = False

    img_cv = cv2.imread(str(img_path))
    H, W = img_cv.shape[:2]

    if side == "left":
        x0, x1 = 0, int(W * LEFT_TABLE_WIDTH_FRAC)
    else:
        x0, x1 = int(W * LEFT_TABLE_WIDTH_FRAC), W

    table_img = img_cv[int(H * HEADER_HEIGHT_FRAC):H, x0:x1]
    th, tw = table_img.shape[:2]

    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)

    # ── Horizontal lines via Sobel Y-gradient ──────────────
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_grad = np.mean(np.abs(sobel_y), axis=1)

    if _have_scipy:
        smoothed_h = uniform_filter1d(row_grad, size=5)
        h_peaks, _ = sp_find_peaks(
            smoothed_h,
            height=smoothed_h.mean() * 1.3,
            distance=max(60, th // 50),
        )
        h_lines_y = h_peaks.tolist()
    else:
        h_lines_y = _find_peaks_simple(row_grad, min_gap=max(60, th // 50))

    # ── Vertical lines via Sobel X-gradient ───────────────
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    col_grad = np.mean(np.abs(sobel_x), axis=0)

    if _have_scipy:
        smoothed_v = uniform_filter1d(col_grad, size=5)
        v_peaks, _ = sp_find_peaks(
            smoothed_v,
            height=smoothed_v.mean() * 1.5,
            distance=max(30, tw // 30),
        )
        v_lines_x = v_peaks.tolist()
    else:
        v_lines_x = _find_peaks_simple(col_grad, min_gap=max(30, tw // 30))

    # ── Fallback: uniform grid if detection is poor ────────
    EXPECTED_ROWS = 32
    if len(h_lines_y) < 5:
        log.warning("detect_grid: only %d h-lines detected; using uniform grid", len(h_lines_y))
        row_h = th // EXPECTED_ROWS
        h_lines_y = [i * row_h for i in range(EXPECTED_ROWS + 1)]

    EXPECTED_LEFT_COLS  = len(LEFT_COLS)
    EXPECTED_RIGHT_COLS = len(RIGHT_COLS)
    expected_cols = EXPECTED_LEFT_COLS if side == "left" else EXPECTED_RIGHT_COLS
    if len(v_lines_x) < 3:
        log.warning("detect_grid: only %d v-lines detected; using uniform grid", len(v_lines_x))
        col_w = tw // expected_cols
        v_lines_x = [i * col_w for i in range(expected_cols + 1)]

    # ── Ensure edges are included ──────────────────────────
    if not h_lines_y or h_lines_y[0] > 30:
        h_lines_y = [0] + h_lines_y
    if h_lines_y[-1] < th - 30:
        h_lines_y.append(th)

    if not v_lines_x or v_lines_x[0] > 30:
        v_lines_x = [0] + v_lines_x
    if v_lines_x[-1] < tw - 30:
        v_lines_x.append(tw)

    return sorted(set(h_lines_y)), sorted(set(v_lines_x)), table_img, x0


def _find_peaks_simple(arr: np.ndarray, min_gap: int = 60) -> list[int]:
    """Fallback peak finder (no scipy required)."""
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
            in_peak = False
            mid = (peak_start + i) // 2
            if not positions or mid - positions[-1] >= min_gap:
                positions.append(mid)
    if in_peak:
        mid = (peak_start + len(arr)) // 2
        if not positions or mid - positions[-1] >= min_gap:
            positions.append(mid)
    return positions


def crop_cells(table_img: np.ndarray, h_lines: list[int], v_lines: list[int]) -> list[list]:
    """
    Crop individual cells from table image given row/col boundaries.
    Returns 2D list of PIL images: cells[row][col].
    """
    cells = []
    pad = 4  # pixels padding to avoid cutting into text
    for r in range(len(h_lines) - 1):
        row_cells = []
        y0 = max(0, h_lines[r] + pad)
        y1 = min(table_img.shape[0], h_lines[r + 1] - pad)
        if y1 - y0 < 5:
            continue
        for c in range(len(v_lines) - 1):
            x0 = max(0, v_lines[c] + pad)
            x1 = min(table_img.shape[1], v_lines[c + 1] - pad)
            if x1 - x0 < 5:
                row_cells.append(None)
                continue
            cell_cv = table_img[y0:y1, x0:x1]
            cell_pil = Image.fromarray(cv2.cvtColor(cell_cv, cv2.COLOR_BGR2RGB))
            row_cells.append(cell_pil)
        if row_cells:
            cells.append(row_cells)
    return cells


def crop_column_strips(table_img: np.ndarray, v_lines: list[int]) -> list:
    """Crop each column as a full-height PIL image strip."""
    strips = []
    for c in range(len(v_lines) - 1):
        x0, x1 = v_lines[c], v_lines[c + 1]
        col_cv = table_img[:, x0:x1]
        strips.append(Image.fromarray(cv2.cvtColor(col_cv, cv2.COLOR_BGR2RGB)))
    return strips


# ──────────────────────────────────────────────────────────
# KRAKEN OCR
# ──────────────────────────────────────────────────────────

def _kraken_recognize_cell(cell_img: Image.Image, model_path: str) -> str:
    """Recognize text in a single cell (no-segmentation mode)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = os.path.join(tmpdir, "cell.jpg")
        out_path = os.path.join(tmpdir, "cell.txt")
        cell_img.save(in_path, format="JPEG", quality=95)
        cmd = [
            KRAKEN_BIN, "-i", in_path, out_path,
            "binarize", "ocr", "-s", "-m", model_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not os.path.exists(out_path):
            return ""
        return open(out_path).read().strip()


def _kraken_recognize_strip(strip_img: Image.Image, model_path: str) -> list[tuple]:
    """
    Recognize text in a column strip (auto-segmentation).
    Returns list of (y_center, text) tuples for alignment.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = os.path.join(tmpdir, "strip.jpg")
        out_path = os.path.join(tmpdir, "strip.alto")
        strip_img.save(in_path, format="JPEG", quality=95)
        cmd = [
            KRAKEN_BIN, "-i", in_path, out_path,
            "binarize", "segment", "ocr", "-m", model_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        lines = []
        if result.returncode == 0 and os.path.exists(out_path):
            lines = _parse_alto(out_path)
        if not lines:
            # Fallback: try plain text output
            out_txt = out_path.replace(".alto", ".txt")
            cmd2 = [
                KRAKEN_BIN, "-i", in_path, out_txt,
                "binarize", "segment", "ocr", "-m", model_path,
            ]
            subprocess.run(cmd2, capture_output=True, timeout=60)
            if os.path.exists(out_txt):
                text_lines = open(out_txt).read().strip().splitlines()
                h = strip_img.height
                step = h // max(len(text_lines), 1)
                lines = [(step // 2 + i * step, t) for i, t in enumerate(text_lines) if t.strip()]
        return lines


def _parse_alto(alto_path: str) -> list[tuple]:
    """Parse ALTO XML output, return list of (y_center, text) tuples."""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(alto_path)
        ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
        lines = []
        for el in tree.iter():
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if tag == "TextLine":
                vpos  = int(el.get("VPOS",  0))
                height = int(el.get("HEIGHT", 1))
                y_ctr = vpos + height // 2
                text_parts = []
                for child in el.iter():
                    ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if ctag == "String":
                        text_parts.append(child.get("CONTENT", ""))
                lines.append((y_ctr, " ".join(text_parts)))
        return lines
    except Exception:
        return []


def run_kraken_cells(approach: str, model_path: str, page_num: int) -> list[dict]:
    """Approach G or H: Kraken OCR on cell-level crops."""
    cached = load_cache(approach, page_num)
    if cached is not None:
        return cached

    img_path = page_image_path(page_num)
    rows = []

    # Left table
    h_lines_l, v_lines_l, table_l, _ = detect_grid(img_path, "left")
    cells_l = crop_cells(table_l, h_lines_l, v_lines_l)
    n_cols_l = len(LEFT_COLS)

    # Right table
    h_lines_r, v_lines_r, table_r, _ = detect_grid(img_path, "right")
    cells_r = crop_cells(table_r, h_lines_r, v_lines_r)
    n_cols_r = len(RIGHT_COLS)

    n_rows = max(len(cells_l), len(cells_r))
    for r_idx in range(n_rows):
        row = {c: "" for c in ALL_DATA_COLS}

        if r_idx < len(cells_l):
            for c_idx, col in enumerate(LEFT_COLS):
                if c_idx < len(cells_l[r_idx]) and cells_l[r_idx][c_idx] is not None:
                    row[col] = _kraken_recognize_cell(cells_l[r_idx][c_idx], model_path)

        if r_idx < len(cells_r):
            for c_idx, col in enumerate(RIGHT_COLS):
                if c_idx < len(cells_r[r_idx]) and cells_r[r_idx][c_idx] is not None:
                    row[col] = _kraken_recognize_cell(cells_r[r_idx][c_idx], model_path)

        if any(row[c] for c in LEFT_COLS + RIGHT_COLS):
            rows.append(row)

    save_cache(approach, page_num, rows)
    return rows


def run_kraken_columns(approach: str, model_path: str, page_num: int) -> list[dict]:
    """Approach I or J: Kraken OCR on column strips with y-coordinate alignment."""
    cached = load_cache(approach, page_num)
    if cached is not None:
        return cached

    img_path = page_image_path(page_num)

    # Left table column strips
    h_lines_l, v_lines_l, table_l, _ = detect_grid(img_path, "left")
    strips_l = crop_column_strips(table_l, v_lines_l)

    # Right table column strips
    h_lines_r, v_lines_r, table_r, _ = detect_grid(img_path, "right")
    strips_r = crop_column_strips(table_r, v_lines_r)

    # Recognize each strip
    left_col_lines: list[list[tuple]] = []
    for strip in strips_l[:len(LEFT_COLS)]:
        left_col_lines.append(_kraken_recognize_strip(strip, model_path))

    right_col_lines: list[list[tuple]] = []
    for strip in strips_r[:len(RIGHT_COLS)]:
        right_col_lines.append(_kraken_recognize_strip(strip, model_path))

    # Determine row boundaries from horizontal grid lines
    row_y_centers = [
        (h_lines_l[i] + h_lines_l[i + 1]) // 2
        for i in range(len(h_lines_l) - 1)
    ]
    if not row_y_centers:
        save_cache(approach, page_num, [])
        return []

    rows = []
    for r_idx, y_ctr in enumerate(row_y_centers):
        row = {c: "" for c in ALL_DATA_COLS}
        y_lo = h_lines_l[r_idx] if r_idx < len(h_lines_l) else 0
        y_hi = h_lines_l[r_idx + 1] if r_idx + 1 < len(h_lines_l) else table_l.shape[0]

        for c_idx, col in enumerate(LEFT_COLS):
            if c_idx < len(left_col_lines):
                row[col] = _get_line_in_band(left_col_lines[c_idx], y_lo, y_hi)

        for c_idx, col in enumerate(RIGHT_COLS):
            if c_idx < len(right_col_lines):
                row[col] = _get_line_in_band(right_col_lines[c_idx], y_lo, y_hi)

        if any(row[c] for c in LEFT_COLS + RIGHT_COLS):
            rows.append(row)

    save_cache(approach, page_num, rows)
    return rows


def _get_line_in_band(lines: list[tuple], y_lo: int, y_hi: int) -> str:
    """Find the text line whose y_center falls within [y_lo, y_hi]."""
    for y_ctr, text in lines:
        if y_lo <= y_ctr < y_hi:
            return text
    return ""


# ──────────────────────────────────────────────────────────
# APPROACH DISPATCH TABLE
# ──────────────────────────────────────────────────────────

def run_approach(approach: str, page_num: int) -> list[dict]:
    """Run a single approach on a single page, returning list of row dicts."""
    log.info("Running approach %s on page %d ...", approach, page_num)
    try:
        if   approach == "A": return run_claude_full(page_num)
        elif approach == "B": return run_claude_zoomed(page_num)
        elif approach == "C": return run_gemini25_full(page_num)
        elif approach == "D": return run_gemini25_zoomed(page_num)
        elif approach == "E": return run_gemini3x_full(page_num)
        elif approach == "F": return run_gemini3x_zoomed(page_num)
        elif approach == "G": return run_kraken_cells("G",   KRAKEN_USER_MODEL,    page_num)
        elif approach == "H": return run_kraken_cells("H",   KRAKEN_DEFAULT_MODEL, page_num)
        elif approach == "I": return run_kraken_columns("I", KRAKEN_USER_MODEL,    page_num)
        elif approach == "J": return run_kraken_columns("J", KRAKEN_DEFAULT_MODEL, page_num)
        elif approach == "K": return run_gemini3f_full(page_num)
        elif approach == "L": return run_gemini3f_zoomed(page_num)
        elif approach == "M": return run_gemini25_full_fewshot(page_num)
        elif approach == "N": return run_claude_full_fewshot(page_num)
        elif approach == "O": return run_gemini25_cell_hybrid(page_num)
        elif approach == "P": return run_majority_vote_ensemble(page_num)
        elif approach == "Q": return run_fewshot_ensemble(page_num)
        else:
            log.error("Unknown approach: %s", approach)
            return []
    except EnvironmentError as e:
        log.warning("Skipping approach %s (env error): %s", approach, e)
        return []
    except Exception as e:
        log.error("Approach %s page %d failed: %s", approach, page_num, e, exc_info=True)
        return []


# ──────────────────────────────────────────────────────────
# MULTI-MODEL ENSEMBLE
# ──────────────────────────────────────────────────────────

# All Unicode/typographic variants that mean "ditto mark" in this context
_DITTO_VARIANTS = (
    "\u05F4",  # ״  Hebrew punctuation Gershayim
    "\u201C",  # "  Left double quotation mark
    "\u201D",  # "  Right double quotation mark
    ",,",      # two commas used as ditto
)
_DITTO_CANONICAL = '"'  # normalize all ditto variants to plain ASCII quote


def _norm(val: str) -> str:
    """Normalize ditto-mark variants to canonical ASCII quote for comparison."""
    for d in _DITTO_VARIANTS:
        val = val.replace(d, _DITTO_CANONICAL)
    return val


def run_ensemble(primary: str, secondary: str, page_num: int,
                 tiebreaker: Optional[str] = None) -> list[dict]:
    """
    Run two (or three) approaches and merge by cell-level comparison.

    For each cell in each row:
      - primary == secondary → use that value
      - differ, no tiebreaker → annotate [DISAGREE: P=x|S=y]
      - differ, tiebreaker agrees with primary → use primary
      - differ, tiebreaker agrees with secondary → use secondary
      - differ, tiebreaker matches neither (or is empty) → annotate [DISAGREE: P=x|S=y]

    Sets "Disagreements" meta field to semicolon-separated list of column names
    that had unresolved disagreements.
    Rows are aligned positionally (by index).
    """
    rows_p = run_approach(primary,    page_num)
    rows_s = run_approach(secondary,  page_num)
    rows_t = run_approach(tiebreaker, page_num) if tiebreaker else []

    n = max(len(rows_p), len(rows_s), len(rows_t) if rows_t else 0)
    if n == 0:
        return []

    empty = {c: "" for c in ALL_DATA_COLS}
    data_cols = LEFT_COLS + RIGHT_COLS  # only compare data columns, not meta

    merged = []
    for i in range(n):
        rp = rows_p[i] if i < len(rows_p) else empty.copy()
        rs = rows_s[i] if i < len(rows_s) else empty.copy()
        rt = rows_t[i] if rows_t and i < len(rows_t) else empty.copy()

        row: dict = {}
        disagree_cols: list[str] = []

        for col in data_cols:
            vp_raw = rp.get(col, "").strip()
            vs_raw = rs.get(col, "").strip()
            vt_raw = rt.get(col, "").strip()
            # Normalize ditto variants before comparing; output from primary is canonical
            vp = _norm(vp_raw)
            vs = _norm(vs_raw)
            vt = _norm(vt_raw)

            if vp == vs:
                row[col] = vp_raw  # use primary's original representation
            elif tiebreaker and vt:
                if vt == vp:
                    row[col] = vp_raw
                elif vt == vs:
                    row[col] = vs_raw
                else:
                    row[col] = f"[DISAGREE: {primary}={vp_raw}|{secondary}={vs_raw}]"
                    disagree_cols.append(col)
            else:
                row[col] = f"[DISAGREE: {primary}={vp_raw}|{secondary}={vs_raw}]"
                disagree_cols.append(col)

        # Meta columns: prefer primary; if both empty use secondary
        row["Row_Confidence"] = rp.get("Row_Confidence", "") or rs.get("Row_Confidence", "")
        row["Red_Ink"]        = rp.get("Red_Ink", "")        or rs.get("Red_Ink", "")
        row["Disagreements"]  = "; ".join(disagree_cols)

        merged.append(normalize_row(row, ALL_DATA_COLS))

    return merged


# ──────────────────────────────────────────────────────────
# COMPARISON OUTPUT
# ──────────────────────────────────────────────────────────

ALL_APPROACHES = list("ABCDEFGHIJKLMNOPQ")


def save_comparison(results: dict[str, list[dict]], page_num: int):
    """
    Save a long-format comparison CSV:
      columns: Page_Number, Approach, Row_Index, <all data cols>
    """
    out_path = PROJECT_DIR / COMPARISON_TEMPLATE.format(page=page_num)
    fieldnames = ["Page_Number", "Approach", "Row_Index"] + ALL_DATA_COLS

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for approach, rows in results.items():
            for idx, row in enumerate(rows):
                out_row = {"Page_Number": page_num, "Approach": approach, "Row_Index": idx + 1}
                out_row.update(normalize_row(row, ALL_DATA_COLS))
                writer.writerow(out_row)

    log.info("Saved %s", out_path)


# ──────────────────────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────────────────────

def _cer(pred: str, ref: str) -> float:
    """Character Error Rate (edit distance / len(ref))."""
    if not ref:
        return 0.0 if not pred else 1.0
    import difflib
    ops = difflib.SequenceMatcher(None, pred, ref).get_opcodes()
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    return edits / len(ref)


def score_all(ground_truth_csv: Path):
    """Score all cached approach results against ground truth CSV."""
    import csv as csv_mod

    # Load ground truth
    gt_rows: dict[tuple, dict] = {}
    with open(ground_truth_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv_mod.DictReader(f):
            page = int(row["Page_Number"])
            sno  = row.get("Serial_No", "").strip()
            if sno:
                gt_rows[(page, sno)] = row

    results = []
    for approach in ALL_APPROACHES:
        for page_num in TEST_PAGES:
            rows = load_cache(approach, page_num)
            if rows is None:
                continue
            total_cells = 0
            total_cer   = 0.0
            exact_matches = 0
            compared_cells = 0
            for row in rows:
                sno = row.get("Serial_No", "").strip()
                key = (page_num, sno)
                gt  = gt_rows.get(key)
                if not gt:
                    continue
                for col in LEFT_COLS + RIGHT_COLS:
                    pred = row.get(col, "").strip()
                    ref  = gt.get(col, "").strip()
                    if ref == "":
                        continue
                    compared_cells += 1
                    cer = _cer(pred, ref)
                    total_cer += cer
                    if pred == ref:
                        exact_matches += 1

            if compared_cells > 0:
                results.append({
                    "Approach": approach,
                    "Page": page_num,
                    "Cells_Compared": compared_cells,
                    "Exact_Match_Rate": f"{exact_matches / compared_cells:.3f}",
                    "Mean_CER": f"{total_cer / compared_cells:.3f}",
                })

    if not results:
        log.warning("No scoring data found. Run approaches first, then fill ground truth.")
        return

    out_path = PROJECT_DIR / SCORES_FILE
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Approach", "Page", "Cells_Compared",
                                                "Exact_Match_Rate", "Mean_CER"])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print(f"\n{'Approach':<10} {'Page':<6} {'Cells':<8} {'Exact%':<10} {'CER'}")
    print("-" * 45)
    for r in sorted(results, key=lambda x: float(x["Mean_CER"])):
        print(f"{r['Approach']:<10} {r['Page']:<6} {r['Cells_Compared']:<8} "
              f"{float(r['Exact_Match_Rate'])*100:>6.1f}%   {r['Mean_CER']}")
    log.info("Scores saved to %s", out_path)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages",     nargs="+", type=int, default=TEST_PAGES,
                        help="Page numbers to process (default: 3 10 50)")
    parser.add_argument("--approaches", nargs="+", default=ALL_APPROACHES,
                        choices=ALL_APPROACHES, metavar="APPROACH",
                        help="Approaches to run: A B C D E F G H I J")
    parser.add_argument("--score",     action="store_true",
                        help="Score cached results against ground_truth_template.csv")
    parser.add_argument("--no-cache",  action="store_true",
                        help="Ignore cached results and re-run")
    args = parser.parse_args()

    if args.no_cache:
        import shutil
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        CACHE_DIR.mkdir(exist_ok=True)

    if args.score:
        gt_path = PROJECT_DIR / "ground_truth_template.csv"
        if not gt_path.exists():
            log.error("Ground truth file not found: %s", gt_path)
            sys.exit(1)
        score_all(gt_path)
        return

    for page_num in args.pages:
        img_path = page_image_path(page_num)
        if not img_path.exists():
            log.error("Image not found: %s", img_path)
            continue

        page_results: dict[str, list[dict]] = {}
        for approach in args.approaches:
            rows = run_approach(approach, page_num)
            page_results[approach] = rows
            log.info("  → %s rows extracted", len(rows))

        save_comparison(page_results, page_num)

    log.info("Done. Comparison CSVs saved in %s", PROJECT_DIR)


if __name__ == "__main__":
    main()
