"""E3: Set-of-Marks OCR — row indices are PRINTED on the image, so the model reads
a row number instead of counting rows.

Requires `python make_som_pages.py N` first (writes som/Hadita_{N}_som.jpg).

  python run_som_ocr.py 9 10 --model gemini-3.7-flash
Writes exp2608/Hadita_{N}_{tag}.json (+ .xml), appends g3_runs_exp2608.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from google import genai
from google.genai import types

import run_g3v6_local as base
from run_exp2608 import FINAL2, OUT_DIR, RUNS_CSV, estimate_cost
from digit_norm import LEFT_COLS

SOM_DIR = Path("som")
# --som-dir/--xml-dir point the run at a later geometry generation (final3)
# without copying this file; the som/ + final2 defaults are unchanged.
XML_DIR = FINAL2

COLS_DOC = """Column order, RIGHT to LEFT as printed on the page (index: name):
 0 Serial_No
 1 Date
 2 Property_recorded_under_Block_No
 3 Property_recorded_under_Parcel_No
 4 Parcel_Cat_No
 5 Parcel_Area
 6 Nature_of_Entry
 7 New_Serial_No
 8 Reference_to_Register_of_Changes_Volume_No
 9 Reference_to_Register_of_Changes_Serial_No
10 Tax_LP
11 Tax_Mils
12 Total_Tax_LP
13 Total_Tax_Mils
14 Reference_to_Register_of_Exemptions_Entry_No
15 Reference_to_Register_of_Exemptions_Amount_LP
16 Reference_to_Register_of_Exemptions_Amount_Mils
17 Net_Assessment_LP
18 Net_Assessment_Mils"""

PROMPT = f"""This image is a page from a handwritten Arabic tax register (British Mandate
Palestine, 1930s-40s), with a RED ROW NUMBER printed in the margin beside every table
row and a yellow line marking each row boundary. These marks were added by us; they are
not part of the original document.

{COLS_DOC}

Task: for EVERY row that contains any handwriting, output one entry keyed by the RED
ROW NUMBER printed beside it. Do not renumber, do not count rows yourself, and do not
skip a number: read the printed number next to the row you are transcribing.

Output one object per non-empty row:
  row   - the printed red row number (integer)
  cells - object mapping the column INDEX (as a string, e.g. "0", "5") to the text in
          that cell. Include only cells that contain handwriting.

Transcription rules:
  * Keep Arabic-Indic digits as written. Do not convert to Latin digits.
  * Check mark -> the single character U+2713
  * Ditto mark (same as the cell above) -> a single ASCII double-quote
  * Nil / dash -> a single hyphen
  * Thousands separator inside a number -> a comma

IMPORTANT — do not skip sparse rows. Many rows carry only one or two marks, and some
carry nothing but a single dash or a single check mark in one column. Such a row is a
real row and MUST be reported with that one cell filled. Before finishing, scan the red
row numbers in order and confirm you have not silently passed over any number whose row
carries even a single mark."""

SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row": {"type": "integer"},
                    "cells": {
                        "type": "object",
                        "properties": {str(i): {"type": "string"}
                                       for i in range(len(LEFT_COLS) - 1)},
                    },
                },
                "required": ["row", "cells"],
            },
        }
    },
    "required": ["rows"],
}


def run_page(page: int, model: str, tag: str, thinking: str) -> dict | None:
    som_p = SOM_DIR / f"Hadita_{page}_som.jpg"
    xml_p = XML_DIR / f"Hadita_{page}.xml"
    if not som_p.exists() or not xml_p.exists():
        print(f"  ! page {page}: missing inputs (run make_som_pages.py first)")
        return None

    cfg = types.GenerateContentConfig(
        max_output_tokens=32768,
        response_mime_type="application/json",
        response_schema=SCHEMA)
    if thinking != "none":
        cfg.thinking_config = types.ThinkingConfig(
            thinking_budget=base.THINKING_BUDGETS[thinking])

    client = genai.Client()
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=som_p.read_bytes(), mime_type="image/jpeg"),
                  PROMPT],
        config=cfg)
    elapsed = round(time.perf_counter() - t0, 2)
    if not resp.text:
        print(f"    x no output (finish={resp.candidates[0].finish_reason})")
        return None
    try:
        rows_in = json.loads(resp.text).get("rows", [])
    except json.JSONDecodeError as e:
        print(f"    x bad JSON: {e}")
        return None

    n_grid = base.xml_grid_size(xml_p.read_text(encoding="utf-8"))[0]
    grid = [{c: "" for c in LEFT_COLS} for _ in range(n_grid)]
    placed = out_of_range = 0
    for item in rows_in:
        r = item.get("row")
        if not isinstance(r, int) or not (0 <= r < n_grid):
            out_of_range += 1
            continue
        for k, v in (item.get("cells") or {}).items():
            try:
                ci = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= ci < len(LEFT_COLS) and (v or "").strip():
                grid[r][LEFT_COLS[ci]] = v.strip()
                placed += 1

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"Hadita_{page}_{tag}.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    patched, n_p, n_s = base.patch_xml(xml_p.read_text(encoding="utf-8"), grid, LEFT_COLS)
    (OUT_DIR / f"Hadita_{page}_{tag}.xml").write_text(patched, encoding="utf-8")

    um = resp.usage_metadata
    cost = estimate_cost(model, um) if um else float("nan")
    nonempty = sum(1 for r in grid if any(v.strip() for v in r.values()))
    print(f"    ok {len(rows_in)} rows reported, {placed} cells placed, "
          f"{out_of_range} out-of-range; {nonempty} non-empty rows; "
          f"{elapsed}s ${cost:.4f}")

    rec = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), page=page, model=model, tag=tag,
               thinking=thinking, rows=nonempty, grid_rows=n_grid, patched=n_p,
               skipped=n_s, seconds=elapsed,
               prompt_tokens=getattr(um, "prompt_token_count", 0) if um else 0,
               output_tokens=getattr(um, "candidates_token_count", 0) if um else 0,
               thought_tokens=getattr(um, "thoughts_token_count", 0) if um else 0,
               cost_usd=round(cost, 5))
    new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rec))
        if new:
            wr.writeheader()
        wr.writerow(rec)
    return rec


# E14. Measured on the 4 proxy-GT pages: when this model reads ٢ it is wrong 51%
# of the time, against 5-10% for every other digit; 126 of those are a true ٣.
# The two glyphs ARE distinct in this hand (contact sheets of confirmed cells):
# ٢ is a smooth hook, ٣ carries a pointed apex. The base prompt said nothing
# about letterforms at all, so this describes the distinction rather than
# post-processing the output (no asymmetric postprocess).
DIGIT_HINT = """

CRITICAL — the digits ٢ and ٣ in this hand. These are the scribe's most confusable
pair and the single largest source of transcription error. They are written as two
genuinely different shapes:
  * ٢ is a SMOOTH OPEN CURVE — one continuous arc, like a hook or a rotated "C".
    There is no corner anywhere along the stroke.
  * ٣ has a POINTED APEX — two strokes meeting at a visible sharp vertex, forming
    a small peak. When written hurriedly the peak flattens, but the corner remains.
Decide between them by the presence or absence of that apex, not by size or slant.
A flattened ٣ still has a corner; a ٢ never does. Look carefully at every such digit,
including ones inside longer numbers, where crowding makes the apex easy to miss."""

# Tested separately: justified by the 126-vs-61 imbalance, but a thumb on the scale
# risks the same over-correction that sank the blind ٢->٣ replacement rule.
DIGIT_BIAS = """
When you genuinely cannot tell which of the two it is, prefer ٣."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--thinking", default="low", choices=["none", "low", "medium", "high"])
    ap.add_argument("--som-dir", help="folder of stamped SoM images (default som/)")
    ap.add_argument("--xml-dir", help="geometry folder (default final2)")
    ap.add_argument("--digit-hint", default="off", choices=["off", "shape", "shape+bias"],
                    help="append the 2/3 letterform paragraph (E14). 'shape' describes "
                         "the apex; 'shape+bias' also says to prefer 3 when unsure.")
    args = ap.parse_args()

    global SOM_DIR, XML_DIR, PROMPT
    if args.som_dir:
        SOM_DIR = Path(args.som_dir)
    if args.xml_dir:
        XML_DIR = Path(args.xml_dir)
    if args.digit_hint != "off":
        PROMPT = PROMPT + DIGIT_HINT
        if args.digit_hint == "shape+bias":
            PROMPT = PROMPT + DIGIT_BIAS
    tag = args.tag or "som-" + args.model.replace("gemini-", "g").replace("-preview", "").replace(".", "")
    total = 0.0
    for p in args.pages:
        print(f"  page {p}:")
        rec = run_page(p, args.model, tag, args.thinking)
        if rec:
            total += rec["cost_usd"]
    print(f"total: ${total:.4f}")


if __name__ == "__main__":
    main()
