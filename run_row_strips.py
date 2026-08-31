"""E5: one call per INKED row, on a full-width single-row crop.

The row address is known before the call, so the model never counts rows. Full
horizontal context is kept (unlike the rejected vertical column strips), and the
previous row is included above, dimmed, so ditto marks resolve.

Rows with no ink (per gate_textlines.hand_mask) are skipped entirely, which is what
makes this affordable on sparse pages.

  python run_row_strips.py 9 --model gemini-3.7-flash
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

import run_g3v6_local as base
from gate_textlines import hand_mask
from run_exp2608 import FINAL2, OUT_DIR, RUNS_CSV, estimate_cost
from run_som_ocr import COLS_DOC
from digit_norm import LEFT_COLS

CELL_RX = re.compile(
    r'<TableCell id="cell_r(\d+)_c(\d+)"[^>]*>\s*<Coords points="([^"]+)"')
INK_MIN_PX = 12

PROMPT = f"""The image shows ONE row of a handwritten Arabic tax register (British
Mandate Palestine, 1930s-40s), cropped full width. The row to transcribe is the one
below the horizontal red line; anything above the line is the PREVIOUS row, shown dimmed
for context only (use it to resolve ditto marks, but do not transcribe it).

{COLS_DOC}

Return `cells`: an object mapping the column INDEX as a string ("0".."18") to the text
written in that cell of THIS row. Include only cells that contain handwriting.

Transcription rules:
  * Keep Arabic-Indic digits as written. Do not convert to Latin digits.
  * Check mark -> the single character U+2713
  * Ditto mark (repeats the cell above) -> a single ASCII double-quote
  * Nil / dash -> a single hyphen
  * Thousands separator inside a number -> a comma"""

SCHEMA = {
    "type": "object",
    "properties": {
        "cells": {
            "type": "object",
            "properties": {str(i): {"type": "string"} for i in range(len(LEFT_COLS) - 1)},
        }
    },
    "required": ["cells"],
}


def row_geometry(xml_path: Path):
    xml = xml_path.read_text(encoding="utf-8")
    acc: dict[int, list] = {}
    for m in CELL_RX.finditer(xml):
        pts = np.array([[int(a) for a in p.split(",")] for p in m.group(3).split()])
        acc.setdefault(int(m.group(1)), []).append(pts)
    bands = {}
    for r, polys in acc.items():
        allp = np.vstack(polys)
        bands[r] = (int(allp[:, 1].min()), int(allp[:, 1].max()))
    return bands, acc


def inked_rows(img, bands, cell_polys) -> set[int]:
    mask = hand_mask(img)
    out = set()
    for r, polys in cell_polys.items():
        tot = 0
        for poly in polys:
            cm = np.zeros(mask.shape, np.uint8)
            cv2.fillPoly(cm, [poly.astype(np.int32)], 1)
            tot += int((mask & cm).sum())
            if tot >= INK_MIN_PX:
                break
        if tot >= INK_MIN_PX:
            out.add(r)
    return out


def make_strip(img, bands, r: int, pad: int = 8):
    h, w = img.shape[:2]
    y0, y1 = bands[r]
    y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
    prev = bands.get(r - 1)
    if prev:
        py0 = max(0, prev[0] - pad)
        ctx = img[py0:y0].copy()
        ctx = cv2.addWeighted(ctx, 0.45, np.full_like(ctx, 255), 0.55, 0)
        strip = np.vstack([ctx, img[y0:y1]])
        line_y = ctx.shape[0]
    else:
        strip = img[y0:y1].copy()
        line_y = 0
    cv2.line(strip, (0, line_y), (strip.shape[1], line_y), (0, 0, 255), 3)
    return strip


def run_page(page: int, model: str, tag: str, thinking: str, max_rows: int | None):
    img_p = FINAL2 / f"Hadita_{page}.jpeg"
    xml_p = FINAL2 / f"Hadita_{page}.xml"
    if not img_p.exists() or not xml_p.exists():
        print(f"  ! page {page}: missing inputs")
        return None
    img = cv2.imread(str(img_p))
    bands, cell_polys = row_geometry(xml_p)
    todo = sorted(inked_rows(img, bands, cell_polys))
    if max_rows:
        todo = todo[:max_rows]
    n_grid = max(bands) + 1
    print(f"    {len(todo)}/{n_grid} rows have ink -> {len(todo)} calls")

    cfg = types.GenerateContentConfig(
        max_output_tokens=4096, response_mime_type="application/json",
        response_schema=SCHEMA)
    if thinking != "none":
        cfg.thinking_config = types.ThinkingConfig(
            thinking_budget=base.THINKING_BUDGETS[thinking])

    client = genai.Client()
    grid = [{c: "" for c in LEFT_COLS} for _ in range(n_grid)]
    cost = 0.0
    t0 = time.perf_counter()
    ok = 0
    for r in todo:
        strip = make_strip(img, bands, r)
        buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=buf, mime_type="image/jpeg"), PROMPT],
                config=cfg)
        except Exception as e:
            print(f"      row {r}: error {str(e)[:80]}")
            continue
        if resp.usage_metadata:
            cost += estimate_cost(model, resp.usage_metadata)
        if not resp.text:
            continue
        try:
            cells = json.loads(resp.text).get("cells", {})
        except json.JSONDecodeError:
            continue
        for k, v in cells.items():
            try:
                ci = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= ci < len(LEFT_COLS) and (v or "").strip():
                grid[r][LEFT_COLS[ci]] = v.strip()
        ok += 1
    elapsed = round(time.perf_counter() - t0, 2)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"Hadita_{page}_{tag}.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    patched, n_p, n_s = base.patch_xml(xml_p.read_text(encoding="utf-8"), grid, LEFT_COLS)
    (OUT_DIR / f"Hadita_{page}_{tag}.xml").write_text(patched, encoding="utf-8")
    nonempty = sum(1 for r in grid if any(v.strip() for v in r.values()))
    print(f"    ok {ok}/{len(todo)} rows read; {nonempty} non-empty rows; "
          f"{elapsed}s ${cost:.4f}")

    rec = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), page=page, model=model, tag=tag,
               thinking=thinking, rows=nonempty, grid_rows=n_grid, patched=n_p,
               skipped=n_s, seconds=elapsed, prompt_tokens=0, output_tokens=0,
               thought_tokens=0, cost_usd=round(cost, 5))
    new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rec))
        if new:
            wr.writeheader()
        wr.writerow(rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--thinking", default="none",
                    choices=["none", "low", "medium", "high"])
    ap.add_argument("--max-rows", type=int, default=None, help="cap calls (debug)")
    args = ap.parse_args()
    tag = args.tag or "strip-" + args.model.replace("gemini-", "g").replace("-preview", "").replace(".", "")
    total = 0.0
    for p in args.pages:
        print(f"  page {p}:")
        rec = run_page(p, args.model, tag, args.thinking, args.max_rows)
        if rec:
            total += rec["cost_usd"]
    print(f"total: ${total:.4f}")


if __name__ == "__main__":
    main()
