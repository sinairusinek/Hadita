"""E4: grounded OCR — model returns (text, bbox) items; WE assign them to grid cells.

The model is never asked to count rows or name columns. It only reads ink and says
where it saw it. Row/column assignment is a deterministic geometric join against the
final2 PAGE XML cell polygons, so a sparse or skipped row simply receives no items.

  python run_grounded_ocr.py 9 10 --model gemini-3.7-flash
Writes exp2608/Hadita_{N}_{tag}.json (LEFT_COLS rows) + .xml, appends g3_runs_exp2608.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

import run_g3v6_local as base
from run_exp2608 import FINAL2, OUT_DIR, RUNS_CSV, estimate_cost
from digit_norm import LEFT_COLS

CELL_RX = re.compile(
    r'<TableCell id="cell_r(\d+)_c(\d+)"[^>]*>\s*<Coords points="([^"]+)"')

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ymin": {"type": "integer"},
                    "xmin": {"type": "integer"},
                    "ymax": {"type": "integer"},
                    "xmax": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["ymin", "xmin", "ymax", "xmax", "text"],
            },
        }
    },
    "required": ["items"],
}

PROMPT = """This image is a page from a handwritten Arabic tax register (British Mandate
Palestine, 1930s-40s). The page is a large table.

Detect EVERY handwritten entry inside the table body. Do NOT transcribe the printed
column headers or printed form text. One detection per table cell that contains
handwriting.

For each detection return:
  ymin, xmin, ymax, xmax  - the bounding box, normalized to 0-1000 over the FULL image
  text                    - exactly what is written in that cell

Transcription rules:
  * Keep Arabic-Indic digits as written (٠١٢٣٤٥٦٧٨٩). Do not convert to Latin digits.
  * A check mark is the single character U+2713.
  * A ditto mark (repeat of the cell above) is the single ASCII double-quote character.
  * A dash / nil entry is a single hyphen.
  * Thousands separators inside numbers: use a comma.
  * If a cell is empty, do not emit a detection for it.

Be exhaustive: faint and short entries matter as much as long ones. Work down the page
row by row so that no row is skipped, including rows that contain only one or two
entries."""


def load_cells(xml_path: Path) -> dict[tuple[int, int], np.ndarray]:
    xml = xml_path.read_text(encoding="utf-8")
    out = {}
    for m in CELL_RX.finditer(xml):
        pts = np.array([[int(a) for a in p.split(",")] for p in m.group(3).split()],
                       dtype=np.float64)
        out[(int(m.group(1)), int(m.group(2)))] = pts
    return out


def point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def assign(items: list[dict], cells: dict, w: int, h: int) -> tuple[list[dict], dict]:
    """Assign each detection to the cell containing its centre; nearest-centroid fallback."""
    centroids = {k: v.mean(axis=0) for k, v in cells.items()}
    max_r = max(r for r, _ in cells) + 1
    grid = [{c: "" for c in LEFT_COLS} for _ in range(max_r)]
    stats = {"items": len(items), "placed": 0, "fallback": 0, "dropped": 0,
             "collisions": 0}
    for it in items:
        try:
            cx = (int(it["xmin"]) + int(it["xmax"])) / 2000.0 * w
            cy = (int(it["ymin"]) + int(it["ymax"])) / 2000.0 * h
        except (KeyError, TypeError, ValueError):
            stats["dropped"] += 1
            continue
        text = (it.get("text") or "").strip()
        if not text:
            stats["dropped"] += 1
            continue
        hit = None
        for key, poly in cells.items():
            if point_in_poly(cx, cy, poly):
                hit = key
                break
        if hit is None:
            key, best = None, float("inf")
            for k, ct in centroids.items():
                d = (ct[0] - cx) ** 2 + (ct[1] - cy) ** 2
                if d < best:
                    best, key = d, k
            hit = key
            stats["fallback"] += 1
        r, c = hit
        if c >= len(LEFT_COLS):
            stats["dropped"] += 1
            continue
        col = LEFT_COLS[c]
        if grid[r][col]:
            stats["collisions"] += 1
            grid[r][col] = f"{grid[r][col]} {text}".strip()
        else:
            grid[r][col] = text
        stats["placed"] += 1
    return grid, stats


def run_page(page: int, model: str, tag: str, thinking: str) -> dict | None:
    img_p = FINAL2 / f"Hadita_{page}.jpeg"
    xml_p = FINAL2 / f"Hadita_{page}.xml"
    if not img_p.exists() or not xml_p.exists():
        print(f"  ! page {page}: missing inputs")
        return None
    from PIL import Image
    with Image.open(img_p) as im:
        w, h = im.size

    cfg = types.GenerateContentConfig(
        max_output_tokens=32768,
        response_mime_type="application/json",
        response_schema=ITEM_SCHEMA,
    )
    if thinking != "none":
        cfg.thinking_config = types.ThinkingConfig(
            thinking_budget=base.THINKING_BUDGETS[thinking])

    client = genai.Client()
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=img_p.read_bytes(), mime_type="image/jpeg"),
                  PROMPT],
        config=cfg)
    elapsed = round(time.perf_counter() - t0, 2)

    if not resp.text:
        print(f"    x no output (finish={resp.candidates[0].finish_reason})")
        return None
    try:
        items = json.loads(resp.text).get("items", [])
    except json.JSONDecodeError as e:
        print(f"    x bad JSON: {e}")
        return None

    cells = load_cells(xml_p)
    grid, stats = assign(items, cells, w, h)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"Hadita_{page}_{tag}.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    patched, n_p, n_s = base.patch_xml(xml_p.read_text(encoding="utf-8"), grid, LEFT_COLS)
    (OUT_DIR / f"Hadita_{page}_{tag}.xml").write_text(patched, encoding="utf-8")

    um = resp.usage_metadata
    cost = estimate_cost(model, um) if um else float("nan")
    nonempty = sum(1 for r in grid if any(v.strip() for v in r.values()))
    print(f"    ok {stats['items']} detections -> {stats['placed']} placed "
          f"({stats['fallback']} fallback, {stats['collisions']} collisions, "
          f"{stats['dropped']} dropped); {nonempty} non-empty rows; "
          f"{elapsed}s ${cost:.4f}")

    rec = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), page=page, model=model, tag=tag,
               thinking=thinking, rows=nonempty, grid_rows=len(grid), patched=n_p,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--thinking", default="low", choices=["none", "low", "medium", "high"])
    args = ap.parse_args()
    tag = args.tag or "grounded-" + args.model.replace("gemini-", "g").replace("-preview", "").replace(".", "")
    total = 0.0
    for p in args.pages:
        print(f"  page {p}:")
        rec = run_page(p, args.model, tag, args.thinking)
        if rec:
            total += rec["cost_usd"]
    print(f"total: ${total:.4f}")


if __name__ == "__main__":
    main()
