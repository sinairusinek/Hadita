"""E5b: targeted repair — re-read only rows that have ink but came back empty.

Set-of-marks solves row alignment but still drops a few visually near-empty rows
(typically a lone dash). Those are cheap to find without any model: the ink gate knows
which grid rows carry marks. This re-reads just those rows as single-row strips.

  python repair_empty_rows.py 10 --tag som-g37-flash --out som-rep10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import run_g3v6_local as base
from run_exp2608 import FINAL2, OUT_DIR
from run_row_strips import row_geometry, inked_rows, make_strip, PROMPT, SCHEMA
from run_exp2608 import estimate_cost
from digit_norm import LEFT_COLS


def repair(page: int, tag: str, out_tag: str, model: str) -> None:
    src = OUT_DIR / f"Hadita_{page}_{tag}.json"
    if not src.exists():
        print(f"  ! no source output {src}")
        return
    grid = json.load(open(src, encoding="utf-8"))
    img = cv2.imread(str(FINAL2 / f"Hadita_{page}.jpeg"))
    xml_p = FINAL2 / f"Hadita_{page}.xml"
    bands, polys = row_geometry(xml_p)
    ink = inked_rows(img, bands, polys)

    # Pad grid to the geometry's row count so ink rows beyond the model's output
    # are still repairable.
    while len(grid) < max(bands) + 1:
        grid.append({c: "" for c in LEFT_COLS})

    targets = [r for r in sorted(ink)
               if r < len(grid) and not any((grid[r].get(c) or "").strip()
                                            for c in LEFT_COLS)]
    print(f"  page {page}: {len(ink)} inked rows, {len(targets)} inked-but-empty -> repair")
    if not targets:
        return

    from google import genai
    from google.genai import types
    client = genai.Client()
    cfg = types.GenerateContentConfig(
        max_output_tokens=4096, response_mime_type="application/json",
        response_schema=SCHEMA)

    cost = 0.0
    filled = 0
    for r in targets:
        strip = make_strip(img, bands, r)
        buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=buf, mime_type="image/jpeg"), PROMPT],
                config=cfg)
        except Exception as e:
            print(f"    row {r}: {str(e)[:70]}")
            continue
        if resp.usage_metadata:
            cost += estimate_cost(model, resp.usage_metadata)
        if not resp.text:
            continue
        try:
            cells = json.loads(resp.text).get("cells", {})
        except json.JSONDecodeError:
            continue
        got = False
        for k, v in cells.items():
            try:
                ci = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= ci < len(LEFT_COLS) and (v or "").strip():
                grid[r][LEFT_COLS[ci]] = v.strip()
                got = True
        filled += bool(got)

    (OUT_DIR / f"Hadita_{page}_{out_tag}.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    patched, _, _ = base.patch_xml(xml_p.read_text(encoding="utf-8"), grid, LEFT_COLS)
    (OUT_DIR / f"Hadita_{page}_{out_tag}.xml").write_text(patched, encoding="utf-8")
    print(f"    filled {filled}/{len(targets)} rows; ${cost:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--tag", default="som-g37-flash", help="source output tag")
    ap.add_argument("--out", default=None, help="output tag (default: <tag>-rep)")
    ap.add_argument("--model", default="gemini-3.7-flash")
    args = ap.parse_args()
    out = args.out or f"{args.tag}-rep"
    for p in args.pages:
        repair(p, args.tag, out, args.model)


if __name__ == "__main__":
    main()
