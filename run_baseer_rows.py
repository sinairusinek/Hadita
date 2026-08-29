"""E8c: Baseer__Nakba on full-width single-row strips instead of isolated cell crops.

Separates two hypotheses for E8b's deficit: is Baseer bad at this material, or did
isolated cell crops starve it of the horizontal context that a line-level model trained
on full text lines expects? A row strip is much closer to its training input (a line).

Output is free-form text per row, so we can only measure how much of the row's GT
content it recovers — not per-cell placement. That is the right measurement for the
question being asked.

  python run_baseer_rows.py 9 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from run_exp2608 import FINAL2
from run_row_strips import row_geometry, inked_rows, make_strip
from run_baseer_cells import load_model, PROMPT
from score_exp2608 import load_reference
from score_g3_vs_gt import LEFT_COLS, _normalize

OUT = Path("exp2608")


@torch.inference_mode()
def read_image(model, proc, img_bgr, max_new_tokens: int = 256) -> str:
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    msgs = [{"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True).strip()


def run_page(page: int, tag: str) -> None:
    img = cv2.imread(str(FINAL2 / f"Hadita_{page}.jpeg"))
    bands, polys = row_geometry(FINAL2 / f"Hadita_{page}.xml")
    todo = sorted(inked_rows(img, bands, polys))
    model, proc = load_model()
    rows = {}
    t0 = time.perf_counter()
    for i, r in enumerate(todo, 1):
        strip = make_strip(img, bands, r)
        rows[r] = read_image(model, proc, strip)
        if i % 10 == 0:
            print(f"    {i}/{len(todo)} rows ({(time.perf_counter()-t0)/i:.1f}s/row)")
    (OUT / f"Hadita_{page}_{tag}_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # Token-recall against GT: what fraction of the row's GT tokens appear in the output?
    gt = load_reference(page)
    tot = hit = 0
    for i, g in enumerate(gt):
        toks = [_normalize(g[c]) for c in LEFT_COLS
                if c != "Remarks" and _normalize(g.get(c, ""))]
        if not toks:
            continue
        # GT row i corresponds to grid row i (both are the page's non-empty rows in order)
        pred = rows.get(sorted(rows)[i] if i < len(rows) else -1, "")
        for t in toks:
            tot += 1
            if t and t in pred:
                hit += 1
    print(f"  page {page}: {len(todo)} rows in {(time.perf_counter()-t0)/60:.1f}min; "
          f"GT-token recall in free text = {hit}/{tot} "
          f"({100*hit/max(tot,1):.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--tag", default="baseer")
    args = ap.parse_args()
    for p in args.pages:
        run_page(p, args.tag)


if __name__ == "__main__":
    main()
