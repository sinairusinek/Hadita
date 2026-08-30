"""E8: run Misraj/Baseer__Nakba (NAKBA competition winner) on per-cell crops.

Baseer__Nakba is a LINE-LEVEL model (644x644, 1200 tokens) with no table/layout
capability, so it can only contribute recognition — segmentation comes from our
final2 geometry. That is exactly the slot Hadid02/PyLaia occupies.

Prompt is the one from misraj-ai/Nakba-pipeline: "Extract the text from the above
document."

  python run_baseer_cells.py 9 --limit 20      # smoke test
  python run_baseer_cells.py 3 4 5 6 9 10
Writes exp2608/Hadita_{N}_baseer.json (+ .xml)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import run_g3v6_local as base
from run_exp2608 import FINAL2, OUT_DIR

# --xml-dir points recognition at a later geometry generation (final3) without
# copying this file; the final2 default is unchanged.
XML_DIR = FINAL2
from crop_cells import crop_page
from digit_norm import LEFT_COLS

MODEL_DIR = "models/Baseer__Nakba"
PROMPT = "Extract the text from the above document."

_state: dict = {}


def load_model():
    if _state:
        return _state["model"], _state["proc"]
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f"  loading {MODEL_DIR} (bf16, mps) ...")
    t0 = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="mps")
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    model.eval()
    print(f"  loaded in {time.perf_counter()-t0:.0f}s")
    _state.update(model=model, proc=proc)
    return model, proc


@torch.inference_mode()
def read_crop(model, proc, img_path: str, max_new_tokens: int = 64) -> str:
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return proc.decode(gen, skip_special_tokens=True).strip()


def run_page(page: int, tag: str, limit: int | None, max_new_tokens: int) -> None:
    crops = crop_page(page, Path("crops"), inked_only=True, pad=6, src_dir=XML_DIR)
    if limit:
        crops = crops[:limit]
    if not crops:
        return
    model, proc = load_model()
    xml_p = XML_DIR / f"Hadita_{page}.xml"
    n_grid = base.xml_grid_size(xml_p.read_text(encoding="utf-8"))[0]
    grid = [{c: "" for c in LEFT_COLS} for _ in range(n_grid)]

    t0 = time.perf_counter()
    for i, c in enumerate(crops, 1):
        try:
            txt = read_crop(model, proc, c["file"], max_new_tokens)
        except Exception as e:
            print(f"    crop r{c['row']}c{c['col']}: {str(e)[:70]}")
            continue
        if c["col"] < len(LEFT_COLS) and c["row"] < n_grid:
            grid[c["row"]][LEFT_COLS[c["col"]]] = txt
        if i % 25 == 0:
            el = time.perf_counter() - t0
            print(f"    {i}/{len(crops)} cells  ({el/i:.2f}s/cell, "
                  f"eta {(len(crops)-i)*el/i/60:.1f}min)")
    el = time.perf_counter() - t0

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"Hadita_{page}_{tag}.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    patched, _, _ = base.patch_xml(xml_p.read_text(encoding="utf-8"), grid, LEFT_COLS)
    (OUT_DIR / f"Hadita_{page}_{tag}.xml").write_text(patched, encoding="utf-8")
    nonempty = sum(1 for r in grid if any(v.strip() for v in r.values()))
    print(f"  page {page}: {len(crops)} cells in {el/60:.1f}min "
          f"({el/max(len(crops),1):.2f}s/cell); {nonempty} non-empty rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--tag", default="baseer")
    ap.add_argument("--limit", type=int, default=None, help="cap cells per page (debug)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--xml-dir", help="geometry folder (default final2)")
    args = ap.parse_args()

    global XML_DIR
    if args.xml_dir:
        XML_DIR = Path(args.xml_dir)
    for p in args.pages:
        run_page(p, args.tag, args.limit, args.max_new_tokens)


if __name__ == "__main__":
    main()
