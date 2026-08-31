#!/usr/bin/env python3
"""run_kraken_masked.py — Kraken recognition with the output alphabet restricted.

Roughly 90% of this register's cells hold Eastern Arabic numerals, and the
column tells you which cells those are. But the CTC decode is free to pick from
the model's whole codec — 196 characters for arabic_best, 48 of them Latin
letters — so it wanders into `M`, `سا`, stray punctuation. This masks the
per-timestep probabilities down to an allowed label set before decoding, so the
model must choose among characters that can actually occur in that column.

It cannot fix a glyph the model misreads (٣ vs ٢ survives); it only removes
answers that were never possible.

Runs under the anaconda python where kraken is importable, NOT the project venv:
  /opt/anaconda3/bin/python run_kraken_masked.py 3 4 5 --xml-dir "..." --tag k-masked
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent

EASTERN = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
# Marks that legitimately appear in cells: ditto, nil dash, seen-tick, separators.
CONVENTIONS = '"\'-–—.,/٬٫ '
TICKS = "✓"

# Columns that are NOT numeric: keep the full alphabet there.
FREE_COLS = {"Nature_of_Entry", "Remarks"}


def allowed_for(col: str) -> str:
    if col in FREE_COLS:
        return ""            # no mask
    return EASTERN + WESTERN + CONVENTIONS + TICKS


def build_mask(codec, allowed: str) -> np.ndarray:
    """Boolean mask over label indices; index 0 (CTC blank) always allowed."""
    n = max(max(v) for v in codec.c2l.values()) + 1
    m = np.zeros(n + 1, dtype=bool)
    m[0] = True
    for ch, labels in codec.c2l.items():
        if ch in allowed:
            for l in labels:
                m[l] = True
    return m


def main() -> None:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--xml-dir", default="Transkribus upload/final3")
    ap.add_argument("--model", default="arabic_best.mlmodel")
    ap.add_argument("--tag", default="kmask")
    ap.add_argument("--crops", default="crops")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from digit_norm import LEFT_COLS
    from kraken.lib import models
    from kraken.lib.dataset import ImageInputTransforms

    net = models.load_any(str(ROOT / args.model))
    mask_cache: dict[str, np.ndarray] = {}
    tf = ImageInputTransforms(batch=1, height=net.nn.input[2], width=0,
                              channels=net.nn.input[1], pad=16,
                              valid_norm=True, force_binarization=False)

    for page in args.pages:
        idx_p = Path(args.crops) / f"page{page}" / "index.json"
        if not idx_p.exists():
            print(f"page {page}: no crops (run the unmasked runner first)")
            continue
        crops = json.loads(idx_p.read_text())
        out = {}
        for c in crops:
            ci = int(c["col"])
            col = LEFT_COLS[ci] if ci < len(LEFT_COLS) else ""
            allowed = allowed_for(col)
            img = Image.open(c["file"]).convert("L")
            t = tf(img).unsqueeze(0)
            probs, lens = net.forward(t)
            p = probs[0]                                   # (W, C)
            if allowed:
                if col not in mask_cache:
                    mask_cache[col] = build_mask(net.codec, allowed)
                m = mask_cache[col][:p.shape[1]]
                p = np.where(m[None, :], p, -1e9 if p.min() < 0 else 0.0)
            # greedy CTC over the (possibly masked) probabilities
            best = p.argmax(axis=1)
            seq, prev = [], 0
            for b in best:
                if b != prev and b != 0:
                    seq.append(int(b))
                prev = b
            txt = "".join(net.codec.l2c.get((s,), "") for s in seq)
            out[f"r{c['row']}_c{c['col']}"] = unicodedata.normalize("NFC", txt).strip()
        Path("exp2608").mkdir(exist_ok=True)
        (Path("exp2608") / f"Hadita_{page}_{args.tag}_raw.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        nz = sum(1 for v in out.values() if v)
        print(f"page {page}: {len(out)} cells, {nz} non-empty -> {args.tag}")


if __name__ == "__main__":
    main()
