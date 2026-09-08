"""E16: iterative "LLM as annotator" — Calfa's ICL loop, not a fixed few-shot block.

E15 prepended one FIXED block of p50 cell crops to every page and lost (4907 ks
at 8-shot vs 4476 zero-shot), by tracking the pool's 2/3 balance instead of the
letterforms. That is not a test of Calfa's actual method, which differs twice:

  1. RECENCY  - `ICLPool.sample()` sorts by added_at descending and takes the top
     n, so the exemplars are the n most-recently-validated PAGES. The exemplar
     set changes for every target page; E15's was constant.
  2. CORRECTION - the pool is fed by the editor (`source="corrected"`), so the
     model sees its own errors fixed. E15 fed pre-existing typed GT.

This runs (1) faithfully at page granularity. It does NOT run (2): we use each
prior page's GT as its "correction", which is the OPTIMISTIC BOUND — a perfect
RA. If the loop cannot win with flawless corrections it will not win with real
ones.

  python run_som_iter.py --n 2 --tag som-f3-iter2
  python score_exp2608.py --tags som-f3v2 som-f3-iter2 --tolerant

Caveat on power: only 6 proxy-GT pages exist, and page 1 of the sequence is
necessarily zero-shot, so at most 5 pages carry an exemplar block. Against the
+-10% run variance this is indicative, not conclusive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from google.genai import types

import run_som_ocr as som
from score_exp2608 import PAGES, load_reference
from digit_norm import LEFT_COLS


def render_gt_block(rows: list[dict]) -> str:
    """Render a page's GT in the SAME shape the model is asked to emit.

    The exemplar must model the output format as well as the hand; showing a
    different shape would teach the wrong lesson. Mirrors run_som_ocr.SCHEMA:
    {"rows": [{"row": int, "cells": {"<col idx>": "text"}}]}
    """
    out = []
    for r_i, row in enumerate(rows):
        cells = {str(c_i): row[c].strip()
                 for c_i, c in enumerate(LEFT_COLS[:-1])
                 if row.get(c, "").strip()}
        if cells:
            out.append({"row": r_i, "cells": cells})
    return json.dumps({"rows": out}, ensure_ascii=False)


def build_iter_fewshot(prior: list[int], som_dir: Path) -> list:
    """Exemplar block from already-'validated' pages, most recent LAST.

    Recency order matters: `ICLPool.sample()` returns newest-first, but a
    prompt reads top-to-bottom, so the nearest neighbour goes closest to the
    target page. Empty list for the first page = plain zero-shot.
    """
    if not prior:
        return []
    parts = [types.Part.from_text(text=(
        "Before the target page, here are pages from this SAME register that have "
        "already been transcribed and CONFIRMED correct by a human expert. They are "
        "the pages immediately preceding the target, in the same hand and layout. "
        "Study how this scribe forms each digit and how the transcription is "
        "structured, then apply both to the page that follows."))]
    for p in prior:
        img = som_dir / f"Hadita_{p}_som.jpg"
        rows = load_reference(p)
        if not img.exists() or not rows:
            print(f"    ! exemplar page {p} unavailable, skipped")
            continue
        parts.append(types.Part.from_bytes(
            data=img.read_bytes(), mime_type="image/jpeg"))
        parts.append(types.Part.from_text(
            text=f"Confirmed transcription of that page:\n{render_gt_block(rows)}"))
    if len(parts) == 1:
        return []
    parts.append(types.Part.from_text(
        text="End of confirmed examples. Now transcribe the target page below."))
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs="+", type=int, default=PAGES,
                    help="processed in the order given; the sequence IS the curriculum")
    ap.add_argument("--n", type=int, default=2,
                    help="ICL window: how many preceding pages to inject")
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--thinking", default="low",
                    choices=["none", "low", "medium", "high"])
    ap.add_argument("--som-dir", default="som_f3")
    ap.add_argument("--xml-dir", default="Transkribus upload/final3")
    ap.add_argument("--pool", nargs="*", type=int, default=None,
                    help="FIXED pool: use these GT pages as exemplars for every "
                         "target, instead of the sliding window. This is the "
                         "production-shaped question -- can the 6 verified pages "
                         "carry a page that has no corrected neighbours?")
    args = ap.parse_args()

    som_dir = Path(args.som_dir)
    som.SOM_DIR = som_dir
    som.XML_DIR = Path(args.xml_dir)
    tag = args.tag or f"som-f3-iter{args.n}"

    if args.pool is not None:
        print(f"E17 fixed pool {args.pool} -> targets {args.pages}, tag {tag}")
    else:
        print(f"E16 iterative ICL: window n={args.n}, order {args.pages}, tag {tag}")
    done: list[int] = []
    total = 0.0
    for p in args.pages:
        if args.pool is not None:
            # A page never exemplifies itself (leave-one-out keeps it scorable).
            prior = [q for q in args.pool if q != p]
        else:
            # Newest-first selection (ICLPool.sample), reversed so the nearest
            # neighbour sits closest to the target page in the prompt.
            prior = done[-args.n:] if args.n > 0 else []
        fewshot = build_iter_fewshot(prior, som_dir)
        shots = (len(fewshot) - 2) // 2 if fewshot else 0
        print(f"  page {p}:  exemplars={prior or '(none, zero-shot)'} ({shots})")
        rec = som.run_page(p, args.model, tag, args.thinking, fewshot)
        if rec:
            total += rec["cost_usd"]
        # A page joins the pool whether or not it scored well — the loop
        # assumes the human corrected it, so GT is what enters either way.
        done.append(p)
    print(f"total: ${total:.4f}")
    print(f"\nscore with:\n  python score_exp2608.py --tags som-f3v2 {tag} --tolerant")


if __name__ == "__main__":
    main()
