#!/usr/bin/env python3
"""E16: Mistral OCR 4.1 on final3 cell crops, kept for its CONFIDENCE scores.

Why cells and not the page: the geometry problem is already solved by final3, and
every prose-shaped reader we have tried (Baseer, QARI, Kraken rows) failed on
cell-to-column assignment rather than on glyphs. Feeding one crop per cell keeps
the column identity exact, so the only thing measured here is reading + how well
the model knows when it is wrong.

The point of this experiment is NOT keystrokes. It is calibration: does per-cell
confidence predict per-cell error? Everything else in the stack reports a
transcription with no usable uncertainty, so ~2/3 cells cannot be routed to a
human automatically. score_mistral_calibration.py answers that question; this
script only produces the data.

  export MISTRAL_API_KEY=...
  python run_mistral_ocr.py 3 4 5 6 9 10            # ~1700 cells, see --limit
  python run_mistral_ocr.py 3 --limit 40 --dry-run  # shape check, no API calls

Writes exp2608/Hadita_{N}_mistral-ocr.json      (grid, scorer-compatible)
       exp2608/Hadita_{N}_mistral-ocr.conf.json (per-cell confidence + raw text)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from digit_norm import LEFT_COLS  # noqa: E402

FINAL3 = ROOT / "Transkribus upload" / "final3"
OUT_DIR = ROOT / "exp2608"
MODEL = "mistral-ocr-4-1"
PAD = 6
# €3.50/1000 pages, and the API bills one "page" per image submitted.
PRICE_PER_IMAGE = 0.0038


def cell_polys(page: int) -> dict[tuple[int, int], np.ndarray]:
    """TableCell polygons from the final3 PAGE XML, keyed (row, col)."""
    out = {}
    for tc in ET.parse(FINAL3 / f"Hadita_{page}.xml").getroot().iter():
        if not tc.tag.endswith("TableCell"):
            continue
        r, c = tc.get("row"), tc.get("col")
        co = [e for e in tc.iter() if e.tag.endswith("Coords")]
        if r is None or c is None or not co:
            continue
        out[(int(r), int(c))] = np.array(
            [tuple(map(int, q.split(","))) for q in co[0].get("points").split()],
            np.int32)
    return out


def interior_ink(crop: np.ndarray, inset: float = 0.18) -> int:
    """Dark-pixel count inside the cell, excluding the ruled border.

    The threshold is relative to the crop's own paper tone, so it survives the
    per-page exposure differences across the corpus.
    """
    h, w = crop.shape[:2]
    dy, dx = int(h * inset), int(w * inset)
    inner = crop[dy:h - dy, dx:w - dx]
    if inner.size == 0:
        return 0
    g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    thr = max(60, int(np.percentile(g, 60)) - 40)
    return int((g < thr).sum())


def grid_size(polys: dict) -> tuple[int, int]:
    if not polys:
        return 0, 0
    return max(r for r, _ in polys) + 1, max(c for _, c in polys) + 1


def _walk_confidences(obj, found: list) -> None:
    """Collect any confidence-ish number, wherever the response nests it.

    The public schema does not pin the per-block/per-word field names, and they
    differ by granularity, so discover them instead of guessing a path.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and "confidence" in k.lower():
                found.append((k, float(v)))
            else:
                _walk_confidences(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk_confidences(v, found)


def parse_cell_response(resp: dict) -> tuple[str, float | None, dict]:
    """-> (text, confidence, debug). Confidence is the MINIMUM found.

    A cell is only as trustworthy as its least certain character: a 3-digit
    number with one shaky glyph is a wrong number, so min beats mean here.
    """
    pages = resp.get("pages") or []
    text = " ".join((p.get("markdown") or "").strip() for p in pages).strip()
    # OCR markdown wraps stray content; keep it to one line for a table cell.
    text = " ".join(text.split())
    for junk in ("```", "#"):
        text = text.replace(junk, "")
    text = text.strip()

    found: list[tuple[str, float]] = []
    _walk_confidences(resp, found)
    conf = min((v for _, v in found), default=None)
    return text, conf, {"fields": sorted({k for k, _ in found}), "n": len(found)}


def run_page(page: int, client, *, limit: int | None, dry: bool,
             granularity: str, args_ink_min: int = 20) -> dict | None:
    img_p = FINAL3 / f"Hadita_{page}.jpeg"
    if not img_p.exists():
        print(f"  ! page {page}: no image")
        return None
    polys = cell_polys(page)
    if not polys:
        print(f"  ! page {page}: no cells in XML")
        return None
    n_rows, n_cols = grid_size(polys)
    img = cv2.imread(str(img_p))

    grid = [{c: "" for c in LEFT_COLS} for _ in range(n_rows)]
    conf_rows: list[dict] = []
    sent = skipped = 0
    fields_seen: set[str] = set()
    t0 = time.perf_counter()

    for (r, c), poly in sorted(polys.items()):
        if c >= len(LEFT_COLS):
            continue
        if limit is not None and sent >= limit:
            break
        x, y, w, h = cv2.boundingRect(poly)
        x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
        crop = img[y0:y + h + PAD, x0:x + w + PAD]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            skipped += 1
            continue

        # Blank-cell gate. The register is ~44% filled and the API bills per
        # image, so sending blanks would nearly double the cost for nothing.
        # Measured against GT on pp3-6: sampling the cell INTERIOR (inset 18%)
        # separates filled from empty cleanly (median ink 229 vs 0), whereas a
        # whole-crop threshold just counts the ruled border and passes 97% of
        # cells. Threshold 20 keeps ~85% of truly-filled cells.
        ink = interior_ink(crop)
        if ink < args_ink_min:
            skipped += 1
            continue

        if dry:
            sent += 1
            continue

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            skipped += 1
            continue
        b64 = base64.b64encode(buf.tobytes()).decode()
        try:
            resp = client.ocr.process(
                model=MODEL,
                document={"type": "image_url",
                          "image_url": f"data:image/png;base64,{b64}"},
                confidence_scores_granularity=granularity)
            resp = resp if isinstance(resp, dict) else json.loads(resp.model_dump_json())
        except Exception as e:  # keep the page going; one bad cell is not fatal
            print(f"    x r{r} c{c}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        text, conf, dbg = parse_cell_response(resp)
        fields_seen.update(dbg["fields"])
        sent += 1
        if text:
            grid[r][LEFT_COLS[c]] = text
        conf_rows.append({"row": r, "col": c, "col_name": LEFT_COLS[c],
                          "text": text, "confidence": conf, "ink": ink})

    elapsed = round(time.perf_counter() - t0, 1)
    cost = sent * PRICE_PER_IMAGE
    if dry:
        print(f"    dry-run: would send {sent} inked cells "
              f"(skipped {skipped}) ~${cost:.2f}")
        return {"page": page, "sent": sent, "cost": cost}

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"Hadita_{page}_mistral-ocr.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / f"Hadita_{page}_mistral-ocr.conf.json").write_text(
        json.dumps(conf_rows, ensure_ascii=False, indent=1), encoding="utf-8")

    n_conf = sum(1 for r in conf_rows if r["confidence"] is not None)
    print(f"    ok {sent} cells sent, {skipped} skipped, "
          f"{n_conf} with confidence; {elapsed}s ${cost:.2f}")
    if not n_conf:
        print("    ! no confidence field found — calibration cannot be measured. "
              f"granularity={granularity!r}")
    elif fields_seen:
        print(f"    confidence fields: {sorted(fields_seen)}")
    return {"page": page, "sent": sent, "cost": cost, "with_conf": n_conf}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--limit", type=int, default=None,
                    help="max cells per page (cost control while piloting)")
    ap.add_argument("--granularity", default="word",
                    choices=["word", "block", "page"])
    ap.add_argument("--ink-min", type=int, default=20,
                    help="interior-ink threshold for the blank-cell gate")
    ap.add_argument("--dry-run", action="store_true",
                    help="count cells and estimate cost, send nothing")
    args = ap.parse_args()

    client = None
    if not args.dry_run:
        key = os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            sys.exit("MISTRAL_API_KEY is not set. export it, then re-run "
                     "(or use --dry-run to size the job first).")
        try:
            from mistralai import Mistral
        except ImportError:
            sys.exit("pip install mistralai")
        client = Mistral(api_key=key)

    total = 0.0
    for p in args.pages:
        print(f"  page {p}:")
        rec = run_page(p, client, limit=args.limit, dry=args.dry_run,
                       granularity=args.granularity, args_ink_min=args.ink_min)
        if rec:
            total += rec["cost"]
    print(f"total: ${total:.2f}")


if __name__ == "__main__":
    main()
