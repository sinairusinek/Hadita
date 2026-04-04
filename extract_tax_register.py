#!/usr/bin/env python3
"""
extract_tax_register.py — Production extraction pipeline for British Mandate Palestine tax registers.

After running compare_ocr.py and selecting the best approach, use this script to process
all 100 data pages (pages 3–101) and produce a single output CSV.

Usage:
  python3 extract_tax_register.py --approach A
  python3 extract_tax_register.py --approach B --start-page 10 --end-page 50
  python3 extract_tax_register.py --approach G --start-page 3 --resume

Supported approaches (use the ID from compare_ocr.py):
  A: Claude Opus 4.6  — full page
  B: Claude Opus 4.6  — zoomed crops
  C: Gemini 2.5 Pro   — full page
  D: Gemini 2.5 Pro   — zoomed crops
  E: Gemini 3.x Pro   — full page
  F: Gemini 3.x Pro   — zoomed crops
  G: Kraken + gen2_sc_clean_best — cell-level
  H: Kraken + arabic_best        — cell-level
  I: Kraken + gen2_sc_clean_best — column strips
  J: Kraken + arabic_best        — column strips
  K: Gemini 3 Flash   — full page
  L: Gemini 3 Flash   — zoomed crops
  ENSEMBLE            — run two models and merge (cell-level diff)
    --primary E       primary model (default: E)
    --secondary C     secondary model (default: C)
    --tiebreaker K    optional tiebreaker (default: none)

  Disagreeing cells are annotated as [DISAGREE: E=<val>|C=<val>].
  The "Disagreements" column lists which columns had unresolved disagreements.

Output:
  tax_register_output.csv          — final CSV with all extracted rows
  tax_register_checkpoint.jsonl    — checkpoint file for resume support

Setup:
  export ANTHROPIC_API_KEY=...    (for approaches A, B)
  export GOOGLE_API_KEY=...       (for approaches C–F)
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# Import all OCR machinery from compare_ocr.py
sys.path.insert(0, str(Path(__file__).parent))
from compare_ocr import (
    PROJECT_DIR, IMAGE_PATTERN, PAGE_FOLIO,
    ALL_DATA_COLS, LEFT_COLS, RIGHT_COLS, META_COLS,
    run_approach, run_ensemble, normalize_row, page_image_path,
)

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

FIRST_DATA_PAGE = 3
LAST_DATA_PAGE  = 101
OUTPUT_CSV      = PROJECT_DIR / "tax_register_output.csv"
CHECKPOINT_FILE = PROJECT_DIR / "tax_register_checkpoint.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FIELDNAMES = [
    "Page_Number", "Folio_Number",
    "Tax_Payer_Arabic", "Tax_Payer_Romanized",
    "Village_Arabic", "Village_Romanized",
] + ALL_DATA_COLS + ["OCR_Method"]


# ──────────────────────────────────────────────────────────
# CHECKPOINT (resume support)
# ──────────────────────────────────────────────────────────

def load_checkpoint() -> set[int]:
    """Return set of page numbers already processed."""
    done = set()
    if CHECKPOINT_FILE.exists():
        for line in CHECKPOINT_FILE.open():
            try:
                record = json.loads(line)
                done.add(record["page"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def append_checkpoint(page_num: int, row_count: int):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"page": page_num, "rows": row_count, "ts": time.time()}) + "\n")


# ──────────────────────────────────────────────────────────
# DITTO-MARK RESOLUTION
# ──────────────────────────────────────────────────────────

DITTO = '"'


def resolve_ditto_marks(rows: list[dict]) -> list[dict]:
    """
    Replace ditto marks (") with the value from the previous row in the same column.
    Resolves chains: if row N-1 also has a ditto, walk back until a real value is found.
    """
    resolved = []
    for i, row in enumerate(rows):
        new_row = dict(row)
        for col in LEFT_COLS + RIGHT_COLS:
            val = new_row.get(col, "").strip()
            if val == DITTO or val.startswith(DITTO):
                # Walk back through resolved rows
                for j in range(i - 1, -1, -1):
                    prev_val = resolved[j].get(col, "").strip()
                    if prev_val and prev_val != DITTO:
                        new_row[col] = prev_val
                        break
        resolved.append(new_row)
    return resolved


# ──────────────────────────────────────────────────────────
# PAGE-LEVEL METADATA INFERENCE
# ──────────────────────────────────────────────────────────

def infer_folio(page_num: int) -> str:
    """Best-effort folio number from known page→folio mapping or offset heuristic."""
    if page_num in PAGE_FOLIO:
        return PAGE_FOLIO[page_num]
    # Heuristic: folio ≈ page - 2 for page 3, page - 1 for the rest
    return str(page_num - 1)


# ──────────────────────────────────────────────────────────
# MAIN EXTRACTION LOOP
# ──────────────────────────────────────────────────────────

def extract(approach: str, start_page: int, end_page: int, resume: bool = True,
            delay: float = 2.0,
            ensemble_primary: str = "E", ensemble_secondary: str = "C",
            ensemble_tiebreaker: str = None):
    """
    Process pages start_page..end_page (inclusive) with the given approach.
    Appends rows to OUTPUT_CSV; updates CHECKPOINT_FILE after each page.

    When approach == "ENSEMBLE", runs ensemble_primary + ensemble_secondary
    (+ optional ensemble_tiebreaker) and merges results with cell-level diffing.
    """
    done_pages = load_checkpoint() if resume else set()

    # Build a human-readable label for the OCR_Method column
    if approach == "ENSEMBLE":
        parts = [ensemble_primary, ensemble_secondary]
        if ensemble_tiebreaker:
            parts.append(ensemble_tiebreaker)
        ocr_label = "ENSEMBLE:" + "+".join(parts)
    else:
        ocr_label = approach

    # Determine write mode
    write_header = not OUTPUT_CSV.exists() or not resume
    f_out = open(OUTPUT_CSV, "a" if resume else "w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    pages_to_run = [p for p in range(start_page, end_page + 1)
                    if page_image_path(p).exists() and p not in done_pages]

    log.info(
        "Extraction: approach=%s, pages=%d-%d, remaining=%d",
        ocr_label, start_page, end_page, len(pages_to_run),
    )

    try:
        from tqdm import tqdm
        iter_pages = tqdm(pages_to_run, desc=f"Extracting [{ocr_label}]", unit="page")
    except ImportError:
        iter_pages = pages_to_run

    for page_num in iter_pages:
        log.info("Processing page %d ...", page_num)

        if approach == "ENSEMBLE":
            rows = run_ensemble(
                ensemble_primary, ensemble_secondary, page_num,
                tiebreaker=ensemble_tiebreaker,
            )
        else:
            rows = run_approach(approach, page_num)

        if not rows:
            log.warning("  No rows extracted for page %d", page_num)
            append_checkpoint(page_num, 0)
            continue

        rows = resolve_ditto_marks(rows)

        folio = infer_folio(page_num)
        for row in rows:
            out_row = {f: "" for f in FIELDNAMES}
            out_row["Page_Number"]  = page_num
            out_row["Folio_Number"] = folio
            out_row["OCR_Method"]   = ocr_label
            out_row.update(normalize_row(row, ALL_DATA_COLS))
            writer.writerow(out_row)

        f_out.flush()
        append_checkpoint(page_num, len(rows))
        log.info("  → %d rows written (page %d)", len(rows), page_num)

        # Rate limiting — LLM APIs need breathing room
        # For ENSEMBLE, delay applies since we hit 2+ models per page
        if approach in ("A", "B", "C", "D", "E", "F", "K", "L", "ENSEMBLE"):
            time.sleep(delay)

    f_out.close()
    log.info("Extraction complete. Output: %s", OUTPUT_CSV)


# ──────────────────────────────────────────────────────────
# POST-PROCESSING: LOW-CONFIDENCE FILTER
# ──────────────────────────────────────────────────────────

def flag_low_confidence():
    """Print summary of rows needing manual review (low confidence or uncertain values)."""
    if not OUTPUT_CSV.exists():
        log.error("Output file not found: %s", OUTPUT_CSV)
        return

    low_conf = []
    with open(OUTPUT_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if (row.get("Row_Confidence", "").lower() == "low"
                    or any("[?]" in row.get(c, "") for c in ALL_DATA_COLS)):
                low_conf.append(row)

    print(f"\nRows flagged for review: {len(low_conf)}")
    if low_conf:
        review_path = PROJECT_DIR / "review_flagged.csv"
        with open(review_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(low_conf)
        print(f"Saved to: {review_path}")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

VALID_APPROACHES = list("ABCDEFGHIJKL") + ["ENSEMBLE"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--approach", required=True, choices=VALID_APPROACHES,
        help=(
            "OCR approach to use (A–L; run compare_ocr.py first to pick the best one). "
            "Use ENSEMBLE to run two models and merge with cell-level diffing."
        ),
    )
    parser.add_argument("--start-page", type=int, default=FIRST_DATA_PAGE,
                        help=f"First page to process (default: {FIRST_DATA_PAGE})")
    parser.add_argument("--end-page",   type=int, default=LAST_DATA_PAGE,
                        help=f"Last page to process (default: {LAST_DATA_PAGE})")
    parser.add_argument("--resume",     action="store_true", default=True,
                        help="Skip already-checkpointed pages (default: on)")
    parser.add_argument("--no-resume",  action="store_true",
                        help="Ignore checkpoint and re-process all pages")
    parser.add_argument("--delay",      type=float, default=2.0,
                        help="Seconds to wait between API calls (default: 2)")
    parser.add_argument("--flag",       action="store_true",
                        help="After extraction, print low-confidence rows")
    # Ensemble-specific options (only used when --approach ENSEMBLE)
    parser.add_argument("--primary",    default="E",
                        help="Primary approach for ENSEMBLE (default: E)")
    parser.add_argument("--secondary",  default="C",
                        help="Secondary approach for ENSEMBLE (default: C)")
    parser.add_argument("--tiebreaker", default=None,
                        help="Optional tiebreaker approach for ENSEMBLE (e.g. K)")
    args = parser.parse_args()

    resume = args.resume and not args.no_resume

    extract(
        approach=args.approach,
        start_page=args.start_page,
        end_page=args.end_page,
        resume=resume,
        delay=args.delay,
        ensemble_primary=args.primary,
        ensemble_secondary=args.secondary,
        ensemble_tiebreaker=args.tiebreaker,
    )

    if args.flag:
        flag_low_confidence()


if __name__ == "__main__":
    main()
