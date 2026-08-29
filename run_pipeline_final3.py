#!/usr/bin/env python3
"""run_pipeline_final3.py — SoM OCR + ink repair on the final3 corpus.

Repoints the exp2608 runners at final3/ (module-constant monkeypatch — the
runners read FINAL2 as `from run_exp2608 import FINAL2`... no, they bind it at
import, so we patch each module's own global). Outputs land in exp_final3/
with tag `som-f3` (raw) and `som-f3-rep` (after repair).

  python run_pipeline_final3.py 11 12 ... 20
  python run_pipeline_final3.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

FINAL3 = Path("Transkribus upload/final3")
OUT3 = Path("exp_final3")
SOM3 = Path("som3")

import make_som_pages as msp  # noqa: E402
import run_exp2608  # noqa: E402
import run_som_ocr  # noqa: E402
import run_row_strips  # noqa: E402
import repair_empty_rows  # noqa: E402

# repoint every module that captured a path constant at import time
msp.FINAL2 = FINAL3
msp.OUT = SOM3
run_exp2608.FINAL2 = FINAL3
run_som_ocr.FINAL2 = FINAL3
run_som_ocr.OUT_DIR = OUT3
run_som_ocr.SOM_DIR = SOM3
run_row_strips.FINAL2 = FINAL3
run_row_strips.OUT_DIR = OUT3
repair_empty_rows.FINAL2 = FINAL3
repair_empty_rows.OUT_DIR = OUT3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="*", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="gemini-3.7-flash")
    args = ap.parse_args()
    pages = args.pages or (
        sorted(int(p.stem.split("_")[1]) for p in FINAL3.glob("Hadita_*.xml"))
        if args.all else [])
    if not pages:
        ap.error("give pages or --all")

    OUT3.mkdir(exist_ok=True)
    SOM3.mkdir(exist_ok=True)
    total = 0.0
    for p in pages:
        msp.stamp(p)
        rec = run_som_ocr.run_page(p, args.model, "som-f3", "low")
        if rec:
            total += rec.get("cost_usd", 0.0) or 0.0
        try:
            repair_empty_rows.repair(p, "som-f3", "som-f3-rep", args.model)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"  page {p}: repair failed ({exc}); copying raw as -rep")
            import shutil
            for ext in ("json", "xml"):
                src = OUT3 / f"Hadita_{p}_som-f3.{ext}"
                if src.exists():
                    shutil.copy(src, OUT3 / f"Hadita_{p}_som-f3-rep.{ext}")
    print(f"\nSoM total cost: ${total:.3f}")


if __name__ == "__main__":
    main()
