#!/usr/bin/env python3
"""run_kraken_cells.py — Kraken gen2_sc_clean recognition on final3 cell crops.

The same slot Baseer/NAKBA and Hadid02 occupy: segmentation comes from our
geometry, the model only recognises. kraken_experiment.py cannot be reused
directly — it is hardwired to page 3 and does its own segmentation — so this
runs the recognition model over the cell crops that crop_cells.py produces
from a chosen upload folder.

  python run_kraken_cells.py 3 --xml-dir "Transkribus upload/final3"
Writes exp2608/Hadita_{N}_{tag}.json (+ .xml)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import run_g3v6_local as base  # noqa: E402
from crop_cells import crop_page  # noqa: E402
from digit_norm import LEFT_COLS  # noqa: E402
from run_exp2608 import FINAL2, OUT_DIR  # noqa: E402

OCR_MODEL = ROOT / "gen2_sc_clean_best.mlmodel"
XML_DIR = FINAL2


def recognise(files: list[Path], model: Path) -> dict[str, str]:
    """One kraken call over a glob so the model loads once."""
    if not files:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for f in files:
            (tmp_path / f.name).write_bytes(f.read_bytes())
        cmd = ["kraken", "-I", "*.png", "-o", ".txt", "-f", "image",
               "ocr", "-m", str(model), "--no-segmentation"]
        r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    kraken failed: {r.stderr[-300:]}")
            return {}
        return {p.stem: p.read_text(encoding="utf-8").strip()
                for p in tmp_path.glob("*.txt")}


def run_page(page: int, tag: str, model: Path) -> None:
    crops = crop_page(page, Path("crops"), inked_only=True, pad=6, src_dir=XML_DIR)
    if not crops:
        return
    xml_p = XML_DIR / f"Hadita_{page}.xml"
    n_grid = base.xml_grid_size(xml_p.read_text(encoding="utf-8"))[0]
    grid = [{c: "" for c in LEFT_COLS} for _ in range(n_grid)]

    t0 = time.perf_counter()
    texts = recognise([Path(c["file"]) for c in crops], model)
    for c in crops:
        txt = texts.get(Path(c["file"]).stem, "")
        if c["col"] < len(LEFT_COLS) and c["row"] < n_grid:
            grid[c["row"]][LEFT_COLS[c["col"]]] = txt
    el = time.perf_counter() - t0

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"Hadita_{page}_{tag}.json").write_text(
        json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    patched, _, _ = base.patch_xml(xml_p.read_text(encoding="utf-8"), grid, LEFT_COLS)
    (OUT_DIR / f"Hadita_{page}_{tag}.xml").write_text(patched, encoding="utf-8")
    nonempty = sum(1 for r in grid if any(v.strip() for v in r.values()))
    print(f"    ok {len(crops)} cells, {nonempty}/{n_grid} non-empty rows; "
          f"{el:.1f}s ({el/max(1,len(crops)):.2f}s/cell)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--tag", default="kraken")
    ap.add_argument("--xml-dir", help="geometry folder (default final2)")
    ap.add_argument("--model", default=str(OCR_MODEL))
    args = ap.parse_args()

    global XML_DIR
    if args.xml_dir:
        XML_DIR = Path(args.xml_dir)
    model = Path(args.model)
    if not model.exists():
        sys.exit(f"model not found: {model}")
    for p in args.pages:
        print(f"  page {p}:")
        run_page(p, args.tag, model)


if __name__ == "__main__":
    main()
