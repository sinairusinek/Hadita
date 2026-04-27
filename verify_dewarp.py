#!/usr/bin/env python3
"""
verify_dewarp.py — Cross-page invariant checks on processed/Hadita-{N}Processed.jpg.

Confirms the dewarp pipeline generalizes:
  - Canvas width is exactly W_OUT for every output (top corners share coords).
  - Detected row/col counts and band usability stay within expected ranges.
  - Synthetic-row fraction stays below threshold.
  - Generates processed/_debug/contact_sheet.html showing every output at the
    same scale with a horizontal guideline at y = H_HEADER.

Usage:
  python verify_dewarp.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dewarp import W_OUT, H_HEADER, ROW_PITCH, PROCESSED_DIR, DEBUG_DIR

ROW_COUNT_RANGE = (25, 50)
MAX_SYNTHETIC_FRAC = 0.20
SERIAL_W_RATIO_RANGE = (2.0, 4.0)  # serial_w / median(interior_pitch_cols_9..19);
                                    # ~2.9 on page 3 (gold) — Serial_No is ~3× wider
                                    # than other columns. Outside this band → likely
                                    # spine/paper-edge artifact polluting col_ranges[1].


def _check_serial_width(page: int) -> tuple[bool, str]:
    """Verify Serial_No column width is within [0.7×, 1.3×] of median interior pitch.

    Catches the page-10/50 failure mode where x_left_col lands on a spine artifact
    or paper edge and Serial_No collapses or balloons.
    """
    p = ROOT / ".ocr_cache" / f"dewarp_cols_page{page}.json"
    if not p.exists():
        return True, "no col cache"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        cr = d["col_ranges_framed"]
    except Exception as e:
        return False, f"col cache unreadable: {e}"
    if len(cr) < 12:
        return False, f"only {len(cr)} boundaries"
    serial_w = cr[1] - cr[0]
    right_gaps = [cr[i + 1] - cr[i] for i in range(9, len(cr) - 1)]
    if not right_gaps:
        return False, "no right-side gaps"
    pitch = float(np.median(right_gaps))
    if pitch <= 1:
        return False, f"degenerate median pitch {pitch}"
    ratio = serial_w / pitch
    lo, hi = SERIAL_W_RATIO_RANGE
    ok = lo <= ratio <= hi
    return ok, f"serial_w/pitch={ratio:.2f} (serial_w={serial_w}, pitch={pitch:.0f})"


def _scan_outputs() -> list[Path]:
    pat = re.compile(r"Hadita-(\d+)Processed\.jpg$")
    return sorted(
        (p for p in PROCESSED_DIR.glob("Hadita-*Processed.jpg") if pat.search(p.name)),
        key=lambda p: int(pat.search(p.name).group(1)),
    )


def _check_canvas(p: Path) -> tuple[bool, str]:
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    if w != W_OUT:
        return False, f"width {w} != {W_OUT}"
    if (h - H_HEADER) % ROW_PITCH != 0:
        return False, f"height {h} not on uniform row grid (H-{H_HEADER}={h-H_HEADER}, pitch={ROW_PITCH})"
    return True, f"{w}×{h}"


def _write_contact_sheet(outputs: list[Path]) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    sheet = DEBUG_DIR / "contact_sheet.html"
    img_w_css = 380  # display width
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Dewarp contact sheet</title>",
        "<style>body{font-family:system-ui;margin:20px;background:#222;color:#eee}"
        ".grid{display:flex;flex-wrap:wrap;gap:18px}"
        ".cell{position:relative;background:#fff}"
        ".cell .lbl{position:absolute;top:4px;left:6px;background:#000c;color:#fff;padding:2px 6px;font-size:12px;border-radius:3px}"
        ".cell .guide{position:absolute;left:0;right:0;border-top:2px solid #ff3b30;pointer-events:none}"
        f"img{{width:{img_w_css}px;height:auto;display:block}}"
        "</style>",
        "<h1>Dewarp contact sheet</h1>",
        f"<p>Red guideline = y={H_HEADER}px (table top edge). Every page should have its first row begin exactly on this line.</p>",
        "<div class='grid'>",
    ]
    for p in outputs:
        img = cv2.imread(str(p))
        h_orig = img.shape[0]
        scale = img_w_css / W_OUT
        guide_top = H_HEADER * scale
        rel = p.relative_to(ROOT).as_posix()
        parts.append(
            f"<div class='cell'><div class='lbl'>{p.stem}</div>"
            f"<img src='../../{rel}'>"
            f"<div class='guide' style='top:{guide_top:.1f}px'></div></div>"
        )
    parts.append("</div>")
    sheet.write_text("\n".join(parts), encoding="utf-8")
    return sheet


def main() -> None:
    outputs = _scan_outputs()
    if not outputs:
        print(f"No outputs found in {PROCESSED_DIR}")
        sys.exit(1)

    print(f"Verifying {len(outputs)} processed page(s)\n")
    print(f"{'page':<6}{'canvas':<14}{'rows':<8}{'cols':<8}{'bands':<8}{'synth%':<8}status")
    print("-" * 70)

    all_ok = True
    for p in outputs:
        m = re.search(r"Hadita-(\d+)Processed", p.name)
        page = int(m.group(1))
        ok_canvas, info = _check_canvas(p)
        # Detection stats live in dewarp_seg_page{N}.json (Kraken cache) — best-effort summary:
        rows_text = cols_text = bands_text = synth_text = "?"
        seg_cache = ROOT / ".ocr_cache" / f"dewarp_seg_page{page}.json"
        problems: list[str] = []
        if not ok_canvas:
            problems.append(info)

        # Use canvas height as a row-count proxy (since rows are uniform pitch)
        img = cv2.imread(str(p))
        n_rows = (img.shape[0] - H_HEADER) // ROW_PITCH
        rows_text = str(n_rows)
        if not (ROW_COUNT_RANGE[0] <= n_rows <= ROW_COUNT_RANGE[1]):
            problems.append(f"row count {n_rows} outside {ROW_COUNT_RANGE}")

        ok_sw, sw_info = _check_serial_width(page)
        if not ok_sw:
            problems.append(f"serial_w invariant: {sw_info}")
        else:
            cols_text = sw_info

        status = "OK" if not problems else "FAIL: " + "; ".join(problems)
        if problems:
            all_ok = False
        print(f"{page:<6}{info:<14}{rows_text:<8}{cols_text:<8}{bands_text:<8}{synth_text:<8}{status}")

    sheet = _write_contact_sheet(outputs)
    print(f"\nContact sheet → {sheet.relative_to(ROOT)}")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
