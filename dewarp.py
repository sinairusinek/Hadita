#!/usr/bin/env python3
"""
dewarp.py — Notebook-level preprocessor: original scan → deskewed + dewarped JPEG.

Pipeline per page (canonical order):
  1. Load original 000nvrj-*page-{NNNN}.jpg
  2. Deskew (perspective warp of paper boundary)             [image_preprocess.deskew_image]
  3. Crop table (left ~49.4%, below header strip)            [segment_unified.crop_table]
  4. Detect rows (Kraken + interpolate; morph fallback)       [segment_unified.detect_rows]
  5. Detect columns (global + per-band x for bow tracking)    [segment_unified.detect_columns(_banded)]
  6. Build dense cv2.remap and apply                          [segment_unified.build_remap]
  7. Normalize to fixed output canvas (table top-corners at constant coords)
  8. Save → processed/Hadita-{N}Processed.jpg

Outputs are page-wise sequential; failures are logged per page and don't abort the batch.

Usage:
  python dewarp.py --pages 3 10 50
  python dewarp.py --pages 3 --debug             # write per-step images to processed/_debug/page3/
  python dewarp.py --pages 3 --from-cache        # skip step 2, reuse .ocr_cache/deskewed_page{N}.{png,jpg}
  python dewarp.py --all-except-cover            # every original except portrait page-1 (future)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from image_preprocess import deskew_image
from segment_unified import (
    N_BANDS, EXPECTED_COLS,
    crop_table, detect_columns, detect_columns_banded, detect_rows,
    detect_table_frame, build_remap,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGES_DIR    = ROOT / "images"
CACHE_DIR     = ROOT / ".ocr_cache"
PROCESSED_DIR = ROOT / "processed"
DEBUG_DIR     = PROCESSED_DIR / "_debug"
ORIG_PATTERN  = "000nvrj-432316TAX 1-85_page-{:04d}.jpg"

# ── Fixed output canvas (all processed pages share these) ─────────────────────
W_OUT       = 2299   # output width (matches existing Hadita-3Processed.jpg width)
H_META      = 220    # metadata strip (above the printed table)
H_PHEADER   = 200    # printed column-header band (top of the table, above data rows)
H_HEADER    = H_META + H_PHEADER  # = 420; y of the first data row top edge (constant per page)
ROW_PITCH   = 100    # uniform row pitch in dewarped data region
JPEG_Q      = 95
CFG         = {"header_frac": 0.08, "table_width_frac": 0.55}  # wider — must include the page-split line


# ── I/O helpers ───────────────────────────────────────────────────────────────

def find_original(page: int) -> Path | None:
    p = ROOT / ORIG_PATTERN.format(page)
    return p if p.exists() else None


def load_or_make_deskewed(page: int, from_cache: bool) -> np.ndarray:
    """Load deskewed image: bundled → png cache → jpg cache → recompute from original."""
    for p in [
        IMAGES_DIR / f"deskewed_page{page}.jpg",
        CACHE_DIR / f"deskewed_page{page}.png",
        CACHE_DIR / f"deskewed_page{page}.jpg",
    ]:
        if p.exists():
            log.info("    loaded deskewed from %s", p.relative_to(ROOT))
            return cv2.imread(str(p))

    if from_cache:
        raise FileNotFoundError(f"No cached deskewed image for page {page}")

    orig = find_original(page)
    if orig is None:
        raise FileNotFoundError(f"Original scan not found: {ORIG_PATTERN.format(page)}")
    log.info("    deskewing original %s", orig.name)
    img = cv2.imread(str(orig))
    deskewed = deskew_image(img)
    CACHE_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(CACHE_DIR / f"deskewed_page{page}.png"), deskewed)
    return deskewed


# ── Debug overlays ────────────────────────────────────────────────────────────

def _write_debug(page: int, name: str, img: np.ndarray) -> None:
    d = DEBUG_DIR / f"page{page}"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def _overlay_rows(table_bgr: np.ndarray, rows: list[dict]) -> np.ndarray:
    out = table_bgr.copy()
    for r in rows:
        color = (0, 165, 255) if r.get("synthetic") else (0, 0, 220)
        y = int(r["y_center"])
        cv2.line(out, (0, y), (out.shape[1], y), color, 2)
    return out


def _overlay_anchors(deskewed: np.ndarray, table_bgr: np.ndarray,
                     y_offset: int, x_offset: int, frame: dict) -> np.ndarray:
    """Draw anchor lines on a full-page view (deskewed + meta strip + table).

    Coords from `frame` are in TABLE space; we shift by y_offset/x_offset to
    place them on the deskewed image so the metadata strip is visible too.
    """
    tw = table_bgr.shape[1]
    out = deskewed[:, x_offset : x_offset + tw].copy()
    h, w = out.shape[:2]
    y_hb = y_offset + frame["header_bottom_y"]
    # Mark the top of the table (where pheader strip begins)
    cv2.line(out, (0, y_offset), (w, y_offset), (0, 200, 200), 2)  # cyan = table top
    cv2.line(out, (0, y_hb), (w, y_hb), (0, 200, 0), 3)            # green = header bottom
    for x_table, color, label in [
        (frame["x_left_frame"],  (0,   140, 255), "x_left_frame"),
        (frame["x_left_col"],    (200, 0,   200), "x_left_col"),
        (frame["x_right_split"], (255, 0,     0), "x_right_split"),
    ]:
        x = x_table  # x_offset == 0 in current crop
        cv2.line(out, (x, 0), (x, h), color, 2)
        cv2.circle(out, (x, y_hb), 14, color, -1)
        cv2.putText(out, label, (x + 10, y_hb - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def _overlay_cols(table_bgr: np.ndarray, bands: list[dict]) -> np.ndarray:
    out = table_bgr.copy()
    for b in bands:
        y = int(b["y_center"])
        for x in b["col_x"]:
            cv2.circle(out, (int(x), y), 6, (255, 0, 0), -1)
    return out


# ── Canvas normalization ──────────────────────────────────────────────────────

def _resize_table(table_dewarped: np.ndarray, n_rows: int) -> np.ndarray:
    """Resize the dewarped table to W_OUT × (n_rows * ROW_PITCH)."""
    target_h = n_rows * ROW_PITCH
    return cv2.resize(table_dewarped, (W_OUT, target_h), interpolation=cv2.INTER_AREA)


def _resize_meta(deskewed: np.ndarray, y_offset: int, x_left: int, x_right: int) -> np.ndarray:
    """The strip above the printed table (metadata band)."""
    src = deskewed[0:y_offset, x_left:x_right]
    if src.size == 0:
        return np.full((H_META, W_OUT, 3), 255, dtype=np.uint8)
    return cv2.resize(src, (W_OUT, H_META), interpolation=cv2.INTER_AREA)


def _dewarp_pheader(pheader_region: np.ndarray, bands: list[dict]) -> np.ndarray:
    """Dewarp the printed column-header band using the topmost data-band's column-x.

    pheader_region is already cropped horizontally to the frame ([x_left_frame,
    x_right_split]) and vertically to [0, header_bottom_y]. We straighten its
    columns using the topmost band's per-column x positions so they align with
    the dewarped data rows below. Output is (H_PHEADER, W_OUT, 3).
    """
    h, w = pheader_region.shape[:2]
    if h <= 0 or w <= 0 or not bands:
        return np.full((H_PHEADER, W_OUT, 3), 255, dtype=np.uint8)

    src_col_x = np.array(bands[0]["col_x"], dtype=np.float32)  # in framed coords
    n_bounds = src_col_x.size
    out_col_x = np.linspace(0, W_OUT, n_bounds, dtype=np.float32)

    ox = np.arange(W_OUT, dtype=np.float32)
    src_x_row = np.interp(ox, out_col_x, src_col_x).astype(np.float32)

    oy = np.arange(H_PHEADER, dtype=np.float32)
    scale_y = h / max(1, H_PHEADER)
    src_y_col = (oy * scale_y).astype(np.float32)

    map_x = np.broadcast_to(src_x_row[None, :], (H_PHEADER, W_OUT)).copy()
    map_y = np.broadcast_to(src_y_col[:, None], (H_PHEADER, W_OUT)).copy()
    return cv2.remap(pheader_region, map_x, map_y,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ── Per-page processing ───────────────────────────────────────────────────────

def process_page(page: int, debug: bool = False, from_cache: bool = False) -> dict:
    """Run the full pipeline for one page; return metadata dict.

    Order: deskew → wide crop (safety) → detect frame anchors → tight crop to frame
         → split into pheader and data regions → detect rows + columns on data only
         → dewarp data region → dewarp pheader using topmost data band
         → compose meta strip + dewarped pheader + dewarped data.
    """
    log.info("Page %d", page)

    # 1. Deskew (perspective warp of paper boundary)
    deskewed = load_or_make_deskewed(page, from_cache)
    if debug:
        _write_debug(page, "01_deskewed.jpg", deskewed)

    # 2. Wide crop — generous fractions just to bring the table region into view.
    wide_crop, y_offset, x_offset = crop_table(deskewed, CFG)
    wh, ww = wide_crop.shape[:2]
    log.info("    wide crop %d×%d  offset=(x=%d, y=%d)", ww, wh, x_offset, y_offset)
    if debug:
        _write_debug(page, "02_wide_crop.jpg", wide_crop)

    # 3. Detect frame anchors on the wide crop
    frame = detect_table_frame(wide_crop)
    if debug:
        _write_debug(page, "03_anchors.jpg",
                     _overlay_anchors(deskewed, wide_crop, y_offset, x_offset, frame))

    # 4. Tight crop to frame — defines all downstream coordinates
    x_l = frame["x_left_frame"]
    x_r = frame["x_right_split"]
    hb_y = frame["header_bottom_y"]
    framed = wide_crop[:, x_l:x_r]
    pheader_region = framed[0:hb_y, :]
    data_region    = framed[hb_y:, :]
    fh, fw = framed.shape[:2]
    log.info("    framed %d×%d  hb_y=%d  data region %d×%d",
             fw, fh, hb_y, data_region.shape[1], data_region.shape[0])
    if debug:
        _write_debug(page, "04_framed.jpg", framed)
        _write_debug(page, "04a_data_region.jpg", data_region)

    # 5. Row detection on the clean data region
    seg_cache = CACHE_DIR / f"dewarp_seg_page{page}.json"
    rows_data = detect_rows(data_region, cache_path=seg_cache,
                            use_cache=True, method="kraken")
    n_rows = len(rows_data)
    log.info("    rows: %d (synthetic: %d)", n_rows, sum(1 for r in rows_data if r.get("synthetic")))
    if debug:
        _write_debug(page, "05_rows.jpg", _overlay_rows(data_region, rows_data))

    # 6. Column detection on the framed image. The frame already anchors x=0
    # (= x_left_col, leftmost printed line) and x=fw (= page split, rightmost).
    # detect_columns finds the 18 interior printed lines between them and
    # forces exactly EXPECTED_COLS=19 columns.
    col_ranges = detect_columns(framed, table_left_x=0,
                                expected_cols=EXPECTED_COLS)
    bands = detect_columns_banded(framed, col_ranges, n_bands=N_BANDS)
    log.info("    cols: %d (expected %d), bands usable: %d/%d  (frame: x=[0,%d])",
             len(col_ranges) - 1, EXPECTED_COLS, len(bands), N_BANDS, fw)
    if debug:
        _write_debug(page, "06_cols.jpg", _overlay_cols(framed, bands))

    # 7. Build a single remap that covers BOTH pheader and data. Row centers
    # come from detect_rows (in data_region coords) → shift by hb_y for framed.
    rows_framed = [{**r, "y_center": r["y_center"] + hb_y} for r in rows_data]
    map_x, map_y, _, out_h = build_remap(
        framed, rows_framed, bands,
        pheader_anchor=(0, hb_y, H_PHEADER),
    )
    dewarped_full = cv2.remap(
        framed, map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    if debug:
        _write_debug(page, "07_dewarped_full.jpg", dewarped_full)

    # 8. Compose final canvas: meta strip + dewarped (pheader+data) resized to
    # (W_OUT, H_PHEADER + n_rows*ROW_PITCH).
    target_h = H_PHEADER + n_rows * ROW_PITCH
    body_norm = cv2.resize(dewarped_full, (W_OUT, target_h), interpolation=cv2.INTER_AREA)
    meta_norm = _resize_meta(deskewed, y_offset, x_offset + x_l, x_offset + x_r)
    output    = np.vstack([meta_norm, body_norm])

    PROCESSED_DIR.mkdir(exist_ok=True)
    out_path = PROCESSED_DIR / f"Hadita-{page}Processed.jpg"
    cv2.imwrite(str(out_path), output, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    log.info("    wrote %s (%d×%d)", out_path.relative_to(ROOT), output.shape[1], output.shape[0])
    if debug:
        ov = output.copy()
        canvas_h = ov.shape[0]
        # Horizontal: meta/pheader boundary, pheader/data boundary
        cv2.line(ov, (0, H_META),   (W_OUT, H_META),   (0, 200, 200), 2)
        cv2.line(ov, (0, H_HEADER), (W_OUT, H_HEADER), (0, 200, 0), 3)
        # Vertical: uniform column grid — only inside the data region
        for j in range(EXPECTED_COLS + 1):
            x = int(round(j * W_OUT / EXPECTED_COLS))
            cv2.line(ov, (x, H_HEADER), (x, canvas_h), (0, 0, 220), 1)
        _write_debug(page, "08_processed_with_grid.jpg", ov)
    return {
        "path": out_path,
        "n_rows": n_rows,
        "n_cols": len(col_ranges) - 1,
        "bands_usable": len(bands),
        "synthetic_rows": sum(1 for r in rows if r.get("synthetic")),
        "out_w": output.shape[1],
        "out_h": output.shape[0],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _resolve_pages(args) -> list[int]:
    if args.all_except_cover:
        candidates = sorted(int(p.name.split("page-")[1].split(".")[0])
                            for p in ROOT.glob(ORIG_PATTERN.replace("{:04d}", "*")))
        out = []
        for n in candidates:
            p = ROOT / ORIG_PATTERN.format(n)
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape[1] > img.shape[0]:  # landscape only
                out.append(n)
        return out
    return sorted(args.pages or [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, nargs="+",
                        help="Page numbers to process (sequential ascending)")
    parser.add_argument("--all-except-cover", action="store_true",
                        help="Process every original landscape scan (skip portrait cover)")
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip deskew; require .ocr_cache/deskewed_page{N}.{png,jpg}")
    parser.add_argument("--debug", action="store_true",
                        help="Write per-step images to processed/_debug/page{N}/")
    args = parser.parse_args()

    pages = _resolve_pages(args)
    if not pages:
        parser.error("Provide --pages or --all-except-cover")

    log.info("Processing %d page(s): %s", len(pages), pages)
    failures: list[tuple[int, str]] = []
    for n in pages:
        try:
            process_page(n, debug=args.debug, from_cache=args.from_cache)
        except Exception as exc:
            log.warning("page %d failed: %s", n, exc)
            failures.append((n, str(exc)))

    log.info("Done. %d/%d pages succeeded.", len(pages) - len(failures), len(pages))
    if failures:
        for n, msg in failures:
            log.warning("  FAILED page %d: %s", n, msg)


if __name__ == "__main__":
    main()
