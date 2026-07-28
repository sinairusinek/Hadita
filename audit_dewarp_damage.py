#!/usr/bin/env python3
"""audit_dewarp_damage.py — Phase 0 of PLAN_GT_pipeline_2026-07.

Measure, per page in `Transkribus upload/final/`, how much table content the
dewarp remap clipped, smeared or dropped relative to the *source* deskewed crop.

Two independent evidence streams per page:

A. Geometry (source side, from the cached Kraken segmentation)
   Re-derive exactly what `dewarp.process_page()` derived — deskewed image →
   wide crop → table frame → data region → `detect_rows` — without running the
   remap, then measure:
     - top_gap / bot_gap    : distance (in pitch units) from the data-region
                              edge to the first / last detected row center.
                              The remap allots each row a ±½-pitch band, so a
                              gap below ~½ pitch means that band runs off the
                              source image and gets clamped (BORDER_REPLICATE).
     - bottom_overflow_px   : (last_center + ½ pitch) - data_region_height.
                              > 0 ⇒ the bottom row band is extrapolated.
     - ink_below / ink_above: ink still present outside the modelled row band
                              span, expressed as a fraction of a typical row's
                              ink. ≳0.3 means a whole row of content exists
                              that no row band covers ⇒ dropped row.

B. Canvas (output side, from the shipped JPEG)
   BORDER_REPLICATE smear shows up as consecutive near-identical pixel rows.
   Count them at the top of the data region and at the bottom of the canvas.

Outputs
  damage_audit.tsv            one row per page, with flags
  debug/damage_audit/*.jpg    contact sheets (source vs canvas, top + bottom
                              strips), 6 pages per sheet

No API cost; segmentation and deskew both come from `.ocr_cache/`.

Usage:
  python audit_dewarp_damage.py                 # all pages in final/
  python audit_dewarp_damage.py --pages 3 4 9   # subset
  python audit_dewarp_damage.py --no-sheets     # tsv only
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

from dewarp import CFG, CACHE_DIR, H_HEADER, ROW_PITCH  # noqa: E402
from segment_unified import crop_table, detect_rows, detect_table_frame  # noqa: E402

FINAL_DIR = ROOT / "Transkribus upload" / "final"
SHEET_DIR = ROOT / "debug" / "damage_audit"
TSV_OUT = ROOT / "damage_audit.tsv"

# Flag thresholds (all gaps expressed in pitch units; a row band is ±½ pitch)
CLIP_FRAC = 0.30          # gap below this ⇒ the edge row band is clipped
SQUEEZE_FRAC = 0.75       # gap above this ⇒ >1.5× compression into the ½-pitch band
INK_ROW_FRAC = 0.30       # unmodelled band ink ≥ this share of a typical row ⇒ a written row
INK_LUMA_OFFSET = 45      # ink = darker than (paper luma - this)
PARTIAL_INK_FRAC = 0.10   # unmodelled ink ≥ this ⇒ a row is partly cut
MARGIN_FRAC = 0.04        # left fraction of the crop ignored (scan border / gutter)
SMEAR_TOL = 2.0           # mean abs pixel diff below this ⇒ rows are duplicates
SMEAR_MIN = 4             # ≥ this many duplicate rows ⇒ visible smear band
SMEAR_MAX_LUMA = 215      # duplicated near-white rows are blank paper, not smear
PAGES_PER_SHEET = 3

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── source-side geometry ──────────────────────────────────────────────────────

def load_deskewed(page: int) -> np.ndarray | None:
    for p in (ROOT / "images" / f"deskewed_page{page}.jpg",
              CACHE_DIR / f"deskewed_page{page}.png",
              CACHE_DIR / f"deskewed_page{page}.jpg"):
        if p.exists():
            return cv2.imread(str(p))
    return None


def ink_profile(data_region: np.ndarray) -> np.ndarray:
    """Per-source-row count of *handwriting* pixels over the data region.

    The printed grid must be excluded: one full-width horizontal rule carries as
    much ink as a row of writing, and the vertical rules contribute in
    proportion to strip height — either would make an empty strip look occupied.
    Both are removed morphologically before profiling.

    Thresholding is a fixed offset below the page's own paper luma. Local
    (adaptive/CLAHE) thresholding was tried first and rejected: on blank ruled
    paper it amplifies texture until an empty band scores 60% of a written row,
    which is exactly the discrimination this audit needs. Going *darker* than
    ~-45 is also wrong — the printed rules thin out, escape the rule mask, and
    come back counted as handwriting.
    """
    gray = cv2.cvtColor(data_region, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thr = float(np.median(blur)) - INK_LUMA_OFFSET
    binary = ((blur < thr) * 255).astype(np.uint8)
    # The printed rules are faint and come out dotted, so close along each axis
    # before opening — otherwise the fragments survive as "handwriting".
    h_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8))
    v_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15, 1), np.uint8))
    h_rule = cv2.morphologyEx(h_closed, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (61, 1)))
    v_rule = cv2.morphologyEx(v_closed, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, 61)))
    # Scan border, book gutter and page-curl shadow are large solid dark areas;
    # handwriting strokes are thin, so a big square opening isolates the former.
    blobs = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    mask = cv2.dilate(cv2.bitwise_or(cv2.bitwise_or(h_rule, v_rule), blobs),
                      np.ones((5, 5), np.uint8))
    hand = cv2.bitwise_and(binary, cv2.bitwise_not(mask))
    hand = cv2.morphologyEx(hand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # The framed crop starts at x=0, which on most scans includes the black scan
    # border and the book-gutter fringe; that fringe runs the full page height
    # and would register as content in every band.
    hand[:, :int(MARGIN_FRAC * hand.shape[1])] = 0
    return (hand > 0).sum(axis=1).astype(float)


def source_geometry(page: int, allow_kraken: bool = False) -> dict:
    """Re-derive dewarp's source-side geometry for one page (no remap)."""
    deskewed = load_deskewed(page)
    if deskewed is None:
        raise FileNotFoundError(f"no cached deskewed image for page {page}")

    wide, y_offset, x_offset = crop_table(deskewed, CFG)
    frame = detect_table_frame(wide)
    hb_y = frame["header_bottom_y"]
    framed = wide[:, 0:frame["x_right_split"]]
    data = framed[hb_y:, :]

    seg_cache = CACHE_DIR / f"dewarp_seg_page{page}.json"
    if not seg_cache.exists() and not allow_kraken:
        # Never silently re-segment: a fresh Kraken run can differ from the one
        # that built the shipped XML, which would make the audit measure a page
        # that was never actually produced. Pass --allow-kraken deliberately.
        raise FileNotFoundError(f"no cached segmentation for page {page}")
    rows = detect_rows(data, cache_path=seg_cache, use_cache=True,
                       method="kraken", skip_header_y=0)
    if len(rows) < 2:
        raise ValueError(f"only {len(rows)} rows detected")

    centers = np.array([r["y_center"] for r in rows], dtype=float)
    pitch = float(np.median(np.diff(centers)))
    dh = data.shape[0]

    top_gap = float(centers[0])
    bot_gap = float(dh - centers[-1])

    # Ink outside the modelled row-band span [c0 - ½p, cN + ½p].
    prof = ink_profile(data)
    band_lo = int(round(max(0.0, centers[0] - pitch / 2)))
    band_hi = int(round(min(float(dh), centers[-1] + pitch / 2)))
    per_row_ink = []
    for c in centers:
        lo, hi = int(round(c - pitch / 2)), int(round(c + pitch / 2))
        lo, hi = max(0, lo), min(dh, hi)
        if hi > lo:
            per_row_ink.append(prof[lo:hi].sum())
    typical = float(np.median(per_row_ink)) if per_row_ink else 0.0
    ink_above = float(prof[:band_lo].sum()) / typical if typical else 0.0
    ink_below = float(prof[band_hi:].sum()) / typical if typical else 0.0

    # Whole-row accounting. Summed ink over a tall strip is misleading — many
    # pages end in a long tail of blank ruled rows whose paper noise adds up.
    # Instead, tile the outside region into pitch-sized bands and count how many
    # carry a real row's worth of handwriting.
    def written_bands(start: int, stop: int) -> int:
        """Bands of one pitch laid out from `start` towards `stop`.

        Anchoring must start at the row-band edge, not at the image edge: the
        leftover distance is not a whole number of pitches, so an edge-anchored
        grid splits the row adjacent to the cut across two bands and halves it.
        """
        if not typical:
            return 0
        step = pitch if stop > start else -pitch
        n, y = 0, float(start)
        while abs(stop - y) >= pitch / 2:
            a, b = sorted((int(y), int(min(max(y + step, 0), dh))))
            seg = prof[a:b]
            if seg.size and seg.sum() / typical >= INK_ROW_FRAC:
                n += 1
            y += step
        return n

    rows_lost_bottom = written_bands(band_hi, dh)
    rows_squeezed_top = written_bands(band_lo, 0)

    return {
        "framed": framed, "data": data, "hb_y": hb_y,
        "n_rows": len(rows), "pitch": pitch, "data_h": dh,
        "top_gap": top_gap, "bot_gap": bot_gap,
        "band_lo": band_lo, "band_hi": band_hi,
        "c_first": float(centers[0]), "c_last": float(centers[-1]),
        "bottom_overflow": (centers[-1] + pitch / 2) - dh,
        "top_overflow": (pitch / 2) - centers[0],
        "ink_above": ink_above, "ink_below": ink_below,
        "rows_lost_bottom": rows_lost_bottom,
        "rows_squeezed_top": rows_squeezed_top,
        "synthetic_rows": sum(1 for r in rows if r.get("synthetic")),
    }


# ── output-side smear ─────────────────────────────────────────────────────────

def duplicate_run(canvas: np.ndarray, at: str) -> int:
    """Length of the run of near-identical consecutive pixel rows at an edge.

    Blank paper also repeats, so a run whose pixels are near-white is not
    counted — only replicated rows that still carry ink (a genuine
    BORDER_REPLICATE streak of table content) are reported.
    """
    g = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.int16)
    seq = g[::-1] if at == "bottom" else g[H_HEADER:]
    n = 0
    for i in range(1, min(len(seq), 3 * ROW_PITCH)):
        if np.abs(seq[i] - seq[i - 1]).mean() < SMEAR_TOL:
            n += 1
        else:
            break
    if n and float(seq[:n + 1].mean()) > SMEAR_MAX_LUMA:
        return 0  # blank margin, not a smear
    return n


# ── contact sheets ────────────────────────────────────────────────────────────

def _strip(img: np.ndarray, where: str, h: int) -> np.ndarray:
    if where == "top":
        return img[:h]
    return img[max(0, img.shape[0] - h):]


def _label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((26, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return np.vstack([bar, img])


def page_tile(page: int, geo: dict, canvas: np.ndarray, flags: str,
              width: int = 1500) -> np.ndarray:
    """2×2 tile: source vs canvas, top strip and bottom strip.

    On the source strips a coloured line marks the edge of the modelled row-band
    span: green at `c_first - ½p` (content above it is compressed into the first
    row band) and red at `c_last + ½p` (content below it never reaches the
    canvas). The canvas strips show what actually shipped.
    """
    p = int(round(geo["pitch"]))
    data, dh = geo["data"], geo["data_h"]

    lo, hi = geo["band_lo"], geo["band_hi"]
    src_top = data[0:min(dh, int(geo["c_first"] + p))].copy()
    cv2.line(src_top, (0, lo), (src_top.shape[1], lo), (0, 180, 0), 3)
    src_bot = data[max(0, int(geo["c_last"] - p)):dh].copy()
    y_cut = hi - max(0, int(geo["c_last"] - p))
    if 0 <= y_cut < src_bot.shape[0]:
        cv2.line(src_bot, (0, y_cut), (src_bot.shape[1], y_cut), (0, 0, 220), 3)

    can = canvas[H_HEADER:]
    can_top = _strip(can, "top", int(1.6 * ROW_PITCH))
    can_bot = _strip(can, "bottom", int(1.6 * ROW_PITCH))

    def fit(im, tag):
        h = max(1, int(im.shape[0] * width / max(1, im.shape[1])))
        return _label(cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA), tag)

    col_src = np.vstack([fit(src_top, f"p{page} SOURCE top (green = first row band start)"),
                         fit(src_bot, f"p{page} SOURCE bottom (red = canvas ends here)")])
    col_can = np.vstack([fit(can_top, f"p{page} CANVAS top"),
                         fit(can_bot, f"p{page} CANVAS bottom")])
    h = max(col_src.shape[0], col_can.shape[0])
    pad = lambda c: np.vstack(  # noqa: E731
        [c, np.full((h - c.shape[0], c.shape[1], 3), 230, np.uint8)])
    tile = np.hstack([pad(col_src), np.full((h, 8, 3), 60, np.uint8), pad(col_can)])
    header = np.full((30, tile.shape[1], 3), 245, np.uint8)
    cv2.putText(header, f"page {page}   {flags or 'clean'}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 160) if flags else (0, 110, 0), 2)
    return np.vstack([header, tile, np.full((10, tile.shape[1], 3), 60, np.uint8)])


# ── main ──────────────────────────────────────────────────────────────────────

def final_pages() -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in FINAL_DIR.glob("Hadita_*.jpeg")
                  if p.stem.split("_")[1].isdigit())


def flags_for(m: dict) -> list[str]:
    """Damage flags, from most to least consequential.

    The remap maps output row i onto source [c_i - ½p, c_i + ½p] and *stops* at
    c_last + ½p, so anything below that is cut outright; anything between the
    header bottom and c_first - ½p is compressed into the first band instead.
    """
    f = []
    top, bot = m["top_gap"] / m["pitch"], m["bot_gap"] / m["pitch"]
    # Lost content — the decisive evidence.
    if m["rows_lost_bottom"] >= 1:
        f.append(f"LOST_{m['rows_lost_bottom']}_BOTTOM_ROW"
                 + ("S" if m["rows_lost_bottom"] > 1 else ""))
    elif m["ink_below"] >= PARTIAL_INK_FRAC:
        f.append("PARTIAL_BOTTOM_INK")
    if m["rows_squeezed_top"] >= 1:
        f.append(f"SQUEEZED_{m['rows_squeezed_top']}_TOP_ROW"
                 + ("S" if m["rows_squeezed_top"] > 1 else ""))
    # Geometry that causes it.
    if bot > SQUEEZE_FRAC:
        f.append("BOTTOM_CUT")
    if top > SQUEEZE_FRAC:
        f.append("TOP_SQUEEZE")
    if top < CLIP_FRAC:
        f.append("TOP_CLIP")
    if bot < CLIP_FRAC:
        f.append("BOTTOM_CLIP")
    if m["bottom_overflow"] > 0:
        f.append("BOTTOM_EXTRAP")
    if m["top_overflow"] > 0:
        f.append("TOP_EXTRAP")
    # Output-side confirmation.
    if m["smear_bottom"] >= SMEAR_MIN:
        f.append("BOTTOM_SMEAR")
    if m["smear_top"] >= SMEAR_MIN:
        f.append("TOP_SMEAR")
    return f


COLUMNS = ["page", "n_rows", "synthetic_rows", "pitch", "data_h",
           "top_gap_pitch", "bot_gap_pitch", "top_overflow_px", "bottom_overflow_px",
           "smear_top_px", "smear_bottom_px", "ink_above", "ink_below",
           "rows_lost_bottom", "rows_squeezed_top", "flags"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--allow-kraken", action="store_true",
                    help="re-run Kraken segmentation when the cache is missing "
                         "(writes .ocr_cache/dewarp_seg_page{N}.json)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    pages = sorted(args.pages) if args.pages else final_pages()
    print(f"Auditing {len(pages)} pages\n")

    records, tiles, errors = [], [], []
    for i, page in enumerate(pages, 1):
        try:
            geo = source_geometry(page, allow_kraken=args.allow_kraken)
            canvas = cv2.imread(str(FINAL_DIR / f"Hadita_{page}.jpeg"))
            if canvas is None:
                raise FileNotFoundError("no canvas jpeg in final/")
            m = dict(geo)
            m["smear_top"] = duplicate_run(canvas, "top")
            m["smear_bottom"] = duplicate_run(canvas, "bottom")
            fl = flags_for(m)
            records.append({
                "page": page, "n_rows": m["n_rows"],
                "synthetic_rows": m["synthetic_rows"],
                "pitch": round(m["pitch"], 1), "data_h": m["data_h"],
                "top_gap_pitch": round(m["top_gap"] / m["pitch"], 3),
                "bot_gap_pitch": round(m["bot_gap"] / m["pitch"], 3),
                "top_overflow_px": round(m["top_overflow"], 1),
                "bottom_overflow_px": round(m["bottom_overflow"], 1),
                "smear_top_px": m["smear_top"], "smear_bottom_px": m["smear_bottom"],
                "ink_above": round(m["ink_above"], 3),
                "ink_below": round(m["ink_below"], 3),
                "rows_lost_bottom": m["rows_lost_bottom"],
                "rows_squeezed_top": m["rows_squeezed_top"],
                "flags": ",".join(fl),
            })
            if not args.no_sheets:
                tiles.append(page_tile(page, geo, canvas, ",".join(fl)))
            print(f"[{i}/{len(pages)}] page {page:>3}  {len(fl)} flag(s)  "
                  f"{','.join(fl) or 'clean'}")
        except Exception as exc:
            errors.append((page, str(exc).split("\n")[0]))
            print(f"[{i}/{len(pages)}] page {page:>3}  ERROR: {exc}")

    tsv_out = TSV_OUT if not args.pages else TSV_OUT.with_name("damage_audit_subset.tsv")
    with open(tsv_out, "w", encoding="utf-8") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in records:
            fh.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")
    print(f"\nWrote {tsv_out.name} ({len(records)} pages)")

    if tiles:
        SHEET_DIR.mkdir(parents=True, exist_ok=True)
        if not args.pages:  # a subset run must not wipe the full-corpus sheets
            for old in SHEET_DIR.glob("sheet_*.jpg"):
                old.unlink()
        for s in range(0, len(tiles), PAGES_PER_SHEET):
            chunk = tiles[s:s + PAGES_PER_SHEET]
            w = max(t.shape[1] for t in chunk)
            chunk = [t if t.shape[1] == w else
                     np.hstack([t, np.full((t.shape[0], w - t.shape[1], 3), 230, np.uint8)])
                     for t in chunk]
            stem = "subset" if args.pages else "sheet"
            out = SHEET_DIR / f"{stem}_{s // PAGES_PER_SHEET + 1:02d}.jpg"
            cv2.imwrite(str(out), np.vstack(chunk), [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"Wrote {len(range(0, len(tiles), PAGES_PER_SHEET))} contact sheets "
              f"→ {SHEET_DIR.relative_to(ROOT)}/")

    flagged = [r for r in records if r["flags"]]
    print(f"\nFlagged {len(flagged)}/{len(records)} pages")
    counts: dict[str, int] = {}
    for r in flagged:
        for f in r["flags"].split(","):
            counts[f] = counts.get(f, 0) + 1
    for f, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {f:<22} {c}")
    if errors:
        print(f"\n{len(errors)} page(s) errored:")
        for p, e in errors:
            print(f"  page {p}: {e}")


if __name__ == "__main__":
    main()
