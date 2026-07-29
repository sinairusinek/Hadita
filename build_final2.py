#!/usr/bin/env python3
"""build_final2.py — Phase 1 of PLAN_GT_pipeline_2026-07: warp the coordinates.

`build_final.py` flattens the page (cv2.remap) and writes a straight grid onto
the flattened canvas. Phase 0 (`audit_dewarp_damage.py`) showed that costs
content on every page: the canvas ends at `last_row_center + ½ pitch`, so 25
pages lose a whole written row and 31 lose part of one, and the band between the
printed header and `first_row_center - ½ pitch` is compressed on 81 pages.

This script inverts the trade: upload the **undamaged deskewed page** and curve
the cell polygons to follow it. No remap is applied to any pixel.

Coordinate system
  image   = deskewed[:, 0:x_right_split]  — the whole left page, full height
  table   = "framed" coords, i.e. image coords shifted up by the wide-crop
            offset y_offset; this is the space `detect_columns_banded` reports
            band positions in, so `write_page_xml(bands=...)` interpolates each
            column boundary's x at each row's y and the quads bow with the page.
  rows    = horizontal edges at the midpoints between detected row centers —
            the same row model the dewarp used, minus the two clamped edges
            that did the cutting.

Row coverage differs from `final/` in one deliberate way: where the audit finds
a written row *below* the last detected one, extra rows are appended (`--no-
extend-rows` to disable). Rows are only ever appended, never prepended — row
indices must stay aligned with the proxy-GT transcripts for pages 3-10. Missing
rows at the *top* are reported as anomalies instead of being fixed silently.

Outputs (never touches `Transkribus upload/final/`):
  Transkribus upload/final2/Hadita_{N}.jpeg + .xml
  debug/final2_overlay/page{N}.jpg       (with --overlay)
  final2_build.tsv                       per-page row/col counts and anomalies

Usage:
  python build_final2.py --page 3 --page 4 --overlay
  python build_final2.py --gt-pages --overlay      # the 6 proxy-GT pages
  python build_final2.py                           # every page in final/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from audit_dewarp_damage import INK_LUMA_OFFSET, INK_ROW_FRAC, ink_profile  # noqa: E402
from dewarp import CACHE_DIR, CFG, load_or_make_deskewed  # noqa: E402
from patch_baselines import patch_xml  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from segment_unified import (  # noqa: E402
    EXPECTED_COLS, N_BANDS, PAGE_CONFIG, W2E, _page_cfg,
    crop_table, detect_columns, detect_columns_banded, detect_rows,
    detect_table_frame, fix_x_left_col_geometric, interp_col_x,
    load_text_rows, recognize_top_strip, write_page_xml,
)

FINAL_DIR = ROOT / "Transkribus upload" / "final"
FINAL2_DIR = ROOT / "Transkribus upload" / "final2"
OVERLAY_DIR = ROOT / "debug" / "final2_overlay"
REPORT_TSV = ROOT / "final2_build.tsv"
GT_PAGES = [3, 4, 5, 6, 9, 10]
MAX_EXTRA_ROWS = 6   # guard: never invent a whole page of rows from noise
NARROW_FROM = 7      # boundary index from which columns are uniformly narrow
SPLIT_RATIO = 1.6    # gap >= this x median narrow pitch = merged columns
ALIGN_TOL = 8        # px: boundary counts as on a printed line within this
BAND_OUTLIER_PX = 12 # band x further than this from its neighbours is a latch-on
FIRST_BAND_WIN = 40  # ± window for the top band, anchored on the global grid
TRACK_WIN = 22       # ± window when following a line from the band above
SHIFT_RANGE = 100    # px swept either way when fitting a grid onto the ruling
AGREE_PX = 10        # a shift is only trusted if detection agrees within this
MAX_BOW_PX = 130     # hard cap on a boundary's total drift from the global grid
HEAD_GAP_FLAG = 1.5  # first row this far below the header line = suspicious

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── columns ───────────────────────────────────────────────────────────────────

def repair_col_ranges(col_ranges: list[int]) -> tuple[list[int], int]:
    """Insert interior column boundaries that `detect_columns` missed.

    `detect_columns` forces exactly EXPECTED_COLS columns, but when it misses a
    faint printed line it pads the count elsewhere; `fix_x_left_col_geometric`
    then drops the spurious boundary and the page ends up one column short. In
    `final/` that left 59 of 98 pages with 18 columns: the last column
    (Net_Assessment_Mils) absent and columns 14-17 tagged one place off, since
    `write_page_xml` assigns col_tags positionally from LEFT_COLS.

    Columns from New_Serial_No rightwards are uniform in width, so a gap there
    of k × the median is k merged columns. Columns 0-6 (Serial_No, Date,
    Nature_of_Entry …) are legitimately wide and are never split.

    Returns (boundaries, n_inserted).
    """
    if len(col_ranges) < 12:
        return list(col_ranges), 0
    gaps = [col_ranges[i + 1] - col_ranges[i] for i in range(NARROW_FROM, len(col_ranges) - 1)]
    if not gaps:
        return list(col_ranges), 0
    pitch = float(np.median(gaps))
    if pitch <= 1:
        return list(col_ranges), 0

    # Split the widest offending gap first and stop as soon as the page has its
    # 19 columns: an unconditional pass over every wide gap overshoots (pages
    # that were already correct gained a 20th column).
    out = list(col_ranges)
    inserted = 0
    while len(out) - 1 < EXPECTED_COLS:
        widest, ratio = -1, 0.0
        for i in range(NARROW_FROM, len(out) - 1):
            r = (out[i + 1] - out[i]) / pitch
            if r > ratio:
                widest, ratio = i, r
        if widest < 0 or ratio < SPLIT_RATIO:
            break
        a, b = out[widest], out[widest + 1]
        k = min(int(round(ratio)), EXPECTED_COLS - (len(out) - 1) + 1)
        for j in range(k - 1, 0, -1):
            out.insert(widest + 1, int(round(a + j * (b - a) / k)))
            inserted += 1
    return out, inserted


def band_peaks(framed: np.ndarray,
               n_bands: int = N_BANDS) -> tuple[list[np.ndarray], list[int]]:
    """Vertical-line peak positions per horizontal band, and the band centers."""
    th, tw = framed.shape[:2]
    band_h = th // n_bands
    peaks, centers = [], []
    for b in range(n_bands):
        y0 = b * band_h
        y1 = th if b == n_bands - 1 else (b + 1) * band_h
        strip = framed[y0:y1, :]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        binary = cv2.adaptiveThreshold(
            norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5)
        v_mask = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)), iterations=1)
        v_mask = cv2.dilate(v_mask, np.ones((1, 2), np.uint8), iterations=1)
        proj = np.sum(v_mask, axis=0).astype(float)
        pk, _ = find_peaks(proj, height=proj.mean() + 0.3 * proj.std(),
                           distance=max(20, tw // 40))
        peaks.append(np.asarray(pk, dtype=float))
        centers.append((y0 + y1) // 2)
    return peaks, centers


def alignment_score(bands: list[dict], peaks: list[np.ndarray]) -> float:
    """Fraction of (boundary × band) positions sitting on a printed line.

    The direct measure of the thing that has been wrong most often — whether
    the grid follows the ruling — so candidate column sets can be compared
    instead of guessed between.
    """
    hits = total = 0
    for b, band in enumerate(bands):
        if b >= len(peaks) or peaks[b].size == 0:
            continue
        for x in band["col_x"]:
            hits += int(np.min(np.abs(peaks[b] - x)) <= ALIGN_TOL)
            total += 1
    return hits / total if total else 0.0


def best_shift(framed: np.ndarray, col_ranges: list[int],
               peaks: tuple[list[np.ndarray], list[int]],
               detected: list[int]) -> tuple[list[int], int, float]:
    """Slide a cached grid sideways onto the printed lines.

    The cached boundaries were computed against an older deskew, so on some
    pages the whole grid sits tens of pixels off the ruling even though its
    internal spacing is right (page 12 scores 0.06 as cached, 0.96 shifted).

    A shift may only be accepted when it brings the grid into agreement with
    the *independently detected* boundaries. Line-alignment alone is not
    enough evidence: on page 77 the best-scoring offset lands the whole grid a
    uniform 50px from where detection puts it — a higher score achieved by
    sitting on different lines. Requiring two methods to agree rules that out.

    Returns (boundaries, offset, score).
    """
    tw = framed.shape[1]

    def apply(offset: int) -> list[int]:
        return [int(min(max(x + offset, 0), tw - 1)) for x in col_ranges]

    def score(offset: int) -> float:
        r = apply(offset)
        b = clamp_bands(smooth_bands(track_bands(framed, r, peaks=peaks), r), tw)
        return alignment_score(b, peaks[0])

    def agrees(offset: int) -> bool:
        if not detected:
            return offset == 0
        shifted = apply(offset)[1:-1]
        if not shifted:
            return offset == 0
        return float(np.median([min(abs(x - y) for y in detected)
                                for x in shifted])) <= AGREE_PX

    allowed = [o for o in range(-SHIFT_RANGE, SHIFT_RANGE + 1, 4) if agrees(o)] or [0]
    coarse = max(allowed, key=score)
    fine = max([o for o in range(coarse - 3, coarse + 4) if agrees(o)] or [coarse],
               key=score)
    return apply(fine), fine, score(fine)


def track_bands(framed: np.ndarray, col_ranges: list[int],
                n_bands: int = N_BANDS,
                peaks: tuple[list[np.ndarray], list[int]] | None = None) -> list[dict]:
    """Follow each printed column line down the page, band by band.

    `detect_columns_banded` matches every band independently against the
    *global* boundary within ±60px. Near the spine the bow reaches that limit
    and the search grabs the neighbouring line instead: on pages 3 and 6 the
    per-band deviation runs smoothly to −49px and then jumps to +57px for the
    last three bands, which drags the bottom rows' cells a half-column off —
    precisely the rows this rebuild exists to recover.

    Tracking from the previous band's position instead keeps the window small
    (±TRACK_WIN), so a line can bow arbitrarily far overall but can never jump
    to its neighbour. Bands where a line isn't found carry the previous
    position forward at the previous rate.
    """
    peaks_per_band, centers = peaks or band_peaks(framed, n_bands)
    tw = framed.shape[1]
    bands: list[dict] = []
    n_bounds = len(col_ranges)
    prev = [float(x) for x in col_ranges]
    vel = [0.0] * n_bounds
    for b in range(n_bands):
        peaks = peaks_per_band[b]
        col_x, used = [], set()
        for j in range(n_bounds):
            target = prev[j] + vel[j]
            win = FIRST_BAND_WIN if b == 0 else TRACK_WIN
            cand = [p for p in peaks if abs(p - target) <= win and p not in used]
            if cand:
                x = min(cand, key=lambda p: abs(p - target))
                used.add(x)
            else:
                x = target                      # carry forward at the same rate
            x = float(min(max(x, col_ranges[j] - MAX_BOW_PX),
                          col_ranges[j] + MAX_BOW_PX))
            # The outer boundaries sit on the image edges, so their bow would
            # otherwise put cell corners at negative x / past the right edge —
            # Transkribus rejects coordinates outside the page.
            x = float(min(max(x, 0), tw - 1))
            vel[j] = x - prev[j] if b else 0.0
            prev[j] = x
            col_x.append(int(round(x)))
        bands.append({"y_center": centers[b], "col_x": col_x})
    return bands


def clamp_bands(bands: list[dict], width: int) -> list[dict]:
    """Keep every boundary inside the image (the median filters can push out)."""
    for b in bands:
        b["col_x"] = [int(min(max(x, 0), width - 1)) for x in b["col_x"]]
    return bands


def smooth_bands(bands: list[dict], col_ranges: list[int]) -> list[dict]:
    """Fit each column boundary's x-vs-y across bands, rejecting outliers.

    `detect_columns_banded` snaps every boundary to the nearest peak within
    ±60px per band, so a boundary can latch onto a handwriting stroke instead
    of the printed rule and zigzag by tens of pixels (visible on page 3).

    A running median over three adjacent bands removes exactly that — a single
    band out of line with its neighbours — while leaving the page bow intact.
    Fitting a curve instead was tried and rejected: bands where no line is found
    fall back to the *global* boundary, and a least-squares fit through that mix
    tilts every boundary off the printed rule.
    """
    if len(bands) < 3:
        return bands
    n_bounds = len(bands[0]["col_x"])
    out = [{"y_center": b["y_center"], "col_x": list(b["col_x"])} for b in bands]
    for j in range(n_bounds):
        xs = [b["col_x"][j] for b in bands]
        for i in range(1, len(bands) - 1):
            med = int(sorted(xs[i - 1:i + 2])[1])
            if abs(xs[i] - med) > BAND_OUTLIER_PX:
                out[i]["col_x"][j] = med

    # A boundary can also be latched in *every* band — a faint printed rule with
    # handwriting beside it, which no filter along y can see. The bow is smooth
    # across the page too, so a boundary whose offset from the global grid
    # disagrees with both its neighbours' offsets is following the wrong line.
    if len(col_ranges) == n_bounds:
        for band in out:
            off = [band["col_x"][j] - col_ranges[j] for j in range(n_bounds)]
            for j in range(1, n_bounds - 1):
                med = int(sorted(off[j - 1:j + 2])[1])
                if abs(off[j] - med) > BAND_OUTLIER_PX:
                    band["col_x"][j] = col_ranges[j] + med
    return out


# ── geometry ──────────────────────────────────────────────────────────────────

def row_edges(centers: np.ndarray, pitch: float, limit: int) -> list[int]:
    """Row boundaries: midpoints between centers, half a pitch at each end."""
    mids = (centers[:-1] + centers[1:]) / 2
    edges = np.concatenate([[centers[0] - pitch / 2], mids, [centers[-1] + pitch / 2]])
    return [int(round(min(max(e, 0), limit))) for e in edges]


def written_rows_outside(prof: np.ndarray, centers: np.ndarray, pitch: float,
                         start: int, stop: int) -> int:
    """Count pitch-sized bands of handwriting between `start` and `stop`.

    Bands are anchored at the row-band edge and walk outward; see
    audit_dewarp_damage for why edge-anchoring matters. Counting stops at the
    first empty band, so a page's blank ruled tail is not mistaken for content.
    """
    n_px = len(prof)
    per_row = []
    for c in centers:
        a, b = max(0, int(round(c - pitch / 2))), min(n_px, int(round(c + pitch / 2)))
        if b > a:
            per_row.append(prof[a:b].sum())
    typical = float(np.median(per_row)) if per_row else 0.0
    if not typical:
        return 0
    step = pitch if stop > start else -pitch
    n, y = 0, float(start)
    while abs(stop - y) >= pitch and n < MAX_EXTRA_ROWS:
        a, b = sorted((int(y), int(y + step)))
        seg = prof[max(0, a):min(n_px, b)]
        if not seg.size or seg.sum() / typical < INK_ROW_FRAC:
            break
        n += 1
        y += step
    return n


def page_geometry(page: int, extend_rows: bool = True,
                  extend_top: bool = True) -> dict:
    """Everything the XML needs, all in deskewed/framed coordinates."""
    deskewed = load_or_make_deskewed(page, from_cache=True)
    wide, y_offset, x_offset = crop_table(deskewed, CFG)
    frame = detect_table_frame(wide)
    x_r, hb_y = frame["x_right_split"], frame["header_bottom_y"]
    framed = wide[:, 0:x_r]
    data = framed[hb_y:, :]
    fh = framed.shape[0]

    seg_cache = CACHE_DIR / f"dewarp_seg_page{page}.json"
    if not seg_cache.exists():
        raise FileNotFoundError(f"no cached segmentation for page {page}")
    rows = detect_rows(data, cache_path=seg_cache, use_cache=True,
                       method="kraken", skip_header_y=0)
    if len(rows) < 2:
        raise ValueError(f"only {len(rows)} rows detected")

    # Work in framed coords from here: the rows lost at the top can sit above
    # hb_y, where a profile of the data region alone would never see them.
    centers = np.array([r["y_center"] for r in rows], dtype=float) + hb_y
    pitch = float(np.median(np.diff(centers)))
    prof = ink_profile(framed)

    band_hi = min(fh, int(round(centers[-1] + pitch / 2)))
    band_lo = max(0, int(round(centers[0] - pitch / 2)))
    n_below = written_rows_outside(prof, centers, pitch, band_hi, fh)
    # Stop at the header line: above it the profile picks up printed header
    # text, which would read as a row of handwriting on every page.
    n_above = written_rows_outside(prof, centers, pitch, band_lo, hb_y)
    added = added_top = 0
    if extend_rows and n_below:
        extra = centers[-1] + pitch * np.arange(1, n_below + 1)
        extra = extra[extra + pitch / 2 <= fh]
        centers = np.concatenate([centers, extra])
        added = len(extra)
    if extend_top and n_above:
        # Prepending shifts row indices, so it is confined to pages with no RA
        # work — the six proxy-GT pages have no unmodelled top rows.
        extra = centers[0] - pitch * np.arange(n_above, 0, -1)
        extra = extra[extra - pitch / 2 >= hb_y]
        centers = np.concatenate([extra, centers])
        added_top = len(extra)

    centers_framed = centers
    edges = row_edges(centers_framed, pitch, fh)
    row_ranges = list(zip(edges[:-1], edges[1:]))

    # Columns: prefer the cache that produced final/ over fresh detection —
    # re-detecting drifts (page 4 loses a real boundary and doubles a column).
    cols_cache = CACHE_DIR / f"dewarp_cols_page{page}.json"
    cached = json.loads(cols_cache.read_text()) if cols_cache.exists() else None
    # Neither source wins everywhere. The cache reproduces the shipped columns,
    # but on pages 1-12 it was computed against a ~75px narrower frame, so its
    # boundaries miss the printed rules entirely. Fresh detection tracks the
    # current image but sometimes drops a column. So build both, measure each
    # against the printed lines, and prefer 19 columns, then alignment.
    peaks = band_peaks(framed)
    detected_raw = fix_x_left_col_geometric(
        detect_columns(framed, table_left_x=0, expected_cols=EXPECTED_COLS))
    candidates = []
    if cached:
        raw_cache = list(cached["col_ranges_framed"])
        candidates.append(("cache", raw_cache))
        fw = cached.get("framed_w") or framed.shape[1]
        if fw != framed.shape[1]:
            candidates.append(("cache-scaled",
                               [int(round(x * framed.shape[1] / fw)) for x in raw_cache]))
    candidates.append(("detected", detected_raw))

    scored = []
    for name, raw in candidates:
        repaired, inserted = repair_col_ranges(raw)
        repaired, offset, _ = best_shift(framed, repaired, peaks, detected_raw)
        if offset:
            name = f"{name}{offset:+d}px"
        bands_c = clamp_bands(
            smooth_bands(track_bands(framed, repaired, peaks=peaks), repaired),
            framed.shape[1])
        scored.append({
            "source": name, "col_ranges": repaired, "bands": bands_c,
            "inserted": inserted, "n_cols": len(repaired) - 1,
            "score": alignment_score(bands_c, peaks[0]),
        })
    best = max(scored, key=lambda s: (s["n_cols"] == EXPECTED_COLS, s["score"]))
    col_ranges, bands = best["col_ranges"], best["bands"]
    col_source, n_inserted = best["source"], best["inserted"]
    col_score = round(best["score"], 3)

    # A row written across the detected header line cannot be measured (the
    # printed header sits in the same band), so report it rather than guess.
    head_gap = (centers[0] - hb_y) / pitch

    return {
        "head_gap": head_gap,
        "image": deskewed[:, 0:x_r],
        "y_offset": y_offset, "hb_y": hb_y, "framed_h": fh,
        "row_ranges": row_ranges, "col_ranges": col_ranges, "bands": bands,
        "n_rows": len(row_ranges), "pitch": pitch,
        "rows_added": added, "rows_added_top": added_top,
        "rows_missing_top": n_above,
        "rows_recoverable_below": n_below,
        "col_source": col_source, "cols_inserted": n_inserted,
        "col_score": col_score,
    }


# ── overlay ───────────────────────────────────────────────────────────────────

def write_overlay(page: int, geo: dict) -> Path:
    """Draw the curved cell grid on the deskewed page for eyeball checking."""
    ov = geo["image"].copy()
    yo, bands = geo["y_offset"], geo["bands"]
    n_cols = len(geo["col_ranges"]) - 1
    n_added = geo["rows_added"]
    n_rows = geo["n_rows"]

    for r, (y0, y1) in enumerate(geo["row_ranges"]):
        recovered = r >= n_rows - n_added
        colour = (0, 140, 255) if recovered else (0, 0, 220)
        for y in (y0, y1):
            pts = [(interp_col_x(c, y, bands), y + yo) for c in range(n_cols + 1)]
            cv2.polylines(ov, [np.array(pts, np.int32)], False, colour, 2)
    # Column boundaries, sampled per band so the bow is visible.
    ys = list(range(geo["row_ranges"][0][0], geo["row_ranges"][-1][1], 20))
    for c in range(n_cols + 1):
        pts = [(interp_col_x(c, y, bands), y + yo) for y in ys]
        cv2.polylines(ov, [np.array(pts, np.int32)], False, (220, 60, 0), 2)
    cv2.line(ov, (0, geo["hb_y"] + yo), (ov.shape[1], geo["hb_y"] + yo), (0, 190, 0), 2)
    cv2.line(ov, (0, yo), (ov.shape[1], yo), (200, 200, 0), 2)

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = OVERLAY_DIR / f"page{page}.jpg"
    cv2.imwrite(str(out), ov, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return out


# ── per page ──────────────────────────────────────────────────────────────────

def build_page(page: int, extend_rows: bool = True, overlay: bool = False,
               top_strip: bool = True, extend_top: bool = True) -> dict:
    if page not in PAGE_CONFIG:
        PAGE_CONFIG[page] = _page_cfg(page)

    geo = page_geometry(page, extend_rows=extend_rows, extend_top=extend_top)
    FINAL2_DIR.mkdir(parents=True, exist_ok=True)

    img = geo["image"]
    jpeg = FINAL2_DIR / f"Hadita_{page}.jpeg"
    cv2.imwrite(str(jpeg), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    strip = None
    if top_strip:
        try:
            strip = recognize_top_strip(img, geo["y_offset"], page)
        except Exception as exc:                      # Kraken is optional here
            log.warning("page %d: top strip failed (%s)", page, exc)

    xml = FINAL2_DIR / f"Hadita_{page}.xml"
    write_page_xml(
        geo["col_ranges"], geo["row_ranges"],
        y_offset=geo["y_offset"],
        page_w=img.shape[1], page_h=img.shape[0],
        image_filename=jpeg.name,
        out_path=xml,
        text_rows=load_text_rows(page, PAGE_CONFIG[page]),
        bands=geo["bands"],
        text_fn=lambda t: t.translate(W2E),
        col_tags=True,
        row_baseline_y=None,
        top_strip=strip,
    )
    patch_xml(xml)

    if overlay:
        write_overlay(page, geo)
    return geo


def final_rows(page: int) -> int:
    """Row count in the shipped final/ XML, for comparison."""
    p = FINAL_DIR / f"Hadita_{page}.xml"
    if not p.exists():
        return -1
    import re
    m = re.search(r'<TableRegion[^>]*rows="(\d+)"', p.read_text(encoding="utf-8"))
    return int(m.group(1)) if m else -1


COLUMNS = ["page", "n_rows", "n_cols", "final_rows", "rows_added",
           "rows_added_top", "rows_missing_top", "col_source", "col_score",
           "page_w", "page_h", "notes"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--page", type=int, action="append", dest="pages")
    ap.add_argument("--gt-pages", action="store_true",
                    help=f"build only the proxy-GT pages {GT_PAGES}")
    ap.add_argument("--overlay", action="store_true",
                    help="write debug/final2_overlay/page{N}.jpg")
    ap.add_argument("--no-extend-rows", action="store_true",
                    help="do not append rows the dewarp cut off the bottom")
    ap.add_argument("--no-extend-top", action="store_true",
                    help="do not prepend rows lost above the detected header line "
                         "(prepending shifts row indices)")
    ap.add_argument("--no-top-strip", action="store_true",
                    help="skip Kraken recognition of the metadata strip")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if args.gt_pages:
        pages = GT_PAGES
    elif args.pages:
        pages = sorted(args.pages)
    else:
        pages = sorted(int(p.stem.split("_")[1]) for p in FINAL_DIR.glob("Hadita_*.jpeg")
                       if p.stem.split("_")[1].isdigit())

    print(f"Building {len(pages)} page(s) → {FINAL2_DIR.relative_to(ROOT)}/\n")
    records, failed = [], []
    for i, page in enumerate(pages, 1):
        try:
            geo = build_page(page, extend_rows=not args.no_extend_rows,
                             overlay=args.overlay, top_strip=not args.no_top_strip,
                             extend_top=not args.no_extend_top)
            fr = final_rows(page)
            notes = []
            if geo["rows_added"]:
                notes.append(f"+{geo['rows_added']} recovered rows")
            if geo["head_gap"] >= HEAD_GAP_FLAG:
                notes.append(f"first row starts {geo['head_gap']:.1f} pitches below "
                             "the header line — a row may straddle it, unmeasurable")
            if geo["rows_added_top"]:
                notes.append(f"+{geo['rows_added_top']} recovered top row(s) "
                             "(row indices shift by that much)")
            elif geo["rows_missing_top"]:
                notes.append(f"{geo['rows_missing_top']} written row(s) above the "
                             "first band NOT modelled")
            if fr >= 0 and geo["n_rows"] - geo["rows_added"] - geo["rows_added_top"] != fr:
                notes.append(f"row count differs from final/ ({fr}) beyond recovery")
            records.append({
                "page": page, "n_rows": geo["n_rows"],
                "n_cols": len(geo["col_ranges"]) - 1, "final_rows": fr,
                "rows_added": geo["rows_added"],
                "rows_added_top": geo["rows_added_top"],
                "rows_missing_top": geo["rows_missing_top"],
                "col_source": geo["col_source"], "col_score": geo["col_score"],
                "page_w": geo["image"].shape[1], "page_h": geo["image"].shape[0],
                "notes": "; ".join(notes),
            })
            print(f"[{i}/{len(pages)}] page {page:>3}  {geo['n_rows']}r × "
                  f"{len(geo['col_ranges'])-1}c  (final/: {fr}r)  "
                  f"cols={geo['col_source']}/{geo['col_score']:.2f}  "
                  f"{'; '.join(notes) or 'ok'}")
        except Exception as exc:
            failed.append((page, str(exc).split("\n")[0]))
            print(f"[{i}/{len(pages)}] page {page:>3}  FAILED: {exc}")
            for ext in (".jpeg", ".xml"):
                f = FINAL2_DIR / f"Hadita_{page}{ext}"
                if f.exists():
                    f.unlink()

    with open(REPORT_TSV, "w", encoding="utf-8") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in records:
            fh.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")

    print(f"\n{len(records)}/{len(pages)} pages built; report → {REPORT_TSV.name}")
    added = sum(r["rows_added"] for r in records)
    miss = sum(1 for r in records if r["rows_missing_top"] and not r["rows_added_top"])
    print(f"  {added} rows recovered at the bottom across "
          f"{sum(1 for r in records if r['rows_added'])} pages")
    top = sum(r["rows_added_top"] for r in records)
    print(f"  {top} rows recovered at the top across "
          f"{sum(1 for r in records if r['rows_added_top'])} pages "
          "(row indices shift on those pages)")
    print(f"  {miss} pages still have written row(s) above the first modelled band")
    if failed:
        print(f"  {len(failed)} failed:")
        for p, e in failed:
            print(f"    page {p}: {e}")


if __name__ == "__main__":
    main()
