#!/usr/bin/env python3
"""build_final3.py — rebuild page geometry with harm-based validation.

Fixes the two final2 defects diagnosed 2026-08-29 (E12 + column review):
  1. ROWS: the printed table starts directly under the header, so the first row
     edge is anchored there — rows are prepended at the global pitch until the
     gap to the header line is < TOP_GAP_MAX pitches. final2 prepended only
     where it saw ink, and under-recovered on ~half the corpus.
  2. COLUMNS: candidates now include detection on a background-normalized
     (pencil-enhanced) image, and the winner is chosen by the harm metric —
     the fraction of boundary-path pixels that cross handwriting (ink_cross)
     — not by distance-to-detectable-peaks alone, which approved final2 grids
     that slice values on faint-ruled pages (e.g. page 20).

Outputs (never touches final/ or final2/):
  Transkribus upload/final3/Hadita_{N}.jpeg + .xml
  debug/final3_overlay/page{N}.jpg   (--overlay)
  final3_build.tsv

Usage:
  python build_final3.py --page 20 --page 40 --overlay
  python build_final3.py                       # all pages in final2
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import build_final2 as b2  # noqa: E402
from build_final2 import (  # noqa: E402
    ALIGN_TOL, EXPECTED_COLS, GT_PAGES, alignment_score, band_peaks,
    best_shift, clamp_bands, repair_col_ranges, row_edges, smooth_bands,
    track_bands, write_overlay)
from dewarp import CACHE_DIR, CFG, load_or_make_deskewed  # noqa: E402
from gate_textlines import hand_mask  # noqa: E402
from make_som_enhanced import enhance  # noqa: E402
from patch_baselines import patch_xml  # noqa: E402
from audit_dewarp_damage import ink_profile  # noqa: E402

from segment_unified import (  # noqa: E402
    PAGE_CONFIG, W2E, _page_cfg, crop_table, detect_columns,
    detect_table_frame, detect_rows, fix_x_left_col_geometric, interp_col_x,
    load_text_rows, recognize_top_strip, write_page_xml)

log = logging.getLogger("final3")
FINAL3_DIR = ROOT / "Transkribus upload" / "final3"
OVERLAY_DIR = ROOT / "debug" / "final3_overlay"
REPORT_TSV = ROOT / "final3_build.tsv"
TOP_GAP_MAX = 0.8    # first row edge must sit within this many pitches of header
MAX_PREPEND = 6
HB_MAX = 350         # detected header_bottom_y above this is a misdetection (p100: 392 = first rule, lost a row)
HB_FALLBACK = 300    # healthy corpus range is 266-323 (constant printed form)
INK_DILATE = 3       # px dilation of the hand mask before boundary crossing test

# Median relative column widths over 59 healthy pages (wide Serial col first).
# The printed form is constant; a candidate whose widths stray far from this,
# or whose wide column is not index 0, has inserted/dropped a boundary and
# shifts every value's column index — the p17 failure class.
TEMPLATE = np.array([0.1577, 0.0623, 0.044, 0.0438, 0.0442, 0.044, 0.0682,
                     0.0521, 0.0446, 0.0443, 0.05, 0.0363, 0.0448, 0.0364,
                     0.0391, 0.0459, 0.0366, 0.0451, 0.0605])
DEV_MAX = 0.25       # max sum|w - TEMPLATE| for a candidate to be admissible
SPAN_MIN, SPAN_MAX = 2350, 2650   # px: table span across the corpus is ~2440-2540
R_BACK, R_FWD = 60, 450           # right-edge search window around detected x_r
FIT_MIN_HITS = 14    # of 18 header lines within ALIGN_TOL: fit trusted to set x_r
EDGE_LEFT_MIN = 10   # px: fitted edge this far left of the split = split on next page
ROW_PITCH_FALLBACK = 93.0   # px: printed row pitch, constant form (corpus median)
ROW_SNAP = 0.2       # of a pitch: a Kraken centre this close to the prediction is taken
ROW_PROM = 0.0015    # peak prominence on the strip-median darkness profile
ROW_STRIPS = 30      # vertical strips whose darkness profiles are medianed


def width_dev(col_ranges: list[int]) -> tuple[float, bool]:
    """(template deviation, wide-col-at-0) for a boundary list."""
    w = np.diff(np.asarray(col_ranges, dtype=float))
    if len(w) != len(TEMPLATE) or w.sum() <= 0:
        return 9.9, False
    w = w / w.sum()
    return float(np.abs(w - TEMPLATE).sum()), int(np.argmax(w[:5])) == 0


def header_line_peaks(framed: np.ndarray, hb_true: int) -> np.ndarray:
    """x-positions of printed vertical rulings inside the header band.

    The header band is print-only (no handwriting), so its verticals are the
    cleanest column anchors on the page — the user's observation that the
    Serial/Date line 'starts in the header and continues downwards'.
    """
    # the sub-column splits (LP/Mils etc.) are short lines that exist only in
    # the bottom ~100px of the header, so the strip stays inside that band
    strip = framed[max(0, hb_true - 110):max(1, hb_true - 4), :]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5)
    v = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))
    proj = np.sum(v, axis=0).astype(float)
    pk, _ = find_peaks(proj, height=max(proj.max() * 0.2, 1.0),
                       distance=25)
    pk = np.asarray(pk, dtype=float)
    # binding shadows produce dense peak clusters at the far left: the table
    # is never wider than ~2450px, so anything left of that is an artifact
    return pk[pk >= framed.shape[1] - 2450] if pk.size else pk


def fit_template_frame(wide: np.ndarray, hb_true: int, x_r: int) -> dict:
    """Fit the 19-column template to the header-band verticals of the *wide*
    crop, with BOTH the right edge and the span free.

    Why both: `detect_table_frame` takes the leftmost long vertical in the
    rightmost 12% of the crop as the page split, and on narrow crops (p17)
    that is the NetLP|NetMils rule, ~150px left of the binding — the image
    then loses its last column and every right-anchored fit is wrong. The
    header band has 18 interior printed lines at fixed proportions; 14+ of
    them hitting within ALIGN_TOL is unambiguous, so the right edge is taken
    from the fit whenever it is that confident.

    The left edge is derived (r - span) and may fall left of the crop (p12:
    x=-3): the span range must not be limited by it.
    """
    cum = np.concatenate([[0.0], np.cumsum(TEMPLATE)])
    hpk = header_line_peaks(wide, hb_true)
    hpk = hpk[hpk >= x_r - 2600] if hpk.size else hpk
    wide_w = wide.shape[1]
    if not hpk.size:
        return {"r": x_r, "span": x_r - 80, "score": 0.0, "hits": 0, "peaks": hpk}
    spans = np.arange(SPAN_MIN, SPAN_MAX + 1, 2, dtype=float)
    best = None
    for r in range(max(400, x_r - R_BACK), min(wide_w - 3, x_r + R_FWD) + 1, 2):
        inter = r - spans[:, None] * (1.0 - cum[None, 1:-1])          # (S, 18)
        dmin = np.abs(inter[:, :, None] - hpk[None, None, :]).min(axis=2)
        hits = (dmin <= ALIGN_TOL).sum(axis=1)
        sc = (hits + 0.5 * (dmin <= 2 * ALIGN_TOL).sum(axis=1)
              - dmin.clip(0, 40).sum(axis=1) / 2000.0)
        if np.abs(hpk - r).min() <= ALIGN_TOL:                         # edge on a line
            sc = sc + 1.0
        i = int(sc.argmax())
        key = (round(float(sc[i]), 2), -abs(r - x_r))
        if best is None or key > best[0]:
            best = (key, r, int(spans[i]), float(sc[i]), int(hits[i]))
    _, r, span, score, hits = best
    return {"r": r, "span": span, "score": round(score, 2), "hits": hits, "peaks": hpk}


def resolve_right_edge(x_r: int, fit: dict) -> int:
    """Choose between the detected page split and the template's right edge.

    Checked visually on 16 pages (2026-08-29 montage, both print batches):
      * split within ALIGN_TOL of fitted boundary 18 (NetLP|NetMils rule) —
        the detector took the last column rule for the binding (p17, +152px):
        the fit is right and the image would otherwise lose a column.
      * fit >= EDGE_LEFT_MIN left of the split — the detector latched onto
        the right page's first rule across the fold (p27/p75/p98/p100, -30 to
        -58px); the fit sits on the fold's left edge, no content lies between.
      * fit right of the split (p3/p19/p20/p45, +16 to +66px) — the split is
        on the fold and the last Mils column is simply narrower there (bound
        into the spine); the template overshoots into the right page. Keep
        the detector.
    """
    if fit["hits"] < FIT_MIN_HITS:
        return x_r
    cum = np.concatenate([[0.0], np.cumsum(TEMPLATE)])
    b18 = fit["r"] - fit["span"] * (1.0 - cum[-2])
    if abs(x_r - b18) <= 1.5 * ALIGN_TOL:
        return fit["r"]
    if fit["r"] <= x_r - EDGE_LEFT_MIN:
        return fit["r"]
    return x_r


def template_candidate(framed: np.ndarray, peaks_all, fit: dict) -> list[int]:
    """Header-anchored template boundaries from a `fit_template_frame` result.

    Interior boundaries snap to a header line within ALIGN_TOL, else to an
    any-band ruling within 2*ALIGN_TOL; the left edge is derived, never
    detected — whatever lies left of the Serial/Date line is the Serial column.
    """
    cum = np.concatenate([[0.0], np.cumsum(TEMPLATE)])
    hpk = fit["peaks"]
    framed_w = framed.shape[1]
    bounds = [int(round(fit["r"] - fit["span"] * (1.0 - c))) for c in cum]
    flat_all = (np.unique(np.concatenate([p for p in peaks_all[0] if p.size]))
                if any(p.size for p in peaks_all[0]) else np.array([]))
    out = []
    for i, b in enumerate(bounds):
        if i not in (0, len(bounds) - 1):
            for src, tol in ((hpk, ALIGN_TOL), (flat_all, 2 * ALIGN_TOL)):
                if src.size:
                    near = src[np.argmin(np.abs(src - b))]
                    if abs(near - b) <= tol:
                        b = int(near)
                        break
        out.append(min(max(int(b), 0), framed_w - 1))
    # Serial numbers are routinely written in the left margin, outside the
    # printed table: everything left of the Serial/Date line belongs to the
    # Serial column, so its drawn edge extends to the paper edge.
    out[0] = min(out[0], 25)
    return out


def best_shift_tiebreak0(framed: np.ndarray, col_ranges: list[int],
                         peaks_all) -> tuple[list[int], int, float]:
    """`best_shift` with ties broken toward offset 0.

    build_final2.best_shift takes the first maximal offset in ascending order,
    so when tracked bands score the same for every small shift (they snap to
    the same rulings) the grid is moved to the most negative allowed offset —
    every final3 page came out "-10px". The template grid is header-anchored
    already; it should stay put unless a shift really scores higher.
    """
    tw = framed.shape[1]
    shifted, fine, sc = best_shift(framed, col_ranges, peaks_all, col_ranges)
    if fine == 0:
        return shifted, fine, sc
    base = clamp_bands(smooth_bands(track_bands(framed, col_ranges, peaks=peaks_all),
                                    col_ranges), tw)
    sc0 = alignment_score(base, peaks_all[0])
    if sc0 >= sc - 1e-9:
        return list(col_ranges), 0, sc0
    return shifted, fine, sc


def pin_outer_bands(bands: list[dict], col_ranges: list[int]) -> list[dict]:
    """Hold the two outer boundaries fixed in every band.

    They are crop edges (paper margin, page split), not printed rules, so
    there is nothing for `track_bands` to follow: on p12 the right edge
    caught the fold shadow in one band and then — carried forward "at the
    previous rate" — slid left band after band to the MAX_BOW_PX cap, closing
    the Net Mils column over the values written in the bottom rows. The
    interior metrics could not see it (ink_cross scores interior boundaries
    only). Pinning also gives smooth_bands' neighbour test a fixed anchor for
    boundaries 1 and n-1.
    """
    for b in bands:
        b["col_x"][0] = int(col_ranges[0])
        b["col_x"][-1] = int(col_ranges[-1])
    return bands


def rule_profile(framed: np.ndarray) -> np.ndarray:
    """Per-row darkness of the PRINTED RULES, isolated from the handwriting.

    The image is cut into ROW_STRIPS vertical strips; each strip gets a
    per-row mean darkness relative to local paper (capped, so strong ink
    cannot dominate); the profile is the MEDIAN across strips. A printed rule
    runs through every strip and survives the median; a written line touches
    a few strips and is suppressed. Verified 2026-08-29 on the 10 preview
    pages: 33/33 rules on p100 (the densest page) at 5.8 px max residual to a
    rigid lattice, and the peaks coincide with Kraken's line centres (median
    offset 0-6 px), i.e. the scribe writes with the baseline on the rule.
    """
    g = cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY).astype(np.float32)
    w = g.shape[1]
    profs = []
    for i in range(ROW_STRIPS):
        x0, x1 = int(i * w / ROW_STRIPS) + 3, int((i + 1) * w / ROW_STRIPS) - 3
        if x1 - x0 < 10:
            continue
        reg = g[:, x0:x1]
        bg = cv2.medianBlur(reg.astype(np.uint8), 51).astype(np.float32) + 1
        profs.append(np.clip(1.0 - reg / bg, 0, 0.06).mean(axis=1))
    prof = np.median(np.array(profs), axis=0)
    return np.convolve(prof, np.ones(3) / 3, mode="same")


def rule_peaks(framed: np.ndarray, hb_true: int) -> np.ndarray:
    prof = rule_profile(framed)
    pk, _ = find_peaks(prof, prominence=ROW_PROM, distance=60)
    return pk[pk >= hb_true + 40].astype(float)


def table_bottom(framed: np.ndarray, hb_true: int,
                 peaks: np.ndarray | None = None) -> int:
    """Last y where the printed column rules are still present: the verticals
    overshoot the last horizontal rule by ~40 px, and below them is the margin
    where the scribe writes the page totals (not a table row)."""
    g = cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 5)
    vm = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150)))
    cnt = (vm > 0).sum(axis=1).astype(float)
    fh = framed.shape[0]
    ref = np.percentile(cnt[hb_true + 200:fh - 300], 50) if fh - 300 > hb_true + 200 else cnt.max()
    ys = np.where(cnt >= 0.3 * max(ref, 1))[0]
    end = int(ys.max()) if ys.size else fh
    # The verticals can fade before the last ruled row (p12: verticals stop 3505,
    # printed rule at 3588 with two written rows above it). A rule found BELOW the
    # vertical end, within one pitch, is still table: extend to it.
    if peaks is not None and len(peaks):
        # Walk the rules down: the verticals can fade a row or more before the
        # ruling stops, so extend repeatedly rather than once (p87: one hop
        # reached 3481 while the table and its writing continue past it).
        while True:
            below = peaks[(peaks > end) & (peaks < end + 1.3 * ROW_PITCH_FALLBACK)]
            if not below.size:
                break
            end = int(below.max())
    return min(end, fh)


def lattice_rows(peaks: np.ndarray, kraken: np.ndarray, hb_true: int,
                 fh: int) -> tuple[list[int], float, int, int]:
    """Row edges as a rigid-pitch lattice phased on the WRITTEN text lines.

    Why not Kraken rows directly: their centres follow the handwriting, which
    drifts and merges (p12 bottom: 3 lines in 2 bands), and the cache fills
    empty stretches with synthetic rows at a wrong pitch (p16/p40: runs of
    69-78 px rows). A rigid pitch fixes both while keeping the writing's phase.

    Why not the printed rules, though `rule_profile` finds them accurately
    (strip-median, 33/33 on p100): **the scribe writes ON the rule, not between
    the rules.** Measured 2026-08-29 over pages 100/12/20/3/19: median signed
    distance from a Kraken centre to its nearest rule peak is -1 to -7 px on a
    92-94 px pitch (<=0.08 of a pitch). So a rule marks a row CENTRE, not a row
    edge; phasing edges on the rules puts every edge half a row out and cuts the
    handwriting it should contain -- geometrically right, useless for reading
    (user review after the rules version scored 34/34 rows on the preview set).
    The rules stay in as a pitch estimate only; the phase must come from the ink.
    (An earlier note in this file blamed writer drift off faint ruling; the
    measurement above contradicts it -- the offset is ~0, not a drift.)

    So: pitch = median plausible spacing over rule peaks AND Kraken centres
    (the printed form is constant); phase = circular median of the Kraken
    centres mod pitch (text sits mid-band); the train walks from the header at
    that pitch and snaps to a Kraken centre only within ROW_SNAP of the
    prediction, so a merged line cannot pull the following rows off. Edges are
    the midpoints between centres; the first edge is the header line.
    Returns (edges, pitch, n_snapped, n_centres).
    """
    def plausible(a):
        d = np.diff(a) if a.size > 1 else np.array([])
        return d[(d > 80) & (d < 110)]
    ok = np.concatenate([plausible(peaks), plausible(kraken)])
    pitch = float(np.median(ok)) if ok.size >= 3 else ROW_PITCH_FALLBACK
    src = kraken[kraken > hb_true + 0.3 * pitch]
    if src.size >= 2:
        ang = 2 * np.pi * ((src - hb_true) % pitch) / pitch
        phase = float((np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))
                       / (2 * np.pi) * pitch) % pitch)
    else:
        phase = 0.5 * pitch
    c = hb_true + phase
    while c < hb_true + 0.5 * pitch:          # first centre at least half a row down
        c += pitch
    centers, snapped = [], 0
    # A row counts while its CENTRE is inside the table, not its whole band: the
    # last ruled row often has its lower half clipped by the page edge or by the
    # verticals fading (p50: 0.98 pitch of table left below the last edge, one
    # whole written row -- serial ١٩ -- that the stricter test discarded).
    while c + 0.15 * pitch <= fh:
        cand = kraken[np.abs(kraken - c) <= ROW_SNAP * pitch] if kraken.size else np.array([])
        if cand.size:
            c = float(cand[np.argmin(np.abs(cand - c))])
            snapped += 1
        centers.append(c)
        c += pitch
    cs = np.asarray(centers)
    mids = ((cs[:-1] + cs[1:]) / 2).tolist() if len(cs) > 1 else []
    edges = [int(hb_true)] + [int(round(m)) for m in mids] + \
            [int(round(min(cs[-1] + pitch / 2, fh)))]
    return edges, pitch, snapped, len(centers)


def text_in_band_frac(centers: np.ndarray, edges: list[int]) -> float:
    """Fraction of Kraken text centres in the middle 60% of their band — the
    check that the lattice and the writing agree (low = writer ignores the
    ruling, or the train slipped). Not independent of the lattice's own phase
    source; read it as "no line is cut", not as proof the rows are right."""
    if not len(centers) or len(edges) < 2:
        return 0.0
    e = np.asarray(edges, float)
    idx = np.clip(np.searchsorted(e, centers) - 1, 0, len(e) - 2)
    rel = (centers - e[idx]) / (e[idx + 1] - e[idx])
    return float(np.mean((rel >= 0.2) & (rel <= 0.8)))


def ink_cross_frac(framed: np.ndarray, bands: list[dict],
                   n_cols: int, row_span: tuple[int, int]) -> float:
    """Fraction of interior-boundary path pixels that cross handwriting."""
    mask = cv2.dilate(hand_mask(framed),
                      np.ones((INK_DILATE, INK_DILATE), np.uint8))
    y0, y1 = row_span
    ys = np.arange(max(0, y0), min(framed.shape[0], y1), 4)
    hits = total = 0
    for ci in range(1, n_cols):  # interior boundaries only
        for y in ys:
            x = interp_col_x(ci, int(y), bands)
            if 0 <= x < mask.shape[1]:
                hits += int(mask[int(y), int(x)] > 0)
                total += 1
    return hits / total if total else 1.0


def page_geometry3(page: int) -> dict:
    deskewed = load_or_make_deskewed(page, from_cache=True)
    wide, y_offset, x_offset = crop_table(deskewed, CFG)
    frame = detect_table_frame(wide)
    x_r, hb_y = frame["x_right_split"], frame["header_bottom_y"]
    hb_true = hb_y if hb_y <= HB_MAX else HB_FALLBACK
    # Header-anchored frame fit on the full crop: overrides a page split that
    # latched onto the last column rule (p17: 2467 -> 2621).
    fit = fit_template_frame(wide, hb_true, x_r)
    x_r_detected = x_r
    x_r = resolve_right_edge(x_r, fit)
    if x_r != x_r_detected:
        log.warning("page %d: x_right_split %d -> %d (template fit, %d/18 lines)",
                    page, x_r_detected, x_r, fit["hits"])
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

    # ROWS: lattice phased on the ink. Use the RAW Kraken lines from the cache,
    # not detect_rows' output: the latter interpolates synthetic rows across empty
    # stretches, and on a sparse page those outvote the real ones (p17: 8 real vs
    # 27 synthetic -> phase offset spread over a full row). Synthetic rows carry no
    # evidence about where the writing sits.
    raw_lines = json.loads(seg_cache.read_text()).get("lines", [])
    kraken_raw = np.array([l["y_center"] for l in raw_lines], dtype=float) + hb_y
    kraken_centers = np.array([r["y_center"] for r in rows], dtype=float) + hb_y
    if kraken_raw.size < 2:
        kraken_raw = kraken_centers
    peaks_y = rule_peaks(framed, hb_true)
    bottom = table_bottom(framed, hb_true, peaks_y)
    edges, pitch, n_snap, n_edges = lattice_rows(peaks_y, kraken_raw, hb_true, bottom)
    row_ranges = list(zip(edges[:-1], edges[1:]))
    row_span = (edges[0], edges[-1])
    head_gap = (edges[1] - edges[0]) / pitch          # first band, in pitches
    text_ok = text_in_band_frac(kraken_centers, edges)
    added = added_top = 0

    # COLUMN candidates: cache, detected, and enhanced-detected
    enh = enhance(framed)
    peaks_std = band_peaks(framed)
    peaks_enh = band_peaks(enh)
    # union peak set per band: enhanced finds faint rulings, standard the dark ones
    peaks_all = ([np.unique(np.concatenate([a, b_])) if a.size or b_.size
                  else a for a, b_ in zip(peaks_std[0], peaks_enh[0])],
                 peaks_std[1])

    candidates = []
    cols_cache = CACHE_DIR / f"dewarp_cols_page{page}.json"
    if cols_cache.exists():
        cached = json.loads(cols_cache.read_text())
        raw = list(cached["col_ranges_framed"])
        candidates.append(("cache", raw))
        fw = cached.get("framed_w") or framed.shape[1]
        if fw != framed.shape[1]:
            candidates.append(("cache-scaled",
                               [int(round(x * framed.shape[1] / fw)) for x in raw]))
    for name, img in (("detected", framed), ("enh-detected", enh)):
        try:
            det = fix_x_left_col_geometric(
                detect_columns(img, table_left_x=0, expected_cols=EXPECTED_COLS))
            candidates.append((name, det))
        except Exception as exc:
            log.warning("page %d: %s failed (%s)", page, name, exc)
    # header-anchored template fit: right edge fixed, span fitted against the
    # print-only header-band verticals; left edge derived, never detected
    candidates.append(("template", template_candidate(framed, peaks_all, fit)))

    scored = []
    for name, raw in candidates:
        repaired, inserted = repair_col_ranges(raw)
        repaired, offset, _ = best_shift_tiebreak0(framed, repaired, peaks_all)
        if offset:
            name = f"{name}{offset:+d}px"
        bands_c = pin_outer_bands(clamp_bands(
            smooth_bands(track_bands(framed, repaired, peaks=peaks_all), repaired),
            framed.shape[1]), repaired)
        n_cols = len(repaired) - 1
        dev, wide_ok = width_dev(repaired)
        scored.append({
            "source": name, "col_ranges": repaired, "bands": bands_c,
            "inserted": inserted, "n_cols": n_cols,
            "align": alignment_score(bands_c, peaks_all[0]),
            "ink_cross": ink_cross_frac(framed, bands_c, n_cols, row_span),
            "dev": round(dev, 3), "wide_ok": wide_ok,
        })
    # admissible: right column count, Serial widest at index 0, near-template
    admissible = [s for s in scored
                  if s["n_cols"] == EXPECTED_COLS and s["wide_ok"]
                  and s["dev"] <= DEV_MAX]
    pool = admissible or scored
    best = min(pool, key=lambda s: (s["n_cols"] != EXPECTED_COLS,
                                    not s["wide_ok"], round(s["dev"], 2),
                                    round(s["ink_cross"], 3), -s["align"]))
    return {
        "image": deskewed[:, 0:x_r], "y_offset": y_offset,
        "hb_y": hb_y, "framed_h": fh, "pitch": pitch,
        "x_r": x_r, "x_r_detected": x_r_detected,
        "fit_hits": fit["hits"], "fit_span": fit["span"],
        "row_ranges": row_ranges, "n_rows": len(row_ranges),
        "rows_added": added, "rows_added_top": added_top,
        "rows_missing_top": 0, "head_gap": head_gap,
        "row_snap": round(n_snap / max(1, n_edges), 3), "text_ok": round(text_ok, 3),
        "n_kraken": int(len(kraken_centers)),
        "col_ranges": best["col_ranges"], "bands": best["bands"],
        "col_source": best["source"], "col_score": round(best["align"], 3),
        "ink_cross": round(best["ink_cross"], 4),
        "width_dev": best["dev"], "wide_ok": best["wide_ok"],
        "all_candidates": [(s["source"], s["n_cols"], round(s["align"], 3),
                            round(s["ink_cross"], 4), s["dev"], s["wide_ok"])
                           for s in scored],
    }


def build_page(page: int, overlay: bool = False) -> dict:
    if page not in PAGE_CONFIG:
        PAGE_CONFIG[page] = _page_cfg(page)
    geo = page_geometry3(page)
    FINAL3_DIR.mkdir(parents=True, exist_ok=True)
    img = geo["image"]
    jpeg = FINAL3_DIR / f"Hadita_{page}.jpeg"
    cv2.imwrite(str(jpeg), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    strip = None
    try:
        strip = recognize_top_strip(img, geo["y_offset"], page)
    except Exception as exc:
        log.warning("page %d: top strip failed (%s)", page, exc)
    xml = FINAL3_DIR / f"Hadita_{page}.xml"
    write_page_xml(
        geo["col_ranges"], geo["row_ranges"],
        y_offset=geo["y_offset"],
        page_w=img.shape[1], page_h=img.shape[0],
        image_filename=jpeg.name, out_path=xml,
        text_rows=None,                       # final3 ships geometry only
        bands=geo["bands"], text_fn=lambda t: t.translate(W2E),
        col_tags=True, row_baseline_y=None, top_strip=strip)
    patch_xml(xml)
    if overlay:
        b2.OVERLAY_DIR = OVERLAY_DIR  # write_overlay uses module global
        OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
        write_overlay(page, geo)
    return geo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--page", type=int, action="append", dest="pages")
    ap.add_argument("--gt-pages", action="store_true")
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()
    pages = args.pages or (GT_PAGES if args.gt_pages else
                           sorted(int(p.stem.split("_")[1]) for p in
                                  (ROOT / "Transkribus upload" / "final2").glob("Hadita_*.xml")))
    hdr = not REPORT_TSV.exists()
    with open(REPORT_TSV, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if hdr:
            w.writerow(["page", "n_rows", "rows_added_top", "head_gap", "width_dev", "wide_ok",
                        "col_source", "col_score", "ink_cross", "x_r", "x_r_detected",
                        "fit_hits", "fit_span", "pitch", "row_snap", "text_ok",
                        "n_kraken", "candidates"])
        for page in pages:
            try:
                geo = build_page(page, overlay=args.overlay)
            except Exception as exc:
                print(f"page {page}: FAILED {exc}")
                w.writerow([page, "", "", "", "", "FAILED"] + [""] * 11 + [str(exc)[:100]])
                continue
            print(f"page {page}: {geo['n_rows']} rows (pitch {geo['pitch']:.1f}, snap {geo['row_snap']}, "
                  f"text_ok {geo['text_ok']}, kraken {geo['n_kraken']})  cols={geo['col_source']} "
                  f"align={geo['col_score']} ink_cross={geo['ink_cross']} "
                  f"x_r={geo['x_r']}({geo['x_r_detected']}) fit={geo['fit_hits']}/18")
            w.writerow([page, geo["n_rows"], geo["rows_added_top"],
                        round(geo["head_gap"], 2), geo["width_dev"], geo["wide_ok"],
                        geo["col_source"], geo["col_score"], geo["ink_cross"],
                        geo["x_r"], geo["x_r_detected"], geo["fit_hits"], geo["fit_span"],
                        round(geo["pitch"], 2), geo["row_snap"], geo["text_ok"], geo["n_kraken"],
                        json.dumps(geo["all_candidates"])])


if __name__ == "__main__":
    main()
