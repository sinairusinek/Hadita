"""Generate the figure set for the Vienna 2026 talk.

Each function builds one slide's image into img/. Run with no arguments for all
of them, or name figures to rebuild just those:

    python make_figures.py                 # everything
    python make_figures.py fig06 fig11     # just these

Conventions: photographic panels are JPEG at slide width (1600px long edge is
plenty for a projector); charts are SVG so they stay crisp and re-colourable.
Annotation colours come from PALETTE so the deck reads as one system.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
IMG = HERE / "img"
DATA = HERE / "data"
F3 = ROOT / "Transkribus upload" / "final3"
HAIFA = (Path.home() / "Documents" / "GitHub" / "Hospital-Registers"
         / "data" / "private" / "page-cache")

# One accent per role, used consistently across every figure.
PALETTE = {
    "bad": "#d1495b",      # the failure being shown
    "good": "#2a9d8f",     # the fix
    "print": "#e9c46a",    # printed form / geometry
    "ink": "#264653",      # handwriting / measured signal
    "muted": "#8d99ae",
}
PAPER = "#f4f1ea"
FIGS: dict[str, callable] = {}


def figure(fn):
    FIGS[fn.__name__] = fn
    return fn


# ---------------------------------------------------------------- helpers

def load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def fit(img: np.ndarray, w: int | None = None, h: int | None = None) -> np.ndarray:
    ih, iw = img.shape[:2]
    if w and not h:
        h = max(1, round(ih * w / iw))
    elif h and not w:
        w = max(1, round(iw * h / ih))
    interp = cv2.INTER_AREA if (w or 0) < iw else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def hex2bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def label(img: np.ndarray, text: str, org: tuple[int, int],
          colour: str = "#ffffff", scale: float = 1.0, thick: int = 2) -> None:
    """Caption with a dark halo so it reads over paper or over ink."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                hex2bgr(colour), thick, cv2.LINE_AA)


def caption_bar(img: np.ndarray, text: str, height: int = 62,
                colour: str = "#ffffff") -> np.ndarray:
    """Dark strip under a panel carrying its caption."""
    bar = np.full((height, img.shape[1], 3), hex2bgr("#1d2229"), np.uint8)
    label(bar, text, (18, height - 22), colour, 0.78, 2)
    return np.vstack([img, bar])


def pad(img: np.ndarray, l=0, t=0, r=0, b=0, colour: str = PAPER) -> np.ndarray:
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT,
                              value=hex2bgr(colour))


def save(img: np.ndarray, name: str, quality: int = 92) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    out = IMG / name
    if out.suffix == ".png":
        cv2.imwrite(str(out), img)
    else:
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"  wrote {out.relative_to(HERE)}  {img.shape[1]}x{img.shape[0]}")


def save_svg(svg: str, name: str) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    (IMG / name).write_text(svg, encoding="utf-8")
    print(f"  wrote img/{name}")


def cells_from_xml(path: Path) -> dict[tuple[int, int], np.ndarray]:
    """{(row, col): polygon} for every TableCell in a PAGE XML."""
    rx = re.compile(r'<TableCell id="cell_r(\d+)_c(\d+)"[^>]*>\s*'
                    r'<Coords points="([^"]+)"')
    out = {}
    for r, c, pts in rx.findall(path.read_text(encoding="utf-8")):
        out[(int(r), int(c))] = np.array(
            [[int(a) for a in p.split(",")] for p in pts.split()], np.int32)
    return out


def cell_texts(path: Path) -> dict[tuple[int, int], str]:
    rx = re.compile(r'<TableCell id="cell_r(\d+)_c(\d+)"[^>]*>(.*?)</TableCell>', re.S)
    out = {}
    for r, c, body in rx.findall(path.read_text(encoding="utf-8")):
        t = " ".join(u.strip() for u in
                     re.findall(r"<Unicode>(.*?)</Unicode>", body, re.S) if u.strip())
        out[(int(r), int(c))] = t
    return out


# ---------------------------------------------------------------- figures

@figure
def fig06_two_registers() -> None:
    """slide 6 — Haifa vs Hadita, the argument in one picture"""
    left = load(HAIFA / "nb01_p003_2000_0.jpg")
    right = load(F3 / "Hadita_9.jpeg")
    H = 1500
    left, right = fit(left, h=H), fit(right, h=H)
    left = caption_bar(
        left, "Haifa Government Hospital  ·  every row filled  ·  serials 99-109 unbroken")
    right = caption_bar(
        right, "Al-Haditha tax register  ·  five filled rows, then scatter  ·  page bows into the gutter")
    gut = np.full((left.shape[0], 26, 3), hex2bgr(PAPER), np.uint8)
    save(pad(np.hstack([left, gut, right]), 26, 26, 26, 26), "fig06_two_registers.jpg")


@figure
def fig06b_detail_strip() -> None:
    """slide 6 — four crops, one per failure mode"""
    p9, p3 = load(F3 / "Hadita_9.jpeg"), load(F3 / "Hadita_3.jpeg")
    h9, w9 = p9.shape[:2]
    h3, w3 = p3.shape[:2]

    # Absolute pixel windows: the ditto/nil crop is taken from cell coordinates
    # in the page's own XML, so it frames real marks rather than blank paper.
    panels = [
        (p9[900:1900, 0:520], "no left border"),
        (p3[2300:2750, 700:1900], "the clerk drew lines too"),
        (p3[730:1300, 390:720], 'ditto " = same as above'),
        (p3[1640:1810, 1480:1760], "nil dash - : meaning, almost no ink"),
    ]
    # Crops differ in aspect, so normalise each tile to the same box before the
    # caption bar goes on; otherwise hstack gets mismatched heights.
    H, W = 430, 620
    tiles = []
    for img, cap in panels:
        t = fit(img, h=H)
        if t.shape[1] > W:
            t = t[:, (t.shape[1] - W) // 2:(t.shape[1] - W) // 2 + W]
        dx = W - t.shape[1]
        t = pad(t, dx // 2, 0, dx - dx // 2, 0)
        tiles.append(caption_bar(t, cap, 92))
    gut = np.full((tiles[0].shape[0], 18, 3), hex2bgr(PAPER), np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, gut, t])
    save(pad(row, 22, 22, 22, 22), "fig06b_detail_strip.jpg")


@figure
def fig08_dewarp_damage() -> None:
    """slide 8 — where the dewarp canvas ended, drawn on the page it truncated"""
    sys.path.insert(0, str(ROOT))
    from audit_dewarp_damage import source_geometry, load_deskewed  # noqa: E402

    PAGE = 71                      # worst case: 382px = 4+ row pitches discarded
    geo = source_geometry(PAGE)
    img = load_deskewed(PAGE)
    # The audit measures y within the data region, which starts at the header
    # bottom (hb_y) in the deskewed frame.
    cut = geo["hb_y"] + geo["band_hi"]   # last scanline the dewarp canvas kept

    # Everything below `cut` existed on the page and never reached the canvas.
    ov = img.copy()
    cv2.rectangle(ov, (0, cut), (img.shape[1], img.shape[0]), hex2bgr(PALETTE["bad"]), -1)
    img = cv2.addWeighted(ov, 0.25, img, 0.75, 0)
    cv2.line(img, (0, cut), (img.shape[1], cut), hex2bgr(PALETTE["bad"]), 7)

    lost_px = geo["data_h"] - geo["band_hi"]
    label(img, "dewarp canvas ends here", (40, cut - 40), PALETTE["bad"], 2.6, 6)
    label(img, f"{lost_px}px below the cut - {lost_px / geo['pitch']:.1f} row pitches,",
          (40, cut + 100), PALETTE["bad"], 2.3, 5)
    label(img, "written and discarded", (40, cut + 200), PALETTE["bad"], 2.3, 5)

    # Crop to the left leaf: the page split is the long vertical the audit found.
    out = img[max(0, cut - 1500):, :img.shape[1] // 2]
    out = fit(out, w=1500)
    out = caption_bar(
        out, f"page {PAGE}: the straightened image simply stopped - content below was cut, not smeared", 74)
    save(pad(out, 24, 24, 24, 24), "fig08_dewarp_damage.jpg")


@figure
def fig09_column_template() -> None:
    """slide 9 — the header band is print-only, so fit one form to its 18 rules"""
    PAGE = 3
    xml = F3 / f"Hadita_{PAGE}.xml"
    img = load(F3 / f"Hadita_{PAGE}.jpeg")
    cells = cells_from_xml(xml)

    # The shipped XML already carries the fitted template; its cell boundaries
    # are the column lines, so draw those rather than re-running the fit.
    ncol = max(c for _, c in cells) + 1
    r0 = min(r for r, _ in cells)
    xs = sorted({int(cells[(r0, c)][:, 0].min()) for c in range(ncol)
                 if (r0, c) in cells}
                | {int(cells[(r0, ncol - 1)][:, 0].max())})
    top = int(min(cells[(r0, c)][:, 1].min() for c in range(ncol) if (r0, c) in cells))

    # Panel A: the header band alone, magnified, with the rules it anchors on.
    # It is a shallow strip across a tall page, so it only reads when enlarged.
    band = img[max(0, top - 330):top + 20, :].copy()
    for x in xs:
        cv2.line(band, (x, 0), (x, band.shape[0]), hex2bgr(PALETTE["print"]), 5)
    band = caption_bar(fit(band, w=2100),
                       "the header band: 18 clean interior rules, essentially free of handwriting", 86)

    # Panel B: the fitted template carried down the whole page.
    full = img.copy()
    ov = full.copy()
    for x in xs:
        cv2.line(ov, (x, top), (x, full.shape[0]), hex2bgr(PALETTE["print"]), 5)
    full = cv2.addWeighted(ov, 0.75, full, 0.25, 0)
    cv2.line(full, (xs[0], top), (xs[0], full.shape[0]), hex2bgr(PALETTE["good"]), 9)
    label(full, "left edge: derived, never detected", (xs[0] + 24, top + 120),
          PALETTE["good"], 2.0, 5)
    full = caption_bar(full, "two free parameters - right edge and scale - place all 19 columns", 74)

    # Two very different aspect ratios: a wide shallow strip and a tall page.
    # Kept as separate files so each can be sized to its own slide.
    save(pad(band, 24, 24, 24, 24), "fig09a_header_band.jpg")
    save(pad(fit(full, h=2000), 24, 24, 24, 24), "fig09_column_template.jpg")


@figure
def fig09b_one_miss() -> None:
    """slide 9 — a dropped rule mislabels every cell to its right"""
    PAGE = 3
    img = load(F3 / f"Hadita_{PAGE}.jpeg")
    cells = cells_from_xml(F3 / f"Hadita_{PAGE}.xml")
    ncol = max(c for _, c in cells) + 1
    r0 = min(r for r, _ in cells)
    xs = sorted({int(cells[(r0, c)][:, 0].min()) for c in range(ncol) if (r0, c) in cells}
                | {int(cells[(r0, ncol - 1)][:, 0].max())})
    top = int(min(cells[(r0, c)][:, 1].min() for c in range(ncol) if (r0, c) in cells))
    bot = int(max(p[:, 1].max() for p in cells.values()))

    DROP = 6          # pretend the detector missed this rule
    kept = [x for i, x in enumerate(xs) if i != DROP]

    def panel(lines, colour, title):
        p = img[top:bot].copy()
        for i in range(len(lines) - 1):
            x0, x1 = lines[i], lines[i + 1]
            ov = p.copy()
            # alternate tint so column *identity* is visible, not just the lines
            if i % 2 == 0:
                cv2.rectangle(ov, (x0, 0), (x1, p.shape[0]), hex2bgr(colour), -1)
                p = cv2.addWeighted(ov, 0.16, p, 0.84, 0)
            cv2.line(p, (x0, 0), (x0, p.shape[0]), hex2bgr(colour), 4)
            label(p, str(i), (x0 + 12, 54), colour, 1.3, 3)
        return caption_bar(fit(p, w=1150), title, 74)

    a = panel(xs, PALETTE["good"], "19 rules found: every cell lands in its own column")
    b = panel(kept, PALETTE["bad"], "one rule missed: every column to its right is off by one")
    gut = np.full((a.shape[0], 24, 3), hex2bgr(PAPER), np.uint8)
    save(pad(np.hstack([a, gut, b]), 24, 24, 24, 24), "fig09b_one_miss.jpg")


@figure
def fig10a_morphology_finds_nothing() -> None:
    """slide 10 — the standard table recipe returns handwriting, not rules"""
    img = load(F3 / "Hadita_100.jpeg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # The textbook horizontal-rule extraction, and the same one that works on
    # the verticals: adaptive threshold, then a long horizontal opening.
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 5)
    h = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (121, 1)))
    out = cv2.cvtColor(255 - h, cv2.COLOR_GRAY2BGR)
    save(pad(caption_bar(fit(out, h=1700),
                         "adaptive threshold + long horizontal opening: the printed rules do not binarise", 74),
             24, 24, 24, 24), "fig10a_morphology.jpg")


@figure
def fig10b_strip_median() -> None:
    """slide 10 — one strip is noise; the median across 30 strips is a ladder"""
    sys.path.insert(0, str(ROOT))
    from build_final3 import ROW_STRIPS, ROW_PROM, rule_profile  # noqa: E402
    from scipy.signal import find_peaks  # noqa: E402

    img = load(F3 / "Hadita_100.jpeg")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    w = g.shape[1]
    profs = []
    for i in range(ROW_STRIPS):
        x0, x1 = int(i * w / ROW_STRIPS) + 3, int((i + 1) * w / ROW_STRIPS) - 3
        if x1 - x0 < 10:
            continue
        reg = g[:, x0:x1]
        bg = cv2.medianBlur(reg.astype(np.uint8), 51).astype(np.float32) + 1
        profs.append(np.clip(1.0 - reg / bg, 0, 0.06).mean(axis=1))
    med = rule_profile(img)
    pk, _ = find_peaks(med, prominence=ROW_PROM, distance=60)
    # rule_peaks() drops everything above the header bottom; without the same
    # filter the count would include the header's own printed lines. The table
    # top in the shipped XML is that boundary, in the frame this image is in.
    cells = cells_from_xml(F3 / "Hadita_100.xml")
    hb = min(p[:, 1].min() for p in cells.values())
    pk = pk[pk >= hb - 20]

    # Plot y down the page on the x axis: the reader is looking at a page.
    # Trim to the table itself - the margin above the header is noise that
    # compresses the part of the profile the slide is about.
    lo, hi = max(0, int(hb) - 60), min(len(med), int(pk[-1]) + 120)
    med = med[lo:hi]
    profs = [p[lo:hi] for p in profs]
    pk = pk - lo

    W, H, PADL, PADB = 2000, 620, 70, 54
    sx = (W - PADL - 20) / len(med)
    top = med.max() * 1.05

    # One point per output pixel: at ~3000 samples across 1900px of plot the
    # extra vertices are invisible and cost ~1MB of markup.
    def path(v, every=1):
        pts = [f"{PADL + i * sx:.1f},{H - PADB - (val / top) * (H - PADB - 30):.1f}"
               for i, val in enumerate(v) if i % every == 0]
        return "M" + " L".join(pts)

    # The 30 background strips carry most of the file size and none of the
    # detail; every 3rd sample still reads as the same noise band.
    faint = "".join(
        f'<path d="{path(p, 3)}" fill="none" stroke="{PALETTE["muted"]}" '
        f'stroke-width="0.7" opacity="0.16"/>' for p in profs)
    marks = "".join(
        f'<line x1="{PADL + x * sx:.1f}" y1="{H - PADB}" x2="{PADL + x * sx:.1f}" '
        f'y2="18" stroke="{PALETTE["print"]}" stroke-width="1.4" opacity="0.75"/>'
        for x in pk)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
<g>{faint}</g>
<g>{marks}</g>
<path d="{path(med)}" fill="none" stroke="{PALETTE["ink"]}" stroke-width="2.6"/>
<line x1="{PADL}" y1="{H - PADB}" x2="{W - 20}" y2="{H - PADB}" stroke="{PALETTE["ink"]}" stroke-width="1.6"/>
<text x="{PADL}" y="{H - 16}" font-family="Helvetica,Arial" font-size="21" fill="{PALETTE["ink"]}">y down the page &#8594;</text>
<text x="{PADL}" y="30" font-family="Helvetica,Arial" font-size="22" fill="{PALETTE["muted"]}">30 individual strips (faint)</text>
<text x="{PADL}" y="58" font-family="Helvetica,Arial" font-size="22" font-weight="bold" fill="{PALETTE["ink"]}">median across strips</text>
<text x="{W - 20}" y="30" text-anchor="end" font-family="Helvetica,Arial" font-size="22" font-weight="bold" fill="#b08d1f">{len(pk)} printed rules found</text>
</svg>'''
    save_svg(svg, "fig10b_strip_median.svg")


@figure
def fig10c_rules_are_not_boundaries() -> None:
    """slide 10 — the scribe writes ON the rule, so rules-as-boundaries cuts the writing"""
    sys.path.insert(0, str(ROOT))
    from build_final3 import ROW_PROM, rule_profile  # noqa: E402
    from scipy.signal import find_peaks  # noqa: E402

    img = load(F3 / "Hadita_100.jpeg")
    med = rule_profile(img)
    pk, _ = find_peaks(med, prominence=ROW_PROM, distance=60)
    pk = pk[pk > 400]
    y0, y1 = int(pk[3]) - 60, int(pk[11]) + 60
    x0, x1 = 200, min(img.shape[1], 1700)

    def panel(lines, colour, title):
        p = img[y0:y1, x0:x1].copy()
        for y in lines:
            yy = int(y) - y0
            if 0 <= yy < p.shape[0]:
                cv2.line(p, (0, yy), (p.shape[1], yy), hex2bgr(colour), 4)
        return caption_bar(fit(p, w=1500), title, 78, colour)

    a = panel(pk, PALETTE["bad"],
              "rules as boundaries: every line cuts through the writing")
    mid = [(pk[i] + pk[i + 1]) / 2 for i in range(len(pk) - 1)]
    b = panel(mid, PALETTE["good"],
              "boundaries midway between: the rule is the middle of a row, not its edge")
    save(pad(np.vstack([a, np.full((20, a.shape[1], 3), hex2bgr(PAPER), np.uint8), b]),
             24, 24, 24, 24), "fig10c_rules_not_boundaries.jpg")


@figure
def fig11_ink_histogram() -> None:
    """slide 11 — ink per cell is bimodal, so a threshold separates the classes"""
    import csv
    rows = list(csv.DictReader((DATA / "ink_per_cell.tsv").open(), delimiter="\t"))
    text = [int(r["ink_px"]) for r in rows if r["has_text"] == "1"]
    blank = [int(r["ink_px"]) for r in rows if r["has_text"] == "0"]

    # Log-spaced bins with a dedicated zero bucket: the blank class piles up at
    # exactly 0, which a linear axis would render as an invisible spike.
    THR = 6
    edges = np.concatenate([[0, 1], np.geomspace(2, 4000, 34)])
    ht, _ = np.histogram(text, edges)
    hb, _ = np.histogram(blank, edges)

    W, H = 2000, 700
    PADL, PADR, PADB, PADT = 90, 30, 96, 60
    pw, ph = W - PADL - PADR, H - PADB - PADT
    n = len(edges) - 1
    bw = pw / n
    # Square-root height scale: the blank class piles ~2,000 cells into the zero
    # bucket, which on a linear axis flattens the whole ink continuum to nothing.
    top = max(ht.max(), hb.max()) * 1.06

    def bars(h, colour, dx, label_):
        out = []
        for i, v in enumerate(h):
            if v <= 0:
                continue
            frac = (v / top) ** 0.5
            x = PADL + i * bw + dx
            out.append(f'<rect x="{x:.1f}" y="{PADT + ph - frac * ph:.1f}" '
                       f'width="{bw * 0.44:.1f}" height="{frac * ph:.1f}" '
                       f'fill="{colour}" opacity="0.88"/>')
        return "".join(out)

    # threshold marker sits between the zero/one buckets and the ink continuum
    ti = int(np.searchsorted(edges, THR)) - 1
    tx = PADL + ti * bw
    ticks = ""
    for val in (0, 1, 10, 100, 1000):
        i = int(np.searchsorted(edges, max(val, 0))) - (0 if val == 0 else 1)
        x = PADL + max(i, 0) * bw + bw / 2
        ticks += (f'<line x1="{x:.1f}" y1="{PADT + ph}" x2="{x:.1f}" y2="{PADT + ph + 8}" '
                  f'stroke="{PALETTE["ink"]}" stroke-width="1.5"/>'
                  f'<text x="{x:.1f}" y="{PADT + ph + 32}" text-anchor="middle" '
                  f'font-family="Helvetica,Arial" font-size="21" fill="{PALETTE["ink"]}">{val}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
<rect x="{PADL}" y="{PADT}" width="{tx - PADL:.1f}" height="{ph}" fill="{PALETTE['bad']}" opacity="0.06"/>
{bars(hb, PALETTE["muted"], 0, "blank")}
{bars(ht, PALETTE["ink"], bw * 0.48, "text")}
<line x1="{tx:.1f}" y1="{PADT - 12}" x2="{tx:.1f}" y2="{PADT + ph}" stroke="{PALETTE['bad']}" stroke-width="3" stroke-dasharray="8 5"/>
<text x="{tx + 12:.1f}" y="{PADT - 20}" font-family="Helvetica,Arial" font-size="23" font-weight="bold" fill="{PALETTE['bad']}">cutoff at {THR}px &#8594; drop the TextLine</text>
<line x1="{PADL}" y1="{PADT + ph}" x2="{W - PADR}" y2="{PADT + ph}" stroke="{PALETTE["ink"]}" stroke-width="1.6"/>
{ticks}
<text x="{W - PADR}" y="{H - 22}" text-anchor="end" font-family="Helvetica,Arial" font-size="22" fill="{PALETTE["ink"]}">handwriting pixels under the cell polygon (log scale) &#183; bar height &#8730;count</text>
<rect x="{PADL}" y="{PADT + 6}" width="18" height="18" fill="{PALETTE["muted"]}"/>
<text x="{PADL + 26}" y="{PADT + 21}" font-family="Helvetica,Arial" font-size="22" fill="{PALETTE["ink"]}">blank cells ({len(blank):,}) &#183; median 0px</text>
<rect x="{PADL}" y="{PADT + 36}" width="18" height="18" fill="{PALETTE["ink"]}"/>
<text x="{PADL + 26}" y="{PADT + 51}" font-family="Helvetica,Arial" font-size="22" fill="{PALETTE["ink"]}">cells with text ({len(text):,}) &#183; median {int(np.median(text))}px</text>
</svg>'''
    save_svg(svg, "fig11_ink_histogram.svg")


@figure
def fig13_baseer_days() -> None:
    """slide 13 — days of the week, in a tax ledger, in cells with no ink"""
    import csv
    import json
    PAGE = 10
    DAYS = ("السبت", "الخميس", "الاربعاء", "الاحد", "الاثنين", "الثلاثاء", "الجمعة")
    # Romanised for the caption: OpenCV's Hershey fonts cannot draw Arabic, and
    # the Arabic itself is set in the HTML page next to this figure.
    ROMAN = {"السبت": "al-sabt (Saturday)", "الخميس": "al-khamis (Thursday)",
             "الاربعاء": "al-arbi'a (Wednesday)", "الاحد": "al-ahad (Sunday)",
             "الاثنين": "al-ithnayn (Monday)", "الثلاثاء": "al-thulatha (Tuesday)",
             "الجمعة": "al-jum'a (Friday)"}
    rows = json.load((ROOT / "exp2608" / f"Hadita_{PAGE}_baseer-f3.json").open())
    cols = list(rows[0].keys())
    ink = {(int(r["row"]), int(r["col"])): int(r["ink_px"])
           for r in csv.DictReader((DATA / "ink_per_cell.tsv").open(), delimiter="\t")
           if r["page"] == str(PAGE)}

    hits = [(ri, ci, v, ink.get((ri, ci), -1))
            for ri, row in enumerate(rows) for ci, c in enumerate(cols)
            if (v := (row.get(c) or "").strip()) in DAYS]
    hits = [h for h in hits if h[3] >= 0]
    hits.sort(key=lambda h: h[3])

    img = load(F3 / f"Hadita_{PAGE}.jpeg")
    cells = cells_from_xml(F3 / f"Hadita_{PAGE}.xml")
    tiles = []
    for ri, ci, word, px in hits[:6]:
        poly = cells.get((ri, ci))
        if poly is None:
            continue
        x0, x1 = poly[:, 0].min(), poly[:, 0].max()
        y0, y1 = poly[:, 1].min(), poly[:, 1].max()
        crop = img[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):x1 + 8].copy()
        # Cells vary in width by column, so normalise to one tile box - a ragged
        # grid reads as sloppiness rather than as evidence.
        crop = fit(crop, h=430)
        crop = (crop[:, :520] if crop.shape[1] > 520
                else pad(crop, 0, 0, 520 - crop.shape[1], 0))
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1),
                      hex2bgr(PALETTE["bad"]), 3)
        crop = caption_bar(crop, f"{px}px of ink  ->  {ROMAN[word]}", 66, PALETTE["bad"])
        tiles.append(crop)

    h = max(t.shape[0] for t in tiles)
    tiles = [pad(t, 0, 0, 0, h - t.shape[0]) for t in tiles]
    gut = np.full((h, 16, 3), hex2bgr(PAPER), np.uint8)
    grid = []
    for i in range(0, len(tiles), 3):
        band = tiles[i]
        for t in tiles[i + 1:i + 3]:
            band = np.hstack([band, gut, t])
        grid.append(band)
    W = max(g.shape[1] for g in grid)
    grid = [pad(g, 0, 0, W - g.shape[1], 0) for g in grid]
    out = grid[0]
    for g in grid[1:]:
        out = np.vstack([out, np.full((16, W, 3), hex2bgr(PAPER), np.uint8), g])
    out = caption_bar(
        out, f"page {PAGE}: {len(hits)} days of the week invented in near-empty cells", 76)
    save(pad(out, 24, 24, 24, 24), "fig13_baseer_days.jpg")


def _som_vs_gt(pages=(3, 4, 5, 6, 9, 10)):
    """Substitutions between the SoM run and the proxy GT, on equal-length cells."""
    import json
    from collections import Counter
    subs, examples, cols = Counter(), [], None
    for p in pages:
        f = ROOT / "exp2608" / f"Hadita_{p}_som-f3v2.json"
        g = ROOT / "exp2608" / f"Hadita_{p}_gt-f3.xml"
        if not (f.exists() and g.exists()):
            continue
        rows = json.load(f.open())
        cols = cols or list(rows[0].keys())
        gt = cell_texts(g)
        for ri, row in enumerate(rows):
            for ci, c in enumerate(cols):
                pv = (row.get(c) or "").strip()
                gv = (gt.get((ri, ci)) or "").strip()
                if pv and gv and pv != gv and len(pv) == len(gv):
                    for a, b in zip(gv, pv):
                        if a != b:
                            subs[(a, b)] += 1
                            examples.append((p, ri, ci, a, b, gv, pv))
    return subs, examples


@figure
def fig16_three_to_two() -> None:
    """slide 16 — the top substitution, at reading size: can you tell them apart?"""
    subs, examples = _som_vs_gt()
    n32 = subs[("٣", "٢")]
    picks = [e for e in examples if e[3] == "٣" and e[4] == "٢"][:8]

    tiles = []
    for p, ri, ci, _, _, gv, pv in picks:
        cells = cells_from_xml(F3 / f"Hadita_{p}.xml")
        poly = cells.get((ri, ci))
        if poly is None:
            continue
        img = load(F3 / f"Hadita_{p}.jpeg")
        x0, x1 = poly[:, 0].min(), poly[:, 0].max()
        y0, y1 = poly[:, 1].min(), poly[:, 1].max()
        crop = fit(img[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6].copy(), h=340)
        crop = (crop[:, :420] if crop.shape[1] > 420
                else pad(crop, 0, 0, 420 - crop.shape[1], 0))
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1),
                      hex2bgr(PALETTE["muted"]), 2)
        tiles.append(crop)

    tiles = tiles[:6]
    gut = np.full((tiles[0].shape[0], 16, 3), hex2bgr(PAPER), np.uint8)
    grid = []
    for i in range(0, len(tiles), 3):
        band = tiles[i]
        for t in tiles[i + 1:i + 3]:
            band = np.hstack([band, gut, t])
        grid.append(band)
    W = max(g.shape[1] for g in grid)
    grid = [pad(g, 0, 0, W - g.shape[1], 0) for g in grid]
    out = grid[0]
    for g in grid[1:]:
        out = np.vstack([out, np.full((16, W, 3), hex2bgr(PAPER), np.uint8), g])
    out = caption_bar(
        out, f"the scribe's 3 read as 2 - {n32} times, by every model family tried", 78)
    save(pad(out, 24, 24, 24, 24), "fig16_three_to_two.jpg")


@figure
def fig15_set_of_marks() -> None:
    """slides 14-15 — print the row index in the margin and the counting task disappears"""
    som = load(ROOT / "som_f3" / "Hadita_9_som.jpg")
    save(pad(caption_bar(fit(som, h=2000),
                         "set-of-marks: the row index is printed from geometry we already had, "
                         "so the model keys rows instead of counting them", 74),
             24, 24, 24, 24), "fig15_set_of_marks.jpg")


@figure
def fig07_the_cast() -> None:
    """slide 7 — the pipeline, coloured by WHO does each stage"""
    ROLE = {"code": "#4a7c9b", "spec": "#9b6a4a", "vlm": "#6a8f5e", "human": "#8a5f8f"}
    KEY = [("code", "rule-based code"), ("spec", "specialist ATR model"),
           ("vlm", "VLM"), ("human", "editors")]
    STAGES = [
        ("photographed\npage", "code"),
        ("coordinate\nwarp", "code"),
        ("column\ntemplate fit", "code"),
        ("row lattice:\nKraken + rules", "spec"),
        ("ink gate", "code"),
        ("SoM read\n(Gemini 3)", "vlm"),
        ("agreement\n× PyLaia", "spec"),
        ("RA correction\nin Transkribus", "human"),
    ]
    W, H = 2000, 470
    bw, bh, gap = 198, 150, 34   # gap wide enough for the arrows to read
    x0 = (W - (len(STAGES) * bw + (len(STAGES) - 1) * gap)) / 2
    y0 = 150
    boxes, arrows = "", ""
    for i, (name, role) in enumerate(STAGES):
        x = x0 + i * (bw + gap)
        boxes += (f'<rect x="{x:.0f}" y="{y0}" width="{bw}" height="{bh}" rx="12" '
                  f'fill="{ROLE[role]}" opacity="0.92"/>')
        for j, ln in enumerate(name.split("\n")):
            boxes += (f'<text x="{x + bw / 2:.0f}" y="{y0 + bh / 2 - 8 + j * 27:.0f}" '
                      f'text-anchor="middle" font-family="Helvetica,Arial" '
                      f'font-size="21" font-weight="600" fill="#fff">{ln}</text>')
        if i:
            arrows += (f'<path d="M{x - gap + 4:.0f},{y0 + bh / 2} L{x - 8:.0f},{y0 + bh / 2}" '
                       f'stroke="{PALETTE["ink"]}" stroke-width="2.6" marker-end="url(#a)"/>')
    keys = ""
    kx = x0
    for role, lab in KEY:
        keys += (f'<rect x="{kx:.0f}" y="{y0 + bh + 52}" width="20" height="20" rx="4" '
                 f'fill="{ROLE[role]}"/>'
                 f'<text x="{kx + 30:.0f}" y="{y0 + bh + 68}" font-family="Helvetica,Arial" '
                 f'font-size="22" fill="{PALETTE["ink"]}">{lab}</text>')
        kx += 60 + len(lab) * 12
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
<path d="M0,0 L7,3 L0,6 z" fill="{PALETTE["ink"]}"/></marker></defs>
<rect width="{W}" height="{H}" fill="{PAPER}"/>
<text x="{W / 2}" y="72" text-anchor="middle" font-family="Helvetica,Arial" font-size="34" font-weight="bold" fill="{PALETTE["ink"]}">No stage is done by one kind of reader</text>
<text x="{W / 2}" y="108" text-anchor="middle" font-family="Helvetica,Arial" font-size="23" fill="{PALETTE["muted"]}">specialist ATR models &#183; VLMs &#183; rule-based code &#183; editors</text>
{arrows}{boxes}{keys}
</svg>'''
    save_svg(svg, "fig07_the_cast.svg")


@figure
def fig12_agreement_funnel() -> None:
    """after slide 12 — what the pipeline actually buys the editor"""
    STEPS = [
        (58122, "cells in the corpus", PALETTE["muted"]),
        (18718, "carry ink (32%) - the rest need no reading", PALETTE["ink"]),
        (6140, "auto-accepted: SoM and PyLaia agree, 0.88% error", PALETTE["good"]),
        (12578, "left for the RA to correct", PALETTE["print"]),
    ]
    W, H = 1700, 520
    PADL, PADT, bh, gap = 60, 90, 74, 26
    top = STEPS[0][0]
    bars = ""
    for i, (n, lab, col) in enumerate(STEPS):
        y = PADT + i * (bh + gap)
        w = (n / top) * (W - PADL - 460)
        bars += (f'<rect x="{PADL}" y="{y}" width="{w:.0f}" height="{bh}" rx="7" '
                 f'fill="{col}" opacity="0.9"/>'
                 f'<text x="{PADL + 18}" y="{y + bh / 2 + 9:.0f}" font-family="Helvetica,Arial" '
                 f'font-size="27" font-weight="bold" fill="#fff">{n:,}</text>'
                 f'<text x="{PADL + w + 18:.0f}" y="{y + bh / 2 + 8:.0f}" '
                 f'font-family="Helvetica,Arial" font-size="23" fill="{PALETTE["ink"]}">{lab}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<rect width="{W}" height="{H}" fill="{PAPER}"/>
<text x="{PADL}" y="52" font-family="Helvetica,Arial" font-size="31" font-weight="bold" fill="{PALETTE["ink"]}">What the pipeline hands the editor</text>
{bars}
<text x="{PADL}" y="{H - 26}" font-family="Helvetica,Arial" font-size="21" fill="{PALETTE["muted"]}">Auto-accept rate measured on the six proxy-GT pages (32.8%) and projected across the inked cells.</text>
</svg>'''
    save_svg(svg, "fig12_agreement_funnel.svg")


@figure
def fig04_tel_hadid() -> None:
    """slide 4 — the place: the tell today, and the register that records it"""
    land = load(HERE / "img" / "src" / "tel_hadid_landscape.webp")
    page = load(F3 / "Hadita_3.jpeg")
    H = 1300
    land = fit(land, h=H)
    page = fit(page, h=H)
    land = caption_bar(land, "Tel Hadid today - olive terraces over the village site", 70)
    page = caption_bar(page, "the property-tax register: who held which parcel", 70)
    h = max(land.shape[0], page.shape[0])
    land, page = pad(land, 0, 0, 0, h - land.shape[0]), pad(page, 0, 0, 0, h - page.shape[0])
    gut = np.full((h, 26, 3), hex2bgr(PAPER), np.uint8)
    save(pad(np.hstack([land, gut, page]), 26, 26, 26, 26), "fig04_tel_hadid.jpg")


@figure
def fig14_counting_problem() -> None:
    """slide 14 — on a sparse page the model drops rows and everything below shifts"""
    img = load(F3 / "Hadita_9.jpeg")
    cells = cells_from_xml(F3 / "Hadita_9.xml")
    nrow = max(r for r, _ in cells) + 1
    ink = {}
    import csv
    for r in csv.DictReader((DATA / "ink_per_cell.tsv").open(), delimiter="\t"):
        if r["page"] == "9":
            ink[(int(r["row"]), int(r["col"]))] = int(r["ink_px"])
    # A row counts as written when some single cell clears the gate comfortably;
    # summing across 19 cells lets rule noise accumulate into a false positive.
    written = {r for r in range(nrow)
               if max((ink.get((r, c), 0) for c in range(19)), default=0) >= 60}

    out = img.copy()
    for r in range(nrow):
        ys = [cells[(r, c)] for c in range(19) if (r, c) in cells]
        if not ys:
            continue
        y0 = min(p[:, 1].min() for p in ys)
        y1 = max(p[:, 1].max() for p in ys)
        colour = PALETTE["ink"] if r in written else PALETTE["muted"]
        ov = out.copy()
        cv2.rectangle(ov, (0, y0), (out.shape[1], y1), hex2bgr(colour), -1)
        out = cv2.addWeighted(ov, 0.10 if r in written else 0.05, out,
                              0.90 if r in written else 0.95, 0)
        cv2.line(out, (0, y0), (out.shape[1], y0), hex2bgr(colour), 2)
        label(out, f"{r}", (12, (y0 + y1) // 2 + 12), colour, 1.1, 3)

    out = caption_bar(
        fit(out, h=1900),
        f"page 9: only {len(written)} of {nrow} ruled rows carry writing", 74)
    save(pad(out, 24, 24, 24, 24), "fig14_counting_problem.jpg")


def main() -> None:
    want = sys.argv[1:] or list(FIGS)
    for name in want:
        if name not in FIGS:
            print(f"! no figure {name}; have: {', '.join(FIGS)}")
            continue
        print(f"{name}: {FIGS[name].__doc__ or ''}".rstrip())
        FIGS[name]()


if __name__ == "__main__":
    main()
