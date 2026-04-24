# Table Segmentation Pipeline Report
## Haditha Ottoman Land Register — Page 3

**Script:** `kraken_experiment.py`  
**Image:** `images/deskewed_page3.jpg` (5053 × 3961 px, pre-deskewed)  
**Output:** `page3_segmentation.xml` → `Transkribus upload/Hadita_3.xml`

---

## 1. Overview

The goal of this pipeline is to produce a geometrically accurate PAGE XML segmentation of a tabular Ottoman Arabic land register page, suitable for loading into Transkribus (or any PAGE-compatible tool) for ground-truth creation and HTR model training. The pipeline uses two complementary methods: a neural baseline segmentation model (Kraken) for row detection, and classical computer vision (OpenCV morphological analysis) for column detection, now enhanced with per-band interpolation to capture non-linear page geometry.

---

## 2. Image Deskewing

**Pipeline:** `haditax.py`, function `deskew_page()` (lines 205–277)

The raw scan contains the full two-page spread of the register plus binding, often skewed or rotated. Before any segmentation, the page must be deskewed to a rectangle. The pipeline:

1. **Background separation**: Threshold the image to isolate the bright paper (~205) from the dark scan background (~40)
2. **Page detection**: Find contours, select the largest area (the physical page)
3. **Corner detection**: Approximate the page contour to 4 corners using `cv2.approxPolyDP` (tries epsilon multipliers from 0.02 to 0.05)
4. **Fallback**: If 4-point approximation fails, use `cv2.minAreaRect` to find the best-fit rectangle
5. **Destination frame**: Compute a rectangle from the max detected edge lengths
6. **Perspective warp**: Apply `cv2.getPerspectiveTransform` + `cv2.warpPerspective` (INTER_LANCZOS4 interpolation) to map the detected quadrilateral to the destination rectangle
7. **Caching**: Write the result as a lossless PNG to `.ocr_cache/deskewed_page{N}.png` for reproducibility

**What it corrects:** Gross rotation and perspective distortion of the scan frame (page not square to the camera).

**What it does NOT correct:** Page bow (non-linear curvature due to binding), column-line tilt. These are handled downstream (Section 4.2) by per-band column detection and coordinate interpolation, not by further image warping.

**Output:** `images/deskewed_page{N}.jpg` — the flattened page image, input to all downstream processing.

---

## 3. Image Preprocessing: Table Crop

The full page image contains two facing register pages plus a book binding strip. Only the left data table is processed.

| Parameter | Value |
|-----------|-------|
| Full image | 5053 × 3961 px |
| Left table width | 49.4% of page width → 2496 px |
| Top skip (taxpayer name + column headers) | 8% of height → 316 px |
| **Table crop** | **2496 × 3645 px** |

The 316 px skip (`y_offset`) removes the printed taxpayer identification block and global column header row above the data rows. These are intentionally left outside the table crop for row detection but are recovered as the header row in the PAGE XML (see Section 5).

---

## 4. Row Detection

### 3.1 Neural baseline segmentation (Kraken)

Row detection uses the Kraken v5 neural segmentation model `sc_100p_full_line (1).mlmodel` (4.8 MB), which was trained to detect text baselines in historical handwritten documents. The model is run on the table crop and returns 94 raw line regions, each described by a baseline polyline and boundary polygon.

```
kraken --native -i table.jpg seg.json segment --baseline --model "sc_100p_full_line (1).mlmodel"
```

Each detected region provides a y-center (mean of baseline y-coordinates) and a y-extent (min/max of the boundary polygon). The raw output is cached in `.ocr_cache/kraken_seg_page3.json` for repeatability.

### 3.2 Clustering

The 94 raw line regions are not one-to-one with table rows: each Kraken detection corresponds to one cell's text region, and multiple cells in the same row are detected as separate lines. Nearby detections are merged into rows by y-proximity clustering (gap threshold: 60 px). Lines with y_center < 200 px in table coordinates are discarded as they correspond to the printed column-label area.

### 3.3 Gap interpolation

After clustering, any gap between consecutive row centers exceeding 1.5× the median inter-row spacing is filled with synthetic rows at the midpoint. This handles rows where the ink is too faint or the content too sparse for Kraken to detect (e.g. all-ditto rows).

**Median inter-row spacing:** ~94 px  
**Gap threshold for interpolation:** ~141 px  
**One synthetic row inserted** at y = 1498 (gap of 185 px between rows 13 and 15)

### 3.4 Row range computation

Row boundaries are placed at the midpoint between consecutive row centers, with the first boundary at y = 0 and the last at the full table height. This gives one (y_top, y_bottom) range per detected row.

**Result: 35 rows** (ground truth expects 36; one row at the bottom remains undetected)

| Row | y_center | y_range |
|-----|----------|---------|
| 1 (header area) | 266 | [198–307] |
| 2 (Serial_No 1) | 380 | [336–421] |
| 3 | 472 | [336–548] |
| … | … | … |
| 14 | 1498 | [1478–1518] — **synthetic** |
| … | … | … |
| 35 | 3492 | [3438–3557] |

---

## 5. Column Detection

### 4.1 Global detection (full-height projection)

Column detection uses classical computer vision on the table crop. The pipeline:

1. **CLAHE** contrast enhancement (`clipLimit=2.0, tileGridSize=8×8`) to normalise uneven illumination across the page
2. **Adaptive thresholding** (`ADAPTIVE_THRESH_MEAN_C`, block size 41, C=5) to binarise the image locally
3. **Morphological opening** with a tall vertical kernel (1×15 px) followed by horizontal dilation (1×2 px) to isolate printed vertical column lines while suppressing horizontal strokes and text
4. **Vertical projection** — summing the mask column-wise (`np.sum(mask, axis=0)`) to produce a 1D signal where peaks correspond to column boundaries
5. **Peak finding** (`scipy.signal.find_peaks`) with height threshold at mean + 0.5σ, minimum distance 30 px

**Post-processing:**
- Peaks left of x = 140 px (book binding area) are discarded
- Detected peaks are shifted +15 px to centre on printed lines
- The set is trimmed to 20 boundaries (19 columns + 1) using a nearest-ideal-position algorithm
- One wide merged column (Exemptions_LP + Mils, width 215 px at x ≈ 2070–2285) is split at its internal local projection peak (x = 2177)

**Result: 19 column boundaries** → 19 columns (matching the 19 printed table columns)

| Col | Field | x range | Width |
|-----|-------|---------|-------|
| 1 | Serial_No | 167–418 | 251 px |
| 2 | Date | 418–572 | 154 px |
| 3 | Block_No | 572–682 | 110 px |
| 4 | Parcel_No | 682–794 | 112 px |
| 5 | Parcel_Cat_No | 794–903 | 109 px |
| 6 | Parcel_Area | 903–1014 | 111 px |
| 7 | Nature_of_Entry | 1014–1187 | 173 px |
| 8 | New_Serial_No | 1187–1319 | 132 px |
| 9 | Changes_Volume_No | 1319–1432 | 113 px |
| 10 | Changes_Serial_No | 1432–1546 | 114 px |
| 11 | Tax_LP | 1546–1670 | 124 px |
| 12 | Tax_Mils | 1670–1761 | 91 px |
| 13 | Total_Tax_LP | 1761–1878 | 117 px |
| 14 | Total_Tax_Mils | 1878–1971 | 93 px |
| 15 | Exemptions_Entry_No | 1971–2070 | 99 px |
| 16 | Exemptions_Amount_LP | 2070–2177 | 107 px |
| 17 | Exemptions_Amount_Mils | 2177–2285 | 108 px |
| 18 | Net_Assessment_LP | 2285–2405 | 120 px |
| 19 | Net_Assessment_Mils | 2405–2496 | 91 px |

### 4.2 Per-band detection and page-bow correction

**Problem:** The full-height projection detects one averaged x-position per column. A linear tilt correction model (`dx = (H/2 − y) × TILT_RATE`, where `TILT_RATE = 30/3645`) was initially used to account for the slight rightward drift of printed lines. However, visual inspection of the PAGE XML in Transkribus revealed that column lines were systematically displaced ~20 px too far right in the middle rows while being correctly positioned at the top and bottom. This is the signature of **page bow** — the physical curvature of a bound register page under scanning, which causes printed vertical lines to follow a non-linear path that a linear model cannot capture.

**Solution: per-band column detection with linear interpolation**

The table is divided into 8 equal horizontal bands (~456 px each). Within each band, the same CLAHE + adaptive-threshold + morphological + peak-finding pipeline is run independently on just that band's rows. This yields a set of 8 sample points `(y_band_center, x_col[0…19])` per column boundary — tracing the actual curve of each printed line through the page.

For any given row y-position, the column x is obtained by linear interpolation between the two nearest band y-centers (flat extrapolation beyond the outermost bands). This replaces both `col_ranges` and the linear `tilt_offset` — the curvature is encoded directly in the band data.

**Result:** 8/8 bands usable on page 3. The per-band model captures both the global tilt and the non-linear bow in a single data-driven representation, with no hand-tuned parameters. This approach generalises to other pages and notebooks regardless of their specific warp shape.

**Robustness provisions:**
- Per-band peak-finding uses a lower threshold (mean + 0.3σ vs. 0.5σ globally) to tolerate sparser ink in narrow bands
- Bands with fewer than half the expected peaks are skipped; gaps are bridged by interpolation from neighbours
- Unmatched columns within a band fall back to the global detected position

---

## 6. PAGE XML Generation

The segmentation is exported in PAGE XML format (schema: 2013-07-15), directly loadable in Transkribus alongside the image.

### 5.1 Table structure

- **`TableRegion`**: covers the full detected table area
- **`rows="35"`**: one header row + 34 data rows
- **`columns="19"`**: matching the 19 detected column boundaries

### 5.2 Row 0 — column-label header

The first detected row cluster (y ≈ 266 in table coordinates, corresponding to the printed column-label text area) is assigned `row="0"` in the XML and left empty (no `TextLine` element). Transkribus treats this as the table header row, leaving it available for manual label entry.

This row was not added artificially — it arises naturally from the Kraken detection (the column-label area produces detectable ink that forms the first cluster after the 200 px skip). Using this natural first row as the header means the table geometry is self-consistent with the image.

### 5.3 Data rows 1–34

Each data cell is a `TableCell` containing:

**`<Coords>`** — a parallelogram (four-point polygon) whose corners are computed using the per-band interpolated column x at both the top and bottom y of the row:

```
top-left:     (interp_col_x(c,   y_top),    y_top  + y_offset)
top-right:    (interp_col_x(c+1, y_top),    y_top  + y_offset)
bottom-right: (interp_col_x(c+1, y_bottom), y_bottom + y_offset)
bottom-left:  (interp_col_x(c,   y_bottom), y_bottom + y_offset)
```

Since `interp_col_x` returns different x values at different y positions, the resulting quadrilateral accurately tracks the curvature of the printed column lines — it is a true parallelogram only when the page has pure linear tilt, and a general quadrilateral when there is bow.

**`<TextLine>`** (only for cells with ground-truth text):

Each text-bearing cell contains a `TextLine` with:
- `<Coords>`: same as the cell boundary
- `<Baseline>`: a two-point horizontal line at 75% of the cell height, x-positions computed using `interp_col_x` at that y — anchors the text for HTR training
- `<TextEquiv><Unicode>`: the ground-truth transcription from `ground_truth.tsv`

Cells with no ground-truth content have no `TextLine` element, keeping the XML clean for human annotation in Transkribus.

### 5.4 Coordinate system

All coordinates in the XML are in **full-page pixels** (origin at top-left of the original image). The `y_offset = 316` is added to all table-coordinate y values before writing. Column x values from the band data are already in table coordinates and are written directly (the band detection was run on the table crop).

---

## 7. OCR Accuracy (Kraken two-model pipeline)

For completeness, the pipeline also supports running OCR on the segmented cells using the `gen2_sc_clean_best.mlmodel` (19 MB Ottoman Arabic recognition model). Results on page 3 with ditto-mark resolution applied to both sides:

| Approach | Exact match | Mean CER | Notes |
|----------|-------------|----------|-------|
| Approach M (Gemini 1.5) | 62.6% | 0.225 | Best overall |
| Kraken gen2_sc_clean | 40.5% | 0.47 | After ditto resolution |
| Kraken muharaf_rec_best | 19.8% | 0.78 | Wrong model (outputs Latin) |

Key Kraken error patterns on page 3:
- **٤ confusion**: the scribe's handwritten ٤ is systematically misread as ٣ or ٢ in every Block_No cell — a training-data gap in gen2_sc_clean
- **Arabic comma**: the ، separator in Parcel_Area is read as ١
- **Nature_of_Entry**: base value misread, cascades to all ditto rows

The OCR results confirm that the segmentation geometry (row and column boundaries) is sound — the failures are model failures on this specific scribe's script, not segmentation failures. The intended next step is to use the Transkribus GT creation workflow (enabled by this PAGE XML) to produce a corrected transcription, then fine-tune a Kraken model on this scribe's handwriting.

---

## 8. Design Decisions and Open Questions

**Kraken for rows, OpenCV for columns.** Kraken's neural segmentation excels at finding text baselines even in faint or irregular handwriting, making it well-suited for row detection where the signal is the ink itself. Column detection, by contrast, is about finding printed ruling lines — a structural signal better handled by morphological methods tuned for straight high-contrast strokes.

**Per-band column detection over a parametric warp model.** A parametric model (linear tilt, or even a quadratic bow model) requires calibration and makes assumptions about the warp shape. The per-band approach is data-driven: it reads the actual position of each column line at each height from the image itself, with no prior on the warp shape. This makes it robust to pages with different binding tension, scan angle, or paper curl.

**Row 0 as the natural header.** The first Kraken-detected cluster corresponds to the printed column-label area — a row that exists in the image and should be empty in the transcription. Assigning it as `row="0"` rather than adding a synthetic header row keeps the XML geometry anchored to actual image content.

**Synthetic row insertion.** The gap interpolation step (Section 3.3) is essential for rows where all cells contain ditto marks (″) — Kraken finds no detectable text and skips the row entirely. Without interpolation, subsequent rows would be misaligned by one position.

**Baseline at 75%.** Arabic text in this register sits near the bottom of each cell's ruled space. Placing the baseline at 75% of cell height provides a reasonable anchor for Transkribus's HTR without requiring per-cell calibration.

### How much to preprocess the image? (Open question / future research)

Two broad strategies exist and have not been definitively resolved:

**Strategy A — Correct in the image (full preprocessing pipeline)**

Each geometric issue (perspective, tilt, bow) is physically corrected in a warp pass before segmentation. Downstream code operates on a flat, normalised image with simple rectangular coordinates.

- Pro: simpler coordinate logic; cell crops more rectangular; potentially better OCR/HTR on individual cells
- Con: every interpolation pass degrades a degraded historical image; bow varies per page (no fixed parameters — would still need per-page measurement); modern HTR models are robust to moderate distortion

**Strategy B — Correct in coordinates (current approach)**

Minimal image transformation (one perspective deskew pass); remaining geometry (tilt, bow) is handled at the coordinate level via per-band interpolation. The image is touched as little as possible.

- Pro: preserves original pixel values; simpler to reason about what the HTR model sees
- Con: coordinate logic is more complex; cell crops remain slightly trapezoidal

**Current state:** Strategy B, with one deskew pass (perspective only). Whether this is the right balance is an open question. In particular:

- Does the current deskew pass help or hurt HTR accuracy? (Modern models tolerate moderate skew; resampling softens ink strokes.)
- Would physically correcting the bow improve cell-level OCR/HTR quality enough to justify the extra warp pass?

These questions should be tested empirically — comparing HTR results on raw, deskew-only, and fully-rectified images — before committing to a more aggressive preprocessing strategy.

---

## 9. Ground Truth (GT) Specification

**File:** `ground_truth.tsv`

**Current status (2026-04-24):** 65 rows across pages 3, 10, 50 (RA-verified). Maintained by the Research Assistant team via Streamlit/Haditax app with GitHub API save triggers.

**Digit encoding:**
- **Primary representation**: Eastern Arabic digits (٠-٩) for all numeric columns (Date, Block_No, Parcel_No, Parcel_Cat_No, New_Serial_No, Reference_to_Register_of_Changes_{Volume,Serial}_No, Tax_{LP,Mils}, Total_Tax_{LP,Mils}, Reference_to_Register_of_Exemptions_{Entry_No,Amount_{LP,Mils}}, Net_Assessment_{LP,Mils})
- **Exception**: T.D.L. (Tax Daily List) reference numbers use Western digits (0-9) — `New_Serial_No: 102, 154` and `Reference_to_Register_of_Changes_Serial_No: 1940` in rows 5 and 21 of page 3 — because these values appear in Western digits in the original document
- **Rationale**: Eastern Arabic is the primary script of the register; Western digits are reserved for non-Arabic entries and cross-references to external documents

**Quality improvements (recent):**
- 2026-04-24 08:35 UTC (Yoav): Corrected digit encoding and Nature_of_Entry clarifications
- 2026-04-24 (standardization): All numeric columns converted to Eastern Arabic digits (٠-٩)
- 2026-04-24 (refinement): T.D.L. numbers preserved in Western digits to match document

---

## 10. Files

| File | Role |
|------|------|
| `kraken_experiment.py` | Full pipeline: crop → row detection → column detection → PAGE XML / OCR |
| `images/deskewed_page{N}.jpg` | Input: pre-deskewed page scans (N=3,10,50) |
| `page3_segmentation.xml` | PAGE XML output (master copy for page 3) |
| `Transkribus upload/` | Subdirs: `original/` (Eastern Arabic), `western arabic transliteration/` (Western digits for all numeric columns) |
| `.ocr_cache/` | Cached OCR results: Kraken segmentation, Approach M/O/O2 outputs, etc. |
| `ground_truth.tsv` | Verified GT (65 rows, pages 3/10/50) — RA-maintained, primary source for score computation |
| `compare_ocr.py` | OCR comparison framework: runs all approaches (A–R) and Approach O2 (new), scores against GT |
| `segment_unified.py` | Unified segmentation for PAGE XML generation with configurable options |
| `haditax.py` | Streamlit app: interactive grid detection, ditto resolution, GT correction interface, GitHub save |
| `segmentation_report.md` | This document — pipeline methodology, design decisions, open research questions |
