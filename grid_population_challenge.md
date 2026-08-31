# Hadita HTR Project: The Grid-Population Problem

## Background

We are digitizing a set of Ottoman-era British Mandate tax registers (the "Hadita" notebooks). Each page is a structured table with ~19 named columns and ~30–40 data rows, handwritten in Arabic/Ottoman script. The pipeline has two separate parts that work well individually but are hard to combine:

---

## Part 1: Segmentation — Building the Grid (SOLVED)

`segment_unified.py` takes a scanned page image and produces a regularized PAGE XML file with correct geometric cells. The pipeline:

1. Kraken detects row y-ranges; gaps are interpolated using a median-pitch estimator to recover missed rows
2. Columns are detected per horizontal band (8 bands per page) to handle the non-linear bow/curvature of bound registers, then interpolated between bands
3. Dewarping corrects the left-column boundary using a geometric heuristic (confirmed as the production default)
4. `patch_baselines.py` sets the baseline of each cell near the bottom (frac=0.90) for legibility in Transkribus
5. The result: a complete PAGE XML with bounding boxes and baselines for every cell of a ~19-column × ~35-row table

This was run across the entire 91-page corpus and the output lives in `Transkribus upload/final/`.

---

## Part 2: HTR — Reading the Cells (PARTIAL)

The best approach is **Approach M**: Gemini 2.5 Pro receives the full page image plus a detailed prompt, and returns a JSON array of rows with all column fields filled. On page 3 (the "clean" page), it achieves ~62.6% exact-cell match, CER ~0.225. The prompt includes:

- Explicit column ordering with right-to-left reading instruction
- Symbol reference guide (ditto marks, checkmarks, dashes, uncertainty markers)
- Rules against leftward column shifting
- Few-shot examples in JSON
- Instructions for special row types (multi-category continuation rows, tax-year breakdown rows, sub-total rows, carry-forward rows)

The main failures of Approach M: on complex pages (10, 50), rows with sparse content cause Gemini to drift — it anchors the first visible value to the leftmost field rather than counting columns from position. Example: a year (٩٣٩) in the Date column gets written into Serial_No because Serial_No is empty.

---

## The Core Challenge: Combining Segmentation + HTR

The PAGE XML gives us a precise cell address for every cell (row index × column name). Approach M gives us a row-indexed JSON array but with unreliable column assignment. The question is: **how to get the text into the right cells of the grid?**

Three hybrid strategies were tried, and all failed:

### Approach R — Banded crops (rejected)
Split the page into 8 horizontal bands, send each band to Gemini separately. Hypothesis: smaller images → fewer rows → less room to drift. Result: band-boundary rows are duplicated or split, empty bands hallucinate rows, and results are actually *worse* than full-page (62.6% → lower). ~4× slower.

### Approach O / O2 — M-base + column-strip overrides (rejected)
Use Approach M's output as the base, then for specific "hard" columns (Parcel_Cat_No, Parcel_Area, Tax_Mils, Tax_LP), crop a vertical strip of that column and re-read it with a column-specific prompt. Hypothesis: narrow prompts on narrow crops fix systematic column-specific errors. Result: Gemini cannot reliably count rows in a vertical strip with no horizontal context. Row counts differ per column, so aligning column-strip results back to the M rows is not possible.

### Approach S-lite / S-full — XML-scaffold strips (rejected)
Same idea as O/O2, but now the column boundaries come from the PAGE XML geometry (deskewed, correct) instead of heuristic detection. S-full: all 20 columns as separate vertical strips, each sent to Gemini independently. S-lite: M-base + 4 column overrides using PAGE XML boundaries. Result: same failure mode as O/O2. Correct geometry did not solve the row-count discipline problem. Gemini in a vertical strip has no horizontal row anchors and simply miscounts rows.

---

## Why the Combination Is Hard

The fundamental tension:

- **Gemini reads well with context.** When given the full page, it can infer row boundaries from visual cues (horizontal lines, whitespace, ink patterns). But it outputs a flat JSON list of rows with no guarantee that row *N* of its output corresponds to row *N* of the PAGE XML grid.
- **The PAGE XML grid is coordinate-based.** A cell is addressed by (row_index, col_name). To populate it, you need an answer that is *both* correctly row-indexed AND correctly column-indexed.
- **Column strips remove horizontal context.** A vertical image of a single column is a sequence of cell crops stacked vertically, with no visible row boundaries between them. Gemini cannot tell where one row ends and the next begins, so its count drifts.
- **Cell crops lose semantic context.** Reading individual cell crops (e.g. 19 columns × 35 rows = 665 API calls per page) is expensive and also removes the inter-column context that helps Gemini interpret ambiguous glyphs (e.g. knowing from the Date column that this is a 3-digit year helps decode a blurry Tax_Mils column).
- **Row alignment is the missing link.** Approach M returns the right *content* but cannot guarantee the right *row index* for every row, especially in pages where some rows are visually faint, some are blank, and special row types (sub-totals, carry-forwards) interrupt the regular pattern.

---

## Current State

The decision as of 2026-04-28: **give up on automated grid-population for now**. The full 91-page corpus has been uploaded to Transkribus with:
- Correct dewarped JPEG images
- Regularized PAGE XMLs with correct row/column geometry, empty `<Unicode>` cells, and lowered baselines

Research Assistants (RAs) will transcribe directly in Transkribus, using the pre-segmented grid as scaffolding. This produces ground truth that can later be used to train or fine-tune a model.

---

## The Open Research Question

**Is there a prompt strategy, model architecture, or pipeline design that can reliably populate a known-geometry cell grid from a full-page handwritten table image — without losing either (a) the full-page horizontal context Gemini needs for column discipline, or (b) the row-level precision needed to align output to PAGE XML cell addresses?**

Specific sub-questions:
1. Can a multimodal model be given the PAGE XML bounding boxes as input constraints (e.g. "row 7, column Tax_Mils is at coordinates [x1,y1,x2,y2] — what is written there?") and answer reliably at cell-level granularity?
2. Can a two-pass approach work: pass 1 = full-page M for content, pass 2 = use bounding-box crops of individual cells only for *verification/correction* of specific columns where M is known to drift?
3. Can row-boundary detection be made more robust by combining the visual baseline positions from the PAGE XML segmentation with Gemini's JSON output (matching M's rows to XML rows by row count + content similarity)?
4. Are there structured-output / tool-use API features in Gemini or Claude that could enforce a fixed output schema keyed to (row_index, col_name) pairs rather than a free-form JSON array?
