# Plan: GT-grade seeded corpus in Transkribus on undamaged images

**Prepared 2026-07-27, to be executed by Claude (Opus) in this repo.**
**Read this whole file before touching anything. Ask the user only at the gates marked 🚪.**

## Goal

Produce, for every usable register page, the closest-possible-to-ground-truth artifact in
Transkribus: an **undamaged page image** + correct table segmentation + the best available
seed transcription (Gemini + newest Hadid model, cross-validated), with uncertain cells
flagged `[?]` so RA correction converges to GT with minimal keystrokes.

**Why the pivot:** the current canonical corpus (`Transkribus upload/final/`) uses
*dewarped* (remapped) images. The user reports that **many pages** — not just page 4 — have
dewarp damage (clipped/smeared top/bottom rows, lost bits). Damaged images poison both RA
correction and Hadid HTR inference (Hadid runs inside Transkribus on the uploaded image).
Therefore the upload form must switch from **warp-the-image** to **warp-the-coordinates**:
upload an undamaged image and curve the cell polygons to follow the page, instead of
flattening the image.

## Current state (verified 2026-07-27)

- `Transkribus upload/final/` — 98 pages, dewarped canvas + straight-grid PAGE XML.
  Built by `build_final.py` (pipeline: `dewarp.py` → `segment_unified.py` →
  `patch_baselines.py`). Empty pages 2, 7, 8, 70, 102 permanently excluded.
- Dewarp geometry: `segment_unified.build_remap()` returns `map_x`, `map_y` — per
  output-pixel **source coordinates**. So any point of the straight output grid can be
  projected back onto the source image by lookup/interpolation: `src = (map_x[oy,ox],
  map_y[oy,ox])`. `dewarp.py` also receives a `src_to_out` mapping from `build_remap`.
  The deskew step is a global `warpPerspective` (invertible homography); content damage
  comes from the row/column **remap**, not the deskew. ⇒ The safe upload image is the
  **deskewed table crop** (undamaged content), with curved coordinates.
- Gemini pipeline: `Hadita_Gemini3_to_PAGEXML_v5.ipynb` is the **frozen** production
  seeder (LOW thinking, ~$0.08/page). Local CLI equivalent: `run_g3v6_local.py`
  (public SDK, `--no-raw` = v5-style single image, `--model` override). v6 dual-image
  mode was scored on 2026-07-27: **11% worse than v5 — do not use `--raw` mode.**
- Newest Hadid model: second-iteration PyLaia model in Transkribus (files named
  `Hadid02`, trained checkpoint 592709; first iteration Hadid01 = model id 592509,
  train ~8% / val ~22% CER, overfit). Verify the exact newest model id in the
  Transkribus UI before use. Hadid inference XMLs already fetched for pages 11–13:
  `g3_results/Hadita_{11,12,13}_Hadid0{1,2}.xml`.
- Agreement layer: `agreement_layer.py` is written (states MATCH / MISMATCH /
  ONLY_<src> / BOTH_EMPTY; patches `[?]` into MISMATCH cells) but has **never been run**.
- Proxy GT (RA-corrected, treat as GT): pages **3, 4, 5, 6, 9, 10** — pull latest
  transcript per page. Local copies: `g3_results/Hadita_{N}_Transkribus_latest.xml`.
- Transkribus API: `TrpClient.from_env()` from the Dybbuk repo client
  (`TRANSKRIBUS_USER`/`TRANSKRIBUS_PASS` env vars); collection 2377415, doc 15829823.
  `fetch_hadid01.py` shows the fetch pattern.
- Scoring methodology: `score_g3_vs_gt.py` (cell classes perfect/single/multi/wrong/
  missed_row/phantom_row; RA cost in keystrokes). Reuse it — do not invent new metrics.
- Known model failure mode still open: Gemini drops rows on **sparse** pages
  (page 9: 21/24 rows; page 10: 22/27). Row *order* is stable.

## House rules (violations have burned us before)

1. **Never modify or overwrite v5 artifacts** (`Hadita_Gemini3_to_PAGEXML_v5.ipynb`,
   `build_g3_notebook.py`, `g3_results/Hadita_{N}_G3.{json,xml}`, `g3_runs.csv`).
   New code goes in new sibling scripts; new outputs get new suffixes + their own CSV log.
2. **🚪 Ask the user before any paid API run** (Gemini, Claude) and before re-running
   OCR/scoring that could use cache instead — state cost estimate and cache/fresh choice.
3. **Commit current XMLs before regenerating any** — regeneration can shift columns;
   use cached segmentation (`--from-cache` pattern) rather than fresh detection wherever
   possible.
4. PAGE XML `<Unicode>` content **must escape `<`, `>`, `&`** or Transkribus returns 500.
5. Digit regexes: Python `\d` matches Eastern Arabic digits — use `[0-9]` for
   ASCII-only checks. Column-aware digit normalization lives in `digit_norm.py`
   (`New_Serial_No` and `T.D.L/T.P.L/D.L` cells keep Western digits; all else Eastern).
6. Canonical ditto mark is ASCII `"`. The Remarks column is app metadata, **not** a
   register column — exclude from OCR/HTR/scoring.
7. No asymmetric postprocessing heuristics inside one approach's runner; fix via prompts
   or shared normalization.
8. Every uploaded XML must declare `imageFilename` matching the exact paired image file,
   or Transkribus mis-scales all coordinates.

---

## Phase 0 — Damage inventory (no API cost)

Only the user knows the full damage extent; build the evidence for them to confirm.

1. Write `audit_dewarp_damage.py`: for each page in `final/`, compare the dewarped
   image's data-region row coverage against the source deskewed crop (row centers from
   the cached `dewarp_seg_page{N}.json`): flag pages where the first/last detected row
   center falls within ~½ pitch of the canvas edge, or where remap anchors extrapolate
   (`fill_value` region used). Output `damage_audit.tsv` (page, flags, suspected
   top/bottom loss).
2. Render a contact sheet (top-strip + bottom-strip crops of dewarped vs deskewed image,
   side by side, ~6 pages per sheet) into `debug/damage_audit/`.
3. 🚪 Present the flagged list + contact sheets; the user marks the confirmed-damaged
   set. Record the confirmed list in the session summary and in
   `project_corpus_page_irregularities` terms (top row / bottom row / smear).

**Gate:** if confirmed damage is rare (<10 pages), consider patching only those pages and
keeping `final/` for the rest — ask the user. The default assumption (per the user) is
that damage is widespread ⇒ proceed to Phase 1 for the whole corpus.

## Phase 1 — Coordinate-warp corpus (`Transkribus upload/final2/`)

Build the warp-the-coordinates variant: undamaged image + curved cell polygons.

1. New script `build_final2.py` (do not modify `build_final.py`):
   - Image: the **deskewed table crop** saved before remapping (verify what
     `dewarp.process_page()` already writes; persist it if it doesn't).
   - Coordinates: take the straight-grid cell corner points and baseline points used
     today for the dewarped XML, and map each through `map_x`/`map_y` (bilinear lookup)
     back to deskewed-crop coordinates. Cells become quads/polylines that follow the
     page curvature. Reuse the existing PAGE XML writer + `patch_baselines.py`
     fractions (cells/name 0.90, index 0.97) — baselines map through the same lookup.
   - Sample each cell edge at ≥3 points (corners + midpoint) so curvature is captured,
     then round; Transkribus accepts arbitrary polygons.
   - Keep row/col ids (`cell_r{r}_c{c}`), col_tags, and Eastern-digit text conventions
     identical to `final/` so all downstream tooling (agreement_layer, scorers) works
     unchanged.
2. Validate on the 6 proxy-GT pages first: render overlay images (grid drawn on the
   deskewed crop) into `debug/final2_overlay/` and eyeball-check row/column alignment;
   compare per-cell text-region crops between final/ and final2/ for a sample of cells
   (IoU of ink content) as a sanity metric.
3. Then build all ~98 pages. Log anomalies (pages where remap caches are missing or
   row counts differ from `final/`) — do not silently drop pages.
4. 🚪 Show the user 3–4 overlay samples (one known-damaged page, one dense, one sparse)
   before bulk upload.

**Deliverable:** `Transkribus upload/final2/Hadita_{N}.jpeg + .xml`, flat, same naming.

## Phase 2 — Transkribus re-upload + newest-Hadid inference

1. 🚪 Upload strategy is the user's call: recommend a **new Transkribus document**
   ("Hadita-final2") in collection 2377415, leaving the old doc untouched (RA work on
   pages 3–10 lives there; do not orphan it). Prepare the upload zip(s); the user
   uploads via UI (or use the REST client if credentials are provided).
2. The user runs the **newest Hadid model** (Hadid02 / checkpoint 592709 — verify id in
   UI) on the new document — at minimum on: the 6 proxy-GT pages (for scoring) + the
   next production batch (pages 11–20). Transkribus UI action; cannot be scripted here.
3. Fetch Hadid transcripts via the REST client (pattern in `fetch_hadid01.py`), save as
   `g3_results/Hadita_{N}_Hadid02_final2.xml`, and score the 6 GT pages with the
   `score_g3_vs_gt.py` methodology. This yields the first clean measurement of whether
   undamaged images improve Hadid (previous inference ran on damaged dewarped images —
   if val CER drops materially, that alone justified the rebuild).

## Phase 3 — LLM seed on undamaged geometry

1. 🚪 Model A/B (paid, ~$1–3 total — get approval): run `run_g3v6_local.py --no-raw`
   (v5-style single image) on pages 3, 9, 10 twice: current `gemini-3-flash-preview`
   vs the best available Gemini 3.1-family model id (check the public API model list;
   3.1 Pro led 2026 handwritten-form benchmarks). Input image: the **dewarped canvas**
   is still fine as *model input* for undamaged pages; for confirmed-damaged pages use
   the deskewed crop. Score with `score_g3_vs_gt.py`; the decisive metrics are
   `missed_row`/`phantom_row` on sparse pages 9/10, then keystrokes.
2. Adopt whichever model wins as the seeder for the batch; log to a new
   `g3_runs_final2.csv`. If rows are still dropped on 9/10, add the **skeleton-first
   two-pass** (pass 1: return only row skeleton — serials + row count, locked; pass 2:
   fill cells given the skeleton) as a new sibling script; test on pages 10, 4, 3
   before adopting.
3. Patch winning-model text into the `final2/` XMLs (existing patch pattern from
   `build_g3_notebook.py`'s `patch_xml()`), producing seed XMLs.

## Phase 4 — Agreement triage → flagged seed push

1. Run `agreement_layer.py` per page with sources: `g3` (Phase 3 JSON) + `hadid02`
   (Phase 2 XML). Primary = g3. Output: `agreement_{N}.tsv` +
   `Hadita_{N}_g3_flagged.xml` (MISMATCH cells get `[?]`).
2. Calibrate on the 6 proxy-GT pages: report auto-accept rate (fraction MATCH) and the
   error rate *inside* MATCH cells vs GT. Published reference points: two-independent-
   model consensus reached >85% auto-accept at WER 0.003 (arXiv 2605.25781); Consensus
   Entropy (arXiv 2504.11101) if a graded score is wanted later — start with the
   existing binary states.
   **Gate:** MATCH-cell error rate must be ≤1% to trust the flags; otherwise add a
   third reader (🚪 Claude Opus 4.5/4.6 vision on cell crops — best-in-class on historic
   Arabic manuscripts per METATR — costs approval).
3. 🚪 Push flagged seed XMLs to the new Transkribus doc for the production batch
   (pages 11–20 first), tagged with a tool name like `Hadita-final2-agree-<date>`.
   RA instruction: correct `[?]` cells first, spot-check the rest.
4. Keep a **random 2% cell sample** per batch for expert verification, independent of
   flags — this estimates residual error in auto-accepted cells (debiasing, arXiv
   2606.28063) and is the number that certifies the corpus as "GT-grade".

## Phase 5 — Rollout + reporting

- Batch pages 14+ in groups of ~10 using the winning recipe; after each batch, recompute
  auto-accept rate and MATCH-error on the growing corrected set.
- Every phase ends with: a short written summary of results (numbers, not adjectives),
  files committed (XMLs **before** any regeneration), and an updated
  `PLAN_GT_pipeline_2026-07.md` checklist below.
- Negative results are results — record them (see v6 example) so nothing is retried
  blindly.

## Checklist (update in place)

- [x] Phase 0: damage_audit.tsv + contact sheets + user-confirmed damage list
      (2026-07-28, `audit_dewarp_damage.py`, commit 2165dd1). **All 98 pages
      flagged.** 25 lose ≥1 whole written row at the bottom (19, 53, 71 lose 2),
      31 lose part of a row, 20 have squeezed top rows, 15 show replicate smear;
      92/98 have the bottom-cut geometry irrespective of what was written there.
      Mechanism: the remap gives each row a ±½-pitch band and the canvas ends at
      `last_row_center + ½ pitch` — content below is cut, not smeared; content
      between the header line and `first_row_center − ½ pitch` is compressed into
      one band. **Gate decision: rebuild the whole corpus** (damage is
      structural; the 23 undamaged pages are undamaged only by luck of what was
      written in the sacrificed band).
      Caveat: `.ocr_cache/dewarp_seg_page10.json` was missing and was
      regenerated with Kraken (`--allow-kraken`), so page 10's segmentation may
      differ slightly from the one that built the shipped `final/Hadita_10.xml`.
- [x] Phase 1: final2/ built for 6 GT pages, overlays approved
- [x] Phase 1: final2/ built for full corpus (2026-07-29, `build_final2.py`,
      commit 53cedb2). 98/98 pages, 3069 rows × 19 cols. **20 rows recovered at
      the bottom (17 pages) + 15 at the top (12 pages).** Three defects fixed:
      (a) 59/98 pages in `final/` had only 18 columns — a missed faint line plus
      the geometric drop left `Net_Assessment_Mils` absent and columns 14–17
      mis-tagged; `repair_col_ranges` fixes 95/98 (1, 21, 75 remain);
      (b) `detect_columns_banded` grabbed the neighbouring line near the spine
      (deviation −49px → +57px at the ±60px limit), pulling bottom-row cells
      half a column off — replaced by `track_bands` (±22px, band-to-band);
      (c) on 12 pages `header_bottom_y` sits below the first data row, so it was
      never segmented — those rows are prepended (row indices shift there; no
      proxy-GT page affected).
      `validate_final2.py`: final2/ 95/98 clean vs final/ 37/98.
      Open: page 69 may have a row straddling the header line (unmeasurable —
      printed header text sits in the same band); pages 1, 21, 75 under 19 cols.
      Alignment risk: page 9 gains 10 rows (its `final/` build predates the
      2026-05-04 bottom-rows fix) — confirm GT rows 0–24 still align when scored.
- [x] Phase 1: column geometry fitted to the printed ruling (commit 2d7d378).
      `score_col_alignment.py` measures every boundary against the printed rules
      (±8px). The builder now scores cached vs freshly-detected grids and picks
      19-columns-first, then alignment; a shift search corrects grids computed
      against an older deskew, but only where independent detection agrees
      within 10px (without that guard page 77's best-scoring offset sat 50px off
      on the wrong lines). **Median alignment 0.970**; only pages 1 (0.32) and
      75 (0.63) below 0.75 — both known-broken layouts. Page 101 has an extra
      table column beyond the standard 19 and is mis-tagged by any 19-col grid.
      Per-page `col_source`/`col_score` are in `final2_build.tsv`.
- [x] Phase 2: upload zips prepared — `make_upload_zips.py` →
      `Transkribus upload/final2_zips/`, 7 batches (GT pages first, then 11–20,
      then 20s), 129 MB, each verified for image/XML pairing and imageFilename.
      **User uploads via the UI** to a NEW document "Hadita-final2" in
      collection 2377415 (gate decision 2026-07-29); doc 15829823 untouched.
- [ ] Phase 2: uploaded to new Transkribus doc; Hadid02 run; transcripts fetched + scored
- [ ] Phase 3: Gemini A/B done, winner adopted; (skeleton-first if needed)
- [ ] Phase 4: agreement calibrated on GT pages (auto-accept %, MATCH-error %)
- [ ] Phase 4: flagged seeds pushed for pages 11–20
- [ ] Phase 5: batch rollout started; 2% random verification sample logged

## What only the user can do

- Confirm the damaged-page list (Phase 0) and the upload strategy (Phase 2).
- Upload to Transkribus / run Hadid inference in the UI (unless REST credentials are
  supplied: `TRANSKRIBUS_USER`/`TRANSKRIBUS_PASS`).
- Approve every paid API run before it happens, with cost estimate.
- Decide GO/NO-GO at each 🚪 gate.
