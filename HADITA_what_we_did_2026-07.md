# Hadita — what we did, per challenge, and whether it worked

*30 July 2026. The eventual approach in each case, not the first attempt.*
*Transkribus refs: collection `2377415`; documents `17829738` (current), `15829823` (previous); models `592509` (hadid01), `592709` (hadid02).*

## Page geometry

- **Deskewing** each scan by a perspective warp of the paper boundary (OpenCV) to deal with photographed, tilted pages — **succeeded**, and is the basis of everything downstream.
- **Uploading the undamaged deskewed page and curving the cell polygons to follow the page**, instead of flattening the pixels with `cv2.remap`, to deal with the dewarp cutting content — **succeeded**: recovered 20 written rows at the bottom of 17 pages and 15 at the top of 12 pages; structural validation went from 37/98 to 95/98 pages clean.
- **An ink-profile audit** (handwriting only: ink minus printed rules, minus scan-border blobs, band-counted from the cut outward) to establish *whether* the dewarp was damaging pages and how much — **succeeded as diagnosis**: found damage on all 98 pages, which is what justified the rebuild.

## Row detection

- **Kraken baseline segmentation**, clustered by y-proximity, with synthetic rows interpolated into gaps and a morphological pass to supplement the page bottom — **partly succeeded**: fine on dense pages, still unsolved on sparse ones (pages 9 and 10), where blank ruled rows and real rows are hard to tell apart.

## Column detection

- **Band-to-band tracking of each printed rule** (±22px window, following the line down from the band above) instead of matching each band against a global grid, to deal with the page bow near the spine — **succeeded**: removed a systematic error that pulled bottom-row cells half a column sideways.
- **Splitting the widest gap until 19 columns are present**, to deal with a faint rule being missed and the last column disappearing — **succeeded**: 95/98 pages now have 19 correctly tagged columns, up from 37/98. Pages 1, 21, 75 still fail, and page 101 genuinely has a 20th column.
- **Scoring candidate grids against the detected printed rules and picking the winner** (cached geometry vs fresh detection, shift-corrected only where the two agree), to deal with not knowing which source to trust — **succeeded**: median alignment 0.97 against the ruling.

## Getting segmentation into Transkribus

- **Pushing PAGE XML page by page over the REST text endpoint**, after the zip upload ingested the images but silently discarded the `page/` XMLs — **succeeded**: 98/98 pages carry the segmentation without re-uploading a single image.

## Running the model

- **Submitting PyLaia jobs directly** via `POST /pylaia/{col}/{model}/recognition` and polling `/jobs/{id}`, instead of driving the web UI by hand — **succeeded**; the assumption that this was a UI-only action was wrong.

## Hallucination on sparse pages

- **Emitting TextLines only where an OpenCV ink measure finds handwriting** (paper-luma threshold minus printed rules and page-edge blobs; threshold 12px, calibrated on the GT pages where the distribution is cleanly bimodal — median 222px in cells with text, 0 in cells without), to deal with PyLaia writing text into every blank cell it is handed a line for — **succeeded**: invented cells on page 11 went from 416 of 532 to zero.

## The actual transcription

- **Kraken's Ottoman Arabic models** — **failed** (40.5% and 19.8% cell accuracy; wrong scribe style, one outputs Latin).
- **QARI-OCR on cell crops** — **failed** (CER 5.46; ignores the pixels and hallucinates document structure).
- **Gemini 3 at LOW reasoning, full page to structured JSON** — **best result so far, still insufficient**: 56–70% perfect cells, $0.08/page, 720 RA keystrokes per page.
- **Fine-tuning PyLaia on our own GT in Transkribus** (hadid01, then hadid02 on normalised GT) — **failed to beat Gemini**: 45.8% perfect, 33.8% wrong, 970 keystrokes per page; both models overfit on ~3,000 training lines.
- ⇒ **The transcription challenge is unsolved.** 65% of the remaining errors are genuine misreads, dominated by this scribe's numeral confusions (٢/٣, ٣/٤, ٤/٦, ٦/٨) — not layout, and not something better segmentation will fix.

## Measurement

- **Per-cell scoring with an RA-keystroke cost model** (the frozen `score_g3_vs_gt.py` methodology, reused unmodified) to make model comparisons decidable — **succeeded**: it is what showed Gemini beating the fine-tuned model, rather than an impression.

## Ground truth

- **RA correction in Transkribus plus the Streamlit correction view** — **partly succeeded**: 6 pages corrected, 1 verified, ~3,000 lines. That volume is now the binding constraint on training, and it is why the model overfits.
