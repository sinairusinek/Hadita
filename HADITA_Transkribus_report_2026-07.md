# Hadita tax register — HTR/table-extraction status and a collaboration proposal

**To:** Transkribus support / engineering
**From:** Sinai Rusinek (Hadita project)
**Date:** 30 July 2026
**Transkribus refs:** collection `2377415`; documents `17829738` (current, "Hadita-final2"), `15829823` (previous, "Hadita-Processed"); models `592509` (hadid01), `592709` (hadid02)

## 1. The source and the corpus

A bound village property-tax register: printed English column headers, all entries handwritten in Arabic script. Amounts are in Palestine pounds and mils; entries reference Tax Distribution Lists dated 1938–1945. Each page is a 19-column table.

| | |
|---|---|
| Scanned pages | 102 (98 with table content; 2, 7, 8, 102 are blank) |
| Segmented rows | 3,069 |
| Cells | 58,122 |
| Cells actually carrying handwriting | 18,718 (32%) |
| Per-cell reference transcriptions | 6 pages RA-corrected; 1 page independently verified |

Three properties make it hard, and interesting:

- **Two digit systems coexist and both are authentic.** Eastern Arabic-Indic (٠–٩) in most columns, Western in `New_Serial_No` and the Tax/Distribution-List references. We do not normalise them, because the distinction is in the source.
- **Ditto and nil conventions carry meaning.** A cell may hold `"` (same as above), `-` (nil), or a `✓` mark. Visually near-empty cells are not semantically empty — 97 cells in our reference set hold text with essentially zero ink.
- **Row semantics vary within a page.** Multi-row parcels, tax-year breakdown rows with no serial, sub-totals, and carry-forward rows. Any model or prompt tuned on the neat pages degrades on the rest.

## 2. What we have built

**Segmentation.** Kraken baseline segmentation for rows, per-band projection for the 19 printed column rules, exported as PAGE XML with `TableRegion`/`TableCell` and a positional `structure {type:…}` tag per column.

**A geometry rebuild, July 2026.** Our earlier pipeline flattened each page with `cv2.remap` and wrote a straight grid onto the flattened canvas. An audit of all 98 pages found this cost content on **every page**: the canvas ended at `last_row_center + ½ pitch`, so 25 pages lost a whole written row and 31 lost part of one, and the band between the header and the first row was compressed on 81 pages. We inverted the approach — upload the **undamaged deskewed page** and curve the cell polygons to follow it. That recovered 20 rows at the bottom and 15 at the top, and fixed a column defect affecting 59 of 98 pages (a missed faint rule left the last column absent and columns 14–17 mis-tagged). Column boundaries now sit on the printed ruling with a median hit rate of 0.97 (±8px).

The corpus in document `17829738` is the result.

## 3. What we have measured

Scored per cell against reference transcriptions, classified perfect / single-digit / multi-digit / wrong, with an RA-correction cost in keystrokes.

| Approach | Result |
|---|---|
| Kraken `gen2_sc_clean` (Ottoman Arabic) | 40.5% cell accuracy, CER 0.47 — wrong scribe style |
| Kraken `muharaf_rec_best` | 19.8%, CER 0.78 — outputs Latin |
| QARI-OCR v0.3 on cell crops | CER 5.46 — hallucinates structure, ignores pixels |
| **Gemini 3 (LOW reasoning), full page → JSON** | **70.1 / 65.2 / 56.1% perfect cells** (pp. 3/4/5); $0.082 per page; **720 RA keystrokes/page** |
| **Transkribus PyLaia `hadid02`** | **45.8% perfect, 33.8% wrong** over 6 reference pages; **970 RA keystrokes/page** |

The two PyLaia models were trained on our own GT: `hadid01` on 3,102 lines (train ~10% / val 22.2% CER), `hadid02` on 2,964 normalised lines (train 9.5% / val 28.4%). Both show classic small-GT overfitting. Per page, `hadid02` ranges from 66.9% perfect on our cleanest page to 14.0% on a sparse one.

Decomposing `hadid02`'s disagreements: **65% are genuine misreads**, 18% are cells where it wrote text the reference leaves empty, 16.5% cells it left empty that the reference fills. The dominant residual error is digit recognition on this scribe's hand — the documented ٢/٣, ٣/٤, ٤/٦, ٦/٨ confusions — not layout.

**Our conclusion:** a general-purpose vision LLM currently beats our fine-tuned PyLaia model on this material by roughly 26% of downstream correction effort, and neither is yet good enough to seed a GT-grade corpus without cell-by-cell review.

## 4. Why we would like an early try at Smart Extract

Our real task is not line transcription, it is **structured extraction from a ruled table with semantics** — 19 typed columns, ditto/nil conventions, and row types that must be recognised as such. That is precisely the shape of problem Smart Extract addresses, and we think this corpus is an unusually good early test:

1. **It is already in Transkribus**, fully segmented, with per-column type tags — no ingestion work required.
2. **We can measure honestly.** We have per-cell reference transcriptions on 7 pages, a frozen scoring methodology (cell classes plus an RA-keystroke cost model), and published negative results for four other approaches to compare against.
3. **It is hard in an informative way.** Mixed digit systems, semantically-loaded near-empty cells, heterogeneous row types, and a strong page bow — failure modes that a clean printed table would never expose.
4. **We would report back.** We are documenting this project publicly, including what did not work; we would share the evaluation on Smart Extract in the same form, and contribute our GT and PAGE XML back.

**The ask:** early access to Smart Extract for document `17829738`, and if useful to you, a short technical conversation about the results.

## 5. Three questions on model training

Independently of Smart Extract, we would value your view on:

1. **Volume vs. overfitting.** ~3,000 training lines produced a 22–28% validation CER with a large train/val gap. Is more GT the answer here, or a different base model / augmentation regime for Eastern Arabic-Indic numerals?
2. **Cell-level vs. line-level training.** Our lines *are* table cells — often one to four glyphs. Is PyLaia the right choice for such short lines, and does training benefit from cell context or suffer from it?
3. **Digit-specialised training.** Most of our residual error is a handful of numeral confusions. Is there a supported way to weight or specialise training toward a character subset?

## Appendix — three platform issues we hit

1. **Upload silently dropped the PAGE XML.** We uploaded zips in the documented layout (images at the top level, PAGE XML in `page/`). Transkribus ingested the images and created empty `<Page/>` transcripts, discarding the segmentation without a warning. We recovered by pushing the XMLs onto the existing pages via `POST /collections/{c}/{d}/{p}/text`, which worked correctly — but an upload that partially succeeds silently is dangerous.
2. **The same upload duplicated 95 of 98 pages**, producing a 193-page document. There is no page-delete in the legacy REST API, so the duplicates remain.
3. **PyLaia transcribes every `TextLine` it is given, with no per-cell confidence to filter on.** Our XML initially placed a TextLine in every cell; on one sparse page the model returned text for 416 of 532 cells that are blank paper. We fixed it on our side by emitting TextLines only where ink is present (dropping 39,404 of 58,122), and the invented output went to zero. A confidence value per line, or an explicit "empty" prediction, would let users detect this rather than discover it by inspection.
