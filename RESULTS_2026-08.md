# Hadita HTR — Results Synthesis (August 2026)

Paper-ready synthesis of every approach tried on the Hadita Mandate-era tax-register
corpus, closing with the August 2026 "last try" round (E0–E8, `EXPERIMENTS_2026-08.md`).

**Corpus:** 98 pages, handwritten Arabic tabular tax registers (British Mandate
Palestine, 1930s–40s), 19 data columns, Arabic-Indic numerals, heavy use of ditto
marks and nil-dashes, most pages sparsely filled. Grid geometry is known per page
(final2 PAGE XMLs, cell polygons). Proxy GT: 6 RA-corrected pages (3, 4, 5, 6, 9, 10);
sparse offenders are 9 and 10.

**Primary metric:** RA keystrokes to correct a page (`ra_cost`, frozen in
`score_g3_vs_gt.py`); structural metric: missed/phantom rows. ⚠ Run-to-run variance
of the Gemini pipeline is **±10% with identical inputs** (5-run test, E3c): any
single-run difference under ~10% in the tables below is uninterpretable noise. Row-loss
differences (23 → 1) are far outside the noise band.

---

## 1. Headline result

**The sparse-row problem that blocked the project for four months is solved — not by a
bigger model, but by removing row-counting from the model.**

`gemini-3.7-flash` + set-of-marks overlay + Needleman-Wunsch row alignment +
ink-guided repair:

| | v5 baseline (2026-06) | final pipeline (2026-08) |
|---|---|---|
| missed rows (6 GT pages) | 23 | **1** |
| phantom rows | 4 | 4 |
| RA keystrokes (6 pages) | ~5,900 (offset-scored) | **3,692** |
| cost / page | $0.08 | **$0.012** |
| cost / 98-page corpus | ~$8 | **~$1.20** |

Residual error is now genuine recognition error (Arabic-Indic digit shapes), no longer
structural. That is a different and more tractable problem.

## 2. Master comparison — every approach ever tried

| # | approach (date) | what it was | key result on proxy GT | verdict |
|---|---|---|---|---|
| 1 | **Kraken segmentation + cell HTR** (2026-04) | Kraken baselines + OpenCV columns → per-cell crops → Kraken HTR | gen2 ≈ 40.5%; cell HTR clearly worse than Gemini on the same crops | ❌ kept only for row detection |
| 2 | **QARI v0.3** (2026-04) | open Arabic OCR VLM on handwritten cell crops | CER 5.46 — hallucinates fluent prose | ❌ explained by provenance (E8): 100% synthetic *printed* fonts, zero handwriting |
| 3 | **Column strips S-lite/S-full/R** (2026-04) | vertical single-column strips to Gemini | model cannot count rows in a strip; misalignment everywhere | ❌ do not revisit |
| 4 | **Gemini v5 agentic** (2026-05/06) | `gemini-3-flash-preview` + code_execution, full page, frozen prompt | production baseline: ~720 ks/page on dense GT pages; p9 21/24 rows, p10 22/27 | ✅ was SOTA; now superseded (and its API path is dead on 3.x models) |
| 5 | **Gemini v6 dual-image** (2026-07) | original + dewarped image together | +11% *worse* than v5 | ❌ |
| 6 | **Transkribus PyLaia Hadid01/02** (2026-06/07) | fine-tuned field model, 592509/592709, on final2 | 45.8% perfect cells; ~970 ks/page vs Gemini's 720 on shared pages; nonsensical digit strings | ❌ as primary; ✅ reborn as *agreement partner* (E6) |
| 7 | **E1: newer Gemini models** (2026-08) | same v5 prompt, structured output, final2 | 3.7-flash −17% ks vs 3-flash; row loss reduced (p9: 7→2-3 missed) **but not eliminated** | ✅ model refresh helps; doesn't solve sparsity |
| 8 | **E2: NW row alignment** (2026-08) | Needleman-Wunsch over rows in scoring/merging, gaps anywhere | −20–28% ks, zero API cost; much "OCR error" was misalignment artifact | ✅ FREE WIN |
| 9 | **E3: set-of-marks** (2026-08) | row indices printed in red on the image; model keys output to printed number | missed rows 23→4; 0 missed on pages 3,4,5,6,9; $0.011/page | ✅ **WINNER** |
| 10 | E3b: SoM + bigger model / more thinking | 3.1-Pro, medium thinking | Pro 64% worse; medium thinking 18% worse at 1.6× cost | ❌ counter-scaling result, paper-worthy |
| 11 | **E4: grounded OCR** (2026-08) | text+bbox detections, geometric cell assignment | placement essentially perfect; recall too low (86 detections vs 120 inked cells) | ❌ |
| 12 | **E5: single-row strips** (2026-08) | one call per inked row | works, but $0.30/page = 18× SoM | ❌ whole-page; ✅ as repair tool |
| 13 | **E5b: ink-guided repair** (2026-08) | re-read only rows that are inked but came back empty | missed rows 4→1, ~$0.01/page only where needed | ✅ completes the fix |
| 14 | **E6: agreement triage + third reader** (2026-08) | SoM × Hadid02 per-cell agreement; Claude reads flagged crops | see §4 | ✅ conditional pass; third reader ❌ |

## 3. Sparse vs dense — the decisive breakdown

Rows returned vs GT rows on the two sparse pages (the project-killers) and one dense page:

| approach | p9 (24 GT rows) | p10 (27 GT rows) | p6 dense (33 GT rows) |
|---|---|---|---|
| v5 model, full page | 17 | 15 | 33 |
| 3.7-flash, full page | 21 | 21 | 27 |
| 3.1-pro, full page | 22 | 23 | 33 |
| grounded OCR | (7 missed total, all pages) | | |
| **set-of-marks** | **24** | 23 | 33 |
| **SoM + ink repair** | **24** | **26** | 33 |

Dense pages were never the problem: every full-page approach reads page 6's 33 rows.
Sparse pages defeat anything that asks the model to count. Printing the row index on
the image is the single intervention that closed the gap.

This mirrors the only published precedent (Can & Kabadayı, Ottoman *nüfus* registers):
99.76% on isolated digits collapsing to ~60% on full pages — third-party confirmation
that **segmentation/localization, not the classifier, is the bottleneck** in tabular
numeral HTR.

## 4. Agreement triage (E6) — what the GT pipeline can auto-accept

Per-cell agreement on the 6 GT pages, SoM output × a second source, gate from
`PLAN_GT_pipeline_2026-07.md` (MATCH-cell error ≤1%):

| pair | auto-accept | MATCH error (strict) | MATCH error (substantive) |
|---|---|---|---|
| **SoM × Hadid02** (cross-family) | 32.8% | 3.18% | **0.88% — PASS** |
| SoM × SoM rerun (same model) | 66.5% | 20.8% | 17.2% — FAIL |

*Substantive* = after harmonizing notation families GT treats inconsistently: dash
forms (`-`/`--`/`__`), dropped `✓` prefixes, digit script, thousands separators — 13 of
18 strict MATCH "errors" were conventions, not misreadings.

**Finding 1 — independence beats accuracy.** Hadid02 reads only 45.8% of cells
correctly, but it fails *differently* from Gemini, so its agreement is informative:
0.88% error. A same-model rerun agrees twice as often but wrongly 17% of the time —
correlated errors. Self-consistency (Consensus-Entropy-style, single model) is **not**
a valid triage layer on this material.

**Finding 2 — the third reader on crops fails.** Claude reading 240 flagged cell crops
blind: 40% vs GT (86% on self-rated high-confidence, 21% on low); adjudication does
not beat simply trusting the SoM value (32.6% vs 34.7%). Claude makes the *same*
mistakes as Gemini: ٢/٣ confusion (36/144 errors), faint pencil ✓ dismissed as empty.
In flagged MISMATCH cells, *neither* source was right 38% of the time — flags mark
genuinely hard ink, and isolated crops discard the context that makes it legible.

**Finding 3 — the invisible remainder.** ~4% of cells have GT content missed by *both*
readers (sparse-row ticks and dashes); agreement can never flag them. The ink gate
(E5b) marks candidates for free; RA review of inked-but-empty cells stays mandatory.

**Rollout verdict:** flagged seeds are viable in exactly one configuration — SoM
primary, Hadid02 agreement, harmonized comparison: ~⅓ of cells auto-accepted at 0.88%
error, the rest RA-corrected, ink-gated empties marked. Raising auto-accept coverage
requires a second *independent model family*, not a Gemini rerun.

## 5. Negative results (keep these in the paper)

1. **Bigger models are worse under set-of-marks.** 3.1-Pro: 64% more keystrokes than
   3.7-flash, truncated page 3 at 14/33 rows. Under v5 prompting it was merely equal.
2. **More thinking is worse.** Medium budget: +18% error at 1.6× cost vs low.
3. **Same-model self-consistency is invalid as an agreement layer** (17% error inside
   agreements) — despite ±10% run variance, errors are correlated.
4. **Grounded OCR under-detects** faint/short entries (recall, not placement, fails).
5. **A same-generation VLM third reader adds no orthogonal signal** on cell crops.
6. **Column strips, dual-image input, QARI, Kraken cell HTR** — all fail for reasons now
   understood (counting, distribution shift, provenance, model capacity respectively).
7. **`code_execution` agentic OCR is dead on Gemini 3.x** (MALFORMED_FUNCTION_CALL);
   structured output is the replacement. The v5 pipeline is unreproducible on current APIs.
8. **Prompt-only fixes for near-empty rows don't stick** (som-v2 within noise); geometry
   + ink gating fixes them for $0.01/page.

## 6. Cost

| pipeline | $/page | 98-page corpus |
|---|---|---|
| Gemini v5 agentic (2026-06) | ~$0.08 | ~$8 |
| 3.7-flash full page (E1) | $0.017 | $1.70 |
| **set-of-marks + repair (final)** | **$0.012** | **$1.20** |
| grounded OCR | $0.017 | — |
| single-row strips | ~$0.30 | (rejected) |
| Transkribus PyLaia | paid credits per run | — |
| whole August round, 67 logged runs | | **$1.79 total** |

## 7. Where this leaves the project

The corpus sits in a documented gap: **Arabic script × tabular layout × Arabic-Indic
numerals × degraded forms** — no published model or dataset occupies that intersection
(E8 provenance audit: QARI is synthetic printed text; GLM-OCR/PaddleOCR-VL never
benchmark Arabic handwriting; the Muharaf→Nakba lineage is prose). The nearest
neighbour is the ERC LOOP project (Late Ottoman Palestinians, ISA *nüfus* registers —
tabular, Arabic-script, digit-rich): its page scans are public and it publishes
extracted data, but the data entry is Latin-transliterated database records and its
HTR ground truth / models are, per its own research notes, not yet released. So the
defensible claim is narrower and hedged: **to our knowledge, no public dataset yet
pairs page images of handwritten Arabic-script tabular registers with cell-level
Arabic-script transcriptions** — Hadita GT would be the first, noting that LOOP's HTR
work is in progress and that their registers are Ottoman-Turkish-language where ours
are Arabic-language.

Recommended path: run the final pipeline over all 98 pages (~$1.20), push flagged
seeds (SoM × Hadid02, harmonized) to Transkribus, RA-correct with ~⅓ auto-accepted,
and treat the resulting GT as both the paper's dataset contribution and the training
set for a properly-matched field model.
