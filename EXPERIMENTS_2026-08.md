# Hadita "Last Try" Experiments — August 2026

Running log. Every experiment appends here. Plan: `~/.claude/plans/please-make-a-plan-keen-yao.md`.

**Core question:** can any model or strategy reliably populate a known-geometry cell grid
from a sparse handwritten table page — i.e. solve the row-alignment problem that killed
every previous approach?

**Baseline (v5, `gemini-3-flash-preview`, dewarped `final/` corpus):**

| page | gt_rows | v5 keystrokes |
|---|---|---|
| 3 | 33 | 505 |
| 4 | 35 | 545 |
| 5 | 35 | 1156 |
| 6 | 33 | 1921 |
| 9 | 24 | 415 |
| 10 | 27 | (no v5 artifact) |
| **total (5 pages)** | | **4542** |

Known v5 sparse-row failure: page 9 → 21/24 rows, page 10 → 22/27 rows.

---

## E0 — Setup & model discovery (2026-08-29) ✅

- `GOOGLE_API_KEY` present; `google-genai` 1.47.0; Python 3.9 venv.
- final2 corpus intact: 98 pages, `Transkribus upload/final2/Hadita_{N}.{jpeg,xml}`.
- v5 baseline re-derived (table above) via `score_v6_vs_v5.py`.

**Available Gemini models (generateContent), newer than expected:**

| model id | note |
|---|---|
| `gemini-3-flash-preview` | v5 production baseline |
| `gemini-3.1-pro-preview` | top Pro tier, 1M in / 65k out |
| `gemini-3.1-flash-lite-preview` | cheap tier |
| `gemini-3.5-flash`, `gemini-3.5-flash-lite` | |
| ~~`gemini-3.5-transcribe`~~ | **NOT usable — speech-to-text.** Sending an image returns 400 "Image input modality is not enabled for this model". The name is misleading; it is not a document-transcription model. |
| `gemini-3.6-flash`, `gemini-3.7-flash` | newest flash line |
| `gemini-pro-latest`, `gemini-flash-latest` | aliases |

Selected for E1: `gemini-3-flash-preview` (continuity), `gemini-3.7-flash` (newest flash),
`gemini-3.1-pro-preview` (top tier). `gemini-3.5-transcribe` was probed and **ruled out —
it is a speech model and rejects image input**.

---
## E1 — Model A/B on final2, v5 prompt (2026-08-29) ✅

Runner: `run_exp2608.py` (sibling of run_g3v6_local.py; final2 corpus, single image).
Scorer: `score_exp2608.py --tags ...` (imports frozen metric from score_g3_vs_gt).

**Two API findings that blocked the old path:**
1. `code_execution` + the v6 prompt returns `FinishReason.MALFORMED_FUNCTION_CALL` on
   every 3.x model — the agentic-vision path used by v5/v6 is effectively dead on the
   current API. Structured output (`response_schema`) is the working replacement.
2. v6's `_strict_schema()` emits `additionalProperties`, which the current API rejects
   with 400. `run_exp2608.clean_schema()` strips it.

**Results — RA keystrokes (lower = better), standard constant-offset alignment:**

| tag | p3 | p4 | p5 | p6 | p9 | p10 | total |
|---|---|---|---|---|---|---|---|
| `gemini-3-flash-preview` (v5 baseline model) | 313 | 632 | 1512 | 1953 | 730 | 772 | 5912 |
| `gemini-3.7-flash` | (pending) | 464 | 916 | 1876 | 578 | 803 | 4637* |
| `gemini-3.1-pro-preview` | 381 | 496 | 1223 | 1506 | 811 | 832 | 5249 |

\* over pages 4,5,6,9,10 only. On those same 5 pages the baseline totals 5599 → **3.7-flash is ~17% better**.

**Rows returned vs GT rows (the sparse-row problem, unchanged in kind):**

| page | GT | 3-flash | 3.7-flash | 3.1-pro |
|---|---|---|---|---|
| 9 | 24 | 17 | 21 | 22 |
| 10 | 27 | 15 | 21 | 23 |
| 6 | 33 | 33 | 27 | 33 |

Newer models **reduce** row loss (page 9: 7 missed → 2–3) but **do not eliminate** it.
Cost collapsed: ~$0.017/page for flash (vs the $0.08/page assumed for v5), ~$0.06/page for Pro.

## E2 — Ink/occupancy row alignment (2026-08-29) ✅ FREE WIN

`score_exp2608.py --ink`: Needleman–Wunsch over rows (gaps allowed anywhere) replaces
the frozen scorer's single constant offset. Similarity = 3×(exact cell agreement) +
0.15×(occupancy agreement). Pure post-processing — no new API calls.

| tag | offset-aligned ks | NW-aligned ks | Δ |
|---|---|---|---|
| `gemini-3-flash-preview` | 5912 | **4705** | −20.4% |
| `gemini-3.7-flash` (5 pages) | 4637 | **3349** | −27.8% |
| `gemini-3.1-pro-preview` | 5249 | **4170** | −20.6% |

Biggest single gain: page 6 / 3-flash 1953 → 1179 ks. Interpretation: **a large fraction
of measured "OCR error" was never recognition error at all — it was row misalignment**
downstream of a dropped row. The constant-offset assumption in the frozen scorer
overstates error whenever a row is dropped mid-page, which is exactly the sparse-page case.

**Best so far: `gemini-3.7-flash` + NW alignment = 3349 ks (5 pages)** vs 5599 for the
v5-model/offset-aligned baseline on the same pages — a **40% reduction** in RA effort.

---
## E3 — Set-of-Marks (row numbers printed on the image) (2026-08-29) ✅ WINNER

`make_som_pages.py` draws each grid row's index (from the final2 cell polygons) in both
margins plus a separator line on every row boundary; `run_som_ocr.py` asks the model to
key each output row to the **printed** red number and to emit `{column_index: text}`.
The model therefore never counts rows and never names columns — both come from geometry.

Visual check of `som/Hadita_9_som.jpg` confirmed marks land on the true rulings.

## E4 — Grounded OCR (text + bbox, cells assigned geometrically) (2026-08-29) ❌

`run_grounded_ocr.py`: model returns `{ymin,xmin,ymax,xmax,text}` per handwritten item;
we assign each detection to the final2 cell polygon containing its centre (point-in-poly,
nearest-centroid fallback). **Placement works essentially perfectly** — 0 dropped, 2-5
fallbacks and 1-14 collisions per page out of 94-354 detections — so the geometric join
is not the problem. The problem is **recall**: page 9 produced 86 detections where GT has
120 non-empty cells. The model under-detects faint/short entries, so grounded OCR loses
on transcription completeness even though it fixes alignment.

## E5 — Single-row strips — deprioritised on cost

`run_row_strips.py` works (ink gate finds 33/35 inked rows on page 9, strips include the
dimmed previous row for ditto context), but a 4-row probe cost $0.0374 → ~$0.30/page,
about **18× the set-of-marks cost** for the same problem set-of-marks already solves.
Kept as a targeted repair tool, not a whole-page strategy.

### Results — all approaches, NW-aligned, RA keystrokes

| tag | p3 | p4 | p5 | p6 | p9 | p10 | total |
|---|---|---|---|---|---|---|---|
| `g37-flash` (full page, v5 prompt) | — | 464 | 916 | 921 | 514 | 534 | 3349* |
| **`som-g37-flash` (set-of-marks)** | 449 | 583 | 987 | **721** | **452** | **437** | **3629** |
| `grounded-g37-flash` | 758 | 762 | 1359 | 1226 | 518 | 647 | 5270 |

\* pages 4,5,6,9,10 only. **On those same 5 pages: set-of-marks 3180 vs 3349 — SoM wins.**

### The decisive metric — row loss across all 6 pages

| approach | missed rows | phantom rows |
|---|---|---|
| `gemini-3-flash-preview` (v5 model, full page) | **23** | 4 |
| `gemini-3.7-flash` (full page, 5 pages) | 18 | 1 |
| `gemini-3.1-pro-preview` (full page) | 10 | 3 |
| `grounded-g37-flash` | 7 | 3 |
| **`som-g37-flash`** | **4** | 3 |

Set-of-marks reaches **0 missed rows on pages 3, 4, 5, 6 and 9** — including page 9, the
sparse page that had defeated every previous approach. Only page 10 still loses 4 rows.
Cost: ~$0.011/page, the cheapest of the three strategies.

**Conclusion: printing the row index onto the image is what breaks the sparse-row
problem.** Not a bigger model, not better geometry downstream — removing the counting
task from the model.

---
## E3b — Set-of-Marks: model and prompt variants (2026-08-29)

| variant | model | thinking | total ks (6 pages) | $/page |
|---|---|---|---|---|
| **`som-g37-flash`** | gemini-3.7-flash | low | **3629** | $0.011 |
| `som-v2` (explicit sparse-row instruction) | gemini-3.7-flash | low | 3720 | $0.010 |
| `som-g37-med` | gemini-3.7-flash | medium | 4291 | $0.017 |
| `som-g31-pro` | gemini-3.1-pro-preview | low | 5938 | $0.046 |
| `som-g36-flash` | gemini-3.6-flash | low | 7078 | $0.026 |

Two counter-intuitive results, both worth reporting in the paper:
* **The biggest model is much worse.** 3.1-Pro under-transcribes badly under set-of-marks
  (page 9: 18 perfect cells vs Flash's 55) and truncated page 3 at 14 of 33 rows.
* **More thinking is worse.** Medium thinking budget costs 1.6× and scores 18% worse
  than low.

## E3c — Run-to-run variance (2026-08-29) ⚠ METHODOLOGICAL

Five runs of the same config on pages 6+9: **1173, 1277, 1361, 1405, 1415 ks** — a spread
of ±10% with identical inputs. Consequence: **differences under ~10% between variants in
this project are not interpretable from a single run.** The `som-v2` prompt "regression"
above is within this band and is noise. This also retro-justifies the 2026-07 note that
v6 run-to-run variance was real, and means the E1/E3 headline comparisons should be read
as "large effects only" (SoM's row-loss fix, 23 → 4, is far outside the noise band; the
3629-vs-3349 keystroke gap is not).

### Remaining set-of-marks weakness

Page 10's 3-4 residual missed rows are almost all rows whose ONLY content is a single
dash (nil marker) — e.g. `{Tax_LP: '-'}`, `{Tax_LP: '-', Net_Assessment_LP: '-'}`. These
are visually near-empty rows. An explicit prompt instruction about them (som-v2) did not
reliably fix it. This is the natural target for a cheap targeted repair pass using
`run_row_strips.py` on ink-positive rows the model returned empty.

---
## E5b — Targeted repair of inked-but-empty rows (2026-08-29) ✅ COMPLETES THE FIX

`repair_empty_rows.py`: after a set-of-marks run, the ink gate identifies grid rows that
carry marks but came back empty; only those rows are re-read as single-row strips. No
model is asked to find them — geometry plus ink does that for free.

| page | inked-but-empty | filled | cost |
|---|---|---|---|
| 3, 4, 5 | 0 | — | $0 |
| 6 | 1 | 1 | $0.010 |
| 9 | 8 | 0 | $0.015 |
| 10 | 4 | 3 | $0.027 |

(Page 9's 8 targets were genuinely blank rows the ink gate over-flagged — correctly
left empty, which is the desired conservative behaviour.)

## FINAL PIPELINE — result of Sessions 1+2

**`gemini-3.7-flash` + set-of-marks + NW row alignment + ink-guided repair**

| page | GT rows | pred rows | perfect cells | missed | phantom | ks |
|---|---|---|---|---|---|---|
| 3 | 33 | 34 | 248 | 0 | 1 | 449 |
| 4 | 35 | 35 | 267 | 0 | 0 | 583 |
| 5 | 35 | 35 | 158 | 0 | 0 | 987 |
| 6 | 33 | 35 | 165 | 0 | 2 | 724 |
| 9 | 24 | 25 | 55 | 0 | 1 | 452 |
| 10 | 27 | 26 | 65 | 1 | 0 | 497 |
| **total** | | | | **1** | **4** | **3692** |

**Row loss across the 6 proxy-GT pages: 23 missed (v5 baseline) → 1.**
Cost ≈ $0.012/page (~$1.20 for the 98-page corpus), vs the $0.08/page assumed for v5.

The residual cost is now genuine **recognition** error (reading the Arabic digits), not
structural error. That is a different, and more tractable, problem than the one that
blocked this project for four months — and it is the right target for E6 (Claude as a
second reader on disagreeing cells).

### What actually solved it

Ranked by contribution:
1. **Printing row indices on the image** (set-of-marks) — removes row counting from the
   model. 23 → 4 missed rows. ~$0.011/page.
2. **Needleman-Wunsch row alignment in scoring/merging** — removes the constant-offset
   assumption. −20-28% keystrokes, free.
3. **Ink-guided targeted repair** — 4 → 1 missed rows. ~$0.01/page, only on pages needing it.
4. Newer model (3.7-flash over the v5 model) — real but smaller effect, and 5× cheaper.

Not what solved it: bigger models (3.1-Pro is *worse*), more thinking (worse), grounded
bbox OCR (placement perfect, recall too low), single-row strips as a whole-page strategy
(works, 18× too expensive).

---
## E6 — Agreement triage + Claude as in-session third reader (2026-08-29) ✅ MEASURED

### Setup

`agreement_layer.py` (written 2026-06, first ever run today; extended to parse the
PyLaia `line_cell_r{r}_c{c}` TextLine format) on the 6 proxy-GT pages. Primary =
set-of-marks winner (`exp2608/Hadita_{N}_som-g37-flash-rep.json`), second source =
`g3_results/Hadita_{N}_Hadid02_final2.xml`. All three artifacts are grid-indexed by the
same final2 geometry, so the join is per-cell with offset 0 everywhere. Grid rows were
mapped to proxy-GT rows with the E2 NW alignment; scorer = `score_e6_agreement.py`.

### The Phase-4 gate (PLAN_GT_pipeline_2026-07.md: MATCH-cell error ≤ 1%)

| agreement pair | auto-accept (MATCH) rate | MATCH err, strict | MATCH err, substantive* |
|---|---|---|---|
| **SoM × Hadid02** (cross-family) | 566/1727 = **32.8%** | 18/566 = **3.18%** FAIL | 5/566 = **0.88% PASS** |
| SoM × SoM rerun (same model, $0.076) | 1054/1584 = 66.5% | 219/1054 = 20.8% | 181/1054 = **17.2% FAIL** |

\* substantive = after harmonizing notation families the RAs treat as equivalent-but-
uncorrected in GT: dash forms (`-`/`--`/`__`), dropped `✓` prefixes (`✓"`→`"`),
Eastern/Western digit script, thousands separators. 13 of the 18 strict MATCH errors
are these conventions; only 2–3 are true misreadings (e.g. `٢٠,٠٠٩`→`٢,٠٠٩`).

**The two headline findings:**
1. **Reader independence beats reader quality.** Hadid02 is a *bad* reader (45.8%
   perfect cells) but an independent one — when it agrees with Gemini the cell is right
   99.1% of the time. A second run of the *same* Gemini model agrees twice as often
   (66.5%) but its errors are correlated: 17% of its MATCH cells are wrong. Same-model
   self-consistency is **not** a usable agreement layer; the ±10% run variance (E3c)
   does not decorrelate the mistakes.
2. **The gate passes only conditionally.** Cross-family MATCH cells are trustworthy
   (0.88% substantive) *if* the comparison harmonizes notation conventions; strictly
   scored they fail the 1% bar (3.18%).

Caveat for rollout: 74 aligned cells (≈4%) have GT content but are empty in *both*
sources (shared misses, mostly sparse-page tick/dash cells). Agreement triage cannot
flag these — RAs must still scan visually-inked empty cells; the ink gate from E5b can
mark candidates for free.

### Claude as third reader on flagged cells

Stratified blind sample of 240 flagged cells (144 MISMATCH, 48 ONLY_som,
48 ONLY_hadid02) cropped from the final2 JPEGs by cell polygon
(`make_e6_crops.py`, contact sheets in `exp2608/e6_sheets/`), read in-session with
column context but no source values visible (`exp2608/e6_sample.tsv` +
`e6_reader_errors.json`).

| metric | result |
|---|---|
| Claude vs GT, all 240 flagged cells | 96/240 = **40.0%** |
| … self-rated high-confidence (n=44) | **86.4%** |
| … medium (n=42) / low (n=154) | 59.5% / 21.4% |
| ONLY_hadid02 cells (mostly "is it really empty?") | 66.7% |
| MISMATCH cells | 32.6% |
| Adjudication rule (pick source matching Claude, else Claude) | 32.6% — *worse than* trusting SoM (34.7%) |
| Tie-break only (fall back to SoM when Claude matches neither) | 40.3% — +5.6pt, ~noise at n=144 |

Within sampled MISMATCH cells the truth was: SoM right 50, Hadid02 right 39, **neither
right 55** — flagged cells are dominated by cases where *no* reader has the answer.
Claude's error pattern is the *same* as Gemini's: 50/144 errors are one digit off, 36
involve the handwritten ٢/٣ confusion, 21 are faint pencil marks (esp. `✓` in Tax_LP)
dismissed as empty. Isolated cell crops discard the row/column context that makes these
legible; a third reader of the same VLM generation adds no orthogonal signal.

**E6 verdict:** flagged-seed rollout is viable in exactly one configuration — SoM
primary + Hadid02 agreement, notation-harmonized comparison, MATCH cells (≈33%)
auto-accepted at 0.88% error, everything else (67%) left to the RA, with ink-gated
empty-cell candidates marked. Claude-as-adjudicator on cell crops is **rejected**: it
does not beat trusting the primary. What *would* raise auto-accept coverage is a second
independent model family (not a Gemini rerun) — e.g. a cross-vendor VLM as second
reader — since independence, not accuracy, is what makes agreement trustworthy.

---
## E8 — Specialist Arabic models: provenance audit (2026-08-29)

Two research passes on who built the candidate models and — the question that actually
predicts success — **what they were trained on**. Full details in this section; the
short version is that our corpus sits in a documented gap.

### The structural finding

Our material is the intersection of four properties: **Arabic script × tabular
administrative layout × Arabic-Indic numerals × degraded government forms.**
**No published model or dataset occupies that intersection.** Every candidate was
trained on either Arabic *prose* or *non-Arabic* tables. Transkribus itself has both
table models and Arabic models — and has never combined them.

A second finding worth stating plainly: **every "SOTA handwriting" benchmark score in
the current VLM literature (GLM-OCR 87.0, PaddleOCR-VL 87.4) comes from an in-house,
unreleased, Chinese/English-only test set. None contains Arabic.** Those numbers are not
evidence about our material in either direction.

### Misraj / Baseer__Nakba — the NAKBA winner

**Who:** Misraj AI, the research arm of Misraj Technology, a Saudi commercial group in
Riyadh (CEO Safwan AlModhayan, also last author on both papers). Commercial, not
academic, but with a genuine open-research output. Weights are public: 4B params,
CC-BY-NC-SA-4.0, Qwen2.5-VL-3B architecture.

**Lineage and training data:**
* **Baseer (base)** — 500k image-text pairs: 300k *synthetic* (Common Crawl markdown →
  Word → PDF → image, 39 Arabic fonts) + 200k real (books, magazines, academic papers,
  40% labelled by another VLM). **Zero handwriting. Nothing historical.** It is a modern
  printed-Arabic document-to-Markdown converter. Critically, its synthetic pipeline
  **discarded any document whose tables had >25% empty cells** — precisely our
  distribution — and it emits tables as HTML in a Markdown stream, i.e. structure by
  autoregression with no geometric supervision.
* **Muharaf** (warm-up stage) — 1,644 pages / 36,311 lines of historical Arabic
  manuscripts, NC State + Holy Spirit University of Kaslik (Lebanon), 19th–21st c.,
  CC-BY-NC-SA. Genres: letters, diaries, church records, legal correspondence. Notably
  "financial" is a top-4 keyword in ~15 of its ~45 collections, and the authors mention
  "many lines with isolated numbers or single words" — the only numeral-adjacent signal
  in the whole lineage. Whether those financial pages are ruled ledgers is **undocumented**.
* **Nakba** (final stage) — Omar al-Saleh al-Barghouti's memoirs, 6,395 pages,
  1951–1965, Palestine Memory Project. **Continuous prose**, ~250 words/page.

**Competition conditions that matter for us:** NAKBA Subtask 2 handed every participant
**pre-segmented line images**. None of the reported scores reflect any layout or
segmentation difficulty — the part of our problem we spent four months on. Also, Misraj
won on the *corpus-level* (length-weighted) metric but **lost per-line** to two other
teams; our unit is the short cell, so the per-line column is the one that predicts our
experience.

**Predicted fit — testable:** best-matched public model on script, period and image
condition; worst-matched on document genre. Expect it to be **strong on Arabic-word
columns (names, places) and weak-to-hallucinatory on Arabic-Indic numeral columns and
on empty/ditto cells** — it was never trained to output nothing, and a prose language
prior is a liability where no linguistic context exists.

### QARI — our 2026-04 failure, now explained from primary sources

This closes a loose end. QARI (NAMAA + Prince Sultan University, Saudi Arabia) is
**100% synthetic, 100% font-rendered printed Arabic**: text corpora → HTML → WeasyPrint →
PDF → image, in 12 Arabic fonts. **There is no handwritten training data in the pipeline
at all.** The v0.3 "handwriting support" claim rests on a single qualitative figure with
no quantitative handwriting evaluation anywhere in the paper.

The paper's own Limitations section concedes: *"the model's performance on historical or
non-standard **Arabic numeral systems has not been extensively validated** and may be
suboptimal"*, and Future Work commits to *"improving numeral recognition… and extending
capabilities to Arabic handwriting recognition"* — i.e. the authors classify our exact
task as not yet done. Two further explanations of our CER 5.46: v0.3 is the *weakest*
version on raw character accuracy (CER 0.300 vs v0.2's 0.061), and 4-bit quantization
is catastrophic for it (8-bit CER 0.133 → 4-bit CER **3.228**).

Independent corroboration: QARI placed **7th of 8** in the NAKBA competition
(CER 0.195 vs Misraj's 0.079).

### Other candidates — ruled out or downgraded

| model | who | verdict |
|---|---|---|
| **Qwen3-VL** | Alibaba, Apache-2.0, 2B–235B | Only locally-runnable candidate with plausible breadth (4B or 8B-4bit via MLX on the 18GB M3 Pro). But **no training-data disclosure at all**, and **Arabic is never explicitly named** among its "32 OCR languages". |
| **GLM-OCR** | Zhipu AI, MIT, 0.9B | **The word "Arabic" does not appear anywhere in its technical report.** Language list is 8, all LTR. Its own worst self-reported category is Multilingual (69.3). Third parties publish Arabic fine-tunes of it — which implies the base cannot. |
| **PaddleOCR-VL** | Baidu, Apache-2.0, 0.9B | Arabic measured once (edit distance 0.122, ~9× the Latin rate, almost certainly *printed* lines), then **disappears entirely from v1.5 and v1.6**. Named handwriting corpora are Chinese and English only. |
| **HATFormer** | NC State (same group as Muharaf) | Right script and period, wrong genre (prose, line-level, RoBERTa decoder = lexical priors that hurt on isolated numerals). **No public weights could be found.** |

### The one genuine precedent — and the number that vindicates our architecture

Can & Kabadayı (Koç University, ERC UrbanOccupationsOETR) are the only research line
targeting handwritten Arabic-Indic numerals in ruled administrative registers
(19th-c. Ottoman *nüfus* population registers).

* Isolated digits, [Applied Sciences 10(16):5430, 2020]: **99.76% accuracy** — but the
  digits were isolated by a **red-ink colour filter**, a trick unavailable on our
  monochrome Mandate forms.
* Same pipeline on **real full pages**, [Electronics 10(18):2253, 2021]: **~60%.**

That **99.76% → 60% collapse is published third-party confirmation that in tabular
numeral HTR the bottleneck is segmentation/localization, not the classifier** — exactly
what our own 2026-07-30 verdict concluded, and exactly what set-of-marks addressed. For
their production dataset the same group **fell back to manual transcription**, using CNN
work only for validation, which justifies human-in-the-loop design rather than
apologising for it.

Their dataset (`UrbanOccupationsOETR_hdr_Nicaea_6k`, ~6,000 digits — the only historical
Arabic handwritten digit set in existence) appears **link-rotted**; worth emailing the
authors. MADBase/ADBase (70k isolated digits) is the practical pretraining source.

### Consequence for the paper

Three citable, evidence-backed claims:
1. QARI's failure is explained by provenance, not tuning — with the authors' own
   limitations text conceding numerals and handwriting.
2. Every current "SOTA handwriting" VLM score is Chinese/English in-house data; no
   published Arabic-benchmark number exists for GLM-OCR or PaddleOCR-VL at all.
3. The Ottoman 99.76%→60% drop confirms segmentation dominates classification here.

And: **our ground truth, if released, would — to our knowledge — be the first public
dataset pairing page images of handwritten Arabic-script tabular registers with
cell-level Arabic-script transcriptions.** The gap is the contribution.
(Checked 2026-08-29 against the nearest neighbour, the ERC LOOP project on the ISA
Ottoman *nüfus* registers of Palestine: scans public, extracted data published as
Latin-transliterated database records, HTR ground truth/models explicitly not yet
released. LOOP's HTR pipeline is in progress with eLijah-Lab and Teklia, so this
priority claim has a shelf life; also theirs is Ottoman Turkish, ours Arabic.)

---
## E7 — Synthesis for the paper (2026-08-29) ✅

Full comparison of every approach ever tried (Kraken, QARI, column strips, PyLaia
Hadid, Gemini v5/v6, E1–E6), sparse-vs-dense breakdown, cost table, and the
negative-results catalogue written to **`RESULTS_2026-08.md`** (also published as a
shareable web page). All single-run comparisons in it carry the E3c ±10% variance
caveat. Total August-round spend: $1.79 of the $30 budget.

---
## E8b — Baseer__Nakba measured on the Hadita registers (2026-08-29) ❌ but INFORMATIVE

First evaluation of the NAKBA competition winner outside prose, as far as we can tell.

**Setup.** `crop_cells.py` cuts per-cell crops from the final2 polygons (ink-gated);
`run_baseer_cells.py` runs `Misraj/Baseer__Nakba` locally (bf16 on MPS, M3 Pro 18GB) with
the official pipeline prompt, "Extract the text from the above document." 1,857 cells
across the 6 proxy-GT pages, ~0.5 s/cell, ~15 min total, **$0** (local inference).
Scored with the same frozen metric; `score_by_column_type.py` splits numeral vs word
columns and measures empty-cell hallucination.

### Headline

| tag | keystrokes (6 pages) |
|---|---|
| **`som-g37-flash`** (our pipeline) | **3629** |
| `baseer` (cell crops) | 6656 |

Baseer costs **~1.8× more RA effort** than gemini-3.7-flash + set-of-marks.

### The prediction, tested

The provenance audit predicted: *strong on Arabic-word columns, weak/hallucinatory on
numeral columns and empty cells.* Result — **half right, and the wrong half is the
interesting one**:

| metric | som-g37-flash | baseer |
|---|---|---|
| numeral cells correct | **892/1496 (59.6%)** | 470/1493 (31.5%) |
| word cells correct (`Nature_of_Entry`) | **66/129 (51%)** | 12/128 (9%) |
| empty-GT cells filled anyway (hallucination) | **39/1852 (2.1%)** | 158/1913 (8.3%) |
| GT-text cells returned empty | 140 | 145 |

The hallucination prediction was **confirmed and then some** — 4× the empty-cell
invention rate. But the "strong on word columns" half was **wrong in the opposite
direction**: Baseer is *worse* on the word column (9% vs 51%) than on numerals. A
prose-trained model does not transfer to isolated administrative abbreviations
(`تح`, `شاع`) any better than to digits.

### The most publishable single finding

Baseer's empty-cell hallucinations are **days of the week**:

```
Tax_LP            GT=""  Baseer='السبت'     (Saturday)
Tax_Mils          GT=""  Baseer='الخميس'    (Thursday)
Serial_No         GT=""  Baseer='الاربعاء'  (Wednesday)
Tax_Mils          GT=""  Baseer='الثلاثاء'  (Tuesday)
Date              GT=""  Baseer='الخميس'    (Thursday)
```

This is the **diary/memoir prior made visible**: shown a nearly-blank ruled cell in a
tax ledger, the model reaches for the vocabulary of the Nakba memoir corpus it was tuned
on. It is a clean, legible demonstration that the language-model prior is a *liability*
where no linguistic context exists — the argument the provenance audit made in the
abstract, caught in the act.

### Numeral confusions — the cross-family finding

| pair | som-g37-flash | baseer |
|---|---|---|
| ٢/٣ | 137 | 67 |
| ٣/٤ | 13 | 13 |
| ٦/٨ | 3 | 0 |
| ٤/٦ | 2 | 3 |

Top substitution for BOTH: **٣→٢** (135 for Gemini, 67 for Baseer). Two entirely
independent model families — a Google frontier VLM and a Saudi 4B Qwen2.5-VL fine-tune
trained on Palestinian manuscripts — make **the same dominant error on the same scribe's
hand**. That points at the source rather than the model: this scribe's ٣ genuinely looks
like a ٢. Combined with the parallel session's finding that Claude (a third family) also
confuses ٢/٣ in 36 of 144 errors, the ٢/٣ confusion looks like a property of the
document, not of any model — and therefore **not fixable by model choice**. It is the
strongest argument in the whole project for targeted human review of that one confusion.

### Verdict

Baseer__Nakba is not a replacement for the production pipeline. Its value here is
threefold: (1) it confirms the genre gap empirically, (2) it is model-family-independent
of Gemini, which the parallel session's E6 showed is the property that matters for
agreement triage — Baseer is a *better* agreement partner candidate than a Gemini rerun
(which failed at 17% MATCH error through correlated errors), and (3) the day-of-the-week
hallucination is a memorable, citable illustration of prior-driven failure.

Untested and worth one run if time permits: Baseer on **full-width single-row strips**
rather than isolated cells, which would restore the horizontal context the parallel
session found Claude needed (`run_row_strips.py` geometry + local inference = free).

---
## E8c — Baseer on row strips instead of cells: the deficit is the MODEL, not the crop

The obvious objection to E8b was that isolated cell crops starve a line-level model of
context (the parallel session found exactly that for Claude). Tested directly with
`run_baseer_rows.py` — full-width single-row strips, the input shape closest to
Baseer's training data (a text line):

| input shape | page 9 | page 10 |
|---|---|---|
| per-cell crops (E8b) | 31.5% cells correct (corpus-wide) | — |
| full-width row strips | **10.0% GT-token recall** | **15.3%** |

Row strips are **far worse**, not better. Sample output (page 9, row 0):

```
'١٦ ٢٨٨٤ ٢٩ ١٦ ٢٤٨٦  شعبان'      <- GT row is ١ / ٩٣٨ / ٤١٤٧ / ٤٩ / ١٦ / ٢,٤٨٦ / تح
```

It recovers some digits, loses column order entirely (no cell boundaries survive in a
free-text stream), and appends **شعبان** — the Islamic month Sha'ban, invented. Same
prose prior as the days-of-the-week hallucination, from the same cause.

**Conclusion: the deficit is the model-to-genre mismatch, not our cropping.** Giving
Baseer more context makes it worse, because the context is a ruled grid it has no
representation for, and the extra room lets the language prior run further. This
closes the objection and strengthens E8b's finding.

Also 8-10× slower per unit (4-6 s/row vs 0.5 s/cell).

## E8 — final verdict

`gemini-3.7-flash` + set-of-marks + NW alignment + ink repair remains the production
pipeline. No specialist Arabic model beats it, and the audit explains why: **none was
trained on anything resembling a ruled administrative numeral grid**, because no such
public dataset exists.

Baseer's residual value is as an **independent-family agreement partner** — the parallel
session's E6 showed that agreement partners fail when errors are correlated (a Gemini
rerun scored 17.2% MATCH error), and Baseer is architecturally and in training data
maximally independent of Gemini. Its 31.5% numeral accuracy is too low to be a primary,
but MATCH cells between two unrelated families would be strong evidence. Untested.

---
## E9 — Baseer as agreement partner: the E6 × E8 convergence test (2026-08-29) ✅

The two parallel sessions each left one arrow pointing here: E6 showed independence is
what makes agreement trustworthy; E8 produced a maximally independent (non-Gemini,
non-PyLaia, local, $0) reader whose outputs for all 6 GT pages were already on disk.
Question: does adding Baseer extend the auto-accept tier beyond SoM × Hadid02's 33%?

**Answer: no — but it refines it.** Tiering every non-empty SoM cell in aligned rows by
*which* independent reader agrees with it (substantive error vs proxy GT):

| tier | cells | substantive error |
|---|---|---|
| **Hadid02 AND Baseer agree** | 205 (13%) | **0.00%** |
| Hadid02 only agrees | 361 (24%) | 1.39% |
| Baseer only agrees | 210 (14%) | **20.95%** |
| neither agrees | 759 (49%) | 62.58% |

Two findings:
1. **Independence is necessary but not sufficient.** Baseer is architecturally and
   corpus-wise maximally independent of Gemini, yet its solo agreements run 21% error —
   contaminated by the scribe-level ٢/٣ confusion every model family shares (E8) and by
   trivially copyable ditto cells. A second reader must be independent *and* at least
   moderately competent before its agreement certifies anything.
2. **The double-agreement tier is essentially certified** (0/205 substantive). Worth
   using as the "no-review" tier in the RA workflow; SoM × Hadid02 (with convention
   harmonization) remains the auto-accept gate at ~33%; Baseer-only agreement must NOT
   be treated as confirmation.

Bookkeeping: agreement TSV filename collision bit twice — `agreement_{N}.tsv` is
regenerated by every agreement_layer run; the som×som2 copies live as
`agreement2_{N}.tsv`, the som×baseer ones as `agreementB_{N}.tsv`.

---
## E10 — Pilot: production pipeline on pages 11–20 (2026-08-29) ✅ SEEDS BUILT, LESSONS LEARNED

Full pipeline (SoM → ink repair → agreement vs Hadid02 → flagged XMLs) on pages 11–20,
chosen because Hadid02_final2 pulls already exist for them. Cost: $0.31
(OCR $0.065 + repair $0.25). Outputs: `exp2608/Hadita_{N}_som-pilot-rep.{json,xml}`,
`exp2608/Hadita_{N}_som_flagged.xml`, `exp2608/agreementP_{N}.tsv`.

**Pilot-page agreement is much thinner than on the GT pages:** MATCH 380 cells /
MISMATCH 936 / ONLY_som 180 / ONLY_hadid02 1,111 over 10 pages → auto-accept ≈ 14.6%
of non-empty cells (vs 32.8% on GT pages 3–10). Two causes, both diagnosed:

1. **The Hadid02 pulls for 11–20 were never ink-gated** and PyLaia invents text
   wherever a cell carries any speck (hallucinated cells sit at 15–90 ink px; the
   12 px gate passes them, and GT-calibrated thresholds can't separate specks from
   real dashes — gt_text median 68 px vs gt_empty 44 px, hopelessly overlapped;
   `gate_hadid_pull.py` written and measured, dropped 0 cells). Damage is confined:
   MATCH requires SoM non-empty, so the auto-accept tier is structurally immune, and
   ONLY_hadid02 entries surface only as XML comments Transkribus does not render.
2. **Pilot pages are faint-pencil-heavy and SoM under-reads them.** Page 17 is
   genuinely ~3 rows (SoM correct); page 18 carries dozens of faint pencil entries
   (dates, Total Tax, Net Assessment) of which SoM captured only 39 cells — and the
   E5b repair strips also declined (filled 0/13). Faint pencil on sparse pages is the
   next genuine recognition frontier, distinct from the solved row-counting problem.

Consequence for rollout math: the "⅓ auto-accepted" figure from the GT pages does not
transfer to faint pages; corpus-wide auto-accept will land between 15% and 33%. The
[?] MISMATCH flags (936 cells) are live in the flagged XMLs; all 10 validate as XML.
Next lever for faint pages: image preprocessing (contrast/pencil enhancement) before
SoM, and the digit-CNN second reader once pilot GT exists.

---
## E11 — Pencil enhancement + pilot seeds pushed (2026-08-29)

**Grid audit of the pilot pages** (prompted by the p18 SoM overlay): the two
worst-agreement pages have *defective final2 grids* — p18's bands start ~3 row-heights
too low (grid top y=915 vs ~650 healthy; its 4 top entry rows fall outside every cell
polygon) and p15 is ~2 rows low (y=834); p11 has 28 fat rows (height 112 vs ~93). The
"faint-page failure" of E10 is substantially a segmentation defect. Pages 11/15/18 need
grid re-derivation before their GT is trusted.

**Pencil enhancement (background-division + gamma, `make_som_enhanced.py`): clean
negative.** Legibility to the human eye improves dramatically, but SoM recall is flat
(p15 61→49, p17 32→31, p18 39→39 non-empty cells; within ±10% noise) for $0.008.
Gemini already reads faint ink as well as it will; contrast is not the binding
constraint — grid alignment and model reluctance on ghost marks are.

**Push:** all 10 flagged seed pages pushed to doc 17829738 (col 2377415), pageNrs 5–14,
toolName `Hadita-som-seed-2026-08-29`, tsIds 301482582–301482613 (`push_pilot_seeds.tsv`).
Transcripts are NEW versions; prior layers remain in the version picker. Total pushed:
1,360 non-empty cells, 934 [?] flags. Round spend: $2.11 of $30.

---
## E12 — Corpus-wide grid audit (2026-08-29) ⚠ MAJOR FINDING

Prompted by the user's challenge on the pilot's "segmentation" failures. One-pass audit
of all 98 final2 grids (band top-y, row count, row height vs corpus medians):
**50 of 98 pages flagged**; spot-checks CONFIRM real damage (p18: 4 entry rows above
the grid; p40: 3 rows above grid + phase-shifted bands). Dominant pattern: grid top
~150-450px (1.5-5 row-heights) too low, i.e. **the July final2 coordinate-warp rebuild
systematically lost the top rows of ~half the corpus**. The aggregate "alignment 0.97"
validation of 2026-07 could not see per-page band offsets.

Critical context for all 2026-08 results: **the 6 proxy-GT pages are all in the good
half** — every E1-E11 measurement stands, but only under the assumption "grid geometry
correct", which holds for ~half the corpus. Set-of-marks solved grid POPULATION;
grid DETECTION was never closed corpus-wide and is the blocking work item before any
rollout. Audit list in this section; pages 11/15/18 of the pushed pilot are affected.

---

---
## E13 — Page-split misdetection + header-anchored frame fit (2026-08-29, session 3) ✅ p17 FIXED

**p17 regression root cause:** not the column fit itself — `detect_table_frame` takes the
*leftmost* long vertical in the rightmost 12% of the `wide` crop as the page split. On p17
the crop is narrow (2775 px) so the window starts at 2442 and catches the NetLP|NetMils rule
(2465) instead of the binding (~2620). Consequences: the final2/final3 JPEG for p17 was
**cropped 152 px short — the Net Assessment column (with written values) was physically
absent from the image**, the right-anchored template could not fit (span search range
excluded the truth), and `track_bands` sheared band 0 from a grid 100 px off.

**Fix (build_final3.py):** `fit_template_frame` — 2-D fit of the 19-column template to the
header-band verticals of the full `wide` crop, right edge × span both free (span 2350–2650,
left edge derived and allowed off-crop: p12's is x=−3, which the old left∈[120,480] range
could never reach — that is why "template" lost to "detected" on p12/16 last session).
Result corpus-wide: **18/18 header lines hit on 96/98 pages** (p1 half-page, p101 outlier —
neither overridden). `resolve_right_edge` then decides the split, calibrated on a 16-page
montage of both print batches:
  * split ≈ fitted boundary 18 → take fit (p17: 2467→2619);
  * fit ≥10 px LEFT of split → take fit — the detector had latched onto the *right page's*
    first rule across the fold (p27 −30, p75 −46, p98 −56, p100 −56; ~45 pages in −10…−58);
  * fit RIGHT of split (p3 +50, p19 +66, p20 +30, p45 +24) → keep detector: the split is on
    the fold and the last Mils column is simply narrower there (bound into the spine).
Also fixed: `best_shift` tie-break (first-max in ascending order → every page came out
"−10px"; now ties go to offset 0), and the final3_build.tsv column order (head_gap/width_dev
were swapped vs the header). TSV now also records x_r, x_r_detected, fit_hits, fit_span.

**Rebuilt + checked visually:** 12, 16, 17, 18, 20, 27, 98 — all pick "template", columns on
the printed rules, first row under the header, recovered top rows (+3 on 18/27/98) aligned.
p18 ink_cross 0.119 on *every* candidate → writer/pencil-over-lines page (p74 class), not a
grid defect. Preview re-uploaded as a new doc (the page image changed, transcript pushes
cannot replace it): **docId 18531980 "Hadita-final3-preview2"**, pageNr 1–5 = Hadita
12/16/17/18/20, geometry only, no seed text. Old preview 18530298 left in place.
Spend this session: $0 (all local).

**E13 addendum (same evening).** User review: columns OK on preview2; reported p12
bottom-right: the Net Mils values ١.٠٩ in the last rows fell outside the grid. Cause:
`track_bands` carries a boundary forward "at the previous rate" when no rule is found;
the right-edge boundary (an image edge, not a rule) caught the fold shadow once and then
slid left band by band to the MAX_BOW_PX cap (2531→2396), closing the last column.
Invisible to ink_cross (interior boundaries only). Fix: `pin_outer_bands` — boundaries 0
and 19 are crop edges and are held fixed in every band. Corrected XMLs pushed as layer
`Hadita-final3-geometry-v2` on preview2 (tsIds 301491523–301491532). Five more column-check
pages (3, 19, 40, 66, 100 — early batch, +66 px split case, hb-broken +4 top rows, two
right-page-latch cases) built, checked, uploaded as **doc 18532357 "Hadita-final3-preview3"**.
Column-detection flow written up for the slides session (hadita-f6).

---
## E14 — Rows: the printed rules, found under the handwriting (2026-08-29, session 5) ✅ 34/34

Resuming the row review that session 4 was cut off mid-analysis (OAuth expiry). Three cases
the user reported: **p12** — the last two rows swallow a row between them; and from the
overlays, the whole-corpus versions of the same defect — first band far too tall, last band
running past the table into the totals margin.

**Root cause: rows were never derived from the form.** Every generation so far phased the row
lattice on *Kraken text lines* (final2, and final3's v1/v2). The handwriting drifts, merges
(p12's bottom: 3 written lines in 2 bands) and the segmentation cache pads empty stretches
with synthetic rows at a wrong pitch (p16/p40 carried runs of 69–78 px bands). Session 4's v2
lattice made this worse in a way its own metric could not see: it treated a Kraken centre as a
*mid-band* and put every edge half a row below the rules (p100: **+40 px on all 33 edges**),
while `text_in_band_frac` scored it 0.97 — the score was tautological, measuring text against
lines derived from that same text.

**The printed rules are visible after all.** They are 1–3% darker than the paper, dashed, and
invisible to any binary line detector — but a *median across 30 vertical strips* of the
per-row local-contrast profile isolates them: a printed rule crosses every strip and survives
the median, a written line touches a few strips and is suppressed. On the densest page (p100)
this finds **33/33 rules at 5.8 px max residual** to a rigid lattice. It also settles the
phase question empirically: the rule peaks coincide with Kraken's line centres (median offset
0–6 px) — **the scribe writes with the baseline on the rule**, so a centre is a rule, not a
mid-band.

**Fix (build_final3.py):** `rule_profile` (strip-median), `lattice_rows` rewritten to fit
pitch+phase to the *rules* by trimmed least squares over the lattice index, then walk from the
header snapping to a detected rule within ROW_SNAP (0.25 — 0.15 missed the 15–22 px paper
shifts on p40/p3/p16) and bridging at pitch where a rule is hidden under ink. `text_ok` →
`text_on_rule_frac`, now an independent check (writing near an edge, not inside a band).
Two more corrections the rules exposed:
  * `table_bottom` — the last band ran past the table because the frame height was the crop's;
    the printed *verticals* stop ~40 px below the last rule, and what is under them is the
    margin where the scribe writes page totals (p12's "1.09" column, p20's two-line total).
    The walk now stops there.
  * `HB_MAX` 420 → 350: p100's detected hb_y=392 was already the *first rule*, costing it a row.

**Result: 34 rows on all 10 preview pages** — the printed form's row count, for the first time
in any generation (final2 and final3-v1/v2 ranged 32–36). Band heights 73–117 px around a
92–94 px pitch, i.e. the lattice tracks the paper's real drift rather than imposing a constant.
Verified visually top and bottom on all 10: first band under the header rule, last band ending
on the last rule, totals margin excluded. Snap rate 0.79–1.0; `text_on_rule` 0.94–1.0 on the
written pages. The two low scores are content, not geometry: **p17** 0.17 (a nearly empty page
whose few Kraken lines are cache artefacts) and **p18** 0.47 / **p40** 0.59 (the p74
writer-anomaly class — the scribe writes across the ruling; ink_cross 0.12 on p18 confirms).

Pushed: preview2 doc 18531980 (12/16/17/18/20) and preview3 doc 18532357 (3/19/40/66) as layer
`Hadita-final3-rows-v3`, tsIds 301493465–301493483. p100's right edge moved 2 px in this build
(column re-score), so its image changed and a transcript push was refused — re-uploaded as
doc **18533016 "Hadita-final3-preview4-p100"**. Spend this session: $0, all local.
