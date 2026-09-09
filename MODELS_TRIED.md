# HTR / OCR models tried on the Hadita tax register

Benchmark unless noted: **pages 3, 4, 5, 6, 9, 10** of `Transkribus upload/final3`,
scored against RA-corrected proxy GT with `score_exp2608.py`.
Metric: **keystrokes (ks)** a human needs to correct the output. Lower = better.

> Two cautions for any slide:
> - Run-to-run variance is **±10%** (measured: 5 identical SoM runs, 1173–1415 ks).
>   Differences under ~10% are not interpretable from a single run.
> - Scores marked † were measured on **final2** geometry or with the constant-offset
>   scorer, and are not strictly comparable to the final3 numbers.

---

## A. Cloud vision-language models

**1. Gemini 3.7 Flash + Set-of-Marks — 5060 ks — BEST OVERALL**
- `gemini-3.7-flash`, thinking LOW, structured output via `response_schema`
- Set-of-Marks: row index stamped in both margins + separator line per row boundary
  (`make_som_pages.py`), model keys output to the *printed* row number and emits
  `{column_index: text}` — it never counts rows or names columns
- Pipeline: `make_som_pages.py` → `run_som_ocr.py` → `repair_empty_rows.py` (ink-gated
  re-read of inked-but-empty rows) → score with NW alignment
- Cost **~$0.012/page** (~$1.20 for 98 pages)
- Numerals 59.5% · words 51% · fills 2.6% of empty cells
- Solved the sparse-row problem: **23 missed rows → 1**

**2. Gemini 3.1 Pro + Set-of-Marks — 7974 ks**
- Same SoM pipeline, `gemini-3.1-pro-preview`
- Numerals 45.7% · words 54% · fills 5.5% of empty cells · **misses 331 GT cells**
- **Bigger is worse — but the failure is incompleteness, not bad reading.** Per-cell
  accuracy is respectable (it beats Flash on words, 54% vs 51%); it truncates,
  missing 331 GT-bearing cells against Flash's 144. Page 3 stopped at 14/33 rows.
- Most ٢-biased model tested: 112 of its 127 target-pair confusions are ٢/٣,
  and ٣→٢ is its top substitution (112)
- Cost ~$0.75/16pp — **6.5× Flash for a worse result**

**3. Gemini 3.7 Flash, medium thinking — 4291 ks† (pages 6+9 subset)**
- **More thinking is worse**, and 1.6× the cost

**4. Gemini 3 agentic pipeline "v5" — baseline production pipeline**
- Multi-page Colab notebook, thinking LOW, ~$0.08/page (~$377/100pp)
- Superseded by SoM at 1/6 the cost

**5. Gemini v6 dual-image — rejected, +11% worse than v5**

**6. Claude, as third reader on cell crops — rejected**
- 36/144 errors were ٣→٢; did not adjudicate reliably enough to gate

---

## B. Local open-weight models

**7. Baseer__Nakba (NAKBA competition winner) — 7158 ks**
- 4B Qwen2.5-VL, CC-BY-NC-SA, 7.5GB, run locally on MPS, **$0**
- ~0.5s/cell on cell crops; row strips are worse (10–15% token recall)
- Numerals 32.2% · **words 7%** · fills 8.5% of empty cells
- Provenance: base saw 500k pairs, **zero handwriting, nothing historical**; its
  synthetic pipeline explicitly discarded tables with >25% empty cells — our exact
  distribution. NAKBA competition handed all entrants **pre-segmented line images**,
  so no score there reflects segmentation difficulty.
- Hallucinated days-of-the-week (السبت/الخميس) into empty cells on final2 —
  **does not reproduce on final3** (ink gate blanks 0 of 1698 filled cells)

**8. QARI-OCR v0.3 — failed, CER 5.46**
- 100% synthetic font-rendered *printed* Arabic (WeasyPrint), no handwritten training
  data at all; authors concede numerals + handwriting as future work

---

## C. Kraken (local, ours)

Common: `ketos train -f path -s 42 --resize new -q early --lag 5 -d cpu`,
base `arabic_best.mlmodel`, 1362 train / 257 val cell crops, 49-char codec.

**9. Kraken fine-tune ft3 — 6534 ks — BEST LOCAL MODEL**
- **`--augment`** (albumentations 2.0.8), best checkpoint **epoch 12**, val acc **0.40**
- Numerals 33.4% · **words 60% — beats Gemini's 51%**
- Top digit confusions ٣/٤ (59), ٢/٣ (37), ٤/٦ (25)

**9b. Kraken fine-tune ft4 (synthetic augmentation) — 9319 ks — REJECTED**
- ft3's data plus 2000 synthetic cells composed from 966 real digit glyphs cut out
  of the GT crops (`harvest_digits.py` + `synth_cells.py`); val left as the real
  held-out pages 9/10, so the comparison is not circular
- Sampling oversampled ٢/٣ to ~41% of characters (vs ~13% real) to attack the
  known confusion. **This backfired: it shifted the output prior rather than
  teaching discrimination.** ٤ collapsed 18.4% → 0.1% of emitted digits, ٧ and ٩
  to ~0, ٣ nearly quadrupled to 30.3%
- Per-character ٢ **47.0% → 6.5%**, ٣ 46.0% → 69.7% — the pretrained ٢-bias simply
  flipped into a ٣-bias
- Early stopping at epoch 6, best epoch 1, val acc 0.139 (ft3: 0.40)
- Lesson: do not class-balance an HTR mix toward the confused pair. Any retry must
  keep the corpus digit distribution and hold synthetic to a minority of the mix

**9c. Kraken fine-tune ft5 (synthetic, natural frequencies) — 7885 ks — REJECTED**
- ft4's compositor but only 340 synthetic cells (**19% of the mix**), sampled at the
  corpus's own digit frequencies and cell-length distribution instead of rebalanced
- Fixed ft4's prior collapse — ٠/١/٤/٥/٨ all land near their real rates — but produced
  a **new skew in the opposite direction**: ٣ fell to 2.5% of emitted digits (GT 11.2%)
  while ٢ rose to 24.0% (GT 9.1%). Per-char ٢ 68.5% / ٣ 18.4%, mirroring ft4
- Cell accuracy 26.0% vs ft3's 33.1%; val acc 0.235, best epoch 2
- **Conclusion: the class balance was not the defect.** Natural-frequency sampling
  still skews ٢/٣ badly, which points at the compositor — synthetic ink is heavier
  than real, spacing is uniform, and single glyphs lose neighbouring-stroke context.
  Any retry must fix the rendering, not the mix

**10. Kraken fine-tune ft2 — 7162 ks**
- Identical but `--no-augment`; best checkpoint **epoch 1**, val acc 0.34
- Augmentation alone is worth **9%**

**11. Kraken fine-tune ft1 — 8666 ks — INVALID, do not cite**
- Same as ft2 but ~700 of 1356 crops were paired with the **wrong row's text**
  (row-alignment bug, fixed by anchoring on Serial_No)

**12. Kraken gen2_sc_clean (ours, earlier generation) — 8896 ks**

**13. Kraken arabic_best (OpenITI, untuned) — 12636 ks**
- 0.00% character accuracy before tuning — the base for #9–11

---

## D. Transkribus server-side models (PyLaia, paid credits)

Run via `POST /pylaia/{col}/{model}/recognition` on doc 18537955.

**14. hadita1 (model 592309) — 9785 ks**
- Hadita-trained; the only one of the four that reads the hand
  (117/137 perfect cells on pp. 3/4)
- Numerals 21.0% · words 47% · **fills 13.7% of empty cells** · misses only 69 GT cells
- Opposite failure mode to Gemini Pro: reads everything, reads it wrong. Lowest
  missed-cell count of any model, at the cost of heavy hallucination into blanks.
- The only model biased toward ٣ rather than ٢: top substitutions ٦→٣ (33), ٢→٣ (27)

**15. agapet5 (model 612549) — 12353 ks — unusable, 0–5 perfect cells/page**

**16. periodicals (model 386877) — 13099 ks — unusable**

**17. garshuni (model 578529) — 13398 ks — unusable**

**18. hadid01 / hadid02 (models 592509 / 592709) — 45.8% perfect, ~970 ks/page†**
- Trained on our GT on final2; loses to Gemini v5 LOW (970 vs 720 ks/page)
- A/B between them was **inconclusive** — Transkribus only allows *doc-level*
  validation sets and the two runs validated on different pages
- PyLaia **hallucinates into empty TextLines**; on sparse pages emits 34 rows
  where GT has 24

---

## E. Approaches rejected before scoring

**19. Column-strip OCR (S-lite / S-full / R)** — Gemini cannot count rows in vertical strips
**20. Grounded OCR (bbox + point-in-polygon)** — placement essentially perfect
(0 dropped of 94–354 detections/page) but recall too low (86 detections vs 120 GT
cells on p9). The geometric join itself is sound and reusable.
**21. Single-row strips** — work, but ~$0.30/page (18× SoM); kept only as targeted repair
**22. Codec masking** — parked; mechanism works but env conflict blocks kraken preprocessing

---

## Summary table

| # | Model | ks | Notes |
|---|-------|----|----|
| 1 | Gemini 3.7 Flash + SoM | **5060** | best; $0.012/page |
| 9 | Kraken ft3 (augmented) | **6534** | best local; best on words |
| 7 | Baseer/NAKBA | 7158 | free, but 7% on words |
| 10 | Kraken ft2 | 7162 | no augmentation |
| 2 | Gemini 3.1 Pro + SoM | 7974 | 6.5× cost, worse |
| 12 | Kraken gen2_sc_clean | 8896 | |
| 14 | Transkribus hadita1 | 9785 | |
| 15 | Transkribus agapet5 | 12353 | unusable |
| 13 | Kraken arabic_best | 12636 | untuned base |
| 16 | Transkribus periodicals | 13099 | unusable |
| 17 | Transkribus garshuni | 13398 | unusable |

**Cross-cutting finding — the ٢/٣ problem, and what fine-tuning did to it.**

Per-*character* accuracy on the two digits (same-length cells, pages 3–6, 9, 10;
`score_digit_23.py`). Denominators differ per row because only same-length GT/pred
cells are alignable, so compare the accuracy columns, not the raw counts.

| model | ٢ accuracy | ٣ accuracy | ٣→٢ errors |
|---|---|---|---|
| Kraken arabic_best (untuned base) | 5.9% | 22.9% | 7 |
| Kraken ft2 (no augmentation) | 17.4% | 48.4% | 5 |
| **Kraken ft3 (augmented)** | **47.0%** | **46.0%** | 25 |
| Gemini 3.7 Flash + SoM | 86.3% | 37.8% | **130 of 238** |
| Baseer/NAKBA | 91.9% | 38.7% | 67 of 163 |

Two results:

1. **Fine-tuning lifts ٢ accuracy 5.9% → 47%, an eightfold gain** (٣ roughly
   doubles, 22.9% → ~47%). Augmentation supplies most of the ٢ gain. It does not
   remove the confusion so much as *rebalance* it: ft2 made 33 ٢→٣ errors and 5
   ٣→٢; ft3 makes 12 and 25. ft3 is the only near-symmetric model.

2. **The pretrained models are not better at this pair — they are biased toward ٢.**
   Gemini reads more than half of all ٣s as ٢ (130 of 238); NAKBA, 67 of 163. This
   inflates their ٢ score, wrecks their ٣ score, and the two roughly cancel in the
   aggregate keystroke totals, which is why it never surfaced in the headline numbers.

So the pair *is* improvable by hand-specific training. What no model does without
such training is discriminate: they default to ٢. Prior sessions recorded ٢/٣ as
the top substitution for every pretrained model tested (Gemini 137/155, NAKBA 66/83,
Claude 36/144 — three architectures, three corpora, same error on the same hand);
this measurement refines that from "equally confused" to "systematically biased".

**No public model or dataset exists at the intersection Arabic × tabular × numerals ×
administrative.** Our GT, if released, would be the first.

---

## D. Tried 2026-09-07/08 — all rejected

Scored with `score_exp2608.py --tolerant` (see below), so none of these is
penalised for encoding conventions the incumbent happens to share with the GT.

**Gemini 3.8 Flash + SoM (2026-09-02 release) — 4559 / 5234 ks — TIE, then worse**
- Identical pipeline to the champion: same `som_f3` images, final3 geometry,
  thinking LOW, prompt unchanged (`--digit-hint off`). Model is the only variable.
- Incumbent 3.7 Flash scores **4476 ks** on the same scoring run.
- Run 1 is a statistical tie (+1.9%, inside the ±10% variance); run 2 is +16.9%.
- Mechanically clean — 0 out-of-range cells on all 6 pages, so SoM row-keying
  holds on 3.8. But **less prompt-compliant on encoding**: one run emitted 674
  ASCII digits (others ~50), ignoring "keep Arabic-Indic digits".
- Better on p3, worse on p5, consistently across both runs — a real behavioural
  difference, not noise, and unexplained.
- **Fourth data point for "newer/bigger is not better on this hand"** after
  3.1 Pro, medium thinking, and v6 dual-image.

**Few-shot / "LLM as annotator" (E15) — 4907 ks (8-shot), 5047 (4-shot) — REJECTED**
- The one new idea in Calfa's `htr-vlm-annotator` Space: rather than *describing*
  the ٢/٣ letterform (E14, failed), *show* confirmed crops of this scribe's hand
  as in-context examples. Never tested before; every prior attempt was zero-shot.
- Crops from **page 50** (typed GT, outside the benchmark). Note every labelled
  crop in `kraken_gt/` comes from the 6 benchmark pages and would have
  contaminated the evaluation.
- On the pair it targets it only **trades one error for the other**:
  8-shot moves ٣ 29.1%→32.4% while dropping ٢ 81.0%→79.1%.
- Cost 2.6× ($0.006 → $0.0145/page). `run_som_ocr.py --fewshot N`, default off.
- Caveat: the p50 pool is ٢-heavy (25 vs 12), so this is "few-shot from p50
  fails", not "few-shot is impossible".

**Iterative ICL / Calfa's actual loop (E16) — 3441 ks (n=2), 3460 (n=4) vs 4380
zero-shot — BEST RESULT IN THE LOG**
- E15 was not a fair test of the method. Calfa's `ICLPool.sample()` sorts by
  `added_at` DESC and takes the top n, so exemplars are the n most-recently
  **validated pages** and the set changes for every target. E15 used one FIXED
  block of p50 cell crops for all six pages. Two differences: recency/curriculum,
  and whole pages instead of crops.
- `run_som_iter.py`: page k gets pages k-1..k-n as image + confirmed-transcription
  pairs, rendered in the same JSON shape the model must emit. Uses each prior
  page's GT as its "correction" = the OPTIMISTIC BOUND (a perfect RA).
- Scored on the 5 exemplar-bearing pages (page 3 is zero-shot in every arm and
  was excluded), 3-4 runs per arm:

  | arm | mean ks | sd | vs base |
  |-----|---------|----|---------|
  | zero-shot baseline (4 runs) | 4380 | 385 | -- |
  | **iter n=2 (4 runs)** | **3441** | 147 | **-21.4%** |
  | iter n=4 (4 runs) | 3460 | 212 | -21.0% |
  | iter n=6 (3 runs) | 3675 | 227 | -16.1% |

- **٣ accuracy 28.7% -> 67.2% (n=4) while ٢ ALSO rises 74.4% -> 84.3%.** Both
  digits improving together is the signature of real discrimination; ft4, ft5 and
  E15 all traded one for the other, which is a shifted prior. Worst iter run beats
  best baseline run on ٣ by 22.4 points — **no overlap across 15 runs**.
- Gain peaks at n=4 and decays by n=6: distant pages dilute the neighbour signal.
- Also cuts sparse-row overreporting — pp9/10 report 24 rows (GT 24/27) vs the
  baseline's 34, and page 9 `miss` drops to 0.
- **Beware the lucky-run trap.** The original `som-f3v2` (3766 ks) turned out to be
  the BEST of four zero-shot runs; three fresh repeats gave 4351/4638/4766. Judging
  the first iter run against it showed a +3% edge instead of the true +21%. Never
  benchmark against a single run of the incumbent.
- Cost ~2.3x zero-shot ($0.014 vs $0.006/page), trading machine cost for RA time.
- **This is a prompt pattern, not a model.** There is no artefact to hand to page
  20; the gain lives in the input and needs corrected pages near the target.

**Fixed 6-page GT pool on an unseen page (E17) — UNSCORABLE, signature positive**
- The production-shaped question: can the 6 verified pages carry a page with no
  corrected neighbours? `run_som_iter.py --pool 3 4 5 6 9 10 --pages 11`
  (a page never exemplifies itself, so the flag stays usable leave-one-out).
- Page 11 has no GT, so this **cannot be scored**. Indirect signal only, 3 runs:
  ٢:٣ ratio moves 8:1 -> ~3:1 (41/5 zero-shot vs ~34/11), the same shift that on
  the GT pages meant ٣ accuracy doubling; and ~25% more cells filled (84 -> 95-111).
- Consistent with E16 but **not proof** — an unscorable page cannot rule out that
  the extra ٣s are in the wrong cells. Page 11 is the natural first RA review:
  confirming it both validates this and seeds the pool for page 12.

**Column set-of-marks (E18), few-shot+marks (E19), layout-stripped exemplars (E20)
— ALL REJECTED. Reading and placement measured separately, 45 runs, pages 4-10:**

| arm | reading acc | phantom cells | missed | mean ks |
|-----|-------------|---------------|--------|---------|
| **iter4 few-shot (layout)** | **72.5%** | 94 (6.0%) | 144 | **3460** |
| iter2 few-shot | 70.7% | 101 (6.4%) | 133 | 3441 |
| few-shot + density-match | 69.8% | 94 (6.1%) | 169 | 3762 |
| E20 layout-stripped | 61.7% | 101 (6.6%) | 222 | 4920 |
| zero-shot | 60.1% | 94 (6.1%) | 204 | 4380 |
| E18 column marks only | 49.0% | **69 (4.5%)** | 241 | 5288 |

- "Reading acc" compares only cells BOTH sides filled, so placement is factored out.
- **E18** (`make_som_grid.py`, blue column numbers + vertical rules above each column,
  `run_som_ocr.py --grid`) does suppress phantoms — the best rate measured, 4.5% vs
  6.1% — but costs **11 points of reading accuracy** and misses the most real cells.
  The marks compete with the transcription for attention. A misread and a phantom
  both cost RA time and there are far more misreads, so the trade is not worth it.
- **E19** (few-shot + column marks together) is WORSE than either parent and
  unstable: Net_Assessment_LP on p11 came out 10/6/12 across identical runs. The
  printed column number and the exemplars' column keys give conflicting layout
  signals and the model resolves them differently each time.
- **E20** (exemplars rendered as bare values, no column keys, to teach the hand
  without the layout) won both user-confirmed p11 facts and 3 of 4 p12 spot checks,
  but does not survive scaling: 61.7% reading, worst missed count, and ks swinging
  4012/4708/6040. Stripping the column keys also removes the output-format model.
- **There is no general column problem.** Few-shot and zero-shot hallucinate at the
  SAME rate (94 cells, 6.0% vs 6.1%). The p11 drift was layout inheritance from
  dense exemplars shown to a sparse page — `--match-density` addresses that case.
- **Lesson: spot checks on 1-2 pages mislead.** E20 swept the hand-checked cells and
  came 5th of 6 when scored. Confirmations are for resolving specific ambiguities,
  not for ranking configs.

**Human-confirmed readings (p11, p12) — Sinai, 2026-09-09**
- p11 r0 Date = ٩٣٨ (few-shot right, zero-shot's ٩٢٨ wrong: the ٢/٣ bias)
- p11 Net_Assessment_LP is EMPTY; the ink there is handwritten row-marks to ignore.
  **This invalidates any "ink occupancy" metric** — it counts marginalia as content.
- p11/p12 Parcel_Area really does carry a bare separator with NO leading digit
  (`،٢٦٠`). Zero-shot "corrects" this to `٠,٢٦٠` and is wrong; few-shot preserves it.
- p12 areas are 4-digit (`٣,٤٠٨`), not 5-digit (`٣٤,٨٠٠` as zero-shot reads).
- p12 r7 Parcel_Area = ٦,٧٧٥ — **all 8 runs of all 4 arms failed** it (best 2/4
  digits). A ٧ problem, not a ٢/٣ one.
- p12 col 9 rows 4-6 is a DELETED number overwritten with a new one, not Arabic
  script. Excluded from scoring by request; flag such cells for human handling.

**PRODUCTION (2026-09-09): pages 12-20 transcribed with `--match-density 4`**
- `run_som_iter.py --pages 12..20 --pool 3 4 5 6 9 10 --match-density 4`
- Exemplars auto-selected per target: dense pages (12, 16, 20) drew 3,4,5,6; sparse
  pages (13-15, 17-19) drew 5,6,9,10. $0.1119 for 9 pages (~1.2c/page).
- Pushed to Transkribus doc 18537955 as `Hadita-ICL-density4-g37flash`. UNSCORED —
  no GT exists for these pages; they need RA review.

**Calfa (`huggingface.co/calfa-ai`) — nothing runnable**
- Published *models* are Armenian only (`hye-paddle`, `hye-tesseract`).
- Six Arabic HTR **datasets** (Baybars 15.5k lines, Iskandar 5.2k, RASAM-1/2,
  Tarima) — all literary prose in running lines, Oriental/Maghrebi manuscript
  hands. Baybars explicitly excludes tabular material.
- Only plausible use is as a **base model** for two-stage training (Baybars →
  our cells), not as a drop-in reader. Untested.

**Qwen (3.8-Flash-Next / 3.8-Max, released ~2026-09-05) — NOT RUN**
- `Qwen3.8-Flash-Next` is 125B total / ~180GB on disk; the Mac has 18GB.
- `Qwen3.8-Max` is API-only. Both need a Qwen Cloud/DashScope key we do not have.

**Mistral OCR 4.1 — BUILT, NOT RUN**
- Runner + calibration scorer committed; needs `MISTRAL_API_KEY`.
- Judged on **confidence calibration**, not keystrokes — it is the only reader
  in the stack that reports uncertainty, which is what the ٢-cell routing
  recommendation needs. Bar: Spearman ρ > 0.4 (set 2026-06-26).
- Pages 3, 4, 9 = $3.24, under the original $5 cap.

---

## Scoring note (2026-09-07)

The frozen metric folds the codepoint variants **Gemini 3.7 happens to emit**.
Newer models spell the same glyph differently — Extended Arabic-Indic digits
(U+06Fx) in 10 tags, `〃` in 3, em-dash mid-string in 11 — and were charged for
cosmetics, which silently favours the incumbent by 150–350 ks.

`score_exp2608.py --tolerant` composes an encoding fold on top of the frozen
metric (never edits it; the default path is byte-identical). **Any future model
comparison should use it**, or the incumbent wins on spelling rather than reading.

It deliberately does *not* fold ASCII digits or Latin letters: the GT genuinely
contains `T.D.L. 1940` and ASCII serials like `102`.
