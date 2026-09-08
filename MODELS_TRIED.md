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
