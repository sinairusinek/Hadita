# Winding Workflows: Editors, Artificial Agents and Specialist ATR Models

Sinai Rusinek (Haifa University, Open University DH-Dev)

OCR/HTR Workshop for Under-resourced and Under-represented Languages Redux
Austrian Academy of Sciences, Vienna — 10 September 2026
Session 2: Workflows & Systems, 10:50–12:10 (~20 min + Q&A)

Status: draft slide list. Framing not yet fixed. Yiddish/DraCor case not currently included.

---

## 1. Title

Two registers, Mandate Palestine, 1930s–40s. One solved, one not.

---

## 2. Haifa Government Hospital, 1930–1948

33 notebooks · 2,548 page images · **29,879 admissions**.
Handwritten English in a ruled table, ~17 columns, 11 admissions per page.
Closed 2 May 1948.

---

## 3. Two phases

| | Transkribus | Gemini |
|---|---|---|
| Notebooks | 10 | **33** |
| Records | ~6,000 | **29,879** |
| Years | 1930–1935 | **1930–1948** |
| Human training pages | 95 (from 3 notebooks) | **0** |
| Metric | CER 4.30% → 3.51% | — |
| Output | text + PAGE-XML, needs parsing | structured TSV, one pass |

**Caveat to state aloud:** no head-to-head accuracy was ever computed. The
comparison is scope and architecture, not CER.

Not a pure upgrade: the slow phase built the knowledge the fast phase runs on.

---

## 4. Al-Haditha

The village near Tel Hadid, close to Lydda, Ramla and Jaffa. Ottoman origins,
destroyed 1948.

The Hadid Expedition (Alon & Koch, ISF 1316/22) reconstructs it from
archaeology, oral history, and Ottoman and Mandate records.

The property-tax register is one of the few surviving documents of who held
which parcel.

---

## 5. The register

102 pages · 19 columns · 3,069 rows · **58,122 cells**.
Handwritten Arabic under printed English headers.
Entries reference Tax Distribution Lists, 1938–1945.
Shelfmark TAX 1-85.

---

## 6. Why this one is hard

**Image: the two registers side by side.**
Left — Haifa, `nb01_p003_2000_0.jpg`: every row filled, cells hold words,
printed rules crisp in red and blue, page flat, serials 99–109 unbroken.
Right — Hadita, `Transkribus upload/final/Hadita_9.jpeg`: five filled rows then
a scatter, ruling almost invisible below, page bowing into the gutter.

*Physical*
- **Photographed bound pages**: tilt, and bow near the spine — column
  boundaries deviate **−49px to +57px** down a single page.
- **The rules are faint** — printed lines under the handwriting, and in the
  lower two-thirds of a sparse page there is nothing else to mark a row.
- **The left border line is missing.** Near the spine the paper edge and
  binding shadow read as printed rules, so detection either invents a boundary
  or drags the table's left edge into the gutter. The narrow money columns on
  the right are the next worst.
- **The clerk drew lines too** — diagonals and rules struck across the page.
  A morphological filter cannot tell those from printed ruling.

*Content*
- **32% of cells carry ink** (18,718 of 58,122).
- Cells hold **1–4 glyphs**, mostly Eastern Arabic numerals. Almost no prose,
  so no lexicon and no redundancy: every stroke is load-bearing.
- **Two digit systems coexist**, both authentic — ٠-٩ in most columns, 0-9 in
  New_Serial_No and in T.D.L references. Not normalised: the distinction is in
  the source.

*Semantics*
- **Ditto marks and nil dashes carry meaning.** `"` = same as above, `-` = nil,
  `✓` = seen. **97 reference cells hold text with essentially zero ink.**
- **A row is not always an entry.** On page 9 the serials stop after row 5
  while dated entries continue below: rows belonging to the parcel above,
  tax-year breakdowns, sub-totals, carry-forwards.

---

## 7. The cast

Specialist ATR models · VLMs · rule-based code · editors.

The title was too short to name them all — and "agents" was already doing
double duty for two of them.

*(May fold into slide 8, where the players first appear.)*

---

## 8. Dewarping destroyed the pages

The pipeline straightened the *image*. The canvas ended at
`last_row_center + ½ pitch`, so content below the last **detected** row was
**cut, not smeared** — and a missed row was silently discarded.

**98 of 98 pages damaged.** 25 lost a whole written row; 92 had the bottom-cut
geometry.

Fix: **warp the coordinates, not the image.** Upload the undamaged page; push
the cell polygons through the inverse remap. PAGE XML polygons can bow — the
format could always express this.

**Recovered 35 rows. Structural validation 37/98 → 95/98.**

The lossy step was *convenience*: flattening the image made every later stage
simpler, and paid for it by destroying data invisibly.

**No models on this slide — but two players not yet named: the community, and
the chat that relayed what the community knew and experimented alongside me
toward the pipeline.**

---

## 9. Columns — stop detecting lines, fit one known form

Detecting the 19 rules one by one cannot work: they are faint, handwriting
crosses them, the binding shadow makes false verticals, the page bows. Any
detector will sometimes miss a rule or accept a stroke.

**And one miss is the whole problem.** Text is placed into cells by column
*index*, so a slightly misdrawn line is cosmetic — but a dropped or invented
one mislabels every cell to its right.

The fix: the printed form is constant, so the 19 relative widths are a
**template**. The header band is print-only — no handwriting — and carries all
18 interior rules as short clean strokes. Fit the template to those, with two
free parameters: right edge and scale. **The left edge is derived, never
detected** — whatever lies left of the Serial/Date rule is the Serial column.

*"Stop detecting 19 lines; fit ONE known form."*

Residual class: pages where the scribe wrote across the printed columns. No
line detector can fix that — it needs content-level flagging and RA judgment.

*(Provisional: preview doc awaiting review; corpus rebuild not yet run.)*

---

## 10. Rows — where the same move failed

Columns worked by ignoring the writing and fitting the printed form. The
obvious next step was to do it again for rows. It did not survive contact with
the scribe.

**The classic detectors find nothing.** The horizontal rules are printed 1–3%
darker than the paper, and dashed. Adaptive threshold + morphological opening
with a long horizontal kernel — the standard table-extraction recipe, and the
same one that works on the verticals — returns nothing usable: the faint dashes
do not binarise, and what does survive is handwriting.

**A median across strips does find them.** Cut the page into 30 vertical
strips, take each strip's darkness profile, and take the **median across
strips**. A printed rule crosses every strip and survives the median; a written
line touches only a few and is suppressed. On the densest page: **33 of 33
rules**, within 6px of a perfectly regular ladder. The signal was always there —
it needed a statistic robust to ink, not a stronger filter.

**And then the scribe.** Building the grid on those rules gave 34 rows on every
preview page — the printed form's own row count, right for the first time. On
visual review, every boundary cut straight through a line of handwriting.

**The scribe writes _on_ the rule, not between the rules.** Measured: the
written lines sit within 1–7px of the printed rules, on a form whose rules are
93px apart — for practical purposes, exactly on them. So a rule is not the edge
of a row. It is the middle of one. Reading the ruling as a set of boundaries put
every boundary half a row out of place: geometrically flawless, and useless for
reading.

So the grid is built on **Kraken baselines** — a generic segmenter, no Arabic,
no notion of a table, which returns one fragment per written cell. It finds
where the writing is; the boundaries go midway between. The printed rules stay
in, demoted to one job: they measure how far apart the rows are, which they do
better than the handwriting can, because a printed form is regular and a scribe
is not. Regular spacing then repairs what Kraken alone gets wrong — writing that
drifts and merges (three lines in two bands), and empty stretches the cache had
padded at the wrong size. Where a gap is wide enough for two rows, a row is
inserted: blank rows are found by arithmetic, not by vision.

**Two sources, two jobs.** The printed form knows the spacing. Only the ink
knows where the rows actually fall.

**And the measurement was the real failure.** The bad grid was scored by asking
what fraction of the writing fell inside its rows — writing measured against
lines derived from that same writing. It read 0.97 while sitting half a row off.
*Never score geometry with a metric drawn from the source it came from.*

## 11. Ink-gating

PyLaia transcribes every TextLine you give it. It has no "empty" prediction.
Our XML placed a TextLine in all 58,122 cells — so on page 11 it returned text
in **416 of 532 blank cells**.

Fix, and it is not a model: measure ink per cell, emit a TextLine only where
there is some. The distribution is bimodal — median 222px where there is text,
0 where there is not. Cutoff at 12px.

**39,404 of 58,122 lines dropped. Inventions: zero.**

Cost: 16.5% of remaining disagreements are now cells the model left empty that
the reference fills — the near-invisible ones.

---

## 12. Who can read the cells?

| | perfect cells | RA keystrokes/page |
|---|---|---|
| hadid02 — PyLaia fine-tuned on this hand, 6 GT pages | 45.8% | ~1,084 |
| Gemini 3 | 56–70% | 720 |
| Baseer/NAKBA — competition winner, CER 0.079 on Palestinian manuscripts | — | **1.8× more** |

Caveats: hadid02 overfits because there are 6 pages of GT; Baseer is a single
run on isolated cell crops.

---

## 13. Why the specialists lose — genre, not script

**Baseer**: base model trained on 500k pairs with **zero handwriting**, and
tables with >25% empty cells **explicitly discarded** — our exact distribution.
Fine-tuned on memoir *prose*.

**Every Arabic Kraken model** is trained on prose *lines*. CTC needs horizontal
context that a 2-glyph cell denies it, and digits are a rounding error in every
training set — so ٢/٣ and ٤/٦ are exactly the classes no model had pressure to
separate.

Baseer's hallucinations in blank cells: **السبت · الخميس · الاربعاء**
Days of the week, in a tax ledger.

---

## 14. The counting problem

PAGE XML addresses a cell by (row, column). The model returns a flat list, with
no guarantee its row N is the grid's row N.

On a sparse page it cannot tell a blank ruled row from a faint written one, so
it drops rows — and everything below shifts. **Page 9: 21 of 24 rows.**

Everything that gave the model *less* context failed: banded crops, column
strips (Block_No 71.4% → 0.0%), cell crops. Even feeding it perfectly correct
geometry did not help.

---

## 15. Two fixes, different in kind

**The measurement was wrong.** The scorer assumed a constant row offset, so one
dropped row charged every row below it as a misread. Needleman–Wunsch alignment
recovered **20–28% — with zero API calls.**

**Then: delete the task.** Print the row index into the margin, from geometry we
already had, and ask the model to key each row to the printed number.

**Missed rows 23 → 1–4.** ~$0.011/page.
Not a better model. No counting task.

*(Measured noise: ±1 row/page, so ±2–3 over six pages. The improvement is
~6–8 sd. "0 missed on 5 of 6 pages" is the best run, not the expected value.)*

---

## 16. ٣→٢

Top substitution for **Gemini** (135), **Baseer** (67), and **Claude** alike.

Three model families, three architectures, one scribe's hand.

Not model-fixable. This is where I would like help.

---

## Open questions for this draft

- Framing / through-line not yet fixed.
- Yiddish vocalization (YiDraCor) case: not currently included — no room
  without cutting something load-bearing.
- Slide 7 (the cast) may work better folded into slide 8.
- Pending: row-strip variant of the Baseer test, to separate model deficit
  from crop deficit.
- Pending: rerun of gen2_sc_clean on final2 for a fair-footing number.
