# Verification assignment — final3

Doc **18537955 "Hadita-final3"**, collection 2377415. 98 pages, geometry + three
recognition layers (Gemini top, then NAKBA, then Kraken), plus a GT layer on
pages 3/4/5/6/9/10.

Everything below is ranked by *how uncertain the machine is*, not by how bad it
looks. Items 1–7 are columns, 8–13 rows, 14–20 the ink test. Where a number is
quoted it comes from `final3_build.tsv` or `audit_final3.tsv`.

---

## A. Columns — pages where column identity is least certain

A wrong column boundary is the worst failure mode we have: text is placed into
cells by column *index*, so one dropped or invented rule mislabels every cell to
its right. These are the pages where the template fit is weakest.

**1. p101 — the least reliable column fit in the corpus.**
`align=0.494` (corpus median ~0.94), only `11/18` header rules found, and the
fit needed a −10px nudge. **Check: are there 19 columns, and does each header
label sit over the right column?**

**2. p1 — half page, and the fit is anchored on almost nothing.**
`fit=2/18` rules, `align=0.619`, `width_dev=0.939` (i.e. the column widths are
nowhere near the printed form's proportions), source `template-89px`. The page
is physically short (14 rows). **Check: is the column grid meaningful at all
here, or should p1 be handled as a special case?**

**3. p17 — the largest right-edge override in the corpus.**
The page-split detector said the table ended at x=2467; the template fit moved it
to 2619 (**+152px**). That is roughly one whole column's width. `align=0.9`.
**Check: is the rightmost column (Net_Assessment_Mils) real and correctly bounded,
or has an extra column been invented?**

**4. p18 — boundaries slice handwriting more than any other page.**
`ink_cross=0.121` — 12% of boundary path pixels cross ink, against a corpus
median near 0.01. Known "writer anomaly" page: the scribe writes across the
ruling. **Check: is this the scribe's fault (unfixable) or the grid's?**

**5. p29, p27, p39, p31 — the next worst ink_cross** (0.084, 0.069, 0.055, 0.053).
p29 and p39 also needed template nudges (+4px, −10px). **Check the same question
as p18: scribe or grid?**

**6. p22 — the only page whose columns came from `cache-scaled`,** not from the
template fit. It is the one page using a different mechanism entirely.
**Check: do its columns line up as well as its neighbours'?**

**7. p95 and p94 — incomplete header fits** (`17/18` and `15/18` rules).
**Check: which rule is missing, and did its absence shift anything?**

---

## B. Rows — where the lattice was least anchored by the writing

`row_snap` = the fraction of row centres that snapped to an actual Kraken text
line. Low snap means the lattice *walked on prediction* — it placed rows by
arithmetic because there was little or no ink to anchor them. These pages can
look perfectly regular and still be wrong, so they need eyes.

**8. p86 — snap 0.059.** Only ~2 of 34 rows anchored on real text. `text_ok=0.233`.
**Check: do the row bands correspond to the printed rules, top to bottom?**

**9. p38 — snap 0.088, text_ok 0.217, only 23 Kraken lines** for 34 rows.

**10. p17 — snap 0.118, text_ok 0.200.** Also item 3 above: this page is
uncertain in *both* dimensions, which makes it the single highest-value page to
inspect. **Check rows and columns together.**

**11. p96, p76, p90 — snap 0.147/0.176/0.235, text_ok all ≤0.29.** Sparse pages
where the grid is mostly interpolated.

**12. p23 and p45 — the only two pages with 36 rows** (corpus is 34 or 35 on 92
of 98 pages). **Check: is there genuinely an extra row, or has one band been
split in two?**

**13. p59 — 33 rows, snap 0.182, and flagged for ink below the last band.**
**Check the bottom of the page specifically: is a final row missing?**

---

## C. Ink test — cells where a TextLine may be wrongly absent or present

Measured on the six GT pages: **234 of 1,631 GT-bearing cells (14.3%) have no
TextLine**. Two different causes, and they need different answers.

**14. Red and reddish-brown pencil — the real bug (~91 cells, 39% of misses).**
`hand_mask()` converts to greyscale and thresholds at `median − 45`. Coloured
pencil on cream paper often does not darken enough to survive that, so genuine
writing measures 0 ink. **Check a sample of these and confirm they are real text:**
p3 r4c7 `١٠٢`, p3 r4c9 `١٩٤٠`, p4 r7c9 `٠٠٠`, p5 r3c9 `٠٠٠`.
(User-reported example: New_Serial_No on p3.)

**15. But a naive redness fix is WRONG — I tested it.** Adding
`redness = R − (B+G)/2 > 25` as an ink term would give TextLines to **~100 extra
cells per page** (p3 +102, p9 +119, p20 +82, p50 +50) when only ~15/page are
real. The paper itself is warm-toned, so most of that is blank page. **Open
question: what discriminates red *pencil* from cream *paper*?** Saturation plus
local contrast is the obvious candidate, but it must be calibrated and measured
before shipping — the same way the 12px grey threshold was.

**16. The 12px threshold is calibrated against the OLD mask.** If the mask
changes, that number and the "97 zero-ink GT cells" note in `gate_textlines.py`
are both stale. **Re-derive, don't inherit.**

**17. Zero-ink conventions — the other ~120 cells, genuinely blank paper.**
Most frequent missing values: `✓` (27), `"` (26), `-` (22), `–` (21), `٠` (12).
No ink threshold can recover these. **Question: should these be emitted from a
column-aware rule (a ✓ belongs in Volume_No on rows with a T.D.L. reference)
rather than from vision?**

**18. Which columns lose most.** Tax_LP (39), Reference_to_Register_of_Changes_
Serial_No (26), Net_Assessment_LP (26), New_Serial_No (16), Block_No (16).
**Check whether the concentration in the money columns is a narrow-column crop
problem rather than an ink problem.**

**19. Sparse pages lose more.** p9 and p10 miss ~25% of GT cells against ~4% on
p3/p4. **Check whether that is the ink test or the sparse-row geometry.**

**20. The header strip is Kraken output and it is nonsense** — p3 reads
`| مد حسىن |` for the taxpayer name, `| | |  | |` for the index (printed rules
read as characters). Already queued to be re-read by Gemini. **Check whether the
name is recoverable at all, or whether the strip crop itself is wrong.**

---

## Corrections to earlier claims in this session

- I first reported **18.8%** of GT cells gated away. That was inflated by my own
  row-alignment bug on p9/p10 (I reported 88%/74% for those pages). The correct
  figure is **14.3%**, and p9/p10 are ~25%.
- I then said "there is no ink to find — lowering the threshold is worthless."
  **Wrong for 39% of cases**: I measured ink *through the same grey mask that is
  blind to red*, so the measurement confirmed its own blind spot. The user's
  counter-example (New_Serial_No, p3) is what exposed it.
