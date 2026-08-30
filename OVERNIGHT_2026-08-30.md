# Overnight run, 2026-08-30 — report

Autonomous session. Nothing here needs action before you read it; anything that
looked off is flagged under **Findings**.

## What was done

1. **Tracked and committed the pipeline.** `build_final3.py` had never been in
   git, which is why an earlier regression could not be compared against its
   predecessor. Now committed with 15 experiment scripts.
   - `a047e49` track pipeline + the two reviewed row fixes
   - `4c4d8e8` audit tool + `--dir` on the ink gate
   - `42d885e` bottom-row fixes (below)

2. **Rebuilt all 98 pages** into `Transkribus upload/final3/` with overlays.

3. **Audited the build** (`audit_final3.py`, writes `audit_final3.tsv`).
   The audit deliberately reports no pass/fail score — a row count cannot tell a
   correct grid from one half a row out of phase. It flags pages to LOOK at.
   Headline: **19 columns on 98/98 pages**, **edge_ink median 0.000** (band
   edges essentially never cross handwriting — the phase check).

4. **Found and fixed two more bottom-edge bugs** (see Findings), rebuilt again.

5. Ink gate, upload, model runs — see the log at the end.

## Findings

### FIXED: whole rows dropped at the page bottom
The audit's `ink_below` check (handwriting below the last row band) caught two
distinct bugs that row counts had hidden:

- **p50 lost a complete written row** — serial ١٩, block ٤١٤١, area ١,٧٩٢ — with
  a full 0.98 pitch of table left unused below the last band. `lattice_rows`
  required a row's *entire* band to fit above the cutoff, but the last ruled row
  on a page normally has its lower half clipped by the page edge. Now a row
  counts while its *centre* is inside the table. p50 34→35 rows, p17 33→34.
- **p87**: `table_bottom` extended past the verticals' end only once. The
  verticals can fade more than a row before the ruling stops, so the walk now
  repeats.

### OPEN: p87 still has ink below its last band
~40px of handwriting sits below the last band and no printed rule is detected
below it to extend to. This is a rule-detection limit, not a cutoff bug. It is
the writer-anomaly class (p18/p40/p74). Not fixed — needs content-level judgment.

### OPEN: p1 is a half page (13 rows)
Expected — the image is 1702px vs ~3900 for a normal page. Flagged for
completeness, no action.

### NOTE: `ground_truth.tsv` covers only pages 3, 10, 50
The real GT for scoring is the RA-corrected Transkribus transcripts
`g3_results/Hadita_{3,4,5,6,9,10}_Transkribus_latest.xml` (proxy GT), which is
what `score_hadid02_final2.py` uses. Scoring below uses those six pages.

### NOTE: Hadid was NOT run, per instruction
The queued "Hadid02 re-run on final3" from the earlier plan was also skipped.

### OPEN (not fixable in software): 10 pages photographed with the bottom row cut off

After the bottom-edge fixes, 10 of 98 pages still carry handwriting below the
last row band. Inspecting them, they fall into two classes:

**(a) The photograph ends mid-row — the source image is cut.** p67 and p87 are
the clear cases: serials ٢١/٢٢ on p67 are physically sliced by the image edge,
with the bottom half of the glyphs simply absent from the capture. The printed
ruling continues past the frame. Nothing downstream can recover this; it needs
a re-photograph of those pages. p67 ink=5810px, p87 ink=4601px below the band.

**(b) Margin page-totals below the table — correctly excluded.** p3 is the model
case: the printed verticals stop, and the ink below is a hand-written total
(٢٤٤٩) in the margin, which is not a register row. No action.

Full list (gap = px from last band to image bottom):
  p3 147/2718(b)  p4 154/3160  p5 147/2055  p9 131/1673  p19 147/2402
  p30 127/1900  p31 80/1958  p58 103/2152  p67 76/5810(a)  p87 108/4601(a)

**Four of these are GT pages (3, 4, 5, 9)** — worth knowing when reading the
scores below, though (b)-class margin totals are not register content and were
never scored.

Only p67/p87 were inspected individually; the other eight are classified by the
same signature (gap < 155px) and should be spot-checked before any claim that
the corpus is complete.

## Pipeline log

- **Rebuild #1** (98 pages): audit flagged 5 pages -> found the p50 dropped-row bug.
- **Rebuild #2** (98 pages, after fixes): audit flags 3 (p1 half page, p67/p87 cut
  photographs). 19 cols on 98/98; edge_ink median 0.000.
- **Committed** `e9f27ed`: all 98 XMLs + `final3_build.tsv` + `audit_final3.tsv`,
  so the next regression can be diffed rather than reconstructed.
- **Ink gate** (`gate_textlines.py --dir "Transkribus upload/final3"`, threshold
  12px): 63,555 cells -> kept 19,950 TextLines, dropped 43,605 (68.6%). Matches
  the corpus's ~32% ink-bearing figure. TableCells are untouched, so the table
  structure and the RA's ability to type into any cell survive.
- **Uploaded** to Transkribus: all 98 pages, jobId **30735580**, doc "Hadita-final3"
  (collection 2377415). Geometry + ink-gated TextLines, no recognition.

## Model runs

Pages: the 6 GT pages (3,4,5,6,9,10) + 10 spread across the corpus
(11,20,28,35,43,52,59,66,75,82), chosen for ordinary geometry (19 cols, 33-35
rows) and avoiding the known-odd pages.

Per your instruction **Hadid was not run**, and the queued "Hadid02 on final3"
re-run was skipped too.

### Gemini 3.7 Flash + set-of-marks (tag `som-f3v2`)
All 16 pages, **$0.0997 total** (~$0.006/page), **0 out-of-range cells on every
page** — the SoM row-keying holds up on the new geometry.

### FINDING: final3 geometry scores WORSE than final2 on the GT pages

    som-g37-flash (final2)  4087 ks   } two runs on the same
    som-g37-flash-rep       4148 ks   } geometry: variance ~61
    som-f3v2      (final3)  4615 ks   <- +528, far outside that

Ink-aligned scoring narrows it (3629 final2 vs 3986 final3) but does not close
it. Per page, the loss concentrates on p3 (+261) and p9 (+177).

**This is at least partly a measurement artifact, not necessarily worse
geometry.** final3 changed the grid ROW COUNT on 4 of the 6 GT pages:

    page:      3    4    5    6    9   10
    final2:   34   35   35   35   35   27
    final3:   34   34   34   35   34   34

The proxy GT was transcribed against **final2 row indices**. Scoring final3
output against it compares two different row numberings, so some of the penalty
is misalignment rather than misreading. p5 is the tell: it gained 48 *perfect*
cells in final3 yet scored 64 ks worse.

**Not resolved.** Deciding whether final3 is actually better needs the GT
re-indexed onto final3 rows (or scoring restricted to pages whose row count did
not change: p3 and p6, where final3 is +261 and -34 — still inconclusive).
I did not want to silently re-index GT overnight; that is a judgment call.

### Kraken gen2_sc_clean on final3 cells (tags `kraken-f3`, `kraken-f3c`)
All 16 pages, local, ~13s/page. New runner `run_kraken_cells.py`
(kraken_experiment.py could not be reused — hardwired to page 3, does its own
segmentation).

    som-g37-flash (final2 Gemini)   4087 ks
    som-f3v2      (final3 Gemini)   4615 ks
    kraken-f3c    (border-cleaned)  8896 ks
    kraken-f3     (raw)            11094 ks

**Kraken loses decisively** — ~2.2x Gemini's keystroke cost even after cleaning,
with only 79 perfect cells across the 6 GT pages (Gemini: 922).

One artifact worth knowing: **89% of raw Kraken cells contained pipe characters**
(`| ١٧٨|`, `||`) read off the printed cell rules. Stripping `|<>_-` and
collapsing whitespace is worth 2198 ks (20%), and `kraken-f3c` is the fair
number. The cleaner is a scoring-time fix, not baked into the runner.

Kraken also fills nearly every row (34/34 on most pages) — like PyLaia it has no
"empty" prediction, so the ink gate matters for it just as much.

### The final3 regression is NOT row re-indexing (revised)

I first assumed the +528 ks was an artifact of final3 changing row counts on 4
of 6 GT pages. Digging into **p3 — where the row count is IDENTICAL (34) in both
geometries and the loss is the largest single page (+261)** — that explanation
does not hold.

The actual mechanism is a **column-assignment shift in one column**:

    Reference_to_Register_of_Changes_Volume_No, over the 6 GT pages:
      som-g37-flash     (final2)  108 filled    8 rows with ✓ merged into Serial_No
      som-g37-flash-rep (final2)  108 filled    8   <- identical, so not run noise
      som-f3v2          (final3)   81 filled   26   <- the ✓ moved one column

On p3, Volume_No goes 30/34 filled -> **0/34**: the ✓ that belongs in Volume_No
lands in the next column instead, merged as `✓ ٩٠`, `✓ ٢١٦`, `✓ - -`.

**And the column geometry is nearly identical between the two** — boundaries for
cols 8/9/10 agree within 1-2px. So this is not a mis-drawn column. Both final2
runs agree exactly with each other, so it is not sampling noise either.

Whatever changed the model's behaviour is in the SoM image (mean abs diff 1.4,
~70k pixels >30, mostly the ~55px shift of row 0 where final3's first band opens
earlier), not in the column boundaries. A repeat run on final3 is in flight to
confirm the effect is stable and not a single bad draw.

**Read the headline number with this in mind:** most of final3's apparent 528 ks
penalty is one column's ✓ landing one cell to the left, which is a
prompt/reading-convention issue and probably cheap to fix — not evidence that
the new geometry is worse.

### Repeat run confirms the effect is stable, and ~half of it is the ✓ column

    final2:  4087, 4148   (spread 61)
    final3:  4615, 4689   (spread 74)

Two tight clusters ~530 apart: a real, reproducible difference, not a bad draw.

Reassigning the misplaced ✓ back to Volume_No (post-hoc, tag `som-f3v2x`)
recovers **251 of the 528 ks — roughly half**:

    som-g37-flash (final2)          4087 ks
    som-f3v2x     (final3, ✓ fixed) 4364 ks
    som-f3v2      (final3, raw)     4615 ks

So the final3 penalty decomposes as ~250 ks of ✓-column misassignment (a
reading-convention problem, fixable in the prompt) and ~280 ks of everything
else, which remains unexplained and is the thing worth looking at next.
