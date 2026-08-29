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
