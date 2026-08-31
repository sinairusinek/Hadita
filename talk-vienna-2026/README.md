# Vienna 2026 talk — figures and site

Presentation page for *Winding Workflows: Editors, Artificial Agents and Specialist
ATR Models* (OCR/HTR for Under-resourced and Under-represented Languages Redux,
Austrian Academy of Sciences, 10 September 2026).

- `SLIDES.md` — the slide draft the page follows.
- `index.html` — the GitHub Pages site.
- `make_figures.py` — generates every figure in `img/` from the working corpus.
- `measure_ink.py` — per-cell ink measurement feeding the slide-11 histogram.
- `img/src/` — the two supplied Tel Hadid images (photo, line drawing).

## Rebuilding

```sh
../.venv/bin/python measure_ink.py            # writes data/ink_per_cell.tsv
../.venv/bin/python make_figures.py           # all figures
../.venv/bin/python make_figures.py fig08_dewarp_damage fig11_ink_histogram   # just these
```

Photographic panels are JPEG; charts and diagrams are SVG so they stay sharp on a
projector. Annotation colours come from one `PALETTE` so the deck reads as a system:
red = the failure being shown, green = the fix, yellow = printed form/geometry,
dark blue = handwriting and measured signal.

## Figure list

| file | slide | shows |
|---|---|---|
| `fig04_tel_hadid.jpg` | 4 | the site today beside a register page |
| `fig06_two_registers.jpg` | 6 | Haifa (solved) vs Hadita (not) |
| `fig06b_detail_strip.jpg` | 6 | four failure modes at reading scale |
| `fig07_the_cast.svg` | 7 | pipeline coloured by *who* does each stage |
| `fig08_dewarp_damage.jpg` | 8 | the dewarp cut line drawn on the page it truncated |
| `fig09a_header_band.jpg` | 9 | the print-only header band and its 18 rules |
| `fig09_column_template.jpg` | 9 | the fitted template down the page; derived left edge |
| `fig09b_one_miss.jpg` | 9 | one dropped rule → every column right of it off by one |
| `fig10a_morphology.jpg` | 10 | the standard recipe finding nothing |
| `fig10b_strip_median.svg` | 10 | 30 noisy strips vs their median comb of 34 peaks |
| `fig10c_rules_not_boundaries.jpg` | 10 | rules-as-boundaries cut the writing; midway does not |
| `fig11_ink_histogram.svg` | 11 | bimodal ink per cell, with the shipped cutoff |
| `fig12_agreement_funnel.svg` | 12 | 58,122 cells → what the RA actually gets |
| `fig13_baseer_days.jpg` | 13 | days of the week invented in empty cells |
| `fig14_counting_problem.jpg` | 14 | page 9: which ruled rows carry writing |
| `fig15_set_of_marks.jpg` | 15 | row indices printed in the margin |
| `fig16_three_to_two.jpg` | 16 | the scribe's ٣, at reading size |

## Numbers on the page

Recomputed from pipeline outputs rather than copied from notes, so a few differ
from `SLIDES.md`:

- **Ink cutoff is 6px, not 12.** `gate_textlines.py` lowered it (user-labelled
  cells sat at 9px). The histogram and the page both say 6.
- **٣→٢ = 98** on the six proxy-GT pages by my recount (`_som_vs_gt`); the slide's
  135 is Gemini's own figure over a different comparison set. Both are stated in
  their own context.
- The histogram is measured on the six proxy-GT pages (3,895 cells), the only ones
  with per-cell text against the current geometry — the corpus-wide 58,122/18,718
  come from the shipped gate report.
- The agreement funnel projects the measured 32.8% auto-accept rate across inked
  cells; the figure says so in its footnote.

## Publishing

The site is self-contained (no build step, no external assets). Enable Pages on
this repo pointing at `talk-vienna-2026/` on the default branch, or copy the
folder into a `docs/` root.
