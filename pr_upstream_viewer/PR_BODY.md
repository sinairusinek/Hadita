# Render PAGE `TableCell` regions, not just `TextRegion`

## The problem

The viewer walks `TextRegion` → `TextLine`. PAGE files whose text lives in
`TableCell` therefore render as empty: no polygons on the image, no
transcription in the text panel.

That is the normal export shape for tabular sources — registers, ledgers,
censuses, account books — where Transkribus puts each cell's `Coords` and
`TextEquiv` on the `TableCell` itself. On such a page the viewer currently
draws only the two page-level text regions and silently misses every cell.

## The change

`TableCell` is treated as a region, reusing the existing region path:

- **`collectRegions()`** gathers `TextRegion` *and* `TableCell`, matching on
  `localName` so it is independent of the PAGE namespace prefix.
- **`ownCoords()`** reads an element's *own* `Coords` child rather than the
  first descendant. Without this a cell's polygon is drawn as its first line.
- Cells render **from their own `Coords`**, so a cell with geometry but no text
  still appears. This is what makes the viewer useful for checking segmentation
  *before* transcription exists.
- Cells are **coloured by column index** and labelled `Cell r3 c8` from the
  `row`/`col` attributes, so a mis-assigned column is visible at a glance.
- The inline region/line walking is factored into `collectRegions()`,
  `collectLines()`, `ownCoords()` and `parsePoints()`. This is what makes the
  `TableCell` case a few lines instead of a fourth copy of the same loop, and
  it accounts for most of the line count in the diff.

## No regression on prose manuscripts

Verified against the sample data already in this repository — the Dalimil,
Cosmas and NKP I D 10 sets:

| | |
|---|---|
| pages checked | 67 |
| regions found | 76 |
| lines found | 2258 |
| pages where output differs | **0** |

Region and line extraction is identical before and after the change.

## Where it was used

Tested on a 98-page Arabic tax register from British Mandate Palestine
(19 columns, ~35 rows per page): 667 cell polygons and 412 line polygons on a
representative page, no console errors. The colour-by-column view is what let
us spot column misalignment across the corpus.

Happy to split the refactor from the `TableCell` feature into two commits, drop
the colouring, or adjust naming to your preference — whatever fits the project
best.
