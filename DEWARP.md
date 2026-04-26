# Image Dewarp Pipeline

Produces a fully flattened page image — straight horizontal rows and vertical columns — suitable for upload to Transkribus alongside a simple straight-grid PAGE XML (no bow-tracking parallelograms needed).

## How it works

```
Original JPEG
    │
    ▼
deskewed_page{N}.png/jpg  ←  pre-computed by haditax.py (perspective correction)
    │
    ▼  crop_table()
Table region (left ~45% of page, below header strip)
    │
    ├──▶ detect_rows()             → row y-centers  [Kraken; morph fallback]
    │                                 + interpolate_gaps() fills missed rows
    │
    └──▶ detect_columns()          → global column x-boundaries
         detect_columns_banded()   → per-band column x (captures page bow, 8 bands)
              │
              ▼
         build_remap()
              │  For every output pixel (ox, oy):
              │    src_y = row_interp(oy)          ← row centers → uniform pitch
              │    src_x = col_interp(ox, src_y)   ← per-band column x → uniform spacing
              ▼
         cv2.remap(table, map_x, map_y)  →  dewarped table
              │
              ▼
         header strip (above table, unchanged) + dewarped table
              │
              ▼
         Hadita-{N}Processed.jpg
```

## Running (page 3)

```bash
python dewarp_page3.py
```

**Prerequisites:**
- `images/deskewed_page3.jpg` or `.ocr_cache/deskewed_page3.png` must exist (produced by the Haditax app's deskew step)
- Kraken segmentation model at `sc_100p_full_line (1).mlmodel` (used for row detection; morphological fallback activates automatically if absent)

**Output:** `Hadita-3Processed.jpg` — header strip + dewarped data table, JPEG quality 95.

## Output structure

| Section | Source | Transform |
|---|---|---|
| Header strip (top ~8%) | deskewed image | None (copied as-is) |
| Data table | deskewed image | Full 2-D remap (rows + columns) |

The output width equals the deskewed table width (~2300 px for Hadita). Output height = `n_rows × median_pitch`.

## Key parameters (in `dewarp_page3.py`)

| Constant | Value | Meaning |
|---|---|---|
| `CFG["header_frac"]` | 0.08 | Top 8% of page = header, excluded from table |
| `CFG["table_width_frac"]` | 0.455 | Left ~45.5% of page = data table |
| `N_BANDS` | 8 | Horizontal bands for per-band column detection |
| `JPEG_Q` | 95 | Output JPEG quality |

## TODO

- [ ] **Multi-page support** — generalise `dewarp_page3.py` into `dewarp.py --pages 3 10 50` (or `--all`), iterating over a list of page numbers and writing `Hadita-{N}Processed.jpg` for each. Per-page deskewed caches must exist; the script should report clearly which pages are missing their cache.
