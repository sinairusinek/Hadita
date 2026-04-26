# Image Dewarp Pipeline

Notebook-level preprocessor that turns each original census scan into a fully flattened, fixed-canvas page image. Every output shares the same width and the same table-top-corner coordinates, so a single PAGE XML grid (uniform row pitch, evenly-spaced column boundaries) is valid across the whole notebook.

## Pipeline (canonical order)

```
Original JPEG  (000nvrj-…page-{NNNN}.jpg)
    │
    ▼  image_preprocess.deskew_image
Deskewed page              ← perspective warp of paper boundary
    │  (cached → .ocr_cache/deskewed_page{N}.png)
    ▼  segment_unified.crop_table
Table region (left ~45.5%, below header strip)
    │
    ├──▶ detect_rows()             → row y-centers   [Kraken; morph fallback]
    │                                + interpolate_gaps fills missed rows
    │
    └──▶ detect_columns()          → global column x-boundaries
         detect_columns_banded()   → per-band column x (captures page bow, 8 bands)
              │
              ▼  segment_unified.build_remap
         dense cv2.remap → flattened table
              │
              ▼  Canvas normalization
         resize header strip → W_OUT × H_HEADER
         resize dewarped table → W_OUT × (n_rows · ROW_PITCH)
         vstack
              ▼
         processed/Hadita-{N}Processed.jpg
```

## Fixed canvas constants ([dewarp.py](dewarp.py))

| Constant | Value | Meaning |
|---|---|---|
| `W_OUT`     | 2299 | Output width (px). Same for every page. |
| `H_HEADER`  | 320  | Top strip height (metadata band + printed column-header row). |
| `ROW_PITCH` | 100  | Row pitch in dewarped table (px). Same for every page. |
| `JPEG_Q`    | 95   | Output JPEG quality. |

**Guarantees on every output:**
- Table top-left corner at `(0, H_HEADER)`.
- Table top-right corner at `(W_OUT, H_HEADER)`.
- Width exactly `W_OUT`; height = `H_HEADER + n_rows · ROW_PITCH`.

## Running

```bash
# Process specific pages
python dewarp.py --pages 3 10 50

# Reuse the deskewed cache (.ocr_cache/deskewed_page{N}.{png,jpg})
python dewarp.py --pages 3 10 50 --from-cache

# Write per-step debug images to processed/_debug/page{N}/
python dewarp.py --pages 3 --debug

# Future: every original landscape scan (skips portrait cover automatically)
python dewarp.py --all-except-cover
```

Failures are logged per page (`WARNING: page N failed: …`) and don't abort the batch.

## PAGE XML output (segment_unified.py)

```bash
python segment_unified.py --page 3 --from-cache
```

For each page, writes:
- `processed/Hadita-{N}Processed.jpg` (image)
- `Transkribus upload/original/Hadita_{N}.{jpeg,xml}` (Eastern Arabic digits)
- `Transkribus upload/western arabic transliteration/Hadita_{N}.{jpeg,xml}` (Western digits)

Both XML variants reference the dewarped JPEG and use a uniform rectangular grid: row 0 has `y ∈ [H_HEADER, H_HEADER+ROW_PITCH]`, columns are `np.linspace(0, W_OUT, 20)`. No per-band bow tracking is needed because the image itself is already flat.

## Verification

```bash
python dewarp.py --pages 3 10 50 --debug
python verify_dewarp.py
open processed/_debug/contact_sheet.html
```

Per-page debug images (when `--debug`):
| File | Step |
|---|---|
| `01_deskewed.jpg` | After perspective warp |
| `02_table.jpg` | After table crop |
| `03_rows.jpg` | Row centers overlaid (synthetic = orange, real = red) |
| `04_cols.jpg` | All 8 bands' column-x ticks |
| `05_dewarped_raw.jpg` | After remap, before canvas normalization |

`verify_dewarp.py` asserts canvas invariants and emits a contact sheet with a red horizontal guideline at `y = H_HEADER`. Visually scan: every page's first row must begin exactly on that line.

## Source map

| File | Role |
|---|---|
| [image_preprocess.py](image_preprocess.py) | `deskew_image(bgr) → bgr` (pure, no Streamlit) |
| [segment_unified.py](segment_unified.py) | `crop_table`, `detect_rows`, `detect_columns(_banded)`, `build_remap`, `dewarped_grid`, PAGE XML writer |
| [dewarp.py](dewarp.py) | CLI entry point, canvas normalization, `process_page` |
| [verify_dewarp.py](verify_dewarp.py) | Cross-page invariants + contact sheet |
| [haditax.py](haditax.py) | Streamlit-aware `deskew_page(N)` wrapper around `deskew_image` |
