"""E6: build blind contact sheets of flagged cells for the in-session third reader.

Stratified sample from exp2608/e6_reader_tasks.tsv (144 MISMATCH, 48 ONLY_som,
48 ONLY_hadid02). Crops each cell from Transkribus upload/final2/Hadita_{N}.jpeg
using the cell polygon in the final2 PAGE XML, upscales 2x, and tiles 24 per
sheet with only the sample index as label (no values -> blind reading).

Outputs:
  exp2608/e6_sheets/sheet_{k}.jpg
  exp2608/e6_sample.tsv   (idx -> page,row,col,state,som,hadid02,gt)
"""
import csv
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path("exp2608")
SHEET_DIR = OUT / "e6_sheets"
IMG = "Transkribus upload/final2/Hadita_{p}.jpeg"
XML = "Transkribus upload/final2/Hadita_{p}.xml"
PER_SHEET = 24
COLS = 3
PAD = 4
SCALE = 2

CELL_COORD_RE = re.compile(
    r'<TableCell\s+id="cell_r(\d+)_c(\d+)"[^>]*>.*?<Coords points="([^"]+)"',
    re.DOTALL)


def cell_bboxes(page: int) -> dict:
    xml = open(XML.format(p=page), encoding="utf-8").read()
    out = {}
    for m in CELL_COORD_RE.finditer(xml):
        pts = [tuple(map(int, p.split(","))) for p in m.group(3).split()]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out[(int(m.group(1)), int(m.group(2)))] = (min(xs), min(ys), max(xs), max(ys))
    return out


def main() -> None:
    rng = random.Random(2608)
    tasks = list(csv.DictReader(open(OUT / "e6_reader_tasks.tsv", encoding="utf-8"),
                                delimiter="\t"))
    by_state = {}
    for t in tasks:
        by_state.setdefault(t["state"], []).append(t)
    sample = []
    for state, n in [("MISMATCH", 144), ("ONLY_som", 48), ("ONLY_hadid02", 48)]:
        pool = by_state.get(state, [])
        sample += rng.sample(pool, min(n, len(pool)))
    rng.shuffle(sample)

    boxes = {p: cell_bboxes(p) for p in {int(t["page"]) for t in sample}}
    imgs = {p: Image.open(IMG.format(p=p)) for p in boxes}

    SHEET_DIR.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()

    with open(OUT / "e6_sample.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["idx", "page", "grid_row", "col", "col_idx", "state",
                    "som", "hadid02", "gt"])
        for i, t in enumerate(sample):
            w.writerow([i, t["page"], t["grid_row"], t["col"], t["col_idx"],
                        t["state"], t["som"], t["hadid02"], t["gt"]])

    crops = []
    for i, t in enumerate(sample):
        p = int(t["page"])
        key = (int(t["grid_row"]), int(t["col_idx"]))
        bb = boxes[p].get(key)
        if bb is None:
            crops.append((i, None))
            continue
        x0, y0, x1, y1 = bb
        im = imgs[p].crop((max(0, x0 - PAD), max(0, y0 - PAD),
                           x1 + PAD, y1 + PAD))
        im = im.resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS)
        crops.append((i, im))

    cell_w = max(im.width for _, im in crops if im) + 20
    cell_h = max(im.height for _, im in crops if im) + 46
    rows = (PER_SHEET + COLS - 1) // COLS
    for s in range(0, len(crops), PER_SHEET):
        chunk = crops[s:s + PER_SHEET]
        sheet = Image.new("RGB", (COLS * cell_w, rows * cell_h), "white")
        d = ImageDraw.Draw(sheet)
        for k, (idx, im) in enumerate(chunk):
            cx, cy = (k % COLS) * cell_w, (k // COLS) * cell_h
            d.text((cx + 8, cy + 4), f"#{idx}", fill="red", font=font)
            if im:
                sheet.paste(im, (cx + 10, cy + 34))
            d.rectangle([cx, cy, cx + cell_w - 1, cy + cell_h - 1],
                        outline="#bbbbbb")
        path = SHEET_DIR / f"sheet_{s // PER_SHEET:02d}.jpg"
        sheet.save(path, quality=92)
        print(f"wrote {path} ({len(chunk)} cells)")


if __name__ == "__main__":
    main()
