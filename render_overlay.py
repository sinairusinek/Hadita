#!/usr/bin/env python3
"""Render a debug overlay: dewarped page + every PAGE-XML region/cell with its text.

Usage: python render_overlay.py [--page 10|50|...]   (default: 10 and 50)
Output: processed/Hadita_{N}_overlay.png
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
UPLOAD = ROOT / "Transkribus upload" / "original"
OUT_DIR = ROOT / "processed"
FONT_PATH = "/System/Library/Fonts/SFArabic.ttf"


def parse_points(s: str) -> list[tuple[int, int]]:
    return [(int(x), int(y)) for x, y in (p.split(",") for p in s.split())]


def parse_pagexml(xml_text: str) -> dict:
    """Return {'cells': [...], 'regions': [{'id', 'coords', 'text'}], 'baselines': [...]}."""
    cells: list[dict] = []
    for m in re.finditer(
        r'<TableCell id="([^"]+)" row="(\d+)" col="(\d+)".*?'
        r'<Coords points="([^"]+)"/>.*?'
        r'<Baseline points="([^"]+)"/>.*?'
        r'<Unicode>(.*?)</Unicode>',
        xml_text, re.S):
        cid, r, c, coords, bl, txt = m.groups()
        cells.append({
            "id": cid, "row": int(r), "col": int(c),
            "coords": parse_points(coords),
            "baseline": parse_points(bl),
            "text": txt,
        })
    regions: list[dict] = []
    for m in re.finditer(
        r'<TextRegion id="([^"]+)"[^>]*>\s*<Coords points="([^"]+)"/>.*?'
        r'<Unicode>(.*?)</Unicode>', xml_text, re.S):
        rid, coords, txt = m.groups()
        regions.append({"id": rid, "coords": parse_points(coords), "text": txt})
    return {"cells": cells, "regions": regions}


def html_unescape(s: str) -> str:
    return (s.replace("&amp;", "&").replace("&lt;", "<")
             .replace("&gt;", ">").replace("&quot;", '"'))


def render_page(page_num: int) -> Path:
    img_path = UPLOAD / f"Hadita_{page_num}.jpeg"
    xml_path = UPLOAD / f"Hadita_{page_num}.xml"
    if not img_path.exists() or not xml_path.exists():
        raise FileNotFoundError(f"Missing {img_path} or {xml_path}")

    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).convert("RGBA")

    # Translucent overlay layer for shapes
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Solid layer for text (drawn last so it sits on top)
    text_layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(text_layer)

    cell_font = ImageFont.truetype(FONT_PATH, 22)
    region_font = ImageFont.truetype(FONT_PATH, 36)

    parsed = parse_pagexml(xml_path.read_text(encoding="utf-8"))

    # Cells: thin blue border; non-empty cells get a yellow fill + text
    for c in parsed["cells"]:
        coords = c["coords"]
        text = html_unescape(c["text"]).strip()
        if text:
            draw.polygon(coords, fill=(255, 230, 90, 90), outline=(0, 80, 200, 220))
            xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
            tx, ty = min(xs) + 4, min(ys) + 2
            # Draw with white halo for readability
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                tdraw.text((tx+dx, ty+dy), text, font=cell_font, fill=(255,255,255,230))
            tdraw.text((tx, ty), text, font=cell_font, fill=(20, 20, 20, 255))
        else:
            draw.polygon(coords, outline=(0, 80, 200, 130))

    # Header TextRegions: thick colored border + recognized text in big font
    region_colors = {
        "taxpayer_name":  ((0, 200, 0, 255),  (200, 255, 200, 110)),
        "taxpayer_index": ((255, 100, 0, 255),(255, 220, 180, 110)),
    }
    for r in parsed["regions"]:
        outline, fill = region_colors.get(r["id"], ((255, 0, 255, 255), (255, 200, 255, 100)))
        draw.polygon(r["coords"], fill=fill, outline=outline, width=4)
        text = html_unescape(r["text"]).strip()
        label = f"{r['id']}: {text}" if text else f"{r['id']}: (empty)"
        xs = [p[0] for p in r["coords"]]; ys = [p[1] for p in r["coords"]]
        tx, ty = min(xs) + 8, max(min(ys) - 44, 4)
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(1,-1),(-1,1)):
            tdraw.text((tx+dx, ty+dy), label, font=region_font, fill=(255,255,255,255))
        tdraw.text((tx, ty), label, font=region_font, fill=outline[:3] + (255,))

    composed = Image.alpha_composite(pil, overlay)
    composed = Image.alpha_composite(composed, text_layer)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"Hadita_{page_num}_overlay.png"
    composed.convert("RGB").save(out_path, "PNG")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, action="append")
    args = ap.parse_args()
    pages = args.page or [10, 50]
    for p in pages:
        try:
            out = render_page(p)
            print(f"page {p}: {out}")
        except Exception as e:
            print(f"page {p}: FAILED  {e}")


if __name__ == "__main__":
    main()
