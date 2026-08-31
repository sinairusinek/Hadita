#!/usr/bin/env python3
"""
fill_page_xml.py — Run Gemini approach M on a page and inject results into
a PAGE XML template using the dewarped image for better OCR quality.

Usage:
  python3 fill_page5_xml.py

Output:
  0004_Hadita_5_gemini.xml  (next to the input XML)
"""
import re
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from compare_ocr import (
    _gemini_client, GEMINI_25_PRO, OCR_PROMPT_FULL_FEWSHOT,
    parse_json, normalize_row, save_cache, load_cache,
    ALL_DATA_COLS, LEFT_COLS, CACHE_DIR,
)
import json

# ── paths ────────────────────────────────────────────────────
INPUT_XML  = Path("/Users/sinairusinek/Downloads/export_job_25533770/"
                  "15829823/Hadita-Processed/page/0004_Hadita_5.xml")
OUTPUT_XML = INPUT_XML.with_name("0004_Hadita_5_gemini.xml")
PAGE_NUM   = 4

# col index → LEFT_COLS key (Remarks is col 19 but not in the XML's 19-col table)
COL_NAMES = LEFT_COLS[:19]   # Serial_No … Net_Assessment_Mils


def xml_escape(text: str) -> str:
    """Escape characters that are invalid in XML text content."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def inject(xml_text: str, rows: list[dict]) -> str:
    """
    Replace every <Unicode></Unicode> inside a TableCell with the value from `rows`.
    TableCell row/col attributes drive the mapping; the row order in `rows` maps 1:1
    to the XML row index (row="0" → rows[0]).
    """
    def replacer(m: re.Match) -> str:
        xml_row  = int(m.group("r"))
        xml_col  = int(m.group("c"))
        col_name = COL_NAMES[xml_col] if xml_col < len(COL_NAMES) else None
        if col_name is None or xml_row >= len(rows):
            return m.group(0)   # leave untouched
        val = rows[xml_row].get(col_name, "")
        escaped = xml_escape(str(val))
        return m.group(0).replace(
            "<Unicode></Unicode>",
            f"<Unicode>{escaped}</Unicode>",
        )

    # Match an entire TableCell block (non-greedy) and capture its row/col.
    # We rely on the XML being well-structured (one Unicode per cell).
    pattern = re.compile(
        r'<TableCell\s[^>]*\brow="(?P<r>\d+)"[^>]*\bcol="(?P<c>\d+)"[^>]*>.*?</TableCell>',
        re.DOTALL,
    )
    return pattern.sub(replacer, xml_text)


def run_gemini_dewarped(page_num: int) -> list[dict]:
    """Approach M variant: use dewarped image instead of original scan."""
    cached = load_cache("M", page_num)
    if cached is not None:
        print(f"  (using cached M_page{page_num}.json)")
        return cached

    deskewed = CACHE_DIR / f"deskewed_page{page_num}.png"
    if not deskewed.exists():
        raise FileNotFoundError(f"Dewarped image not found: {deskewed}")

    client = _gemini_client()
    img = Image.open(deskewed)
    print(f"  Image: {deskewed.name} ({img.size[0]}×{img.size[1]})")
    from compare_ocr import _gemini_ocr
    raw = _gemini_ocr(client, GEMINI_25_PRO, OCR_PROMPT_FULL_FEWSHOT, [img])
    data = parse_json(raw)
    rows = [normalize_row(r, ALL_DATA_COLS) for r in data.get("rows", [])]
    save_cache("M", page_num, rows)

    raw_meta = data.get("page_meta", {})
    if raw_meta:
        meta = {
            "Tax_Payer_Arabic":       raw_meta.get("tax_payer_arabic", ""),
            "Tax_Payer_Romanized":    raw_meta.get("tax_payer_romanized", ""),
            "Tax_Payer_ID_Arabic":    raw_meta.get("tax_payer_id_arabic", ""),
            "Tax_Payer_ID_Romanized": raw_meta.get("tax_payer_id_romanized", ""),
        }
        (CACHE_DIR / f"meta_page{page_num}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

    return rows


def main():
    print(f"Running Gemini approach M (dewarped) on page {PAGE_NUM} …")
    rows = run_gemini_dewarped(PAGE_NUM)
    print(f"  → {len(rows)} rows returned by Gemini")

    xml_text = INPUT_XML.read_text(encoding="utf-8")
    updated  = inject(xml_text, rows)

    OUTPUT_XML.write_text(updated, encoding="utf-8")
    print(f"Saved: {OUTPUT_XML}")

    # Quick sanity check
    non_empty = updated.count("<Unicode>") - updated.count("<Unicode></Unicode>")
    print(f"  {non_empty} non-empty Unicode fields written")


if __name__ == "__main__":
    main()
