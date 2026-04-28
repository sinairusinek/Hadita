#!/usr/bin/env python3
"""
patch_xml_text.py — Patch <Unicode> text in PAGE XMLs from an OCR cache JSON.

Usage:
  python patch_xml_text.py --page 10 --source S_lite
  python patch_xml_text.py --page 50 --source S_full
  python patch_xml_text.py --page 10 50 --source S_lite

Reads .ocr_cache/{SOURCE}_page{N}.json and patches both:
  Transkribus upload/original/Hadita_{N}.xml          (Arabic-Indic digits)
  Transkribus upload/western arabic transliteration/Hadita_{N}.xml  (Western digits)

Geometry (Coords, Baseline) is untouched — only <Unicode> text changes.
"""

import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
CACHE_DIR   = PROJECT_DIR / ".ocr_cache"
UPLOAD_DIR  = PROJECT_DIR / "Transkribus upload"

LEFT_COLS = [
    "Serial_No", "Date",
    "Property_recorded_under_Block_No", "Property_recorded_under_Parcel_No",
    "Parcel_Cat_No", "Parcel_Area",
    "Nature_of_Entry", "New_Serial_No",
    "Reference_to_Register_of_Changes_Volume_No",
    "Reference_to_Register_of_Changes_Serial_No",
    "Tax_LP", "Tax_Mils", "Total_Tax_LP", "Total_Tax_Mils",
    "Reference_to_Register_of_Exemptions_Entry_No",
    "Reference_to_Register_of_Exemptions_Amount_LP",
    "Reference_to_Register_of_Exemptions_Amount_Mils",
    "Net_Assessment_LP", "Net_Assessment_Mils",
    "Remarks",
]

_EASTERN = "٠١٢٣٤٥٦٧٨٩"
_WESTERN = "0123456789"
_E2W = str.maketrans(_EASTERN, _WESTERN)
_W2E = str.maketrans(_WESTERN, _EASTERN)


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def patch_xml(xml_str: str, text_rows: list[dict], to_western: bool = False) -> tuple[str, int, int]:
    """Replace <Unicode> content in TableCells from text_rows.

    Returns (patched_xml, n_patched, n_skipped).
    """
    n_patched = 0
    n_skipped = 0

    for r_idx, row in enumerate(text_rows):
        for c_idx, col_name in enumerate(LEFT_COLS):
            raw = row.get(col_name, "").strip()
            text = raw.translate(_E2W) if to_western else raw

            cell_id = f'id="cell_r{r_idx}_c{c_idx}"'
            cell_start = xml_str.find(cell_id)
            if cell_start == -1:
                n_skipped += 1
                continue

            # Bound search to this cell block only (stop at next <TableCell)
            next_cell = xml_str.find("<TableCell", cell_start + len(cell_id))
            search_end = next_cell if next_cell != -1 else len(xml_str)

            unicode_start = xml_str.find("<Unicode>", cell_start, search_end)
            unicode_end   = xml_str.find("</Unicode>", cell_start, search_end)
            if unicode_start == -1 or unicode_end == -1:
                n_skipped += 1
                continue

            content_start = unicode_start + len("<Unicode>")
            xml_str = xml_str[:content_start] + _escape_xml(text) + xml_str[unicode_end:]
            n_patched += 1

    return xml_str, n_patched, n_skipped


def main():
    parser = argparse.ArgumentParser(description="Patch PAGE XML Unicode text from OCR cache.")
    parser.add_argument("--page", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--source", default="S_lite",
                        help="Cache key prefix, e.g. S_lite, S_full, M")
    args = parser.parse_args()

    for page_num in args.page:
        cache_file = CACHE_DIR / f"{args.source}_page{page_num}.json"
        if not cache_file.exists():
            print(f"[SKIP] Cache not found: {cache_file}")
            continue

        text_rows = json.loads(cache_file.read_text(encoding="utf-8"))
        print(f"Page {page_num}: loaded {len(text_rows)} rows from {cache_file.name}")

        for subfolder, to_western in [
            ("original", False),
            ("western arabic transliteration", True),
        ]:
            xml_path = UPLOAD_DIR / subfolder / f"Hadita_{page_num}.xml"
            if not xml_path.exists():
                print(f"  [SKIP] {xml_path} not found")
                continue

            xml_str = xml_path.read_text(encoding="utf-8")
            patched, n_ok, n_skip = patch_xml(xml_str, text_rows, to_western=to_western)
            xml_path.write_text(patched, encoding="utf-8")
            print(f"  {subfolder}/Hadita_{page_num}.xml — patched {n_ok} cells, skipped {n_skip}")

    print("Done.")


if __name__ == "__main__":
    main()
