"""Batch runner: Gemini-3 showcase OCR on full pages -> haditax .ocr_cache JSON.

Reuses the notebook's SYSTEM_INSTRUCTION, PROMPT and SectionRows schema (extracted
from Hadita_Gemini_Vision_Showcase.ipynb so they stay in sync), but OCRs the WHOLE
dewarped left-table page (processed/Hadita-{N}Processed.jpg) instead of the top crop,
and writes .ocr_cache/G3_page{N}.json in the row-dict format the haditax app loads.

Usage: python run_showcase_pages.py 33 34 35
"""
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types

NB = Path("Hadita_Gemini_Vision_Showcase.ipynb")
CACHE_DIR = Path(".ocr_cache")
MODEL = "gemini-3-flash-preview"
PAGES = [int(p) for p in sys.argv[1:]] or [33, 34, 35]


def cell_src(idx):
    return "".join(json.loads(NB.read_text())["cells"][idx]["source"])


ns = {"USE_ONE_SHOT": False}
exec(cell_src(8), ns)
exec(cell_src(10), ns)
SectionRows = ns["SectionRows"]
_to_sinai_row = ns["_to_sinai_row"]
SYSTEM_INSTRUCTION = ns["SYSTEM_INSTRUCTION"]
PROMPT = ns["PROMPT"]

client = genai.Client()

for page in PAGES:
    img_path = Path("processed") / f"Hadita-{page}Processed.jpg"
    if not img_path.exists():
        print(f"page {page}: {img_path} not found — skipping")
        continue
    img = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)

    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=SectionRows,
            max_output_tokens=24576,
        ),
    )
    elapsed = round(time.perf_counter() - t0, 2)
    section = SectionRows.model_validate_json(resp.text)
    rows = [_to_sinai_row(r) for r in section.rows]

    out = CACHE_DIR / f"G3_page{page}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    um = resp.usage_metadata
    print(f"page {page}: {len(rows)} rows  {elapsed}s  "
          f"tokens in={um.prompt_token_count} out={um.candidates_token_count} "
          f"thought={getattr(um,'thoughts_token_count',None)}  -> {out}")
