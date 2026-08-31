"""Headless runner for Hadita_Gemini_Vision_Showcase.ipynb.

The notebook targets the new `client.interactions` agentic-vision API, which is not
in the public google-genai SDK yet. This script reuses the notebook's SYSTEM_INSTRUCTION,
PROMPT and structured-output schema verbatim (extracted from the .ipynb so they stay in
sync) and drives them through the available `client.models.generate_content` API, on one
sample page crop, one model x thinking variation.
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
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "Hadita-3Processed.jpg"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemini-3-flash-preview"
TOP_FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.24


def cell_src(idx):
    nb = json.loads(NB.read_text())
    return "".join(nb["cells"][idx]["source"])


# --- pull schema (cell 8) and prompts (cell 10) out of the notebook ---
ns = {"USE_ONE_SHOT": False}
exec(cell_src(8), ns)            # LedgerRow, SectionRows, _strict_schema, _to_sinai_row
exec(cell_src(10), ns)           # SYSTEM_INSTRUCTION, PROMPT
SectionRows = ns["SectionRows"]
_strict_schema = ns["_strict_schema"]
_to_sinai_row = ns["_to_sinai_row"]
SYSTEM_INSTRUCTION = ns["SYSTEM_INSTRUCTION"]
PROMPT = ns["PROMPT"]

# --- crop top TOP_FRAC of the page (same as notebook Step 7) ---
img = Image.open(IMAGE_PATH).convert("RGB")
W, H = img.size
crop = img.crop((0, 0, W, max(1, round(H * TOP_FRAC))))
buf = io.BytesIO()
crop.save(buf, format="JPEG", quality=95)
crop_bytes = buf.getvalue()
print(f"page={IMAGE_PATH} full={W}x{H} crop=top {TOP_FRAC:.0%} ({crop.size[0]}x{crop.size[1]})")
print(f"model={MODEL}\n")

client = genai.Client()

t0 = time.perf_counter()
resp = client.models.generate_content(
    model=MODEL,
    contents=[
        types.Part.from_bytes(data=crop_bytes, mime_type="image/jpeg"),
        PROMPT,
    ],
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
print(f"rows={len(rows)}  elapsed={elapsed}s")
um = resp.usage_metadata
if um:
    print(f"usage: prompt={um.prompt_token_count} output={um.candidates_token_count} "
          f"thought={getattr(um,'thoughts_token_count',None)} total={um.total_token_count}\n")

cols = ["Serial_No", "Date", "Property_recorded_under_Block_No", "Property_recorded_under_Parcel_No",
        "Parcel_Cat_No", "Parcel_Area", "Nature_of_Entry", "Tax_LP", "Tax_Mils", "Row_Confidence"]
for i, r in enumerate(rows, 1):
    print(f"--- row {i} ---")
    for c in cols:
        if r.get(c):
            print(f"  {c:38s} {r[c]}")

out = Path("showcase_sample_out.json")
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print(f"\nwrote {out} ({len(rows)} rows)")
