"""Generator for Hadita_Gemini3_to_PAGEXML_v6.ipynb.

v6 changes vs v5:
  - DEWARP-DAMAGE MITIGATION: per page, the model receives BOTH the dewarped
    processed image (canonical grid, what we transcribe) AND the raw scan
    (full margins, used only to recover top/bottom rows that dewarp clipped
    or smeared). New <dewarp_damage> prompt block governs this.
  - Bulk upload now expects 3N files (processed + raw + XML per page).
  - Outputs: Hadita_{N}_G3v6.json / Hadita_{N}_G3v6.xml; log to g3_runs_v6.csv
    (does not clobber v5 artifacts).
  - run_one_page() accepts a LIST of image URIs.

v5 (frozen) stays as Hadita_Gemini3_to_PAGEXML_v5.ipynb at repo root.

Re-run this script to regenerate the .ipynb after tweaking cell text.
"""
import json
from pathlib import Path


cells = []

def md(s: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})

def code(s: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


# ────────────────────────────────────────────────────────────────
# Cell 0 — intro
# ────────────────────────────────────────────────────────────────
md(r"""# Hadita — Gemini 3 → PAGE XML  (v6, multi-page, dewarp-damage recovery)

**v6 delta vs v5:** for each page the model receives BOTH the dewarped processed image (canonical grid) AND the raw scan (full margins). A new `<dewarp_damage>` prompt block tells the model: transcribe from the processed image, but consult the raw scan to recover top/bottom rows that the dewarp pipeline clipped or smeared. Outputs are written to `Hadita_{N}_G3v6.{json,xml}` and logged to `g3_runs_v6.csv` so they don't overwrite v5 artifacts.

Runs **Gemini 3** agentic OCR on **multiple processed/dewarped** Hadita tax-register pages in one session, with the project's tuned prompt rules ported from Approach M plus page-3/4 scoring findings:

* column-order warning, `Nature_of_Entry` vocabulary, multi-category and breakdown-row rules, resume-numbering rule, [RED]/[?]/~~strike~~ markers, "one value per cell — never `X / Y`", "never omit a row"
* `<numerals>` covers ٢/٣, ٣/٤, ٤/٦, **٦/٨**, and the "digit-order can flip" caution
* `<reference_column_conventions>` (new in v5): **preserve leading zeros** in Reference Serial cells (٠٢٤, ٠١٦, ٠٩٠), use `"..."` for "no data" not `"---"`, and **keep `T.D.L`/`T.P.L`/`D.L` in Vol_No separate from the 4-digit year in Serial_No**

Each page produces:
* `Hadita_{N}_G3.json` — raw row dicts in `LEFT_COLS` order
* `Hadita_{N}_G3.xml` — patched PAGE XML, geometry preserved, `<Unicode>` text overwritten

The cumulative cost log is appended to `g3_runs_v6.csv`.

**Model**: `gemini-3.5-flash`, **`THINKING="low"` by default** (decided by page-3/4 cell-level scoring: LOW costs ~$0.08/page vs MEDIUM's ~$0.32 with no clear net RA-time advantage for MEDIUM on these pages).
""")

# ────────────────────────────────────────────────────────────────
# Cell 1 — install
# ────────────────────────────────────────────────────────────────
code(r'''# Pin a known-working google-genai version once you've verified one in your env.
# Without a pin, Colab may auto-upgrade between sessions and reshape the Interaction
# response (the v3/v4 SDK had `.steps`; the v5-era SDK has `.outputs`).
# Both shapes are handled by the helpers below — but pinning prevents surprises.
!pip install -q -U google-genai pydantic pillow
import google.genai
print(f"google-genai version: {google.genai.__version__}")
''')

# ────────────────────────────────────────────────────────────────
# Cell 2 — API key + client
# ────────────────────────────────────────────────────────────────
code(r'''import os
from google import genai

key = None
try:
    from google.colab import userdata
    key = userdata.get("GEMINI_API_KEY")
except Exception:
    key = os.environ.get("GEMINI_API_KEY")
if not key:
    import getpass
    key = getpass.getpass("Paste your GEMINI_API_KEY: ")

os.environ["GEMINI_API_KEY"] = key
os.environ["GOOGLE_API_KEY"] = key
client = genai.Client()
print("client ready")
''')

# ────────────────────────────────────────────────────────────────
# Cell 3 — page list + bulk upload
# ────────────────────────────────────────────────────────────────
code(r'''# === EDIT THIS — the list of pages you want to process in this session ===
PAGES = [3, 4, 5, 6, 9, 10]

MODEL             = "gemini-3.5-flash"   # Gemini 3 family — do NOT downgrade to 2.5
THINKING          = "low"                # LOW is the v5 default (page-3/4 cell-level scoring)
RESOLUTION        = "high"               # auto-retry will lower this on backend size-cap
MAX_OUTPUT_TOKENS = 32768                # auto-retry will raise this on JSON truncation

# Bulk file upload — for each page N upload THREE files:
#   Hadita-{N}Processed.jpg          (the dewarped image — canonical grid)
#   Hadita-{N}Raw.jpg                (raw scan — full margins; original
#                                     "000nvrj-...page-{N:04d}.jpg" also accepted)
#   Hadita_{N}.xml                   (existing PAGE XML from Transkribus upload/final/)
# Drag them all into the single file picker below.
import re
from google.colab import files

print(f"Will process pages: {PAGES}")
print(f"Drag in up to {3*len(PAGES)} files: processed image + raw scan + XML per page.")
print(f"(If you skip the raw scan for a page, v6 falls back to v5 behavior on that page.)\n")
uploaded = files.upload()

# Pair files by page number, by filename pattern.
IMAGES_PROCESSED, IMAGES_RAW, XMLS = {}, {}, {}
for fname in uploaded:
    m_raw_renamed = re.search(r"Hadita[-_](\d+)Raw\.(?:jpg|jpeg|png)$", fname, re.I)
    m_raw_orig    = re.search(r"page[-_](\d{3,4})\.(?:jpg|jpeg|png)$", fname, re.I)
    m_proc        = re.search(r"Hadita[-_](\d+)Processed\.(?:jpg|jpeg|png)$", fname, re.I)
    m_xml         = re.search(r"Hadita[-_](\d+)\.xml$", fname, re.I)
    if m_raw_renamed:
        IMAGES_RAW[int(m_raw_renamed.group(1))] = fname
    elif m_proc:
        IMAGES_PROCESSED[int(m_proc.group(1))] = fname
    elif m_xml:
        XMLS[int(m_xml.group(1))] = fname
    elif m_raw_orig:
        IMAGES_RAW[int(m_raw_orig.group(1))] = fname

print(f"\nDetected {len(IMAGES_PROCESSED)} processed images, {len(IMAGES_RAW)} raw scans, {len(XMLS)} XMLs")
missing_critical, missing_raw = [], []
for p in PAGES:
    if p not in IMAGES_PROCESSED:
        missing_critical.append(f"processed image for page {p}")
    if p not in XMLS:
        missing_critical.append(f"XML for page {p}")
    if p not in IMAGES_RAW:
        missing_raw.append(p)
if missing_critical:
    print(f"\n⚠ Missing required files:\n  " + "\n  ".join(missing_critical))
    print("\nPages without both processed image AND XML will be skipped silently.")
if missing_raw:
    print(f"\nℹ No raw scan for page(s) {missing_raw} — those pages will run in v5 (single-image) mode.")
if not missing_critical and not missing_raw:
    print(f"✓ All {len(PAGES)} pages have processed + raw + XML ready.")
print(f"\nmodel: {MODEL} × thinking={THINKING}, resolution={RESOLUTION}, max_output={MAX_OUTPUT_TOKENS}")
''')

# ────────────────────────────────────────────────────────────────
# Cell 4 — schema (LedgerRow, SectionRows, helpers)
# ────────────────────────────────────────────────────────────────
code(r'''from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

# Column order MUST match patch_xml_text.py:LEFT_COLS so each LedgerRow field
# lines up positionally with cell_r{R}_c{C} in the PAGE XML.
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

class LedgerRow(BaseModel):
    """One row of the left tax-register table. Empty cells → "" (never null)."""
    serial_no: Optional[str] = Field(None, description="Serial_No. Eastern Arabic digits exactly as written; \"\" if blank (e.g. breakdown rows).")
    date: Optional[str] = Field(None, description="Date (column 2 — NOT Block_No). Year-like Eastern Arabic digits (e.g. ٩٣٨), or \" for ditto, or \"\" if blank. Exactly ONE value; never compound like 'X / Y'.")
    block_no: Optional[str] = Field(None, description="Property_recorded_under_Block_No. 4-digit Eastern Arabic (range ٤١٣٢–٤١٥٢). \"\" on multi-category continuation rows (never ditto/fabricate).")
    parcel_no: Optional[str] = Field(None, description="Property_recorded_under_Parcel_No. Eastern Arabic digits. \"\" on multi-category continuation rows.")
    cat_no: Optional[str] = Field(None, description="Parcel_Cat_No (cultivation category). Eastern Arabic digits, usually two.")
    area: Optional[str] = Field(None, description="Parcel_Area. Eastern Arabic digits; standardize thousands separator to ASCII comma.")
    nature_of_entry: Optional[str] = Field(None, description="Nature_of_Entry. Raw Arabic + symbols only — NEVER translate, gloss, or interpret. Common: تح, تسلل, شراء, شرائي, بيع, ضريبة حرب. Combinations: 'تح ✓'. Blank → \"\". Ditto → \".")
    new_serial_no: Optional[str] = Field(None, description="New_Serial_No. Eastern OR Western digits; or short cross-ref like 'انظر ٩٧'.")
    ref_volume_no: Optional[str] = Field(None, description="Reference_to_Register_of_Changes_Volume_No. Eastern digits, ✓, '...', or an English abbreviation (T.D.L / T.P.L / D.L). \"\" if blank.")
    ref_serial_no: Optional[str] = Field(None, description="Reference_to_Register_of_Changes_Serial_No. Eastern digits TYPICALLY 3-digit WITH LEADING ZERO (٠٢٤, ٠١٦, ٠٩٠ — preserve the leading 0). Or '...' (literal, three dots) for no data. Or a 4-digit Western year (e.g. 1940) when Vol_No holds T.D.L/T.P.L. Or \"\".")
    tax_lp: Optional[str] = Field(None, description="Tax — L.P. column. Usually NOT a number: '✓' (assessed), '-' (nil), Eastern digit ONLY if tax ≥ 1 LP, \"\" if blank.")
    tax_mils: Optional[str] = Field(None, description="Tax — Mils. Eastern Arabic digits (usually 3 with leading zero, e.g. ٠٨٥), '-' for nil, or \"\".")
    total_tax_lp: Optional[str] = Field(None, description="L.P. under 'Total Tax'. Same value domain as Tax_LP.")
    total_tax_mils: Optional[str] = Field(None, description="Mils under 'Total Tax'. Eastern Arabic digits or \"\". Do NOT collapse Total_Tax_Mils into Tax_Mils.")
    exemption_entry_no: Optional[str] = Field(None, description="Entry_No under 'Reference to Register of Exemptions'. Eastern Arabic digits or \"\".")
    exemption_amount_lp: Optional[str] = Field(None, description="L.P. under exemption amount. ✓ / - / Eastern digit / \"\".")
    exemption_amount_mils: Optional[str] = Field(None, description="Mils under exemption amount. Eastern Arabic digits or \"\".")
    net_assessment_lp: Optional[str] = Field(None, description="L.P. under 'Net Assessment'. ✓ / - / Eastern digit / \"\".")
    net_assessment_mils: Optional[str] = Field(None, description="Mils under 'Net Assessment'. Eastern Arabic digits or \"\".")
    remarks: Optional[str] = Field(None, description="Free-text annotation. Use ~~old~~ new for visible correction; append [RED] if red ink; [?] if uncertain.")
    confidence: Literal["high", "medium", "low"] = Field("high", description="Reviewer-facing row confidence. 'medium' or 'low' whenever code_execution was used to disambiguate any cell.")
    reviewer_note: Optional[str] = Field(None, description="≤ 120 chars pointer for the human reviewer when confidence != 'high'.")

class SectionRows(BaseModel):
    rows: list[LedgerRow] = Field(description="EVERY row of the left table, top to bottom. NEVER omit a row.")

def _strict_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    def _t(n: Any) -> None:
        if isinstance(n, dict):
            if n.get("type") == "object" and "properties" in n:
                n["required"] = list(n["properties"].keys())
                n["additionalProperties"] = False
            for v in n.values():
                _t(v)
        elif isinstance(n, list):
            for x in n:
                _t(x)
    _t(schema)
    return schema

# Map LedgerRow field → LEFT_COLS column name (positional patch into cell_r{R}_c{C})
_FIELD_TO_COL = {
    "serial_no":          "Serial_No",
    "date":               "Date",
    "block_no":           "Property_recorded_under_Block_No",
    "parcel_no":          "Property_recorded_under_Parcel_No",
    "cat_no":             "Parcel_Cat_No",
    "area":               "Parcel_Area",
    "nature_of_entry":    "Nature_of_Entry",
    "new_serial_no":      "New_Serial_No",
    "ref_volume_no":      "Reference_to_Register_of_Changes_Volume_No",
    "ref_serial_no":      "Reference_to_Register_of_Changes_Serial_No",
    "tax_lp":             "Tax_LP",
    "tax_mils":           "Tax_Mils",
    "total_tax_lp":       "Total_Tax_LP",
    "total_tax_mils":     "Total_Tax_Mils",
    "exemption_entry_no": "Reference_to_Register_of_Exemptions_Entry_No",
    "exemption_amount_lp":"Reference_to_Register_of_Exemptions_Amount_LP",
    "exemption_amount_mils":"Reference_to_Register_of_Exemptions_Amount_Mils",
    "net_assessment_lp":  "Net_Assessment_LP",
    "net_assessment_mils":"Net_Assessment_Mils",
    "remarks":            "Remarks",
}

def _to_sinai_row(row: LedgerRow) -> dict[str, str]:
    d = {col: (getattr(row, f) or "") for f, col in _FIELD_TO_COL.items()}
    if row.reviewer_note and row.confidence != "high":
        d["Remarks"] = (f"[reviewer: {row.reviewer_note}] " + d.get("Remarks", "")).strip()
    return d

print(f"schema ready: {len(LedgerRow.model_fields)} fields per row, {len(LEFT_COLS)} XML columns")
''')

# ────────────────────────────────────────────────────────────────
# Cell 5 — SYSTEM_INSTRUCTION + PROMPT (with v5 reference conventions)
# ────────────────────────────────────────────────────────────────
code(r'''SYSTEM_INSTRUCTION = """<role>
You transcribe rows from British Mandate-era rural land tax registers (Form TR/39) for the village of Hadita (الحديثة). Column headers are pre-printed English; entries are handwritten Arabic with Eastern Arabic numerals (٠١٢٣٤٥٦٧٨٩).

You may receive ONE or TWO images per page:
  IMAGE 1 (always present, "PROCESSED"): the dewarped LEFT TABLE only, already cropped — this is the canonical grid you transcribe from.
  IMAGE 2 (optional, "RAW"): the original photograph, NOT dewarped. It contains both the left and right pages side-by-side. Use it ONLY as a recovery reference for top/bottom rows the dewarp pipeline clipped or smeared on IMAGE 1. NEVER transcribe content from the right page of the raw scan — only the left page exists in our schema.

If only IMAGE 1 is provided, treat it as the sole source.
</role>

<columns>
The LEFT TABLE has 20 columns. In order (right-to-left reading, the model schema preserves this):
   1. Serial_No
   2. Date                    ← year-like (e.g. ٩٣٨), often ditto. NOT Block_No.
   3. Property_recorded_under_Block_No   (4-digit, ٤١٣٢–٤١٥٢)
   4. Property_recorded_under_Parcel_No
   5. Parcel_Cat_No           (cultivation category, usually two digits)
   6. Parcel_Area             (Eastern digits; thousands sep → ASCII comma)
   7. Nature_of_Entry
   8. New_Serial_No
   9. Reference_to_Register_of_Changes_Volume_No
  10. Reference_to_Register_of_Changes_Serial_No
  11. Tax_LP        12. Tax_Mils        13. Total_Tax_LP    14. Total_Tax_Mils
  15. Reference_to_Register_of_Exemptions_Entry_No
  16. Exemption Amount_LP    17. Exemption Amount_Mils
  18. Net_Assessment_LP      19. Net_Assessment_Mils
  20. Remarks (free-text annotation)

COLUMN POSITION IS FIXED. Never shift a value leftward to fill an empty cell. Count columns from the right edge of the table when in doubt.
</columns>

<core_rules>
1. Preserve Arabic script and Eastern Arabic numerals (٠١٢٣٤٥٦٧٨٩) exactly as written. Standardize thousands separator to ASCII comma (١٬٢٠٠ → ١,٢٠٠). Western (0-9) and Eastern (٠-٩) both appear — never convert.
2. Empty cells → "" (empty string). Never invent values.
3. ONE value per cell. NEVER output compound readings like "X / Y", "٩٤٢ / ٩٤٤", or "٤٢ / ٩٤٠". If two readings seem plausible, pick the more legible and append [?].
4. NEVER omit a row. If a row is entirely blank, output it with all fields "" and confidence "low". Omitting rows breaks the positional patch into the PAGE XML.
5. Uncertainty markers (append to the cell value):
   ~~old~~          strikethrough only
   ~~old~~ new      visible correction
   [?]              uncertain read
   [RED]            written in red ink
6. For bilingual cells the ARABIC is canonical — transcribe what you see; never align Arabic to a presumed English value.
</core_rules>

<numerals>
Distinguish similar handwritten Eastern Arabic digits carefully:
  ٢ (2)  one small angular hook/curve, compact
  ٣ (3)  two scallops/bumps, wider/more open
  ٤ (4)  open hook facing right
  ٦ (6)  small closed circle/loop
  ٨ (8)  similar to ٦ but with a vertical extension or wider top — easy to misread as ٦
  ٠ (0)  small dot, usually smaller than ٦

Documented hard cases on this scribe (zoom in via code_execution before committing):
  - ٢/٣ confusion in the leading digit of `area`, `block_no`, `parcel_no`
  - ٣/٤ and ٤/٦ confusion in `tax_mils` and `total_tax_mils`
  - ٦/٨ confusion in the middle digit of `tax_mils` (e.g. ٠٦٩ vs ٠٨٩)
  - The whole 3-digit `tax_mils` cell can read backwards (١٠٧ vs ٠٧١) when faint — read left-to-right; do NOT permute digits.
</numerals>

<reference_column_conventions>
The two "Reference to Register of Changes" columns (`ref_volume_no` and `ref_serial_no`) have specific conventions on this register — these are common error sources:

1. **Preserve leading zeros in `ref_serial_no`.** Cells are typically 3-digit Eastern Arabic numbers WITH a leading zero: ٠٢٤, ٠١٦, ٠٩٠, ٠٠٨, ٠٥٦. Do NOT strip the leading 0 — output ٠٢٤ (three chars), not ٢٤ (two chars).

2. **"No data" placeholder is literal "..." (three ASCII dots), not "---" or "—".** When you see two or three dots in a Reference cell, output exactly: ...  (Use ".." only if there are clearly only two dots written.)

3. **T.D.L / T.P.L / D.L belongs in `ref_volume_no`; the paired year goes in `ref_serial_no`.** When you see "T.D.L 1940" written across the two cells, output:
     ref_volume_no: "T.D.L"
     ref_serial_no: "1940"
   Do NOT merge them into one cell as "T.D.L 1940".

4. **Distinguish T.D.L from T.P.L carefully.** D and P are easy to confuse; zoom in if the letter shape is ambiguous.

5. **`ref_volume_no` is usually a checkmark ✓** when `ref_serial_no` is a numeric serial — that signals "yes, an entry exists in the change register." Don't leave it empty if a checkmark is drawn.
</reference_column_conventions>

<register_priors>
Hadita register (village الحديثة). Use these ONLY to break a genuine tie, NEVER to override a digit you can read:
  Date: years ٩٣٨–٩٤٩ (1938–1949).
  Block_No: ٤١٣٢–٤١٥٢ (4132–4152), a 4-digit number.
</register_priors>

<symbols>
  ✓   Checkmark (U+2713): yes / confirmed / assessed.
  "   Ditto mark (two short parallel vertical ticks, double-comma, or curly quotes): repeat the value from the row above. Output exactly: " (ASCII U+0022).
  -   Horizontal dash (U+002D): nil / exempted / zero.
  ... or ..   Three or two dots: not applicable / no data.
  T.D.L / T.P.L / D.L   English abbreviations: preserve verbatim, keep in ref_volume_no.

Shape guide for the easy-to-confuse trio:
  ditto " = two vertical strokes
  ✓       = single angled / curved stroke
  ١       = single straight vertical stroke
</symbols>

<nature_of_entry>
Vocabulary on this register (transcribe Arabic exactly — DO NOT translate, gloss, or interpret):
  تح          (abbreviation for تحديث, "update")
  تسلل        (infiltration)
  من تسلل
  شراء / شرائي (purchase)
  بيع         (sale)
  ضريبة حرب    (war tax)

Combinations: Arabic text + checkmark → "تح ✓".
Blank → "". Ditto → ". Checkmark only → ✓.

DO NOT output English words like "transfer", "assessed", "confirmed ditto", "Ditto", or "checked" in this field. Raw Arabic + symbols only.
</nature_of_entry>

<tax_lp_rule>
Tax_LP, Total_Tax_LP, Exemption Amount_LP, Net_Assessment_LP are USUALLY NOT NUMBERS because most assessments are < 1 LP:
  ✓     assessed
  -     nil / exempted
  Eastern digit(s)   only when value ≥ 1 LP (rare)
  ""    genuinely blank
NEVER copy the Mils value into the LP column. NEVER leave the LP column "" when ✓ or - is drawn.
</tax_lp_rule>

<multi_category_parcels>
A single physical parcel may span MULTIPLE CONSECUTIVE numbered rows — one per cultivation category. Each row has its OWN Serial_No (consecutive). Block_No and Parcel_No are EMPTY on continuation rows — output ""; do NOT fabricate, do NOT ditto. Other columns (Date, Nature_of_Entry, New_Serial_No) vary case by case.
</multi_category_parcels>

<unnumbered_rows>
Some rows legitimately have NO Serial_No but contain content (tax-year breakdown, sub-total, carry-forward). Set Serial_No:"" and fill only the cells that have content. NEVER merge with neighbours; NEVER skip.

COLUMN POSITION IS FIXED — if a serial-less row contains a year, it belongs in Date, NOT in Serial_No. Tax/Total figures stay in Tax_LP / Tax_Mils / Total_Tax_LP / Total_Tax_Mils.

Example tax-year breakdown row:
  serial_no:"", date:"٩٣٩",
  block_no:"", parcel_no:"", cat_no:"", area:"",
  nature_of_entry:"", new_serial_no:"",
  ref_volume_no:"", ref_serial_no:"",
  tax_lp:"-", tax_mils:"٠٢٢", total_tax_lp:"-", total_tax_mils:"٠٢٢",
  exemption_entry_no:"", exemption_amount_lp:"", exemption_amount_mils:"",
  net_assessment_lp:"", net_assessment_mils:"",
  remarks:"", confidence:"high"
</unnumbered_rows>

<resuming_numbered_rows>
After a sequence of unnumbered breakdown rows (e.g. ٩٣٩، ٩٤٠، ٩٤١، ٩٤٢، ٩٤٣), a row with a fresh handwritten serial number (e.g. ٧, following the earlier 1–6 sequence) is a NUMBERED data row — assign that Serial_No with its own Block/Parcel/Cat/Area. Subsequent rows continue (٨، ٩، ١٠، ١١، …). A resumed row often has a fresh Date (e.g. ٩٤٤, ٩٤٨) and a distinctive Nature_of_Entry like ضريبة حرب or بيع.
</resuming_numbered_rows>

<dewarp_damage>
The dewarp pipeline that produced IMAGE 1 occasionally CROPS or SMEARS the FIRST and/or LAST row of the table. Symptoms in IMAGE 1: the top row is cut horizontally so only the bottom half of the digits is visible; or the bottom row is missing; or a row near the top/bottom appears doubled, smudged, or warped beyond reading.

When you detect any of those symptoms — AND only then — consult IMAGE 2 (the RAW scan):

  1. The left page in IMAGE 2 is the same physical page as IMAGE 1, BEFORE dewarp. Locate the row in IMAGE 2 by matching column positions and surrounding row content.
  2. Read the damaged row from IMAGE 2. Use the same column-by-column transcription rules from <core_rules>.
  3. Emit the row in the SectionRows output at its correct top-to-bottom position. Set confidence to "medium" and add a reviewer_note like "row recovered from raw scan due to dewarp top-crop".
  4. NEVER use IMAGE 2 to override a cell that IMAGE 1 already shows clearly. The grid, column positions, and middle-of-the-page rows always come from IMAGE 1.
  5. NEVER transcribe anything from the right page of IMAGE 2. The right page belongs to a different document section; mixing rows from it will corrupt the output.

If no top/bottom row damage is visible on IMAGE 1, ignore IMAGE 2 entirely.
</dewarp_damage>

<agentic_vision>
You have a Python sandbox (code_execution). USE IT actively wherever a cell is faint, small, or ambiguous — especially:
  - distinguishing ٢ from ٣ in the leading digit of `area`, `block_no`, `parcel_no`
  - distinguishing ٤ from ٦ and ٦ from ٨ in `tax_mils` and `total_tax_mils`
  - telling a ditto " from a checkmark ✓ (shape guide above)
  - reading the resumed-numbering serial in the bottom block
  - verifying T.D.L vs T.P.L
  - confirming a leading 0 is present in 3-digit reference serials

Crop the region at full resolution, examine it, then commit a reading. When code_execution was used on a row, set that row's confidence to "medium" (or "low") and add a brief reviewer_note.
</agentic_vision>
"""

PROMPT = (
    "<task>\n"
    "IMAGE 1 (PROCESSED) is a full LEFT TABLE page from the Hadita tax register, dewarped and cropped. "
    "The printed English column-header band is at the top; handwritten data rows fill the rest. "
    "Transcribe EVERY row of IMAGE 1 top to bottom — numbered data rows, multi-category continuations, "
    "tax-year breakdown rows, sub-totals, and any resumed-numbering block at the bottom. "
    "Map each handwritten cell to its column by horizontal position; column meanings are in the schema. "
    "Return SectionRows with one LedgerRow per physical row. NEVER omit a row.\n"
    "\n"
    "If IMAGE 2 (RAW, the original photograph including both pages) is also provided, follow the "
    "<dewarp_damage> rules: consult it ONLY when IMAGE 1's top or bottom row is clipped or smeared, "
    "and recover the affected row(s) from the LEFT page of IMAGE 2 only. Otherwise ignore IMAGE 2.\n"
    "</task>"
)

print(f"system_instruction: {len(SYSTEM_INSTRUCTION):,} chars  ·  prompt: {len(PROMPT):,} chars")
''')

# ────────────────────────────────────────────────────────────────
# Cell 6 — helpers (extraction, heartbeat, retry, cost, XML patch)
# ────────────────────────────────────────────────────────────────
code(r'''import time, csv, json, threading, re
from pathlib import Path
from pydantic import ValidationError

# ---------- Pricing ----------
PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "cached": 0.05},
    "gemini-3.5-flash":       {"input": 1.50, "output": 9.00, "cached": 0.15},
}

def _estimate_cost(model: str, ud: dict) -> float:
    p = PRICING.get(model)
    if not p:
        return float("nan")
    inp = ud.get("total_input_tokens") or 0
    out = ud.get("total_output_tokens") or 0
    tho = ud.get("total_thought_tokens") or 0
    cac = ud.get("total_cached_tokens") or 0
    billable_input = max(inp - cac, 0)
    return (billable_input * p["input"] + cac * p["cached"] + (out + tho) * p["output"]) / 1e6

# ---------- SDK-shape-robust output extraction ----------
def _interaction_outputs(interaction):
    """Return the list of step/output items across SDK shapes (new: .outputs, old: .steps)."""
    return getattr(interaction, "outputs", None) or getattr(interaction, "steps", None) or []

def _extract_final_text(interaction) -> str | None:
    """Walk the outputs list and return the last JSON-shaped text we find."""
    # Modern shortcut
    out = getattr(interaction, "output_text", None)
    if out and out.strip().startswith(("{", "[")):
        return out
    for o in reversed(_interaction_outputs(interaction)):
        t = getattr(o, "text", None)
        if t and t.strip().startswith(("{", "[")):
            return t
        c = getattr(o, "content", None)
        if isinstance(c, str) and c.strip().startswith(("{", "[")):
            return c
        if isinstance(c, list):
            for blk in c:
                bt = getattr(blk, "text", None)
                if bt and bt.strip().startswith(("{", "[")):
                    return bt
    return None

def _count_zoom_rounds(interaction) -> int:
    return sum(1 for o in _interaction_outputs(interaction)
               if getattr(o, "type", "") == "code_execution_call")

# ---------- One agentic call, with heartbeat ----------
# v6: accepts a LIST of image URIs (processed first, then optional raw).
# Each image is sent as its own content block with a short label so the model
# can resolve "IMAGE 1" / "IMAGE 2" references in the prompt.
def _run_once(image_uris: list[str], resolution: str, max_output_tokens: int):
    t0 = time.perf_counter()
    content = []
    labels = ["PROCESSED (IMAGE 1)", "RAW (IMAGE 2)"]
    for i, uri in enumerate(image_uris):
        tag = labels[i] if i < len(labels) else f"EXTRA (IMAGE {i+1})"
        content.append({"type": "text", "text": f"=== {tag} ==="})
        content.append({"type": "image", "uri": uri, "mime_type": "image/jpeg", "resolution": resolution})
    content.append({"type": "text", "text": PROMPT})
    interaction = client.interactions.create(
        model=MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        input=[{"type": "user_input", "content": content}],
        tools=[{"type": "code_execution"}],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": _strict_schema(SectionRows),
        },
        generation_config={
            "thinking_level": THINKING,
            "thinking_summaries": "auto",
            "max_output_tokens": max_output_tokens,
        },
    )
    return interaction, round(time.perf_counter() - t0, 2)

def _run_with_heartbeat(image_uris: list[str], resolution: str, max_output_tokens: int, label: str = ""):
    """Run on a background thread; print 'elapsed' every 30s so you can tell
    'still running' apart from 'frozen'. Heartbeat — not an ETA."""
    done = threading.Event()
    result = {}
    def _worker():
        try:
            result["interaction"], result["elapsed"] = _run_once(image_uris, resolution, max_output_tokens)
        except Exception as e:
            result["error"] = e
        finally:
            done.set()
    threading.Thread(target=_worker, daemon=True).start()
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}running with {len(image_uris)} image(s) (resolution={resolution!r}, thinking={THINKING!r}) — heartbeat every 30s …", flush=True)
    t0 = time.perf_counter()
    nudge_at = {600, 900}    # 10 min, 15 min
    nudged = set()
    while not done.wait(timeout=30):
        e = int(time.perf_counter() - t0)
        bar = f"{prefix}… {e//60}m{e%60:02d}s elapsed"
        for threshold in nudge_at:
            if e >= threshold and threshold not in nudged:
                bar += f"   ⚠ past {threshold//60} min — most calls finish under 5 min on this config. Consider Stop + lowering THINKING."
                nudged.add(threshold)
                break
        print(bar, flush=True)
    if "error" in result:
        raise result["error"]
    return result["interaction"], result["elapsed"]

# ---------- Adaptive retry loop ----------
# Two failure modes need different remedies:
#   (A) "exceeds the maximum allowed size limit"   → lower resolution
#   (B) ValidationError "EOF while parsing"        → raise max_output_tokens
RES_DOWNGRADE = {"ultra_high": "high", "high": "medium", "medium": "low"}

def run_one_page(image_uris, label: str = "") -> tuple[list[dict], dict]:
    """Returns (rows, meta). meta has the post-retry settings + usage + elapsed + zoom_rounds + cost.

    v6: image_uris is a list — first entry is the processed image (always present),
    second entry (optional) is the raw scan for top/bottom-row recovery. Backwards
    compatible: a single string is wrapped in a list automatically.
    """
    if isinstance(image_uris, str):
        image_uris = [image_uris]
    res, max_out = RESOLUTION, MAX_OUTPUT_TOKENS
    interaction = elapsed = section = None
    for attempt in range(1, 4):
        try:
            interaction, elapsed = _run_with_heartbeat(image_uris, res, max_out, label=label)
        except Exception as e:
            msg = str(e)
            if "exceeds the maximum allowed size" in msg or "image_size" in msg:
                new_res = RES_DOWNGRADE.get(res, "low")
                print(f"  ⚠ (A) backend size-cap at resolution={res!r}. retry at resolution={new_res!r} [attempt {attempt+1}/3]")
                res = new_res
                continue
            raise
        final_text = _extract_final_text(interaction)
        if not final_text:
            # No usable JSON — dump diagnostic
            print(f"  ⚠ could not extract final text. Interaction has these output types: "
                  f"{[getattr(o,'type','?') for o in _interaction_outputs(interaction)]}")
            raise RuntimeError(f"No JSON output recovered for {label or 'page'}.")
        try:
            section = SectionRows.model_validate_json(final_text)
            break
        except ValidationError as ve:
            if "EOF while parsing" in str(ve) or "json_invalid" in str(ve):
                new_max = min(max_out * 2, 65536)
                print(f"  ⚠ (B) JSON truncated at max_output_tokens={max_out} "
                      f"(got {len(final_text):,} chars). retry at max_output_tokens={new_max} [attempt {attempt+1}/3]")
                max_out = new_max
                continue
            raise
    else:
        raise RuntimeError(f"3 attempts exhausted on {label or 'page'} (res={res!r}, max_out={max_out}).")

    rows = [_to_sinai_row(r) for r in section.rows]
    u = getattr(interaction, "usage", None)
    ud = u.model_dump() if hasattr(u, "model_dump") else (u or {})
    meta = {
        "resolution_used": res,
        "max_output_used": max_out,
        "elapsed_s": elapsed,
        "zoom_rounds": _count_zoom_rounds(interaction),
        "usage": ud,
        "cost_usd": _estimate_cost(MODEL, ud),
    }
    return rows, meta

# ---------- PAGE XML patcher (matches patch_xml_text.py convention) ----------
def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def patch_xml(xml_str: str, text_rows: list[dict]) -> tuple[str, int, int]:
    """Overwrite the FIRST <Unicode>…</Unicode> inside each cell_r{R}_c{C} block.
    Cells the model didn't fill (or rows past end-of-output) are overwritten with ""."""
    n_patched, n_skipped = 0, 0
    for r_idx, row in enumerate(text_rows):
        for c_idx, col in enumerate(LEFT_COLS):
            text = (row.get(col, "") or "").strip()
            cell_id = f'id="cell_r{r_idx}_c{c_idx}"'
            cs = xml_str.find(cell_id)
            if cs == -1:
                n_skipped += 1
                continue
            nxt = xml_str.find("<TableCell", cs + len(cell_id))
            end = nxt if nxt != -1 else len(xml_str)
            us = xml_str.find("<Unicode>",  cs, end)
            ue = xml_str.find("</Unicode>", cs, end)
            if us == -1 or ue == -1:
                n_skipped += 1
                continue
            content_start = us + len("<Unicode>")
            xml_str = xml_str[:content_start] + _escape_xml(text) + xml_str[ue:]
            n_patched += 1
    return xml_str, n_patched, n_skipped

def xml_grid_size(xml_str: str) -> tuple[int, int]:
    cells = set(re.findall(r'cell_r(\d+)_c(\d+)', xml_str))
    if not cells:
        return 0, 0
    return max(int(r) for r,_ in cells) + 1, max(int(c) for _,c in cells) + 1

print("helpers ready: run_one_page() · patch_xml() · _extract_final_text() · _estimate_cost()")
''')

# ────────────────────────────────────────────────────────────────
# Cell 7 — main loop: process every page in PAGES
# ────────────────────────────────────────────────────────────────
code(r'''# Process every page in PAGES sequentially. For each: upload image, run agentic OCR,
# patch the XML, save outputs, append to g3_runs_v6.csv.
import json
from pathlib import Path
from IPython.display import display, Markdown

RUNS_CSV = "g3_runs_v6.csv"
results = {}            # page -> {rows, json_path, xml_path, meta, n_patched, n_skipped}
total_cost = 0.0
total_elapsed = 0.0

# Upload-once cache so re-running this cell after edits doesn't double-upload
if "_UPLOAD_CACHE" not in globals():
    _UPLOAD_CACHE = {}

def _upload_cached(path: str) -> str:
    key = (path, Path(path).stat().st_mtime)
    if key not in _UPLOAD_CACHE:
        _UPLOAD_CACHE[key] = client.files.upload(file=path, config={"mime_type": "image/jpeg"}).uri
        print(f"  uploaded {path} → {_UPLOAD_CACHE[key]}")
    else:
        print(f"  reusing upload {path}")
    return _UPLOAD_CACHE[key]

for page in PAGES:
    print(f"\n{'═'*72}")
    print(f"PAGE {page}")
    print('═'*72)
    if page not in IMAGES_PROCESSED or page not in XMLS:
        miss = []
        if page not in IMAGES_PROCESSED: miss.append("processed image")
        if page not in XMLS: miss.append("XML")
        print(f"  ⚠ skipped — missing {' & '.join(miss)}.")
        continue

    proc_path = IMAGES_PROCESSED[page]
    raw_path  = IMAGES_RAW.get(page)
    xml_path  = XMLS[page]

    image_uris = [_upload_cached(proc_path)]
    if raw_path:
        image_uris.append(_upload_cached(raw_path))
        mode_note = "v6 (processed + raw)"
    else:
        mode_note = "v5-fallback (processed only — no raw supplied)"
    print(f"  mode: {mode_note}")

    try:
        rows, meta = run_one_page(image_uris, label=f"page {page}")
    except Exception as e:
        print(f"  ✖ failed: {type(e).__name__}: {e}")
        continue

    # Patch the XML
    src_xml = Path(xml_path).read_text(encoding="utf-8")
    patched, n_patched, n_skipped = patch_xml(src_xml, rows)
    out_xml  = f"Hadita_{page}_G3v6.xml"
    out_json = f"Hadita_{page}_G3v6.json"
    Path(out_xml).write_text(patched, encoding="utf-8")
    Path(out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    # Diagnostic: row-count vs XML grid
    xr, xc = xml_grid_size(src_xml)
    note = ""
    if len(rows) > xr:
        note = f"   ⚠ model emitted {len(rows) - xr} more row(s) than the XML has slots — those won't be patched"
    elif len(rows) < xr:
        note = f"   ℹ model emitted {xr - len(rows)} fewer row(s) — bottom XML rows overwritten with empty"

    # Append to cumulative cost CSV
    ud = meta["usage"]
    is_new = not Path(RUNS_CSV).exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["page","model","thinking","resolution","max_output_tok","rows","zoom_rounds",
                        "elapsed_s","input_tok","cached_tok","output_tok","thought_tok",
                        "tool_use_tok","total_tok","cost_usd"])
        w.writerow([page, MODEL, THINKING, meta["resolution_used"], meta["max_output_used"],
                    len(rows), meta["zoom_rounds"], meta["elapsed_s"],
                    ud.get("total_input_tokens") or 0, ud.get("total_cached_tokens") or 0,
                    ud.get("total_output_tokens") or 0, ud.get("total_thought_tokens") or 0,
                    ud.get("total_tool_use_tokens") or 0, ud.get("total_tokens") or 0,
                    round(meta["cost_usd"], 6)])

    total_cost += meta["cost_usd"]
    total_elapsed += meta["elapsed_s"]
    results[page] = {"rows": len(rows), "json": out_json, "xml": out_xml,
                     "meta": meta, "n_patched": n_patched, "n_skipped": n_skipped}
    print(f"  → {len(rows)} rows · {meta['zoom_rounds']} zoom rounds · {meta['elapsed_s']}s · ${meta['cost_usd']:.4f}")
    print(f"     wrote {out_json} and {out_xml}   (patched {n_patched} cells; XML grid {xr}×{xc}){note}")

# Session summary
print(f"\n{'═'*72}")
print(f"SESSION SUMMARY — {len(results)} pages processed")
print('═'*72)
print(f"  total cost      : ${total_cost:.4f}")
print(f"  total elapsed   : {int(total_elapsed//60)}m {int(total_elapsed%60)}s")
if results:
    print(f"  avg cost / page : ${total_cost/len(results):.4f}")
    print(f"  projection for 100 pages at this rate: ${total_cost/len(results)*100:.2f}")
''')

# ────────────────────────────────────────────────────────────────
# Cell 8 — download all outputs
# ────────────────────────────────────────────────────────────────
code(r'''from google.colab import files
from pathlib import Path

for page in sorted(results):
    files.download(results[page]["xml"])
    files.download(results[page]["json"])

# also download the cumulative cost log
try:
    files.download(RUNS_CSV)
except Exception:
    pass

print(f"Downloaded outputs for {len(results)} page(s) + g3_runs_v6.csv")
print("Drop the Hadita_{N}_G3.xml files into Transkribus upload/final/ (or upload directly to Transkribus)")
print("for RA correction. The existing FINAL/DONE transcripts are kept by Transkribus version history.")
''')

# ────────────────────────────────────────────────────────────────
# Cell 9 — markdown for cost projection cell
# ────────────────────────────────────────────────────────────────
md(r"""## Cost summary + projection (cumulative)

Run this cell any time. It reads `g3_runs_v6.csv` — which now includes every page you've
processed in this session AND any earlier sessions (if you uploaded an older
`g3_runs_v6.csv` at the start of this session, it appends rather than overwrites).
""")

# ────────────────────────────────────────────────────────────────
# Cell 10 — cost summary code
# ────────────────────────────────────────────────────────────────
code(r'''import csv
from pathlib import Path

RUNS_CSV = "g3_runs_v6.csv"
if not Path(RUNS_CSV).exists():
    print(f"{RUNS_CSV} not found — process at least one page first.")
else:
    rows_log = list(csv.DictReader(open(RUNS_CSV)))
    if not rows_log:
        print(f"{RUNS_CSV} is empty.")
    else:
        print(f"{'page':>4} {'thinking':>8} {'rows':>4} {'zoom':>4} {'elapsed':>8} {'in_tok':>7} {'out_tok':>7} {'tho_tok':>7} {'tool_tok':>8} {'cost':>8}")
        for r in rows_log:
            print(f"{r['page']:>4} {r['thinking']:>8} {r['rows']:>4} {r['zoom_rounds']:>4} {float(r['elapsed_s']):>7.1f}s "
                  f"{int(r['input_tok']):>7,} {int(r['output_tok']):>7,} {int(r['thought_tok']):>7,} "
                  f"{int(r['tool_use_tok']):>8,} ${float(r['cost_usd']):>6.4f}")

        costs = [float(r["cost_usd"]) for r in rows_log]
        elapsed = [float(r["elapsed_s"]) for r in rows_log]
        n = len(costs)
        total = sum(costs); avg = total / n; med = sorted(costs)[n // 2]
        cmin, cmax = min(costs), max(costs)
        avg_s = sum(elapsed) / n

        print()
        print(f"Across {n} run(s):")
        print(f"  total cost so far : ${total:.4f}")
        print(f"  avg  per page     : ${avg:.4f}   (median ${med:.4f}, min ${cmin:.4f}, max ${cmax:.4f})")
        print(f"  avg  elapsed      : {avg_s:.1f}s / page")
        print()
        print(f"Projection for the next 100 pages at current rate:")
        print(f"  min  est: ${cmin*100:>6.2f}")
        print(f"  median  : ${med *100:>6.2f}")
        print(f"  mean    : ${avg *100:>6.2f}")
        print(f"  max  est: ${cmax*100:>6.2f}")
        print(f"  wall time at median elapsed (sequential): ~{avg_s*100/60:.0f} minutes")
''')


# ────────────────────────────────────────────────────────────────
# Assemble notebook
# ────────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("Hadita_Gemini3_to_PAGEXML_v6.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
