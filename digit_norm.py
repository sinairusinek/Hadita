"""Shared digit/cell normalization for Hadita scoring + agreement layer.

Column-aware rule (2026-06-26 re-push):
  - `New_Serial_No` (c7): preserve Western digits as-is.
  - Cells containing T.D.L/T.P.L/D.L: preserve Western digits as-is.
  - Year cell paired with a T.D.L/T.P.L/D.L Vol_No: preserve Western digits.
  - Everywhere else: Western (0-9) -> Eastern Arabic (٠-٩).

Regex pitfall: Python `\\d` matches Eastern Arabic digits under Unicode.
Use `HAS_W` (ASCII-only `[0-9]`) for any "does this cell contain Western
digits?" audit.
"""
import re
import unicodedata

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

EASTERN = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
W2E = str.maketrans(WESTERN, EASTERN)
E2W = str.maketrans(EASTERN, WESTERN)

HAS_W = re.compile(r"[0-9]")
TDL_RE = re.compile(r"T\.?D\.?L|T\.?P\.?L|D\.?L", re.IGNORECASE)

WESTERN_PRESERVE_COLS = {"New_Serial_No"}
VOL_COL = "Reference_to_Register_of_Changes_Volume_No"
SERIAL_REF_COL = "Reference_to_Register_of_Changes_Serial_No"


def convert_cell(text: str, col: str, vol_text: str = "") -> str:
    """Apply the column-aware digit-normalization rule. Returns the cell text
    with Western digits either preserved or converted to Eastern."""
    if text is None:
        return ""
    if col in WESTERN_PRESERVE_COLS:
        return text
    if TDL_RE.search(text):
        return text
    if col == SERIAL_REF_COL and vol_text and TDL_RE.search(vol_text):
        return text
    return text.translate(W2E)


def normalize_for_compare(s: str) -> str:
    """Normalize a cell value for cross-source equality comparison.

    Strips agreement/red markers, harmonizes thousands separators, ditto family,
    no-data placeholders, and whitespace. Does NOT do digit conversion — that
    is the caller's responsibility via convert_cell so column context is kept.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).strip()
    for marker in ("[?]", "[RED]"):
        s = s.replace(marker, "")
    s = s.replace("،", ",").replace("٬", ",")
    for d in ("״", "“", "”", "„", ",,"):
        s = s.replace(d, '"')
    for nd in ("---", "--", "—", "–", ".."):
        if s.strip() == nd:
            s = "..."
    if s.strip() in ("–", "—"):
        s = "-"
    return " ".join(s.split())
