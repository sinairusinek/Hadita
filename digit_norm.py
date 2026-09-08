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


# ---------------------------------------------------------------------------
# Encoding-tolerant normalisation (added 2026-09-07).
#
# The frozen metric in score_g3_vs_gt._normalize folds the variants that Gemini
# 3.7 happened to emit. Newer models pick different-but-equivalent codepoints for
# the SAME glyph, so the frozen scorer charges them for cosmetics and silently
# favours the incumbent. Measured across exp2608/: Extended Arabic-Indic digits
# (U+06Fx) in 10 tags, U+3003 DITTO MARK in 3, em-dash mid-string in 11.
#
# Deliberately NOT folded: ASCII digits and Latin letters. The GT genuinely
# contains "T.D.L. 1940" and ASCII serials like "102", so folding ASCII->Arabic
# would corrupt real content rather than unify an encoding choice.
# ---------------------------------------------------------------------------

# U+06F0..U+06F9 -> U+0660..U+0669. Same glyphs, different codepoint block.
_EXT_DIGITS = {chr(0x06F0 + i): chr(0x0660 + i) for i in range(10)}

# Ditto glyphs the frozen normaliser does not already cover.
_DITTO_EXTRA = {"〃": '"', "″": '"', "‟": '"', "〝": '"', "〞": '"', "«": '"', "»": '"'}

# Dash-family -> ASCII hyphen. The frozen rule only fires when the dash is the
# WHOLE cell; these appear inside longer strings too.
_DASHES = {"—": "-", "–": "-", "‐": "-", "‑": "-", "−": "-"}

# A cell that is nothing but tatweel is a written dash, not a letter-stretch.
_LONE_TATWEEL = "ـ"

# Orthographic equivalents seen in model output but never in the GT.
_LETTERS = {"ی": "ي", "ک": "ك"}

# Zero-width marks that carry no transcription meaning.
#
# NB: tatweel (U+0640) is NOT stripped. Models use a bare tatweel as the nil/dash
# mark, so stripping it empties the cell and the scorer then charges a full
# "wrong" (one side empty) instead of a 1-char fix. It is folded to "-" below.
_STRIP = ("‏", "‎", "​", "­")


def encoding_fold(s: str) -> str:
    """Fold codepoint choices that represent the SAME handwritten glyph.

    Apply BEFORE the frozen _normalize; never changes which glyph was read,
    only which codepoint spells it.
    """
    if not s:
        return ""
    for table in (_EXT_DIGITS, _DITTO_EXTRA, _DASHES, _LETTERS):
        for k, v in table.items():
            s = s.replace(k, v)
    for ch in _STRIP:
        s = s.replace(ch, "")
    if s.strip(_LONE_TATWEEL) == "" and s.strip():
        return "-"                      # a cell of only tatweel == a dash
    return s.replace(_LONE_TATWEEL, "")  # elsewhere it is decorative stretching
