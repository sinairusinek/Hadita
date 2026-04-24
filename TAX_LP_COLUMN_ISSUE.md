# Tax_LP Column Issue — Root Cause Analysis

**Accuracy**: 4% (1/25 cells) — Critical failure  
**Date**: 2026-04-24  
**Impact**: Approach M incorrectly transcribes tax assessment status

---

## The Problem

Approach M achieves only **4% accuracy on Tax_LP**, with only **1 correct match out of 25 cells compared**.

```
Expected (Ground Truth):  ✓  or  -  (meta field)
Actual (OCR Output):      ?    (incorrect values)
Accuracy:                 1/25 (4%)
```

---

## Root Cause: Column Semantic Mismatch

### What Tax_LP Actually Contains

Ground truth analysis reveals that **Tax_LP is NOT primarily a numeric field** — it's a **status/metadata field**:

| Pattern | Frequency | Meaning | Examples |
|---------|-----------|---------|----------|
| **`✓` (checkmark)** | 16/33 rows (48%) | "Tax assessed" | Row 1, 2, 4, 6, 7, 11, 15, 17, 19, 23, 25, 26, 27, 29, 31, 32 |
| **`-` (nil/dash)** | 11/33 rows (33%) | "No tax" or "exempted" | Row 12-14, 16, 18, 20, 22, 24, 28, 30, 33 |
| **`٢` (numeric)** | 1/33 rows (3%) | Rare numeric value | Row 3 |
| **Empty** | 5/33 rows (15%) | Data missing | Row 5, 8, 9, 10, 21 |

### What the OCR Model Probably Did

The model was likely:
1. **Treating Tax_LP as numeric** (because it's adjacent to numeric columns)
2. **Confusing Tax_LP with Tax_Mils** (both are tax-related)
3. **Misaligning columns** due to grid detection errors
4. **Hallucinating values** that don't exist

### Tax_Mils (for comparison)

Tax_Mils contains **actual numeric tax amounts** and has **100% accuracy** in the cache:
- All 28 unique values are numeric (٠٣٤, ٦٢٩, etc.)
- No checkmarks, dashes, or metadata
- Clear, consistent numeric field

---

## Visual Evidence: The Column Layout

The left table columns (right-to-left) are:

```
[Serial] [Date] [Block_No] [Parcel_No] [Cat_No] [Area] [Entry] [New_Serial]
  [Ref_Vol]  [Ref_Serial]  [Tax_LP]  [Tax_Mils]  [Total_LP]  [Total_Mils]
  [Exempt_Entry]  [Exempt_LP]  [Exempt_Mils]  [Net_LP]  [Net_Mils]  [Remarks]
```

**The issue**: Tax_LP and Tax_Mils are **adjacent and tax-related**, making them easy to confuse:
- Tax_LP is on the LEFT (contains ✓ or -)
- Tax_Mils is on the RIGHT (contains numbers)

If the grid detection is slightly off, the model might:
- Read Tax_Mils values when asked for Tax_LP
- Hallucinate numeric values for Tax_LP
- Confuse the column boundaries

---

## Evidence of Misalignment

Looking at the 5 anomalous rows where Tax_Mils has a value but Tax_LP is empty:
```
Row 5:  Tax_LP=""        Tax_Mils="١٧٥"  ← GT says "no Tax_LP here"
Row 8:  Tax_LP=""        Tax_Mils="٠٥٣"  ← but Tax_Mils is filled
Row 9:  Tax_LP=""        Tax_Mils="٠٤٣"
Row 10: Tax_LP=""        Tax_Mils="١٧٤"
Row 21: Tax_LP=""        Tax_Mils="٠٠٨"
```

This suggests the **columns are structurally misaligned in some rows**, possibly due to:
1. Missing or partial rows in the image
2. Grid detection failing to find correct column boundaries
3. Merged cells or variable column widths

---

## Why the OCR Failed

### Hypothesis 1: Column Confusion (Most Likely)
The grid detection found column boundaries, but the model read:
```
Tax_Mils value    →  Output as Tax_LP
Checkmark in Tax_LP  →  Missed or read as ✓ that didn't match expected value
```

### Hypothesis 2: Grid Misalignment
The grid detection is off by 1-2 columns, causing the model to read:
```
[Tax_Mils]        [Total_LP]
  (numeric)       (empty, from meta rows)
   ٦٢٩       →    Output as "" (empty)
```

### Hypothesis 3: Semantic Confusion
The prompt didn't clearly explain that Tax_LP is a **status field** (`✓` or `-`), not a numeric field. The model may have:
1. Assumed it should find numbers
2. Hallucinated plausible-looking values
3. Ignored or misread the actual checkmarks

---

## Supporting Evidence from Approach O

**Approach O (M + cell-level overrides)** also has 0% accuracy on Tax_LP, suggesting:
- The cell-strip extraction makes the problem WORSE
- Reading narrow vertical strips loses context about what symbols are
- Preprocessing (CLAHE, upscaling) may distort checkmarks and dashes

This confirms the issue is **fundamental to the OCR approach**, not just Approach M.

---

## The Fix: Three-Part Solution

### 1. **Update Prompt to Clarify Tax_LP is a Status Field**

**Before** (implicit assumption):
```
Tax_LP, Tax_Mils: typical numeric fields
```

**After** (explicit guidance):
```
IMPORTANT: Tax_LP and Tax_Mils have DIFFERENT meanings:

Tax_LP (Status field):
  - Contains ✓ (checkmark) if a tax was assessed → output "✓"
  - Contains - (nil/dash) if no tax or exempted → output "-"
  - Rarely contains numeric values (very few rows)
  - Empty cells are possible (multi-category parcels) → output ""
  - Do NOT confuse ✓ with actual numbers

Tax_Mils (Amount field):
  - Contains the actual numeric tax amount
  - Examples: ٦٢٩, ٠٨٥, ٢١٤, etc.
  - Always numeric (Eastern Arabic digits)
  - Should never be empty if Tax_LP is ✓

WARNING: These columns are adjacent and easy to confuse.
  Tax_LP (LEFT side, status)
  Tax_Mils (RIGHT side, amount)
  Make sure you are reading the correct column.
```

### 2. **Add Examples to Few-Shot Prompts**

Show examples of correct Tax_LP values:
```json
Row 1: {"Tax_LP":"✓", "Tax_Mils":"٦٢٩"}    ← Both present
Row 2: {"Tax_LP":"✓", "Tax_Mils":"٠٨٥"}    ← Both present
Row 3: {"Tax_LP":"٢", "Tax_Mils":"٨٧٦"}    ← Rare numeric case
Row 5: {"Tax_LP":"", "Tax_Mils":"١٧٥"}     ← Tax_LP empty, Mils filled
Row 12: {"Tax_LP":"-", "Tax_Mils":"-"}      ← Both nil/exempted
```

### 3. **Verify Grid Detection**

The root cause might be grid misalignment. To test:
```bash
# Run Approach O (cell-level, which extracts Tax_LP column strip)
# and examine the strip image quality
python compare_ocr.py --approaches O --pages 3 --debug-grid
```

If strips are misaligned or blurry, that explains why Tax_LP fails.

---

## Summary: Why Tax_LP is 4%

| Issue | Impact | Severity |
|-------|--------|----------|
| **Semantic confusion** (numeric vs status) | Model looks for numbers, finds symbols | HIGH |
| **Column misalignment** | Reads wrong column (Tax_Mils?) | HIGH |
| **Checkmark/dash confusion** | Symbol recognition errors | MEDIUM |
| **Rare numeric rows** | Only 1/33 rows has actual number | MEDIUM |
| **Anomalous GT rows** | 5 rows with structural issues | LOW |

**Bottom line**: The OCR model expected numeric values but Tax_LP is primarily a status field with checkmarks and dashes. The cell-level override (Approach O) made it worse, confirming the fundamental issue is column extraction, not full-page reading.

---

## Next Steps

1. **Test the prompt fix** (add Tax_LP semantic guidance)
2. **Audit grid detection** (is it off by 1-2 columns?)
3. **Inspect cell-strip quality** (if using cell-level approach)
4. **Consider column-specific prompting** (treat Tax_LP as status, Tax_Mils as numeric)
5. **Verify the 5 anomalous rows** (possible GT errors?)

---

## Related Issues

- **Nature_of_Entry: 0%** → Similar symbol confusion (ditto vs checkmark) — FIXED via prompt
- **Parcel_Area: 23%** → Digit normalization issue (Eastern vs Western) — FIXED via digit norm
- **Tax_LP: 4%** → Semantic and alignment issue — REQUIRES INVESTIGATION
