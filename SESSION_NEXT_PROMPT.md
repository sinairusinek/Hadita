# Next Session Prompt: Full Approach Re-evaluation Against Updated GT

## Context

The ground truth (`ground_truth.tsv`) has been significantly improved and standardized:

**GT Updates (2026-04-24):**
- **Scope**: 65 rows across pages 3, 10, 50 (RA-verified)
- **Encoding fix**: All numeric columns standardized to Eastern Arabic digits (٠-٩)
- **Document fidelity**: T.D.L. reference numbers kept in Western digits (0-9) as they appear in the original
- **Recent corrections**: Digit encoding (938→٩٣٨, 4132→٤١٣٢), Nature_of_Entry clarifications

**Code Improvements (implemented, not yet tested):**
- **Approach O2**: New OCR approach extending Approach O with column-strip re-reading for Date, Block_No, Parcel_No (hallucination-prone columns)
- **Validation filter**: `validate_ocr_rows()` function detects and flags repeated Block_No/Parcel_No values from early rows

**Previous baseline (page 3 only):**
- Approach M (Gemini 2.5 Pro, full-page few-shot): **62.6% exact match**
- Approach O (M + cell-strip overrides for 4 columns): **64.9% exact match**

---

## Task for Next Session

**Re-evaluate all approaches (A–R) against the updated GT:**

1. **Run complete scoring pass:**
   ```bash
   python compare_ocr.py --score --approaches A B C D E F G H I J K L M N O P Q R --page 3
   python compare_ocr.py --score --approaches M O O2 --page 10
   python compare_ocr.py --score --approaches M O O2 --page 50
   ```

2. **Focus on:**
   - How much did the GT updates change Approach M's score on page 3? (Was it due to digit encoding or other corrections?)
   - Does Approach O2 show measurable improvement over O on page 3?
   - Are the hallucination patterns (repeated Block_No/Parcel_No) visible in O2's outputs for pages 10/50?
   - What's the validation filter's impact — how many values get flagged per page?

3. **Produce:**
   - Updated comparison report (`ocr_benchmark_report_v3.html` or similar)
   - Per-column breakdown for page 3 to see which columns benefited from GT corrections
   - Hallucination analysis for pages 10/50 (before/after validation filter)

4. **Decision points:**
   - Is O2 worth running in production for pages 10/50, or is the extra API cost not justified by the gain?
   - Should the validation filter be applied by default, or only for Approach M to catch obvious hallucinations?
   - Any GT corrections still needed after inspection?

---

## Files to Check/Use

- `compare_ocr.py`: Has Approach O2 and `validate_ocr_rows()` implemented (lines ~940, ~520)
- `ground_truth.tsv`: Updated and standardized, ready for scoring
- `segmentation_report.md`: Section 9 documents GT spec; Section 10 lists all files
- `GEMINI_HALLUCINATION_FINDINGS.md`: Background on the hallucination patterns O2 is designed to prevent

---

## Success Criteria

- All 18 approaches score cleanly (no errors)
- Approach O2 scores exist for pages 3, 10, 50
- Validation filter can be optionally applied and its impact measured
- Comparison metrics match or exceed previous baseline
- Hallucination patterns in pages 10/50 are documented (visible in raw output, flagged by validator)

---

Good luck! The codebase is ready to test.
