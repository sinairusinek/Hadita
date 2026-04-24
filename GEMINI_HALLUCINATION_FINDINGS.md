# Gemini Hallucination Analysis — Pages 10 & 50

## Summary
Approach M (Gemini-generated OCR cache) exhibits clear hallucination patterns where Block_No/Parcel_No data from early rows reappears identically in later rows. This is impossible in a tax register where each entry has unique identifiers.

## Page 50 — Clear Hallucination

### Repeated Block_No/Parcel_No Pairs

| Pair | First Appearance | Repeated At | Status |
|------|------------------|-------------|--------|
| `٤١٤٢/٢` | Row 0 (Serial 1) | Row 16 (Serial 16) | ❌ Hallucinated |
| `٤١٤١/١٣` | Row 3 (Serial 4) | Row 18 (Serial 18) | ❌ Hallucinated |
| `٤١٤٨/٢` | Row 5 (Serial 6) | Row 20 (Serial 19) | ❌ Hallucinated |

### Repeating Ditto Marks
- `"/١٠` appears at rows **1, 7, 9** (should only appear once if ditto intended)
- `"/١٦` appears at rows **2, 10** (same issue)

### Interpretation
The model learned the pattern of ditto marks ("=repeat previous cell") from the first few rows and then replicated those same numeric values in later rows rather than recognizing new unique data.

---

## Page 10 — Degraded Output

- **Rows 14 & 16**: Identical empty data (should be distinct entries)
- **Rows 15 & 17**: Completely empty (duplicate placeholders)
- **Pattern breakdown**: Series of 6 mostly-empty rows at the end (rows 11-17)
- **Data loss**: Only 5 valid entries at start (rows 0-5), then Serial_No jumps to 939, 941, 943-947 with no associated data

### Interpretation
As the model progresses through the page, it loses coherence and resorts to repeating empty entries instead of generating new data.

---

## Root Cause
**Generative hallucination via pattern repetition**: Gemini's Approach M was trained to complete sequences, not to understand register structure. When encountering unfamiliar data or reaching uncertainty, it reverted to copying learned patterns (early rows) rather than inferring new unique entries.

## Implications
1. **Cannot rely on Approach M for pages 10 & 50** without deduplication/validation
2. **Ground truth training is essential** — fine-tuning a model on actual register data will teach it the structural uniqueness constraint
3. **Validation filter needed** — detect when later rows repeat earlier row data signatures

## Next Steps
- [ ] Implement duplicate detection in OCR cache loading (flag/remove repeated Block_No/Parcel_No pairs from same page)
- [ ] Use Kraken + morphological fallback for row segmentation (independent of hallucinated text)
- [ ] Create GT via Transkribus for pages 10 & 50
- [ ] Train/fine-tune OCR model on GT data
