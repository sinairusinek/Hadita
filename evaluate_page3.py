#!/usr/bin/env python3
"""Evaluate OCR approaches against ground truth for page 3.

Improvements over original version:
- Resolves ditto marks before scoring (matches production pipeline behaviour)
- Excludes Hebrew researcher annotations (הערות column, Hebrew in Remarks)
"""

import csv
import difflib
import re
import unicodedata
from collections import defaultdict

# ── Eastern ↔ Western Arabic numeral mapping ──
EASTERN = "٠١٢٣٤٥٦٧٨٩"
WESTERN = "0123456789"
E2W = str.maketrans(EASTERN, WESTERN)
W2E = str.maketrans(WESTERN, EASTERN)

# Canonical ditto mark variants
DITTO_VARIANTS = {'״', '"', '〃', "''", ',,', '"', '\u05F4'}


def is_ditto(val: str) -> bool:
    """Check if a normalized value is a ditto mark."""
    return val in DITTO_VARIANTS or val == '"'


def normalize_for_compare(val: str) -> str:
    """Normalize a cell value for comparison: strip, lowercase, convert Eastern→Western numerals."""
    val = val.strip()
    # Convert Eastern Arabic numerals to Western for uniform comparison
    val = val.translate(E2W)
    # Normalize various ditto marks to a single canonical form
    if is_ditto(val):
        val = '"'
    # Normalize dash variants: '--', '—', '−' all mean nil/zero → canonical '-'
    if val in ('--', '—', '−', '- -'):
        val = '-'
    # Strip leading zeros from numeric values (085 → 85) but not from values like '0'
    if len(val) > 1 and val[0] == '0' and val[1:].isdigit():
        val = val.lstrip('0') or '0'
    # Strip trailing annotations like [RED], [?] for numeric comparison
    # but keep them for full comparison
    return val


def resolve_dittos(rows: list[dict], columns: list[str]) -> list[dict]:
    """Resolve ditto marks by replacing them with the value from the row above.

    This mirrors the production pipeline's ditto resolution so that
    evaluation reflects actual extraction quality, not raw output format.
    """
    resolved = []
    prev_vals: dict[str, str] = {}
    for row in rows:
        new_row = dict(row)
        for col in columns:
            val = normalize_for_compare(new_row.get(col, ""))
            if val == '"' and col in prev_vals:
                new_row[col] = prev_vals[col]
            else:
                prev_vals[col] = val
                new_row[col] = val
        resolved.append(new_row)
    return resolved


def has_hebrew(text: str) -> bool:
    """Check if text contains Hebrew characters (researcher annotations)."""
    return any('\u0590' <= c <= '\u05FF' for c in text)


def cer(pred: str, ref: str) -> float:
    """Character Error Rate."""
    if not ref:
        return 0.0 if not pred else 1.0
    ops = difflib.SequenceMatcher(None, pred, ref).get_opcodes()
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    return edits / len(ref)

# ── Columns to compare (left + right data columns, skip meta) ──
LEFT_COLS = [
    "Serial_No", "Date", "Block_No", "Parcel_No", "Cat_No", "Area",
    "Nature_of_Entry", "New_Serial_No", "Volume_No", "Serial_No_Vol",
    "Tax_LP", "Tax_Mils", "Total_Tax_LP", "Total_Tax_Mils",
    "Entry_No", "Remarks",
]
RIGHT_COLS = [
    "Assessment_Year", "Amount_Assessed_LP", "Amount_Assessed_Mils",
    "Date_of_Payment", "Receipt_No",
    "Amount_Paid_LP", "Amount_Paid_Mils",
    "Balance_LP", "Balance_Mils", "Right_Side_Notes",
]
DATA_COLS = LEFT_COLS + RIGHT_COLS

# ── Load ground truth for page 3 ──
gt_rows = []
with open("ground_truth.tsv", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        if r["Page_Number"] == "3" and r.get("Serial_No", "").strip():
            gt_rows.append(r)

print(f"Ground truth: {len(gt_rows)} rows for page 3\n")

# ── Load comparison results for page 3 ──
comp_rows = defaultdict(list)
with open("comparison_page3.csv", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    comp_cols = reader.fieldnames
    for r in reader:
        comp_rows[r["Approach"]].append(r)

for a in sorted(comp_rows):
    print(f"Approach {a}: {len(comp_rows[a])} rows")
print()

# ── Figure out which columns exist in both GT and comparison ──
# The comparison CSV may not have "Date" column (check)
gt_sample = gt_rows[0]
comp_sample = comp_rows[list(comp_rows.keys())[0]][0]
available_cols = [c for c in DATA_COLS if c in gt_sample and c in comp_sample]
print(f"Columns available in both GT and comparison: {len(available_cols)}")
missing_in_comp = [c for c in DATA_COLS if c in gt_sample and c not in comp_sample]
if missing_in_comp:
    print(f"  Missing in comparison: {missing_in_comp}")
missing_in_gt = [c for c in DATA_COLS if c not in gt_sample and c in comp_sample]
if missing_in_gt:
    print(f"  Missing in GT: {missing_in_gt}")
print()

# ── Evaluate each approach ──
# Align by Serial_No (row 1 = Serial_No 1, etc.)
# Build GT index by Serial_No
gt_by_sno = {}
for r in gt_rows:
    sno = normalize_for_compare(r.get("Serial_No", ""))
    gt_by_sno[sno] = r

# ── Resolve ditto marks in GT rows ──
gt_rows_resolved = resolve_dittos(gt_rows, available_cols)
gt_by_sno_resolved = {}
for r in gt_rows_resolved:
    sno = r.get("Serial_No", "").strip().translate(E2W)
    gt_by_sno_resolved[sno] = r

print("=" * 80)
print("CELL-LEVEL EVALUATION (with ditto resolution)")
print("=" * 80)

summary = {}

for approach in sorted(comp_rows):
    rows = comp_rows[approach]

    # Resolve ditto marks in prediction rows (so '"' → actual value)
    rows_resolved = resolve_dittos(rows, available_cols)

    total_cells = 0
    exact_matches = 0
    total_cer = 0.0
    col_stats = defaultdict(lambda: {"total": 0, "exact": 0, "cer_sum": 0.0})
    mismatches = []  # store sample mismatches
    matched_rows = 0

    for row in rows_resolved:
        pred_sno = row.get("Serial_No", "").strip().translate(E2W)
        gt = gt_by_sno_resolved.get(pred_sno)
        if not gt:
            continue
        matched_rows += 1

        for col in available_cols:
            gt_val = gt.get(col, "")
            pred_val = row.get(col, "")

            # Skip cells where GT is empty (not evaluated)
            if not gt_val:
                continue

            # Skip cells where GT contains Hebrew researcher annotations
            if col == "Remarks" and has_hebrew(gt_val):
                continue

            total_cells += 1
            cell_cer = cer(pred_val, gt_val)
            total_cer += cell_cer
            col_stats[col]["total"] += 1
            col_stats[col]["cer_sum"] += cell_cer

            if pred_val == gt_val:
                exact_matches += 1
                col_stats[col]["exact"] += 1
            else:
                mismatches.append((col, gt_val, pred_val))
    
    if total_cells == 0:
        print(f"\nApproach {approach}: No comparable cells found!")
        continue
    
    exact_rate = exact_matches / total_cells
    mean_cer = total_cer / total_cells
    
    summary[approach] = {
        "matched_rows": matched_rows,
        "total_cells": total_cells,
        "exact_matches": exact_matches,
        "exact_rate": exact_rate,
        "mean_cer": mean_cer,
    }
    
    print(f"\n{'─' * 60}")
    print(f"APPROACH {approach}")
    print(f"{'─' * 60}")
    print(f"  Rows matched to GT: {matched_rows}/{len(rows)}")
    print(f"  Cells compared:     {total_cells}")
    print(f"  Exact matches:      {exact_matches}/{total_cells} ({exact_rate*100:.1f}%)")
    print(f"  Mean CER:           {mean_cer:.4f}")
    
    # Per-column breakdown
    print(f"\n  Per-column accuracy:")
    print(f"  {'Column':<25} {'Exact':>8} {'/ Total':>8} {'Rate':>8} {'CER':>8}")
    for col in available_cols:
        s = col_stats[col]
        if s["total"] == 0:
            continue
        rate = s["exact"] / s["total"]
        col_cer = s["cer_sum"] / s["total"]
        marker = " ◄" if rate < 0.5 else ""
        print(f"  {col:<25} {s['exact']:>8} / {s['total']:<6} {rate*100:>7.1f}% {col_cer:>7.3f}{marker}")
    
    # Show sample mismatches (first 15)
    print(f"\n  Sample mismatches (first 15):")
    for col, gt_v, pred_v in mismatches[:15]:
        print(f"    {col:<25} GT={gt_v!r:<20} PRED={pred_v!r}")

# ── Summary comparison ──
print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")
print(f"{'Approach':<12} {'Rows':>6} {'Cells':>8} {'Exact%':>10} {'CER':>10}")
print("-" * 50)
for a in sorted(summary, key=lambda x: -summary[x]["exact_rate"]):
    s = summary[a]
    print(f"{a:<12} {s['matched_rows']:>6} {s['total_cells']:>8} "
          f"{s['exact_rate']*100:>9.1f}% {s['mean_cer']:>9.4f}")
