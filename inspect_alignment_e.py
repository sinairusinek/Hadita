#!/usr/bin/env python3
"""Show Approach E vs GT for first 5 rows, left columns only."""
import csv

def normalize(val):
    return val.strip().replace('\u200e','')

LEFT_COLS = [
    "Serial_No", "Date",
    "Property_recorded_under_Block_No", "Property_recorded_under_Parcel_No",
    "Parcel_Cat_No", "Parcel_Area",
    "Nature_of_Entry", "New_Serial_No",
    "Reference_to_Register_of_Changes_Volume_No", "Reference_to_Register_of_Changes_Serial_No",
    "Tax_LP", "Tax_Mils", "Total_Tax_LP", "Total_Tax_Mils",
    "Reference_to_Register_of_Exemptions_Entry_No",
    "Reference_to_Register_of_Exemptions_Amount_LP",
    "Reference_to_Register_of_Exemptions_Amount_Mils",
    "Net_Assessment_LP", "Net_Assessment_Mils",
    "Remarks",
]

gt_rows = []
with open("ground_truth.tsv", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        if r["Page_Number"] == "3" and r.get("Serial_No", "").strip():
            gt_rows.append(r)

comp_e = []
with open("comparison_page3.csv", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["Approach"] == "E":
            comp_e.append(r)

print(f"{'Col':<18} | {'GT':<16} | {'E':<16}")
print("-"*60)
for i in range(5):
    print(f"ROW {i+1}")
    for col in LEFT_COLS:
        gt = normalize(gt_rows[i].get(col, ""))
        e = normalize(comp_e[i].get(col, ""))
        print(f"{col:<18} | {gt:<16} | {e:<16}")
    print("-"*60)
