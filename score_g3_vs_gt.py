"""Score G3 output JSON files against ground_truth.tsv for a given page.

Classifies each comparable cell as:
  perfect | single_digit | multi_digit | wrong | missed_row | phantom_row | empty_both

Outputs per-mode counts and an RA_cost estimate:
  RA_cost (keystrokes) = single_digit + 6 * multi_digit + 8 * wrong
                       + 20 * missed_row + 3 * phantom_row
"""
import csv
import json
import sys
import unicodedata
from pathlib import Path

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


def _normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).strip()
    for marker in ("[?]", "[RED]"):
        s = s.replace(marker, "")
    # Thousands separator → ASCII comma
    s = s.replace("،", ",").replace("٬", ",")
    # Ditto family (Hebrew gershayim, ASCII double-quote, curly quotes, double-comma) → "
    for d in ("״", "“", "”", "„", ",,"):
        s = s.replace(d, '"')
    # No-data placeholders → "..."
    for nd in ("---", "--", "—", "–", ".."):
        if s.strip() == nd:
            s = "..."
    # Dash variants for "nil" → -
    if s.strip() in ("–", "—"):
        s = "-"
    # Leading/trailing whitespace + collapse internal whitespace
    s = " ".join(s.split())
    return s


def _digit_only(s: str) -> str:
    return "".join(c for c in s if c.isdigit() or "٠" <= c <= "٩")


def classify(gt: str, pr: str) -> str:
    g, p = _normalize(gt), _normalize(pr)
    if g == p:
        return "empty_both" if g == "" else "perfect"
    if g == "" or p == "":
        return "wrong"  # one is empty, the other isn't
    # Same length, differ in exactly one position → single_digit (best case)
    if len(g) == len(p):
        diffs = sum(1 for a, b in zip(g, p) if a != b)
        if diffs == 1:
            return "single_digit"
        if diffs <= 3:
            return "multi_digit"
        return "wrong"
    # Different lengths but digit-only and close in digit count → multi_digit
    gd, pd = _digit_only(g), _digit_only(p)
    if gd and pd and abs(len(gd) - len(pd)) <= 1:
        # Count common digits
        return "multi_digit"
    return "wrong"


def load_gt(page: int) -> list[dict]:
    rows = []
    with open("ground_truth.tsv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if str(r.get("Page_Number", "")).strip() == str(page):
                rows.append({c: (r.get(c, "") or "").strip() for c in LEFT_COLS})
    return rows


def load_g3(path: str) -> list[dict]:
    d = json.load(open(path, encoding="utf-8"))
    return [{c: (r.get(c, "") or "").strip() for c in LEFT_COLS} for r in d]


def align_by_position(gt: list[dict], pr: list[dict]) -> list[tuple]:
    """Position-align with offset detection (handles missed/phantom top row).

    Tries small offsets (−2..+2) and picks the one that maximizes per-cell agreement
    on non-empty cells. This is robust to a systematic Serial-number misread, which
    Serial-based alignment would treat as massive structural loss.
    """
    best_off, best_score = 0, -1
    for off in range(-2, 3):
        score_ = 0
        for j in range(len(pr)):
            i = j + off
            if 0 <= i < len(gt):
                for c in LEFT_COLS:
                    gv, pv = _normalize(gt[i][c]), _normalize(pr[j][c])
                    if gv and gv == pv:
                        score_ += 1
        if score_ > best_score:
            best_score, best_off = score_, off
    pairs, used_gt, used_pr = [], set(), set()
    for j in range(len(pr)):
        i = j + best_off
        if 0 <= i < len(gt):
            pairs.append(("matched", i, j, gt[i], pr[j]))
            used_gt.add(i)
            used_pr.add(j)
    for i in range(len(gt)):
        if i not in used_gt:
            pairs.append(("missed", i, None, gt[i], None))
    for j in range(len(pr)):
        if j not in used_pr:
            pairs.append(("phantom", None, j, None, pr[j]))
    print(f"  (alignment offset = {best_off:+d}, {best_score} matching cells)")
    return pairs


def score(gt: list[dict], pr: list[dict]):
    counts = {k: 0 for k in
              ["perfect", "empty_both", "single_digit", "multi_digit", "wrong",
               "missed_row", "phantom_row"]}
    per_col = {c: {"perfect": 0, "single_digit": 0, "multi_digit": 0, "wrong": 0,
                   "empty_both": 0} for c in LEFT_COLS}
    sample_errors = []
    pairs = align_by_position(gt, pr)
    for kind, gi, pi, g, p in pairs:
        if kind == "missed":
            counts["missed_row"] += 1
            sample_errors.append((g["Serial_No"], "missed_row", "", ""))
            continue
        if kind == "phantom":
            counts["phantom_row"] += 1
            sample_errors.append((p["Serial_No"] or f"row{pi}", "phantom_row", "", ""))
            continue
        for c in LEFT_COLS:
            klass = classify(g[c], p[c])
            counts[klass] += 1
            per_col[c][klass] = per_col[c].get(klass, 0) + 1
            if klass in ("single_digit", "multi_digit", "wrong"):
                sample_errors.append((g["Serial_No"], f"{c}:{klass}", g[c], p[c]))
    return counts, per_col, sample_errors, pairs


def ra_cost(counts: dict) -> float:
    """Keystroke estimate."""
    return (1 * counts.get("single_digit", 0)
            + 6 * counts.get("multi_digit", 0)
            + 8 * counts.get("wrong", 0)
            + 20 * counts.get("missed_row", 0)
            + 3 * counts.get("phantom_row", 0))


def report(label: str, counts: dict, per_col: dict, sample_errors: list, api_cost: float):
    keystrokes = ra_cost(counts)
    print(f"\n══════════ {label} ══════════")
    print(f"  perfect:      {counts['perfect']:>4}      (cells the RA touches 0×)")
    print(f"  empty_both:   {counts['empty_both']:>4}      (both blank → no work)")
    print(f"  single_digit: {counts['single_digit']:>4}      (~1 keystroke each)")
    print(f"  multi_digit:  {counts['multi_digit']:>4}      (~6 keystrokes each)")
    print(f"  wrong:        {counts['wrong']:>4}      (~8 keystrokes each)")
    print(f"  missed_row:   {counts['missed_row']:>4}      (~20 keystrokes each)")
    print(f"  phantom_row:  {counts['phantom_row']:>4}      (~3 keystrokes each)")
    print(f"  -------- RA_cost: {keystrokes:>4} keystrokes")
    rng_lo, rng_hi = keystrokes / 60 / 1.5, keystrokes / 60 / 1.0  # 60-90 keystrokes/min
    print(f"     ≈ {rng_lo:.1f}-{rng_hi:.1f} minutes of correction at 60-90 ks/min")
    print(f"  API cost:     ${api_cost:.4f}")
    rate_per_hour = 25  # USD/hour assumed RA rate
    ra_usd = keystrokes / 60 / 1.2 / 60 * rate_per_hour
    print(f"  RA cost @ $25/hr ≈ ${ra_usd:.2f}")
    print(f"  TOTAL:        ${api_cost + ra_usd:.2f}")

    print(f"\n  Worst columns by # errors (non-perfect, non-empty):")
    errs_per_col = [(c, sum(v.get(k, 0) for k in ("single_digit", "multi_digit", "wrong"))) for c, v in per_col.items()]
    errs_per_col.sort(key=lambda x: -x[1])
    for c, n in errs_per_col[:6]:
        if n:
            print(f"    {c:42s}  {n}")

    if sample_errors:
        print(f"\n  First 12 errors:")
        for ser, kind, gt, pr in sample_errors[:12]:
            print(f"    Ser {ser:>3s} | {kind:30s} | GT={gt!r:>14s} | G3={pr!r}")


def load_trx_xml(xml_path: str) -> list[dict]:
    """Load Transkribus PAGE XML as list[dict] keyed by LEFT_COLS."""
    import re
    xml = Path(xml_path).read_text()
    cells = {}
    for m in re.finditer(r'id="cell_r(\d+)_c(\d+)"', xml):
        r, c = int(m.group(1)), int(m.group(2))
        s = m.end()
        nxt = xml.find("<TableCell", s)
        e = nxt if nxt != -1 else len(xml)
        parts = re.findall(r'<Unicode>([^<]*)</Unicode>', xml[s:e])
        cells[(r, c)] = next((p for p in parts if p.strip()), "").strip()
    max_r = max(r for r, _ in cells.keys()) + 1
    rows = []
    for r in range(max_r):
        row = {}
        for c, col in enumerate(LEFT_COLS):
            row[col] = cells.get((r, c), "")
        rows.append(row)
    return rows


def filter_nonempty_rows(rows: list[dict]) -> list[dict]:
    """Drop rows where every cell is blank (artifacts of XML grid)."""
    return [r for r in rows if any(v.strip() for v in r.values())]


def main():
    print("┌" + "─"*78 + "┐")
    print("│" + "  PAGE 3 — scored against ground_truth.tsv (verified GT)".ljust(78) + "│")
    print("└" + "─"*78 + "┘")
    gt3 = load_gt(3)
    print(f"GT page 3: {len(gt3)} rows")
    low3 = load_g3("g3_results/low thinking/Hadita_3_G3.json")
    med3 = load_g3("g3_results/medium thinking/Hadita_3_G3.json")
    print(f"G3 LOW   : {len(low3)} rows · G3 MEDIUM: {len(med3)} rows")

    counts_l3, per_col_l3, err_l3, _ = score(gt3, low3)
    counts_m3, per_col_m3, err_m3, _ = score(gt3, med3)
    report("PAGE 3 — LOW thinking",    counts_l3, per_col_l3, err_l3, 0.082)
    report("PAGE 3 — MEDIUM thinking", counts_m3, per_col_m3, err_m3, 0.320)

    print("\n\n┌" + "─"*78 + "┐")
    print("│" + "  PAGE 4 — scored against Trx FINAL (RA-corrected proxy; not gold GT)".ljust(78) + "│")
    print("└" + "─"*78 + "┘")
    gt4_raw = load_trx_xml("g3_results/Hadita_4_Transkribus_current.xml")
    gt4 = filter_nonempty_rows(gt4_raw)
    print(f"Trx FINAL page 4: {len(gt4_raw)} XML rows, {len(gt4)} non-empty")
    low4 = load_g3("g3_results/low thinking/Hadita_4_G3.json")
    med4 = load_g3("g3_results/medium thinking/Hadita_4_G3.json")
    print(f"G3 LOW   : {len(low4)} rows · G3 MEDIUM: {len(med4)} rows")
    counts_l4, per_col_l4, err_l4, _ = score(gt4, low4)
    counts_m4, per_col_m4, err_m4, _ = score(gt4, med4)
    report("PAGE 4 — LOW thinking",    counts_l4, per_col_l4, err_l4, 0.082)
    report("PAGE 4 — MEDIUM thinking", counts_m4, per_col_m4, err_m4, 0.323)

    print("\n\n═══ COMBINED DECISION (page 3 + page 4) ═══")
    low_ks = ra_cost(counts_l3) + ra_cost(counts_l4)
    med_ks = ra_cost(counts_m3) + ra_cost(counts_m4)
    low_ra_usd = low_ks / 60 / 1.2 / 60 * 25
    med_ra_usd = med_ks / 60 / 1.2 / 60 * 25
    low_total = 0.082 * 2 + low_ra_usd
    med_total = 0.320 * 2 + med_ra_usd
    print(f"  2 pages — LOW : {low_ks:>4} keystrokes, RA ${low_ra_usd:.2f}, API $0.164, TOTAL ${low_total:.2f}")
    print(f"  2 pages — MED : {med_ks:>4} keystrokes, RA ${med_ra_usd:.2f}, API $0.640, TOTAL ${med_total:.2f}")
    per_low, per_med = low_total/2, med_total/2
    print(f"\n  per-page avg — LOW: ${per_low:.2f}  MEDIUM: ${per_med:.2f}")
    print(f"  100 pages    — LOW: ${per_low*100:.0f}    MEDIUM: ${per_med*100:.0f}")
    print(f"  Winner: {'LOW' if per_low < per_med else 'MEDIUM'} (margin: ${abs(per_low-per_med)*100:.0f} over 100 pages)")


if __name__ == "__main__":
    main()
