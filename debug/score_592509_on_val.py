"""Compute per-line CER + confusion patterns: model 592509 predictions vs. GT on val page Hadita_10.

Inputs (Transkribus):
  col 2377415, doc 17102575, page 1 (Hadita_10.jpeg)
  - status=GT             → reference
  - status=IN_PROGRESS, toolName contains 'Model: 592509' → prediction

Outputs:
  debug/val_gt.xml
  debug/val_pred.xml
  debug/val_per_line_cer.tsv  — line_id | gt | pred | cer | edit_distance | gt_len
  printed summary: overall CER, worst lines, top character confusions
"""
import sys, os, re, json
from xml.etree import ElementTree as ET
from collections import Counter

sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL = 2377415
DOC = 17102575
OUT_DIR = "/Users/sinairusinek/Documents/GitHub/Hadita/debug"
NS = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}

# ---- pull both transcripts
c = TrpClient.from_env()
fd = c.fulldoc(COL, DOC)
page = fd["pageList"]["pages"][0]
transcripts = page["tsList"]["transcripts"]

gt_ts = next(t for t in transcripts if t.get("status") == "GT")
pred_ts = next(
    t for t in transcripts
    if (t.get("toolName") or "").startswith("PyLaia") and "592509" in (t.get("toolName") or "")
)

gt_xml = c.fetch_transcript(gt_ts["url"])
pred_xml = c.fetch_transcript(pred_ts["url"])
with open(f"{OUT_DIR}/val_gt.xml", "w") as f: f.write(gt_xml)
with open(f"{OUT_DIR}/val_pred.xml", "w") as f: f.write(pred_xml)
print(f"GT  ts: status={gt_ts['status']}  tool={gt_ts.get('toolName')}  user={gt_ts.get('userName')}")
print(f"PRED ts: status={pred_ts['status']} tool={pred_ts.get('toolName')}  user={pred_ts.get('userName')}")

# ---- parse: collect {line_id: text}
def parse_lines(xml_str):
    root = ET.fromstring(xml_str)
    out = {}
    for tl in root.iter("{%s}TextLine" % NS["pc"]):
        line_id = tl.attrib.get("id", "?")
        # take last TextEquiv/Unicode (some XMLs have multiple)
        uni = tl.findall(".//pc:TextEquiv/pc:Unicode", NS)
        text = (uni[-1].text if uni and uni[-1].text else "") or ""
        out[line_id] = text
    return out

gt_lines = parse_lines(gt_xml)
pred_lines = parse_lines(pred_xml)
print(f"\n#GT lines = {len(gt_lines)}, #PRED lines = {len(pred_lines)}")

# ---- align by line id
shared_ids = [lid for lid in gt_lines if lid in pred_lines and gt_lines[lid].strip()]
only_gt = [lid for lid in gt_lines if lid not in pred_lines]
only_pred = [lid for lid in pred_lines if lid not in gt_lines]
print(f"shared with non-empty GT: {len(shared_ids)}  | only_gt={len(only_gt)}  | only_pred={len(only_pred)}")

# ---- edit distance
def lev(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0]*len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1] + (ca != cb))
        prev = cur
    return prev[-1]

rows = []
total_ed = 0
total_len = 0
for lid in shared_ids:
    g = gt_lines[lid]
    p = pred_lines[lid]
    ed = lev(g, p)
    L = max(len(g), 1)
    cer = ed / L
    total_ed += ed
    total_len += len(g)
    rows.append({"line_id": lid, "gt": g, "pred": p, "ed": ed, "gt_len": len(g), "cer": cer})

rows.sort(key=lambda r: -r["cer"])

# ---- write TSV
tsv = f"{OUT_DIR}/val_per_line_cer.tsv"
with open(tsv, "w") as f:
    f.write("line_id\tgt_len\tedit_dist\tcer\tgt\tpred\n")
    for r in rows:
        f.write(f"{r['line_id']}\t{r['gt_len']}\t{r['ed']}\t{r['cer']:.4f}\t{r['gt']}\t{r['pred']}\n")

# ---- summary
overall_cer = total_ed / max(total_len, 1)
exact = sum(1 for r in rows if r["ed"] == 0)
print(f"\n=== Overall ===")
print(f"  shared non-empty lines: {len(rows)}")
print(f"  total GT chars: {total_len}")
print(f"  total edit distance: {total_ed}")
print(f"  micro CER (sum-ed / sum-len): {overall_cer:.4f}")
print(f"  exact matches: {exact} ({100*exact/max(len(rows),1):.1f}%)")

print(f"\n=== Worst 25 lines (highest CER) ===")
print(f"{'line_id':<32}  {'gt_len':>6} {'ed':>4} {'cer':>6}  gt → pred")
for r in rows[:25]:
    g = r["gt"][:40]
    p = r["pred"][:40]
    print(f"{r['line_id']:<32}  {r['gt_len']:>6} {r['ed']:>4} {r['cer']:>6.2f}  {g!r}  →  {p!r}")

# ---- character-level confusion (over aligned single-line strings)
# Use a simple substitution counter — get per-char ops by re-running DP with backtrace
def ops(a, b):
    # returns list of (op, ca, cb) for backtraced edit operations
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    res = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i-1] == b[j-1]:
            res.append(("match", a[i-1], b[j-1])); i-=1; j-=1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            res.append(("sub", a[i-1], b[j-1])); i-=1; j-=1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            res.append(("del", a[i-1], "")); i-=1
        else:
            res.append(("ins", "", b[j-1])); j-=1
    return list(reversed(res))

sub_counter = Counter()
del_counter = Counter()
ins_counter = Counter()
for r in rows:
    for op, ca, cb in ops(r["gt"], r["pred"]):
        if op == "sub": sub_counter[(ca, cb)] += 1
        elif op == "del": del_counter[ca] += 1
        elif op == "ins": ins_counter[cb] += 1

print(f"\n=== Top 20 character substitutions (gt → pred) ===")
for (ca, cb), n in sub_counter.most_common(20):
    print(f"  {n:>4}   {ca!r:>4}  →  {cb!r}")

print(f"\n=== Top 15 deletions (gt char dropped by model) ===")
for ca, n in del_counter.most_common(15):
    print(f"  {n:>4}   {ca!r}")

print(f"\n=== Top 15 insertions (pred char not in gt) ===")
for cb, n in ins_counter.most_common(15):
    print(f"  {n:>4}   {cb!r}")

print(f"\nfull per-line table → {tsv}")
