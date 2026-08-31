"""A/B compare model 592509 (hadid01) vs 592709 (hadid02) on pp. 11-13.

For each page, pull the latest transcript produced by each model, parse TextLines
by cell id, and emit a TSV: page | line_id | old_592509 | new_592709 | agree?

Also: stdout summary — counts of (agree, both-empty, only-old-empty, only-new-empty, both-nonempty-differ),
and the most common new-vs-old character substitutions among the lines where both are nonempty.
"""
import sys, json, re
from collections import Counter
from xml.etree import ElementTree as ET

sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL = 2377415
DOC = 15829823  # Hadita-Processed
PAGES = [11, 12, 13]
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

c = TrpClient.from_env()
fd = c.fulldoc(COL, DOC)
page_by_nr = {p["pageNr"]: p for p in fd["pageList"]["pages"]}

def latest_for_model(transcripts, model_id):
    matching = [t for t in transcripts if model_id in (t.get("toolName") or "")]
    if not matching: return None
    matching.sort(key=lambda t: -int(t.get("timestamp", 0)))
    return matching[0]

def parse_lines(xml_str):
    root = ET.fromstring(xml_str)
    out = {}
    for tl in root.iter("{%s}TextLine" % NS):
        lid = tl.attrib.get("id","?")
        unis = tl.findall(".//{%s}TextEquiv/{%s}Unicode" % (NS, NS))
        text = (unis[-1].text if unis and unis[-1].text else "") or ""
        out[lid] = text.strip()
    return out

all_rows = []
counters = Counter()
sub_counter = Counter()  # per-char substitution counts (old, new)

for pn in PAGES:
    transcripts = page_by_nr[pn]["tsList"]["transcripts"]
    old_ts = latest_for_model(transcripts, "592509")
    new_ts = latest_for_model(transcripts, "592709")
    if not old_ts or not new_ts:
        print(f"page {pn}: MISSING — old_ts={bool(old_ts)} new_ts={bool(new_ts)}")
        continue
    old_xml = c.fetch_transcript(old_ts["url"])
    new_xml = c.fetch_transcript(new_ts["url"])
    old_lines = parse_lines(old_xml)
    new_lines = parse_lines(new_xml)
    shared = sorted(set(old_lines) & set(new_lines))
    print(f"page {pn}: old ts={old_ts['tsId']} ({old_ts['timestamp']}), new ts={new_ts['tsId']} ({new_ts['timestamp']}), shared cells={len(shared)}")
    for lid in shared:
        o = old_lines[lid]
        n = new_lines[lid]
        agree = (o == n)
        if agree and not o: kind = "both_empty"
        elif agree: kind = "agree_nonempty"
        elif not o: kind = "old_empty_new_nonempty"
        elif not n: kind = "new_empty_old_nonempty"
        else: kind = "both_nonempty_differ"
        counters[kind] += 1
        all_rows.append({"page": pn, "line_id": lid, "old_592509": o, "new_592709": n, "kind": kind})
        if kind == "both_nonempty_differ" and len(o) == len(n):
            for co, cn in zip(o, n):
                if co != cn:
                    sub_counter[(co, cn)] += 1

# write TSV
out_path = "/Users/sinairusinek/Documents/GitHub/Hadita/debug/ab_592509_vs_592709.tsv"
with open(out_path, "w") as f:
    f.write("page\tline_id\tkind\told_592509\tnew_592709\n")
    for r in all_rows:
        f.write(f"{r['page']}\t{r['line_id']}\t{r['kind']}\t{r['old_592509']}\t{r['new_592709']}\n")

# summary
print(f"\n=== Summary across pp. 11-13 ({sum(counters.values())} shared cells) ===")
for k in ["both_empty", "agree_nonempty", "both_nonempty_differ",
          "old_empty_new_nonempty", "new_empty_old_nonempty"]:
    print(f"  {k:<30}  {counters.get(k,0):>4}")

print(f"\n=== Most common per-char divergence (old → new) where lines are same-length and nonempty ===")
for (co, cn), n in sub_counter.most_common(25):
    print(f"  {n:>4}   {co!r:>4}  →  {cn!r}")

# show 20 of the most interesting cells where models disagree
print(f"\n=== Sample of disagreements (nonempty both) ===")
nonempty_diffs = [r for r in all_rows if r["kind"] == "both_nonempty_differ"]
for r in nonempty_diffs[:25]:
    print(f"  p{r['page']} {r['line_id']:<32}  OLD={r['old_592509']!r:<30}  NEW={r['new_592709']!r}")

print(f"\nfull TSV → {out_path}")
