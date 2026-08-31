"""Pull all GT transcripts from training + val docs, normalize, save locally with diff report.

Normalization rules (canonical forms) — NON-DIGIT ONLY:
  - Ditto marks → ASCII double quote  "
      ״  ʺ  ″  “  ”   →   "
  - Dashes / empty markers → ASCII hyphen  -
      –  —  ‒  ―  _   →   -
  - Checkmark variants → single ✓
      ✔  ✗  (kept only if user wants — none observed in val data; map empty)
  - Strip leading/trailing whitespace per line (preserve internal spaces)

Does NOT touch:
  - DIGITS — both Western (0-9) and Eastern Arabic-Indic (٠-٩) appear in the source
    manuscript and are preserved as transcribed.
  - Arabic thousands separator ٬  vs comma ,   (semantic)
  - Letter forms (we saw <5 letter confusions; not normalization candidates)
  - Spaces inside text

Outputs:
  debug/norm/gt_<docId>_p<page>_orig.xml
  debug/norm/gt_<docId>_p<page>_norm.xml
  debug/norm/normalization_diff.tsv  — every change: doc, page, line_id, before → after
  debug/norm/summary.txt
"""
import sys, os, re, json
from xml.etree import ElementTree as ET
from collections import Counter

sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL = 2377415
# Live GT lives in the main processed doc, on RA-corrected pages.
# Pages 7 and 8 are empty in the corpus.
DOCS = [(15829823, "Hadita-Processed (live GT)")]
RA_PAGES = {3, 4, 5, 6, 9, 10}
OUT = "/Users/sinairusinek/Documents/GitHub/Hadita/debug/norm"
os.makedirs(OUT, exist_ok=True)
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
ET.register_namespace("", NS)

# --- normalization tables ---
DITTO = {'"': '"', "'": "'", "״": '"', "ʺ": '"', "″": '"', "“": '"', "”": '"'}
# we'll keep ASCII ' as is (not a ditto); only normalize the explicit ditto unicodes:
DITTO_MAP = {"״": '"', "ʺ": '"', "″": '"', "“": '"', "”": '"'}
DASH_MAP  = {"–": "-", "—": "-", "‒": "-", "―": "-", "_": "-"}
# digits intentionally NOT normalized — source has both Western and Eastern forms
LATIN2EAI = {}

def normalize(s: str) -> str:
    if not s:
        return s
    out = []
    for ch in s:
        if ch in DITTO_MAP:
            out.append(DITTO_MAP[ch])
        elif ch in DASH_MAP:
            out.append(DASH_MAP[ch])
        elif ch in LATIN2EAI:
            out.append(LATIN2EAI[ch])
        else:
            out.append(ch)
    return "".join(out).strip() if "".join(out).strip() != "" or not s.strip() else "".join(out)

# Use simpler version: don't lstrip if original wasn't stripped — keep as-is internally, just trim ends
def normalize_clean(s: str) -> str:
    if not s:
        return s
    out = []
    for ch in s:
        out.append(DITTO_MAP.get(ch, DASH_MAP.get(ch, LATIN2EAI.get(ch, ch))))
    s2 = "".join(out)
    # collapse only edge whitespace
    return s2.strip()


c = TrpClient.from_env()

diff_rows = []
summary_per_doc = {}

for docId, label in DOCS:
    fd = c.fulldoc(COL, docId)
    pages = fd["pageList"]["pages"]
    print(f"\n=== doc {docId} ({label}): {len(pages)} pages ===")
    n_pages_changed = 0
    n_lines_changed = 0
    n_lines_total = 0
    for p in pages:
        pageNr = p["pageNr"]
        if pageNr not in RA_PAGES:
            continue
        img = p.get("imgFileName", "?")
        transcripts = p.get("tsList", {}).get("transcripts", []) or []
        # use the LATEST transcript (RA edits live in IN_PROGRESS / FINAL), not status==GT
        ts_sorted = sorted(transcripts, key=lambda t: -int(t.get("timestamp", 0)))
        if not ts_sorted:
            print(f"  page {pageNr} ({img}): NO transcripts — skip")
            continue
        latest = ts_sorted[0]
        print(f"  page {pageNr} ({img}): latest status={latest.get('status')} tool={(latest.get('toolName') or '—')[:40]!r} user={latest.get('userName')}")
        xml_str = c.fetch_transcript(latest["url"])
        orig_path = f"{OUT}/gt_{docId}_p{pageNr}_orig.xml"
        with open(orig_path, "w") as f:
            f.write(xml_str)

        root = ET.fromstring(xml_str)
        page_changed = False
        for tl in root.iter("{%s}TextLine" % NS):
            line_id = tl.attrib.get("id", "?")
            unis = tl.findall(".//{%s}TextEquiv/{%s}Unicode" % (NS, NS))
            if not unis:
                continue
            for uni in unis:
                orig = uni.text or ""
                new = normalize_clean(orig)
                if new != orig:
                    diff_rows.append({
                        "docId": docId, "label": label, "pageNr": pageNr, "imgName": img,
                        "line_id": line_id, "before": orig, "after": new,
                    })
                    n_lines_changed += 1
                    page_changed = True
                    uni.text = new
                if orig.strip():
                    n_lines_total += 1

        norm_path = f"{OUT}/gt_{docId}_p{pageNr}_norm.xml"
        ET.ElementTree(root).write(norm_path, encoding="utf-8", xml_declaration=True)
        if page_changed:
            n_pages_changed += 1
        print(f"  page {pageNr} ({img}): {sum(1 for d in diff_rows if d['docId']==docId and d['pageNr']==pageNr)} changes")

    summary_per_doc[docId] = {
        "label": label, "n_pages": len(pages),
        "n_pages_changed": n_pages_changed,
        "n_lines_changed": n_lines_changed,
        "n_lines_total_nonempty": n_lines_total,
    }

# --- write diff TSV ---
diff_path = f"{OUT}/normalization_diff.tsv"
with open(diff_path, "w") as f:
    f.write("docId\tlabel\tpageNr\timgName\tline_id\tbefore\tafter\n")
    for r in diff_rows:
        f.write("\t".join(str(r[k]) for k in ["docId","label","pageNr","imgName","line_id","before","after"]) + "\n")

# --- character-level change tally ---
char_changes = Counter()
for r in diff_rows:
    b, a = r["before"], r["after"]
    # rough char-by-char if lengths equal else just summary at line level
    if len(b) == len(a):
        for cb, ca in zip(b, a):
            if cb != ca:
                char_changes[(cb, ca)] += 1
    else:
        char_changes[("<line len changed>", "<line len changed>")] += 1

summary_path = f"{OUT}/summary.txt"
with open(summary_path, "w") as f:
    f.write("=== Normalization dry-run summary ===\n\n")
    for docId, s in summary_per_doc.items():
        f.write(f"doc {docId} ({s['label']}): {s['n_pages']} pages\n")
        f.write(f"  pages with at least 1 change: {s['n_pages_changed']}\n")
        f.write(f"  lines changed: {s['n_lines_changed']}\n")
        f.write(f"  non-empty lines total: {s['n_lines_total_nonempty']}\n\n")
    f.write("=== Top character changes (before → after) ===\n")
    for (cb, ca), n in char_changes.most_common(30):
        f.write(f"  {n:>5}  {cb!r}  →  {ca!r}\n")

print("\n" + open(summary_path).read())
print(f"diff TSV → {diff_path}")
print(f"all GT XMLs (orig + normalized) → {OUT}/")
