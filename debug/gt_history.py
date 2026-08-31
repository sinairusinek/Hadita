"""For each page in training+val docs, list all transcripts (newest first)
with status, timestamp (human), toolName, userName.
For the LATEST transcript, also: # non-empty cells + sample digit chars seen.

This tells us:
  (a) which transcript is actually the most recent (regardless of status)
  (b) whether pages like Hadita_65, 77, etc. contain substantive GT
  (c) whether the digit-character set on Hadita_6 is now Eastern (user-corrected) or still Latin
"""
import sys, json
from datetime import datetime
from xml.etree import ElementTree as ET
from collections import Counter
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL = 2377415
DOCS = [(17097375, "training+val"), (17102575, "held-out val")]
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

c = TrpClient.from_env()

def parse_stats(xml_str):
    """Return (n_nonempty_lines, total_chars, latin_digits, eastern_digits, sample_lines)."""
    try:
        root = ET.fromstring(xml_str)
    except Exception as e:
        return (0, 0, 0, 0, [f"parse error: {e}"])
    n_lines = 0
    total_chars = 0
    latin = 0
    eastern = 0
    samples = []
    EASTERN_DIGITS = set("٠١٢٣٤٥٦٧٨٩")
    LATIN_DIGITS = set("0123456789")
    for tl in root.iter("{%s}TextLine" % NS):
        for uni in tl.findall(".//{%s}TextEquiv/{%s}Unicode" % (NS, NS)):
            t = (uni.text or "").strip()
            if t:
                n_lines += 1
                total_chars += len(t)
                for ch in t:
                    if ch in LATIN_DIGITS: latin += 1
                    elif ch in EASTERN_DIGITS: eastern += 1
                if len(samples) < 5:
                    samples.append(t)
    return (n_lines, total_chars, latin, eastern, samples)

for docId, label in DOCS:
    fd = c.fulldoc(COL, docId)
    pages = fd["pageList"]["pages"]
    print(f"\n{'='*100}\n=== doc {docId} ({label}) ===")
    for p in pages:
        pageNr = p["pageNr"]
        img = p.get("imgFileName", "?")
        ts_list = p.get("tsList", {}).get("transcripts", []) or []
        # sort by timestamp desc to find truly latest
        ts_sorted = sorted(ts_list, key=lambda t: -int(t.get("timestamp", 0)))
        latest = ts_sorted[0] if ts_sorted else None
        gt_ts = next((t for t in ts_sorted if t.get("status") == "GT"), None)
        print(f"\n--- page {pageNr} ({img}) — {len(ts_sorted)} transcripts ---")
        for i, t in enumerate(ts_sorted):
            ts = datetime.fromtimestamp(int(t.get("timestamp",0))/1000).strftime("%Y-%m-%d %H:%M")
            marker = ""
            if t is latest: marker += " [LATEST]"
            if t is gt_ts: marker += " [GT-tagged]"
            print(f"  {i}: {ts}  status={t.get('status'):<12} tool={(t.get('toolName') or '—')[:50]!r:<55} user={t.get('userName')!r}{marker}")
        # parse the LATEST and the GT-tagged (if different)
        for label_, t in [("LATEST", latest), ("GT-tagged", gt_ts)]:
            if t is None: continue
            if label_ == "GT-tagged" and t is latest: continue
            xml = c.fetch_transcript(t["url"])
            n_lines, total_chars, latin, eastern, samples = parse_stats(xml)
            print(f"  >> {label_}: lines={n_lines}, chars={total_chars}, latin_digits={latin}, eastern_digits={eastern}")
            for s in samples[:3]:
                print(f"     sample: {s!r}")
