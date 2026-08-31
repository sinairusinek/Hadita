"""List all transcripts per page for the Hadita doc on Transkribus.

For each page: emit (pageNr, transcript_index, status, userName, toolName, timestamp).
This tells us which pages have FINAL transcripts (proxy GT) and which have model 592509 runs.
"""
import sys, json
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL = 2377415
DOC = 15829823

c = TrpClient.from_env()
fd = c.fulldoc(COL, DOC)
pages = fd["pageList"]["pages"]

rows = []
for p in pages:
    pageNr = p["pageNr"]
    transcripts = p.get("tsList", {}).get("transcripts", []) or []
    for i, t in enumerate(transcripts):
        rows.append({
            "pageNr": pageNr,
            "ts_idx": i,
            "status": t.get("status"),
            "userName": t.get("userName"),
            "toolName": t.get("toolName"),
            "timestamp": t.get("timestamp"),
            "key": t.get("key"),
        })

# print summary: per-page transcript statuses
from collections import defaultdict
per_page = defaultdict(list)
for r in rows:
    per_page[r["pageNr"]].append(r)

print(f"# Doc {DOC}: {len(pages)} pages, {len(rows)} total transcripts\n")
print("pageNr | n_ts | FINAL? | tools_used")
print("-" * 60)
for pageNr in sorted(per_page):
    ts = per_page[pageNr]
    has_final = any(t["status"] == "FINAL" for t in ts)
    tools = sorted({(t["toolName"] or "—") for t in ts})
    print(f"{pageNr:>5} | {len(ts):>4} | {'Y' if has_final else '.':>5}  | {', '.join(tools)}")

# also dump full table to JSON
out = "/Users/sinairusinek/Documents/GitHub/Hadita/debug/trx_transcripts_all.json"
with open(out, "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\nfull dump → {out}")
