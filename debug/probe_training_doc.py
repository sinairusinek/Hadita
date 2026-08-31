"""Probe training doc 17097375 (Hadita1 training+val set, 10 pages)."""
import sys, json
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL = 2377415
DOC = 17097375
c = TrpClient.from_env()
fd = c.fulldoc(COL, DOC)

print(f"# Doc {DOC}: {len(fd['pageList']['pages'])} pages\n")
print("pageNr | imgName                  | n_ts | statuses | tools_used")
print("-" * 110)
rows = []
for p in fd["pageList"]["pages"]:
    pageNr = p["pageNr"]
    img = p.get("imgFileName", "?")
    transcripts = p.get("tsList", {}).get("transcripts", []) or []
    statuses = sorted({t.get("status") for t in transcripts})
    tools = sorted({(t.get("toolName") or "—") for t in transcripts})
    print(f"{pageNr:>5} | {img:<24} | {len(transcripts):>4} | {','.join(statuses):<25} | {', '.join(tools)}")
    for i, t in enumerate(transcripts):
        rows.append({"pageNr": pageNr, "imgName": img, **t})

with open("/Users/sinairusinek/Documents/GitHub/Hadita/debug/trx_training_doc_transcripts.json","w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
