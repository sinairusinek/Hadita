"""Probe the validation doc 17102575 in col 2377415.

We want to know:
- how many pages, and which Hadita pages (by imageFilename) they map to
- what transcripts exist per page (status, toolName, userName)
- whether any transcript was produced by an HTR run (i.e., model 592509 predictions exist)
"""
import sys, json
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL = 2377415
DOC = 17102575

c = TrpClient.from_env()
fd = c.fulldoc(COL, DOC)
pages = fd["pageList"]["pages"]

print(f"# Val doc {DOC}: {len(pages)} pages\n")
print("pageNr | imgName                              | n_ts | statuses | tools_used")
print("-" * 110)

rows = []
for p in pages:
    pageNr = p["pageNr"]
    img = p.get("imgFileName", "?")
    transcripts = p.get("tsList", {}).get("transcripts", []) or []
    statuses = sorted({t.get("status") for t in transcripts})
    tools = sorted({(t.get("toolName") or "—") for t in transcripts})
    print(f"{pageNr:>5} | {img:<36} | {len(transcripts):>4} | {','.join(statuses):<25} | {', '.join(tools)}")
    for i, t in enumerate(transcripts):
        rows.append({
            "pageNr": pageNr, "imgName": img,
            "ts_idx": i, "status": t.get("status"),
            "userName": t.get("userName"), "toolName": t.get("toolName"),
            "timestamp": t.get("timestamp"), "key": t.get("key"),
            "url": t.get("url"),
        })

out = "/Users/sinairusinek/Documents/GitHub/Hadita/debug/trx_val_transcripts.json"
with open(out, "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\nfull dump → {out}")
