"""Find all transcripts produced by an HTR run of model 592509 or 592709
across all docs in col 2377415."""
import sys, json
from datetime import datetime
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL = 2377415
TARGETS = ["592509", "592709"]
c = TrpClient.from_env()
docs = c.list_docs(COL)
hits = []
for d in docs:
    did = d.get("docId") or d.get("id")
    title = d.get("title")
    try:
        fd = c.fulldoc(COL, did)
    except Exception as e:
        print(f"doc {did}: error {e}")
        continue
    for p in fd["pageList"]["pages"]:
        for t in p.get("tsList", {}).get("transcripts", []) or []:
            tn = (t.get("toolName") or "")
            if any(tgt in tn for tgt in TARGETS):
                hits.append({
                    "docId": did, "docTitle": title,
                    "pageNr": p["pageNr"], "imgName": p.get("imgFileName"),
                    "tool": tn, "status": t.get("status"),
                    "timestamp": datetime.fromtimestamp(int(t.get("timestamp",0))/1000).strftime("%Y-%m-%d %H:%M"),
                    "tsId": t.get("tsId"), "url": t.get("url"),
                })

print(f"# {len(hits)} HTR-run transcripts from model 592509 or 592709\n")
for h in sorted(hits, key=lambda x: (x["docId"], x["pageNr"], x["timestamp"])):
    print(f"  doc={h['docId']} ({h['docTitle']:<30}) p{h['pageNr']:>3} {h['imgName']:<20} {h['timestamp']} status={h['status']:<12} tool={h['tool']}")

with open("/Users/sinairusinek/Documents/GitHub/Hadita/debug/model_run_transcripts.json","w") as f:
    json.dump(hits, f, ensure_ascii=False, indent=2, default=str)
