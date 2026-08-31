"""List all docs in col 2377415."""
import sys, json
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL = 2377415
c = TrpClient.from_env()
docs = c.list_docs(COL)
print(f"# {len(docs)} docs in col {COL}\n")
for d in docs:
    # be tolerant of schema
    did = d.get("docId") or d.get("id")
    title = d.get("title", "?")
    npages = d.get("nrOfPages", "?")
    print(f"  docId={did}  pages={npages}  title={title!r}")
