"""Push normalized GT XMLs (debug/norm/gt_15829823_p{N}_norm.xml) to Transkribus
as new IN_PROGRESS transcripts. Leaves originals intact (parent_tsid linkage).
"""
import sys, os
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL = 2377415
DOC = 15829823
PAGES = [3, 4, 5, 6, 9, 10]
TOOL = "Hadita-nondigit-normalize-2026-06-26"
NORM_DIR = "/Users/sinairusinek/Documents/GitHub/Hadita/debug/norm"

c = TrpClient.from_env()
fd = c.fulldoc(COL, DOC)
page_by_nr = {p["pageNr"]: p for p in fd["pageList"]["pages"]}

for pn in PAGES:
    norm_path = f"{NORM_DIR}/gt_{DOC}_p{pn}_norm.xml"
    if not os.path.exists(norm_path):
        print(f"page {pn}: SKIP — normalized file missing")
        continue
    with open(norm_path) as f:
        xml = f.read()
    # find parent ts (the latest transcript we normalized from)
    ts_sorted = sorted(page_by_nr[pn]["tsList"]["transcripts"], key=lambda t: -int(t.get("timestamp", 0)))
    parent_tsid = ts_sorted[0].get("tsId") if ts_sorted else None
    note = f"Non-digit normalization: gershayim/en-dash/underscore → ASCII; bidi marks stripped. Source ts={parent_tsid}."
    print(f"page {pn}: pushing as IN_PROGRESS, parent_tsid={parent_tsid}, {len(xml)} bytes …")
    resp = c.push_transcript(COL, DOC, pn, xml, parent_tsid=parent_tsid, status="IN_PROGRESS", note=note, tool_name=TOOL)
    print(f"  → {resp}")
