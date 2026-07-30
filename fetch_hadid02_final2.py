#!/usr/bin/env python3
"""fetch_hadid02_final2.py — pull the hadid02 layers run on the final2 pages.

Same pattern as fetch_hadid01.py, but resolves the layer by toolName instead of
a hand-maintained tsId map: for each page it takes the newest transcript whose
toolName names the model id, so a re-run is picked up automatically.

Page numbers map to Transkribus pageNr through push_final2_seg.tsv, so we always
read the copy that carries the segmentation (each image is in the doc twice).

Writes g3_results/Hadita_{N}_Hadid02_final2.xml. Overwrites by default —
re-running the model makes the local copy stale, and silently keeping the old
one is worse than replacing it (pass --skip-existing for the old behaviour).

Usage:
  python fetch_hadid02_final2.py
  python fetch_hadid02_final2.py --pages 3 4 5
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")

from transkribus.client import TrpClient  # noqa: E402

COL_ID = 2377415
DOC_ID = 17829738
MODEL_ID = 592709          # hadid02
OUT_DIR = ROOT / "g3_results"
PUSH_LOG = ROOT / "push_final2_seg.tsv"
SUFFIX = "Hadid02_final2"


def page_map() -> dict[int, int]:
    with open(PUSH_LOG, encoding="utf-8") as fh:
        return {int(r["page"]): int(r["pageNr"]) for r in csv.DictReader(fh, delimiter="\t")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    client = TrpClient.from_env()
    fd = client.fulldoc(COL_ID, DOC_ID)
    by_nr = {int(p["pageNr"]): p for p in fd["pageList"]["pages"]}
    pmap = page_map()

    pages = sorted(args.pages) if args.pages else sorted(pmap)
    OUT_DIR.mkdir(exist_ok=True)

    got, none, skipped = 0, [], 0
    for page in pages:
        nr = pmap.get(page)
        if nr is None or nr not in by_nr:
            none.append(page)
            continue
        out = OUT_DIR / f"Hadita_{page}_{SUFFIX}.xml"
        if args.skip_existing and out.exists():
            skipped += 1
            continue
        cands = [t for t in by_nr[nr].get("tsList", {}).get("transcripts", [])
                 if str(MODEL_ID) in str(t.get("toolName", ""))]
        if not cands:
            none.append(page)
            print(f"  page {page:>3} (pageNr {nr}): no hadid02 layer")
            continue
        newest = max(cands, key=lambda t: int(t.get("tsId", 0)))
        xml = client.fetch_transcript(newest["url"])
        out.write_text(xml, encoding="utf-8")
        cells = len(re.findall(r"<TableCell ", xml))
        filled = sum(1 for u in re.findall(r"<Unicode>(.*?)</Unicode>", xml, re.S)
                     if u.strip())
        got += 1
        print(f"  page {page:>3} (pageNr {nr}): {out.name}  tsId={newest.get('tsId')}  "
              f"{cells} cells, {filled} non-empty")

    print(f"\nfetched {got}, skipped {skipped}, no layer {len(none)}"
          + (f" -> {none}" if none else ""))


if __name__ == "__main__":
    main()
