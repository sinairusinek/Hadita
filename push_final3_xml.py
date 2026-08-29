#!/usr/bin/env python3
"""push_final3_xml.py — push final3/ PAGE XML (geometry) onto an existing doc.

For geometry-only fixes where the page IMAGE is unchanged: the XML goes up as
a NEW transcript version on the page whose imgFileName matches. If the image
changed (e.g. a corrected page split), this is the wrong tool — upload a new
document with upload_final3_doc.py instead.

Usage:
  python push_final3_xml.py --doc 18531980 --pages 12 16 17 18 20 --dry-run
  python push_final3_xml.py --doc 18531980 --pages 12 16 17 18 20 --tool Hadita-final3-geometry-v2
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")

from transkribus.client import TrpClient  # noqa: E402

COL_ID = 2377415
FINAL3 = ROOT / "Transkribus upload" / "final3"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", type=int, required=True)
    ap.add_argument("--pages", type=int, nargs="+", required=True)
    ap.add_argument("--tool", default="Hadita-final3-geometry")
    ap.add_argument("--note", default="final3 geometry (build_final3.py), no text")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = TrpClient.from_env()
    fd = client.fulldoc(COL_ID, args.doc)
    by_name = {p.get("imgFileName"): p for p in fd["pageList"]["pages"]}
    for n in args.pages:
        xml_path = FINAL3 / f"Hadita_{n}.xml"
        img_name = f"Hadita_{n}.jpeg"
        target = by_name.get(img_name)
        if target is None:
            print(f"page {n}: {img_name} not in doc {args.doc}, skipped")
            continue
        xml = xml_path.read_text(encoding="utf-8")
        m = re.search(r'imageFilename="([^"]+)"', xml)
        if not m or m.group(1) != img_name:
            print(f"page {n}: imageFilename mismatch, skipped")
            continue
        w = re.search(r'imageWidth="(\d+)"', xml)
        if w and int(w.group(1)) != int(target.get("width", w.group(1))):
            print(f"page {n}: image width {target.get('width')} != XML {w.group(1)} — "
                  f"image changed, needs a new upload; skipped")
            continue
        if args.dry_run:
            print(f"page {n} → pageNr {target['pageNr']} [dry-run]")
            continue
        resp = client.push_transcript(COL_ID, args.doc, target["pageNr"], xml,
                                      status="NEW", tool_name=args.tool, note=args.note)
        print(f"page {n} → pageNr {target['pageNr']}  tsId={resp.get('tsId') if isinstance(resp, dict) else resp}")
        time.sleep(0.4)


if __name__ == "__main__":
    main()
