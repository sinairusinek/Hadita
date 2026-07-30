#!/usr/bin/env python3
"""push_final2_seg.py — put the final2/ table segmentation onto the uploaded pages.

Doc 17829738 ("Hadita-final2", collection 2377415) received the 98 final2 images
but not their PAGE XMLs: every transcript is a 615-byte
`<Page imageFilename=… />` stub with no regions. The images are correct and the
coordinates already match their dimensions, so the segmentation can be pushed
onto the existing pages instead of re-uploading anything.

Each image is present twice in that document (193 pages; only Hadita_1, 3 and 10
appear once). We push to the **lowest pageNr** copy of each filename and leave
the duplicates empty.

Table-cell text is cleared before pushing. The local XMLs carry text from
`load_text_rows`, which on pages 3, 4 and 5 is the proxy **ground truth**
(350/385/228 cells) — seeding that into the document would make the Phase 4
auto-accept calibration measure our own answer key. The taxpayer name/index
TextRegions keep their Kraken reading, as in `final/`.

Idempotent: a page that already carries a transcript from TOOL_NAME is skipped.

Usage:
  python push_final2_seg.py --dry-run
  python push_final2_seg.py --pages 11 12
  python push_final2_seg.py
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")

from transkribus.client import TrpClient  # noqa: E402

COL_ID = 2377415
DOC_ID = 17829738
FINAL2_DIR = ROOT / "Transkribus upload" / "final2"
LOG_TSV = ROOT / "push_final2_seg.tsv"
DEFAULT_TOOL = "Hadita-final2-seg-2026-07-30"
STATUS = "NEW"
PAUSE_S = 0.4


def strip_cell_text(xml: str) -> tuple[str, int]:
    """Blank every <Unicode> inside the TableRegion; leave TextRegions alone."""
    head, sep, tail = xml.partition("<TableRegion")
    if not sep:
        return xml, 0
    cleared = 0

    def blank(m: re.Match) -> str:
        nonlocal cleared
        if m.group(1).strip():
            cleared += 1
        return "<Unicode></Unicode>"

    tail = re.sub(r"<Unicode>(.*?)</Unicode>", blank, tail, flags=re.S)
    return head + sep + tail, cleared


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+", help="only these Hadita page numbers")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve targets and report, push nothing")
    ap.add_argument("--tool-name", default=DEFAULT_TOOL,
                    help="toolName recorded on the pushed layer; a new name is "
                         "needed to push a revised segmentation over an old one")
    args = ap.parse_args()

    client = TrpClient.from_env()
    fd = client.fulldoc(COL_ID, DOC_ID)

    # filename -> page records, lowest pageNr first
    by_name: dict[str, list[dict]] = collections.defaultdict(list)
    for p in fd["pageList"]["pages"]:
        by_name[p.get("imgFileName")].append(p)
    for v in by_name.values():
        v.sort(key=lambda p: p["pageNr"])

    xmls = sorted(FINAL2_DIR.glob("Hadita_*.xml"),
                  key=lambda p: int(p.stem.split("_")[1]))
    if args.pages:
        want = set(args.pages)
        xmls = [x for x in xmls if int(x.stem.split("_")[1]) in want]

    records, skipped, failed = [], [], []
    for i, xml_path in enumerate(xmls, 1):
        page_no = int(xml_path.stem.split("_")[1])
        img_name = f"{xml_path.stem}.jpeg"
        copies = by_name.get(img_name, [])
        if not copies:
            failed.append((page_no, "no page with that image in the document"))
            print(f"[{i}/{len(xmls)}] page {page_no:>3}  MISSING in doc")
            continue
        target = copies[0]
        page_nr = target["pageNr"]

        existing = target.get("tsList", {}).get("transcripts", [])
        if any(t.get("toolName") == args.tool_name for t in existing):
            skipped.append(page_no)
            print(f"[{i}/{len(xmls)}] page {page_no:>3}  pageNr={page_nr:>3}  "
                  f"already pushed, skipping")
            continue

        xml, cleared = strip_cell_text(xml_path.read_text(encoding="utf-8"))
        m = re.search(r'imageFilename="([^"]+)"', xml)
        if not m or m.group(1) != img_name:
            failed.append((page_no, f"imageFilename {m.group(1) if m else None!r} "
                                    f"does not match {img_name!r}"))
            print(f"[{i}/{len(xmls)}] page {page_no:>3}  imageFilename mismatch")
            continue
        rows = re.search(r'<TableRegion[^>]*rows="(\d+)" columns="(\d+)"', xml)
        n_cells = len(re.findall(r"<TableCell ", xml))

        if args.dry_run:
            print(f"[{i}/{len(xmls)}] page {page_no:>3} → pageNr={page_nr:>3}  "
                  f"{rows.group(1)}r×{rows.group(2)}c {n_cells} cells  "
                  f"(would clear {cleared} text cells)  [dry-run]")
            continue

        try:
            resp = client.push_transcript(
                COL_ID, DOC_ID, page_nr, xml,
                status=STATUS, tool_name=args.tool_name,
                note="final2 coordinate-warp segmentation (no cell text)")
            ts_id = resp.get("tsId") if isinstance(resp, dict) else None
            records.append({"page": page_no, "pageNr": page_nr,
                            "rows": rows.group(1), "cols": rows.group(2),
                            "cells": n_cells, "cleared_text_cells": cleared,
                            "tsId": ts_id})
            print(f"[{i}/{len(xmls)}] page {page_no:>3} → pageNr={page_nr:>3}  "
                  f"{rows.group(1)}r×{rows.group(2)}c  tsId={ts_id}")
        except Exception as exc:
            failed.append((page_no, str(exc)[:160]))
            print(f"[{i}/{len(xmls)}] page {page_no:>3}  FAILED: {str(exc)[:160]}")
        time.sleep(PAUSE_S)

    if records:
        cols = ["page", "pageNr", "rows", "cols", "cells", "cleared_text_cells", "tsId"]
        write_header = not LOG_TSV.exists()
        with open(LOG_TSV, "a", encoding="utf-8") as fh:
            if write_header:
                fh.write("\t".join(cols) + "\n")
            for r in records:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    print(f"\npushed {len(records)}, skipped {len(skipped)}, failed {len(failed)}")
    if records:
        print(f"  log → {LOG_TSV.name}")
    for p, e in failed:
        print(f"  page {p}: {e}")


if __name__ == "__main__":
    main()
