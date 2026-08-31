#!/usr/bin/env python3
"""push_pilot_seeds.py — push the E10 pilot flagged-seed transcripts to Transkribus.

Pushes exp2608/Hadita_{N}_som_flagged.xml (SoM text + [?] on MISMATCH cells) onto the
lowest-pageNr copy of each image in doc 17829738 (collection 2377415), as a NEW
transcript version. Transkribus keeps prior versions, so this is reversible from the
version picker.

Usage:
  python push_pilot_seeds.py --dry-run
  python push_pilot_seeds.py --pages 11 12 ... 20
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
SEED_DIR = ROOT / "exp2608"
LOG_TSV = ROOT / "push_pilot_seeds.tsv"
TOOL = "Hadita-som-seed-2026-08-29"
PAUSE_S = 0.4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+",
                    default=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = TrpClient.from_env()
    fd = client.fulldoc(COL_ID, DOC_ID)
    by_name: dict[str, list[dict]] = collections.defaultdict(list)
    for p in fd["pageList"]["pages"]:
        by_name[p.get("imgFileName")].append(p)
    for v in by_name.values():
        v.sort(key=lambda p: p["pageNr"])

    records, failed, skipped = [], [], []
    for i, page_no in enumerate(sorted(args.pages), 1):
        xml_path = SEED_DIR / f"Hadita_{page_no}_som_flagged.xml"
        if not xml_path.exists():
            failed.append((page_no, "no flagged XML"))
            continue
        img_name = f"Hadita_{page_no}.jpeg"
        copies = by_name.get(img_name, [])
        if not copies:
            failed.append((page_no, "image not in document"))
            continue
        target = copies[0]
        page_nr = target["pageNr"]
        existing = target.get("tsList", {}).get("transcripts", [])
        if any(t.get("toolName") == TOOL for t in existing):
            skipped.append(page_no)
            print(f"[{i}] page {page_no:>3}  pageNr={page_nr:>3}  already pushed, skipping")
            continue

        xml = xml_path.read_text(encoding="utf-8")
        m = re.search(r'imageFilename="([^"]+)"', xml)
        if not m or m.group(1) != img_name:
            failed.append((page_no, f"imageFilename mismatch: {m.group(1) if m else None!r}"))
            continue
        n_flag = xml.count("[?]")
        n_text = len([1 for t in re.findall(r"<Unicode>([^<]*)</Unicode>", xml) if t.strip()])

        if args.dry_run:
            print(f"[{i}] page {page_no:>3} → pageNr={page_nr:>3}  "
                  f"{n_text} non-empty cells, {n_flag} [?] flags  [dry-run]")
            continue
        try:
            resp = client.push_transcript(
                COL_ID, DOC_ID, page_nr, xml, status="NEW", tool_name=TOOL,
                note="E10 pilot seed: SoM(g37-flash)+ink repair; [?] = SoM/Hadid02 mismatch")
            ts_id = resp.get("tsId") if isinstance(resp, dict) else None
            records.append({"page": page_no, "pageNr": page_nr,
                            "cells_with_text": n_text, "flags": n_flag, "tsId": ts_id})
            print(f"[{i}] page {page_no:>3} → pageNr={page_nr:>3}  "
                  f"{n_text} cells, {n_flag} flags  tsId={ts_id}")
        except Exception as exc:
            failed.append((page_no, str(exc)[:160]))
            print(f"[{i}] page {page_no:>3}  FAILED: {str(exc)[:160]}")
        time.sleep(PAUSE_S)

    if records:
        cols = ["page", "pageNr", "cells_with_text", "flags", "tsId"]
        hdr = not LOG_TSV.exists()
        with open(LOG_TSV, "a", encoding="utf-8") as fh:
            if hdr:
                fh.write("\t".join(cols) + "\n")
            for r in records:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\npushed {len(records)}, skipped {len(skipped)}, failed {len(failed)}")
    for p, e in failed:
        print(f"  page {p}: {e}")


if __name__ == "__main__":
    main()
