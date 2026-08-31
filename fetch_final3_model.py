#!/usr/bin/env python3
"""fetch_final3_model.py — pull any Transkribus model layer off the final3 doc
and write it in the shape score_exp2608.py reads.

Generalises fetch_hadid02_final2.py, which is pinned to one model and to the
final2 push log. Here the model is any id, and pageNr is resolved live from the
document (same as run_trp_model.py), so it follows whatever pages the doc has.

For each page it takes the newest transcript whose toolName names the model id,
so a re-run is picked up automatically. Writes both halves of the exp2608
convention: exp2608/Hadita_{N}_{tag}.xml (the raw transcript) and .json (rows).

  python fetch_final3_model.py --model 386877 --tag trp-periodicals \
      --pages 3 4 5 6 9 10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")

from transkribus.client import TrpClient  # noqa: E402
from agreement_layer import load_source  # noqa: E402

COL_ID = 2377415
DOC_ID = 18537955          # Hadita-final3
OUT_DIR = ROOT / "exp2608"


def page_map(client, doc: int) -> dict[int, int]:
    fd = client.fulldoc(COL_ID, doc)
    out = {}
    for pg in fd["pageList"]["pages"]:
        stem = (pg.get("imgFileName") or "").replace("Hadita_", "").replace(".jpeg", "")
        if stem.isdigit():
            out[int(stem)] = pg["pageNr"]
    return out, fd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--doc", type=int, default=DOC_ID)
    ap.add_argument("--pages", type=int, nargs="+", default=[3, 4, 5, 6, 9, 10])
    args = ap.parse_args()

    client = TrpClient.from_env()
    pmap, fd = page_map(client, args.doc)
    by_nr = {int(p["pageNr"]): p for p in fd["pageList"]["pages"]}
    OUT_DIR.mkdir(exist_ok=True)

    got, none = 0, []
    for page in sorted(args.pages):
        nr = pmap.get(page)
        if nr is None or nr not in by_nr:
            none.append(page)
            continue
        cands = [t for t in by_nr[nr].get("tsList", {}).get("transcripts", [])
                 if str(args.model) in str(t.get("toolName", ""))]
        if not cands:
            none.append(page)
            print(f"  page {page:>3} (pageNr {nr}): no layer for model {args.model}")
            continue
        newest = max(cands, key=lambda t: int(t.get("tsId", 0)))
        xml = client.fetch_transcript(newest["url"])
        xml_p = OUT_DIR / f"Hadita_{page}_{args.tag}.xml"
        xml_p.write_text(xml, encoding="utf-8")
        rows = load_source(xml_p)
        (OUT_DIR / f"Hadita_{page}_{args.tag}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        filled = sum(1 for u in re.findall(r"<Unicode>(.*?)</Unicode>", xml, re.S)
                     if u.strip())
        got += 1
        print(f"  page {page:>3} (pageNr {nr}): tsId={newest.get('tsId')}  "
              f"{len(rows)} rows, {filled} non-empty cells")

    print(f"\nfetched {got}, no layer {len(none)}" + (f" -> {none}" if none else ""))


if __name__ == "__main__":
    main()
