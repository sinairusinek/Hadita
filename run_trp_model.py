#!/usr/bin/env python3
"""run_trp_model.py — run any Transkribus text model on pages of a document.

Generalises run_hadid.py, which is pinned to the hadid models and to the old
final2 page map. Here the model is any id from
`GET /models/text?collId=`, and the pageNr map is read live from the document,
so it follows whatever pages the doc actually has.

Results land as a new transcript layer on each page (additive — nothing is
overwritten). Jobs run at "high priority (using paid credits)", so --dry-run
first and keep batches small.

  python run_trp_model.py --doc 18537955 --model 386877 --pages 3 4 5 --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")

from transkribus.client import TrpClient  # noqa: E402

BASE = "https://transkribus.eu/TrpServer/rest"
COL_ID = 2377415


def page_map(client, doc: int) -> dict[int, int]:
    fd = client.fulldoc(COL_ID, doc)
    out = {}
    for pg in fd["pageList"]["pages"]:
        stem = (pg.get("imgFileName") or "").replace("Hadita_", "").replace(".jpeg", "")
        if stem.isdigit():
            out[int(stem)] = pg["pageNr"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", type=int, required=True)
    ap.add_argument("--model", type=int, required=True)
    ap.add_argument("--pages", type=int, nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    c = TrpClient.from_env()
    pm = page_map(c, args.doc)
    nrs = [pm[p] for p in args.pages if p in pm]
    missing = [p for p in args.pages if p not in pm]
    if missing:
        print(f"not in doc {args.doc}: {missing}")
    print(f"model {args.model} -> doc {args.doc}, {len(nrs)} pages (pageNr {nrs})")
    if args.dry_run:
        return

    jobs = []
    for i in range(0, len(nrs), args.batch):
        chunk = nrs[i:i + args.batch]
        url = f"{BASE}/pylaia/{COL_ID}/{args.model}/recognition"
        r = c.session.post(url, params={"id": args.doc,
                                        "pages": ",".join(map(str, chunk))}, timeout=120)
        if r.status_code >= 300:
            print(f"  batch {chunk}: HTTP {r.status_code} {r.text[:200]}")
            continue
        jid = r.text.strip().strip('"')
        jobs.append(jid)
        print(f"  batch {chunk} -> jobId {jid}")
        time.sleep(1)

    if args.no_wait or not jobs:
        return
    for jid in jobs:
        while True:
            s = c.session.get(f"{BASE}/jobs/{jid}", timeout=60).json()
            st = s.get("state")
            if st in ("FINISHED", "FAILED", "CANCELED"):
                print(f"  job {jid}: {st} — {str(s.get('description'))[:70]}")
                break
            time.sleep(20)


if __name__ == "__main__":
    main()
