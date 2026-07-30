#!/usr/bin/env python3
"""run_hadid.py — run a Hadid PyLaia model over pages of the Transkribus doc.

The plan assumed this was UI-only. It isn't: the legacy TrpServer endpoint

    POST /pylaia/{colId}/{modelId}/recognition?id={docId}&pages={sel}

accepts the same password-grant session as every other call in this repo and
returns a jobId, pollable at GET /jobs/{jobId}. Verified 2026-07-30 on page 11
(pageNr 5): "PyLaia Decoding" finished in ~60s and wrote a layer with 532 table
cells, 523 of them transcribed, parented to the pushed segmentation.

The Metagrapho (processing/v1) API is the modern route but is unusable here —
the account is not enrolled, so it 401s with "No audience in the token" (see
HebHTR/Transkribus/README.md) — and it works on loose images rather than pages
in a document, so its output would not land as a transcript layer.

Model ids in collection 2377415: hadid01 = 592509, hadid02 = 592709 (newest).

Pages are given as Hadita page numbers; the mapping to Transkribus pageNr comes
from push_final2_seg.tsv, so recognition always targets the copy that carries
the segmentation (each image is in the document twice).

Usage:
  python run_hadid.py --gt-pages --dry-run
  python run_hadid.py --pages 12 13 14
  python run_hadid.py --gt-pages --pages 11 12 13 14 15 16 17 18 19 20
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")

from transkribus.client import TrpClient  # noqa: E402

COL_ID = 2377415
DOC_ID = 17829738
MODELS = {"hadid01": 592509, "hadid02": 592709}
PUSH_LOG = ROOT / "push_final2_seg.tsv"
RUN_LOG = ROOT / "hadid_runs.tsv"
GT_PAGES = [3, 4, 5, 6, 9, 10]
POLL_S = 15
POLL_MAX = 240          # 1 hour per job


def page_map() -> dict[int, int]:
    """Hadita page number → Transkribus pageNr carrying the segmentation."""
    if not PUSH_LOG.exists():
        sys.exit(f"{PUSH_LOG.name} missing — run push_final2_seg.py first")
    with open(PUSH_LOG, encoding="utf-8") as fh:
        return {int(r["page"]): int(r["pageNr"]) for r in csv.DictReader(fh, delimiter="\t")}


def wait(client: TrpClient, job_id: str) -> dict:
    last = ""
    for _ in range(POLL_MAX):
        r = client.session.get(f"{client.base}/jobs/{job_id}", timeout=60)
        if r.status_code >= 300:
            return {"state": "UNKNOWN", "success": False,
                    "description": f"status {r.status_code}"}
        j = r.json()
        descr = str(j.get("description"))[:60]
        if descr != last:
            print(f"      {j.get('state')}: {descr}", flush=True)
            last = descr
        if j.get("state") in ("FINISHED", "FAILED", "CANCELED", "ERROR"):
            return j
        time.sleep(POLL_S)
    return {"state": "TIMEOUT", "success": False, "description": "poll limit reached"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+", default=[])
    ap.add_argument("--gt-pages", action="store_true",
                    help=f"include the proxy-GT pages {GT_PAGES}")
    ap.add_argument("--model", choices=sorted(MODELS), default="hadid02")
    ap.add_argument("--batch", type=int, default=5,
                    help="pages per recognition job (one job per batch)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pages = sorted(set(args.pages) | (set(GT_PAGES) if args.gt_pages else set()))
    if not pages:
        ap.error("give --pages and/or --gt-pages")

    pmap = page_map()
    missing = [p for p in pages if p not in pmap]
    if missing:
        sys.exit(f"no pushed segmentation for pages {missing}")

    model_id = MODELS[args.model]
    client = TrpClient.from_env()

    batches = [pages[i:i + args.batch] for i in range(0, len(pages), args.batch)]
    print(f"{args.model} (model {model_id}) over {len(pages)} page(s) "
          f"in {len(batches)} job(s)\n")

    rows = []
    for bi, chunk in enumerate(batches, 1):
        sel = ",".join(str(pmap[p]) for p in chunk)
        label = ", ".join(f"p{p}(nr{pmap[p]})" for p in chunk)
        if args.dry_run:
            print(f"[{bi}/{len(batches)}] would submit pages={sel}  {label}")
            continue
        print(f"[{bi}/{len(batches)}] submitting pages={sel}  {label}", flush=True)
        r = client.session.post(
            f"{client.base}/pylaia/{COL_ID}/{model_id}/recognition",
            params={"id": DOC_ID, "pages": sel}, timeout=120)
        if r.status_code >= 300:
            print(f"      SUBMIT FAILED {r.status_code}: {r.text[:200]}")
            rows.append({"model": args.model, "pages": sel, "jobId": "",
                         "state": "SUBMIT_FAILED", "success": False,
                         "seconds": 0, "hadita_pages": " ".join(map(str, chunk))})
            continue
        job_id = r.text.strip()
        print(f"      jobId={job_id}", flush=True)
        j = wait(client, job_id)
        secs = (int(j.get("endTime", 0)) - int(j.get("startTime", 0))) // 1000 \
            if j.get("endTime") and j.get("startTime") else 0
        print(f"      {j.get('state')} success={j.get('success')} ({secs}s)")
        rows.append({"model": args.model, "pages": sel, "jobId": job_id,
                     "state": j.get("state"), "success": bool(j.get("success")),
                     "seconds": secs, "hadita_pages": " ".join(map(str, chunk))})

    if rows:
        cols = ["model", "hadita_pages", "pages", "jobId", "state", "success", "seconds"]
        header = not RUN_LOG.exists()
        with open(RUN_LOG, "a", encoding="utf-8") as fh:
            if header:
                fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")
        ok = sum(1 for r in rows if r["success"])
        print(f"\n{ok}/{len(rows)} job(s) succeeded; log → {RUN_LOG.name}")


if __name__ == "__main__":
    main()
