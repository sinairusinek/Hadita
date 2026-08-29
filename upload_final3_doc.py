#!/usr/bin/env python3
"""upload_final3_doc.py — create a fresh Transkribus document from final3/.

Uses the structured-upload REST flow: POST /uploads?collId= with a page
manifest, then PUT each image+XML pair. The result is a brand-new document
(no duplicate pages, no stale layers), title "Hadita-final3".

Usage:
  python upload_final3_doc.py --dry-run
  python upload_final3_doc.py
  python upload_final3_doc.py --pages 11 12 13   # partial doc for testing
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
TITLE = "Hadita-final3"


def page_files(pages: list[int] | None) -> list[tuple[int, Path, Path]]:
    out = []
    xmls = sorted(FINAL3.glob("Hadita_*.xml"),
                  key=lambda p: int(p.stem.split("_")[1]))
    for x in xmls:
        n = int(x.stem.split("_")[1])
        if pages and n not in pages:
            continue
        img = FINAL3 / f"Hadita_{n}.jpeg"
        if not img.exists():
            print(f"  ! page {n}: missing jpeg, skipped")
            continue
        # sanity: XML must reference its own image
        m = re.search(r'imageFilename="([^"]+)"', x.read_text(encoding="utf-8"))
        if not m or m.group(1) != img.name:
            print(f"  ! page {n}: imageFilename mismatch ({m and m.group(1)}), skipped")
            continue
        out.append((n, img, x))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+")
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = page_files(args.pages)
    print(f"{len(files)} pages to upload as '{args.title}'")
    if args.dry_run or not files:
        for i, (n, img, xml) in enumerate(files, 1):
            print(f"  pageNr {i}: {img.name} + {xml.name}")
        return

    client = TrpClient.from_env()
    manifest = {
        "md": {"title": args.title},
        "pageList": {"pages": [
            {"fileName": img.name, "pageXmlName": xml.name, "pageNr": i}
            for i, (n, img, xml) in enumerate(files, 1)]},
    }
    r = client.session.post(f"{client.base}/uploads?collId={COL_ID}",
                            json=manifest, timeout=60)
    r.raise_for_status()
    up = r.json()
    upload_id = up.get("uploadId") or up.get("upload", {}).get("uploadId")
    if not upload_id:
        raise RuntimeError(f"no uploadId in response: {str(up)[:300]}")
    print(f"uploadId={upload_id}")

    for i, (n, img, xml) in enumerate(files, 1):
        for attempt in (1, 2, 3):
            try:
                r = client.session.put(
                    f"{client.base}/uploads/{upload_id}",
                    files={
                        "img": (img.name, img.read_bytes(), "application/octet-stream"),
                        "xml": (xml.name, xml.read_bytes(), "application/octet-stream"),
                    }, timeout=300)
                r.raise_for_status()
                break
            except Exception as exc:
                print(f"  page {n} attempt {attempt} failed: {str(exc)[:120]}")
                if attempt == 3:
                    raise
                time.sleep(5)
        done = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        job = done.get("jobId")
        print(f"[{i}/{len(files)}] uploaded Hadita_{n}" + (f"  jobId={job}" if job else ""))
        time.sleep(0.3)
    print("\nAll pages uploaded — Transkribus ingests the document as a job; "
          "check the collection for the new doc in a minute.")


if __name__ == "__main__":
    main()
