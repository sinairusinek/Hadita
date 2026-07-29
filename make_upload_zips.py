#!/usr/bin/env python3
"""make_upload_zips.py — package Transkribus upload/final2/ for upload.

Transkribus imports a zip whose top-level folder holds the page images with the
matching PAGE XMLs in a `page/` subfolder:

    Hadita-final2_b01/
        Hadita_3.jpeg
        page/Hadita_3.xml

Batches keep each zip small enough for the web uploader and let the first batch
(the six proxy-GT pages) go up on its own for scoring before the rest follows.

Every zip is verified after writing: each image has an XML, each XML's
imageFilename names its partner, and nothing else is in the archive.

Usage:
  python make_upload_zips.py                    # GT batch, 11-20, then 20s
  python make_upload_zips.py --batch-size 10
  python make_upload_zips.py --pages 3 4 5      # one ad-hoc zip
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
FINAL2_DIR = ROOT / "Transkribus upload" / "final2"
OUT_DIR = ROOT / "Transkribus upload" / "final2_zips"
DOC_NAME = "Hadita-final2"
GT_PAGES = [3, 4, 5, 6, 9, 10]
NEXT_BATCH = list(range(11, 21))


def available_pages() -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in FINAL2_DIR.glob("Hadita_*.xml")
                  if (FINAL2_DIR / f"{p.stem}.jpeg").exists())


def write_zip(pages: list[int], out: Path, folder: str) -> tuple[int, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in pages:
            jpeg = FINAL2_DIR / f"Hadita_{p}.jpeg"
            xml = FINAL2_DIR / f"Hadita_{p}.xml"
            z.write(jpeg, f"{folder}/{jpeg.name}")
            z.write(xml, f"{folder}/page/{xml.name}")
    return len(pages), out.stat().st_size


def verify_zip(out: Path, folder: str) -> list[str]:
    problems = []
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        images = {Path(n).stem for n in names
                  if n.startswith(f"{folder}/") and n.endswith(".jpeg")}
        xmls = {Path(n).stem for n in names if n.startswith(f"{folder}/page/")}
        for extra in sorted(images ^ xmls):
            problems.append(f"{extra}: image and XML do not pair up")
        for n in names:
            if not n.startswith(f"{folder}/"):
                problems.append(f"unexpected entry {n}")
            if n.endswith(".xml"):
                raw = z.read(n).decode("utf-8")
                m = re.search(r'imageFilename="([^"]+)"', raw)
                want = f"{Path(n).stem}.jpeg"
                if not m or m.group(1) != want:
                    problems.append(
                        f"{n}: imageFilename={m.group(1) if m else None!r}, expected {want!r}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, nargs="+", help="package exactly these pages")
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()

    pages = available_pages()
    if not pages:
        sys.exit(f"no page pairs in {FINAL2_DIR}")

    if args.pages:
        batches = [("adhoc", sorted(args.pages))]
    else:
        rest = [p for p in pages if p not in GT_PAGES and p not in NEXT_BATCH]
        batches = [("b01-gt", [p for p in GT_PAGES if p in pages]),
                   ("b02-pages11-20", [p for p in NEXT_BATCH if p in pages])]
        for i in range(0, len(rest), args.batch_size):
            chunk = rest[i:i + args.batch_size]
            batches.append((f"b{i // args.batch_size + 3:02d}-"
                            f"pages{chunk[0]}-{chunk[-1]}", chunk))

    total_bytes = 0
    for name, chunk in batches:
        if not chunk:
            continue
        folder = f"{DOC_NAME}_{name}"
        out = OUT_DIR / f"{folder}.zip"
        n, size = write_zip(chunk, out, folder)
        problems = verify_zip(out, folder)
        total_bytes += size
        status = "OK" if not problems else f"{len(problems)} PROBLEM(S)"
        print(f"{out.name:<40} {n:>3} pages  {size/1e6:>6.1f} MB  {status}")
        for p in problems[:5]:
            print(f"      {p}")

    print(f"\n{len(batches)} zip(s) → {OUT_DIR.relative_to(ROOT)}/  "
          f"({total_bytes/1e6:.0f} MB total)")
    print(f"Upload target: collection 2377415, NEW document \"{DOC_NAME}\".")
    print("In Transkribus: Upload → choose zip → it keeps the PAGE XML in page/.")


if __name__ == "__main__":
    main()
