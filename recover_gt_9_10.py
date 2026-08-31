"""Transplant the recovered page 9/10 GT text into the final3 geometry.

Pages 9 and 10 were RA-corrected months ago, but the corrections live on doc
15829823 (Hadita-Processed), while exp2608/Hadita_{9,10}_gt-f3.xml was built
from doc 18537955 — where those pages only ever got a thin Hadita-GT-proxy
layer. So the scorers saw 15 and 35 filled cells instead of 120 and 137, and
E13 reported a spurious 100% error rate on both pages.

The recovered transcripts are on the OLD page geometry (2299x2920) with
different polygons, so this copies TEXT ONLY, keyed by the shared
`line_cell_r{r}_c{c}` id scheme, into final3's coordinates. Row alignment was
verified first: offsets -4..+4 were scored against som-f3v2 and offset 0 won
on both pages (58.0% / 52.5% agreement vs ~26% for any shift).

About a quarter of the recovered cells have no TextLine in the final3 file:
the ink gate did not emit one, because the content is a single faint mark
(-, checkmark, ditto, zero), concentrated in columns 17, 10 and 12. Those are
the sparse marks E6 found are missed by every reader. Dropping them would
discard real GT, so this ADDS a TextLine for each, using the TableCell polygon
that final3 already carries for every cell.

  python recover_gt_9_10.py            # writes exp2608/Hadita_{9,10}_gt-f3.xml
  python recover_gt_9_10.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

OUT = Path("exp2608")
SRC = Path("/tmp")
PAGES = (9, 10)

CELL_RX = re.compile(
    r'<TextLine id="(line_cell_r\d+_c\d+)".*?<Unicode>(.*?)</Unicode>', re.S)
TABLECELL_RX = re.compile(
    r'<TableCell id="cell_r(\d+)_c(\d+)"[^>]*>\s*<Coords points="([^"]+)"')


def escape(s: str) -> str:
    """PAGE XML <Unicode> must escape these or Transkribus 500s on upload."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def recovered_text(page: int) -> dict[str, str]:
    xml = (SRC / f"gt_recover_Hadita_{page}.xml").read_text(encoding="utf-8")
    return {m.group(1): m.group(2).strip()
            for m in CELL_RX.finditer(xml) if m.group(2).strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for page in PAGES:
        target = OUT / f"Hadita_{page}_gt-f3.xml"
        xml = target.read_text(encoding="utf-8")
        text = recovered_text(page)

        polys = {f"line_cell_r{m.group(1)}_c{m.group(2)}": m.group(3)
                 for m in TABLECELL_RX.finditer(xml)}
        present = set(re.findall(r'<TextLine id="(line_cell_r\d+_c\d+)"', xml))

        filled = added = 0

        def put(m: re.Match) -> str:
            """Set the Unicode of an existing cell TextLine."""
            nonlocal filled
            tid = m.group(1)
            if tid not in text:
                return m.group(0)
            filled += 1
            return (m.group(0)[:m.start(2) - m.start(0)]
                    + escape(text[tid])
                    + m.group(0)[m.end(2) - m.start(0):])

        xml = CELL_RX.sub(put, xml)

        # Cells the ink gate never emitted a TextLine for: add one, anchored on
        # the TableCell polygon, with a baseline across its lower third.
        missing = [t for t in text if t not in present]
        for tid in sorted(missing):
            pts = polys.get(tid)
            if not pts:
                print(f"  page {page}: {tid} has no TableCell geometry, skipped")
                continue
            xs = [int(p.split(",")[0]) for p in pts.split()]
            ys = [int(p.split(",")[1]) for p in pts.split()]
            by = int(min(ys) + 0.85 * (max(ys) - min(ys)))
            line = (f'      <TextLine id="{tid}">\n'
                    f'        <Coords points="{pts}"></Coords>\n'
                    f'        <Baseline points="{min(xs)},{by} {max(xs)},{by}">'
                    f'</Baseline>\n'
                    f'        <TextEquiv><Unicode>{escape(text[tid])}'
                    f'</Unicode></TextEquiv>\n'
                    f'      </TextLine>\n')
            r, c = re.match(r"line_cell_r(\d+)_c(\d+)", tid).groups()
            anchor = f'<TableCell id="cell_r{r}_c{c}"'
            i = xml.find(anchor)
            if i == -1:
                print(f"  page {page}: no anchor for {tid}, skipped")
                continue
            j = xml.find("</TableCell>", i)
            xml = xml[:j] + line + "    " + xml[j:]
            added += 1

        print(f"page {page}: {len(text)} recovered cells -> "
              f"{filled} filled in place, {added} TextLines added")
        if args.dry_run:
            continue
        backup = target.with_suffix(".xml.pre-recover")
        if not backup.exists():
            shutil.copy(target, backup)
        target.write_text(xml, encoding="utf-8")
        print(f"  wrote {target} (backup {backup.name})")


if __name__ == "__main__":
    main()
