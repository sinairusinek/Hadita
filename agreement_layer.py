"""Per-cell agreement triage across multiple OCR/HTR sources.

Usage:
  python agreement_layer.py --page 11 \\
      --sources g3=g3_results/Hadita_11_G3.json,hadid01=g3_results/Hadita_11_Hadid01.xml \\
      [--primary g3] [--out-dir g3_results]

Outputs:
  agreement_{N}.tsv          row x col x per-source values + agreement state
  Hadita_{N}_{primary}_flagged.xml
                             primary XML patched so MISMATCH cells get [?]
                             appended in <Unicode>; ONLY_<other> cells get
                             an XML comment listing the other source's value.

Agreement states: MATCH | MISMATCH | ONLY_<source> | BOTH_EMPTY
(For N>2 sources: MATCH iff all non-empty sources agree.)
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from digit_norm import LEFT_COLS, convert_cell, normalize_for_compare, VOL_COL

CELL_RE = re.compile(r'(<TableCell\s+id="cell_r(\d+)_c(\d+)"[^>]*>)(.*?)(</TableCell>)',
                     re.DOTALL)
# Transkribus PyLaia output keeps the grid in TextLine ids instead of TableCells.
LINE_CELL_RE = re.compile(
    r'(<TextLine\s+id="line_cell_r(\d+)_c(\d+)"[^>]*>)(.*?)(</TextLine>)',
    re.DOTALL)
UNICODE_RE = re.compile(r'(<Unicode>)([^<]*)(</Unicode>)')


def _load_json(path: Path) -> list[dict]:
    d = json.load(open(path, encoding="utf-8"))
    return [{c: (r.get(c, "") or "").strip() for c in LEFT_COLS} for r in d]


def _load_xml(path: Path) -> list[dict]:
    """Parse a PAGE XML into a row-indexed list of dicts keyed by LEFT_COLS."""
    xml = path.read_text(encoding="utf-8")
    cells: dict[tuple[int, int], str] = {}
    matches = list(CELL_RE.finditer(xml)) or list(LINE_CELL_RE.finditer(xml))
    for m in matches:
        r, c = int(m.group(2)), int(m.group(3))
        body = m.group(4)
        parts = [u for u in UNICODE_RE.findall(body)]
        text = next((u[1] for u in parts if u[1].strip()), "")
        cells[(r, c)] = text.strip()
    if not cells:
        return []
    max_r = max(r for r, _ in cells) + 1
    rows = []
    for r in range(max_r):
        row = {c: cells.get((r, idx), "") for idx, c in enumerate(LEFT_COLS)}
        rows.append(row)
    return rows


def load_source(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() == ".xml":
        return _load_xml(path)
    raise ValueError(f"Unsupported source extension: {path.suffix}")


def _norm_for_compare(text: str, col: str, vol_text: str) -> str:
    return normalize_for_compare(convert_cell(text, col, vol_text))


def align_by_position(sources: dict[str, list[dict]], primary: str) -> dict[str, int]:
    """For each non-primary source, find the offset that maximizes non-empty
    cell agreement with the primary source. Returns {source_name: offset}."""
    pri_rows = sources[primary]
    offsets = {primary: 0}
    for name, rows in sources.items():
        if name == primary:
            continue
        best_off, best = 0, -1
        for off in range(-2, 3):
            score = 0
            for j in range(len(rows)):
                i = j + off
                if 0 <= i < len(pri_rows):
                    for c in LEFT_COLS:
                        pv = _norm_for_compare(pri_rows[i].get(c, ""), c,
                                               pri_rows[i].get(VOL_COL, ""))
                        sv = _norm_for_compare(rows[j].get(c, ""), c,
                                               rows[j].get(VOL_COL, ""))
                        if pv and pv == sv:
                            score += 1
            if score > best:
                best, best_off = score, off
        offsets[name] = best_off
        print(f"  alignment {name} vs {primary}: offset={best_off:+d}, "
              f"{best} agreeing cells")
    return offsets


def compute_grid(sources: dict[str, list[dict]], offsets: dict[str, int],
                 primary: str) -> list[dict]:
    """Build a per-(row, col) record across all sources, indexed by the primary
    source's row positions."""
    pri_rows = sources[primary]
    records = []
    for i, pri_row in enumerate(pri_rows):
        vol_text = pri_row.get(VOL_COL, "")
        for col in LEFT_COLS:
            cell = {"row": i, "col": col}
            values = {}
            normed = {}
            for name, rows in sources.items():
                j = i - offsets[name]
                if 0 <= j < len(rows):
                    raw = rows[j].get(col, "")
                    values[name] = raw
                    normed[name] = _norm_for_compare(raw, col, vol_text)
                else:
                    values[name] = ""
                    normed[name] = ""
            non_empty = {n: v for n, v in normed.items() if v}
            if not non_empty:
                state = "BOTH_EMPTY"
            elif len(non_empty) == 1:
                state = f"ONLY_{next(iter(non_empty.keys()))}"
            elif len(set(non_empty.values())) == 1:
                state = "MATCH"
            else:
                state = "MISMATCH"
            cell["state"] = state
            for name in sources:
                cell[name] = values[name]
            records.append(cell)
    return records


def write_tsv(records: list[dict], sources: list[str], out_path: Path) -> None:
    cols = ["row", "col", "state"] + sources
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in cols})


def patch_primary_xml(xml_path: Path, records: list[dict], primary: str,
                      others: list[str], out_path: Path) -> tuple[int, int]:
    """Append [?] to <Unicode> of MISMATCH cells; add XML comments listing
    ONLY_<other> values for cells where primary is empty. Returns (n_flagged,
    n_only_other)."""
    xml = xml_path.read_text(encoding="utf-8")
    by_cell = {(r["row"], LEFT_COLS.index(r["col"])): r
               for r in records if r["col"] in LEFT_COLS}
    n_flagged = 0
    n_only_other = 0

    def repl(m: re.Match) -> str:
        nonlocal n_flagged, n_only_other
        header, r_str, c_str, body, tail = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        key = (int(r_str), int(c_str))
        rec = by_cell.get(key)
        if not rec:
            return m.group(0)
        state = rec["state"]
        if state == "MISMATCH":
            def add_flag(um: re.Match) -> str:
                txt = um.group(2)
                if txt and "[?]" not in txt:
                    return f"{um.group(1)}{txt} [?]{um.group(3)}"
                return um.group(0)
            new_body, n_sub = UNICODE_RE.subn(add_flag, body, count=1)
            if n_sub:
                n_flagged += 1
                body = new_body
        elif state.startswith("ONLY_") and not state.endswith(primary):
            other = state[len("ONLY_"):]
            if other in others:
                val = rec.get(other, "")
                comment = (f"\n        <!-- agreement: ONLY_{other}="
                           f"{xml_escape(val)} -->")
                body = comment + body
                n_only_other += 1
        return f"{header}{body}{tail}"

    new_xml = CELL_RE.sub(repl, xml)
    out_path.write_text(new_xml, encoding="utf-8")
    return n_flagged, n_only_other


def parse_sources_arg(arg: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for chunk in arg.split(","):
        if "=" not in chunk:
            raise ValueError(f"--sources entry needs name=path: {chunk}")
        name, path = chunk.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--sources", required=True,
                    help="comma-separated name=path entries, e.g. "
                         "g3=path/to.json,hadid01=path/to.xml")
    ap.add_argument("--primary", default="g3",
                    help="Source name whose XML will be patched + whose row "
                         "positions index the grid (default: g3)")
    ap.add_argument("--out-dir", default="g3_results", type=Path)
    args = ap.parse_args()

    src_paths = parse_sources_arg(args.sources)
    if args.primary not in src_paths:
        print(f"--primary {args.primary} not in --sources", file=sys.stderr)
        return 2

    sources = {name: load_source(path) for name, path in src_paths.items()}
    for name, rows in sources.items():
        print(f"  {name}: {len(rows)} rows from {src_paths[name]}")

    offsets = align_by_position(sources, args.primary)
    records = compute_grid(sources, offsets, args.primary)

    state_counts: dict[str, int] = {}
    for r in records:
        state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1
    print(f"\n  state counts: {state_counts}")

    args.out_dir.mkdir(exist_ok=True)
    tsv_path = args.out_dir / f"agreement_{args.page}.tsv"
    write_tsv(records, list(sources.keys()), tsv_path)
    print(f"  wrote {tsv_path}")

    primary_path = src_paths[args.primary]
    if primary_path.suffix.lower() == ".xml":
        out_xml = args.out_dir / f"Hadita_{args.page}_{args.primary}_flagged.xml"
        others = [n for n in sources if n != args.primary]
        n_flag, n_only = patch_primary_xml(primary_path, records, args.primary,
                                           others, out_xml)
        print(f"  wrote {out_xml} (flagged={n_flag}, only-other-comments={n_only})")
    else:
        sibling_xml = primary_path.with_suffix(".xml")
        if sibling_xml.exists():
            out_xml = args.out_dir / f"Hadita_{args.page}_{args.primary}_flagged.xml"
            others = [n for n in sources if n != args.primary]
            n_flag, n_only = patch_primary_xml(sibling_xml, records,
                                               args.primary, others, out_xml)
            print(f"  wrote {out_xml} (flagged={n_flag}, only-other-comments={n_only})")
        else:
            print(f"  no sibling XML for primary {args.primary} at "
                  f"{sibling_xml}; skipping flagged-XML emit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
