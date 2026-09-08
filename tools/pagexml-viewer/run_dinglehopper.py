#!/usr/bin/env python3
"""Render dinglehopper side-by-side diff reports for the Hadita GT pages.

dinglehopper extracts text from TextRegion; this corpus keeps its text in
TableCell/TextLine, so it reads the pages as empty and reports CER 0. We
flatten each page to plain text first, one cell per line over the union of
cell keys -- the same alignment convention run_cerberus.py uses, so a cell a
model skipped stays an empty line instead of shifting every later cell.
"""
import sys, os, glob, re, subprocess, argparse
import xml.etree.ElementTree as ET

REPO = "/Users/sinairusinek/Documents/GitHub/Hadita"
EXP = os.path.join(REPO, "exp2608")
PAGES = [3, 4, 5, 6, 9, 10]
DH = os.path.join(REPO, ".venv-dinglehopper", "bin", "dinglehopper")


def cells(path):
    root = ET.parse(path).getroot()
    ns = {'p': root.tag.split('}')[0][1:]}
    out = {}
    for c in root.findall('.//p:TableCell', ns):
        u = c.find('.//p:Unicode', ns)
        out[(int(c.get('row')), int(c.get('col')))] = (u.text or '').strip() if u is not None else ''
    return out


def write_pair(gt, hyp, gt_path, hyp_path):
    """One cell per line; drop cells empty on both sides (no evidence either way)."""
    keys = sorted(set(gt) | set(hyp))
    g_l, h_l = [], []
    for k in keys:
        g, h = gt.get(k, ''), hyp.get(k, '')
        if not g and not h:
            continue
        g_l.append(g)
        h_l.append(h)
    open(gt_path, 'w', encoding='utf-8').write("\n".join(g_l))
    open(hyp_path, 'w', encoding='utf-8').write("\n".join(h_l))
    return len(g_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="model suffixes; default all found")
    ap.add_argument("--out", default=os.path.join(REPO, "tools/pagexml-viewer/dinglehopper"))
    args = ap.parse_args()

    if not os.path.exists(DH):
        sys.exit("dinglehopper not found. Create it with:\n"
                 "  /usr/bin/python3 -m venv .venv-dinglehopper\n"
                 "  .venv-dinglehopper/bin/pip install dinglehopper")

    models = set(args.models or [])
    if not models:
        for f in glob.glob(os.path.join(EXP, "Hadita_*_*.xml")):
            m = re.match(r'Hadita_(\d+)_(.+)\.xml$', os.path.basename(f))
            if m and int(m.group(1)) in PAGES and m.group(2) != 'gt-f3':
                models.add(m.group(2))

    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "_txt")
    os.makedirs(tmp, exist_ok=True)

    for model in sorted(models):
        for p in PAGES:
            hyp_xml = os.path.join(EXP, f"Hadita_{p}_{model}.xml")
            gt_xml = os.path.join(EXP, f"Hadita_{p}_gt-f3.xml")
            if not (os.path.exists(hyp_xml) and os.path.exists(gt_xml)):
                continue
            g = os.path.join(tmp, f"{p}_{model}_gt.txt")
            h = os.path.join(tmp, f"{p}_{model}_ocr.txt")
            n = write_pair(cells(gt_xml), cells(hyp_xml), g, h)
            subprocess.run([DH, g, h, f"p{p}_{model}", args.out,
                            "--differences", "true"], check=True)
            print(f"  p{p} {model}: {n} cells -> {args.out}/p{p}_{model}.html")


if __name__ == "__main__":
    main()
