#!/usr/bin/env python3
"""Run CERberus over the Hadita GT pages against every model output."""
import sys, os, glob, json, re, collections
import xml.etree.ElementTree as ET

# CERberus (Haverals) provides the CER engine; clone it next to this file or
# set CERBERUS_PATH. https://github.com/WHaverals/CERberus
for _c in [os.environ.get('CERBERUS_PATH'),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CERberus'),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'CERberus')]:
    if _c and os.path.isdir(_c):
        sys.path.insert(0, os.path.abspath(_c)); break
try:
    from cer_module import cer
except ImportError:
    sys.exit("CERberus not found. git clone https://github.com/WHaverals/CERberus "
             "into this directory, or set CERBERUS_PATH.")

REPO = "/Users/sinairusinek/Documents/GitHub/Hadita"
EXP = os.path.join(REPO, "exp2608")
PAGES = [3, 4, 5, 6, 9, 10]

# Unicode ranges that matter for this register.
RANGES = {
    "Arabic-Indic digits":  [(0x0660, 0x0669)],
    "Arabic letters":       [(0x0620, 0x064A), (0x0671, 0x06D3)],
    "Arabic marks/other":   [(0x064B, 0x065F), (0x06D4, 0x06FF)],
    "ASCII digits":         [(0x0030, 0x0039)],
    "Latin letters":        [(0x0041, 0x005A), (0x0061, 0x007A)],
    "Ditto & quotes":       [(0x0022, 0x0022), (0x2018, 0x201F), (0x2032, 0x2036),
                             (0x00AB, 0x00AB), (0x00BB, 0x00BB), (0x02BA, 0x02BA)],
    "Check marks":          [(0x2713, 0x2714), (0x00D7, 0x00D7), (0x2717, 0x2718)],
    "Punctuation/sep":      [(0x0021, 0x0021), (0x0025, 0x002F), (0x003A, 0x0040),
                             (0x060C, 0x061F)],
}


def cells(path):
    """row,col -> text for every TableCell in a PAGE file."""
    root = ET.parse(path).getroot()
    ns = {'p': root.tag.split('}')[0][1:]}
    out = {}
    for c in root.findall('.//p:TableCell', ns):
        u = c.find('.//p:Unicode', ns)
        out[(int(c.get('row')), int(c.get('col')))] = (u.text or '').strip() if u is not None else ''
    return out


def as_lines(gt, hyp):
    """Emit reference/hypothesis line-aligned over the union of cell keys.

    One cell per line in a fixed order, so a cell the model skipped stays an
    empty line rather than shifting every later cell out of alignment.
    Cells empty on BOTH sides are dropped: they are not evidence either way,
    and a reference of only newlines would make CER meaningless.
    """
    keys = sorted(set(gt) | set(hyp))
    ref_l, hyp_l = [], []
    for k in keys:
        g, h = gt.get(k, ''), hyp.get(k, '')
        if not g and not h:
            continue
        ref_l.append(g)
        hyp_l.append(h)
    return "\n".join(ref_l), "\n".join(hyp_l)


def main():
    models = set()
    for f in glob.glob(os.path.join(EXP, "Hadita_*_*.xml")):
        m = re.match(r'Hadita_(\d+)_(.+)\.xml$', os.path.basename(f))
        if m and int(m.group(1)) in PAGES and m.group(2) != 'gt-f3':
            models.add(m.group(2))

    gt_all = {p: cells(os.path.join(EXP, f"Hadita_{p}_gt-f3.xml")) for p in PAGES}
    report = {}

    for model in sorted(models):
        per_page, ref_parts, hyp_parts = {}, [], []
        for p in PAGES:
            f = os.path.join(EXP, f"Hadita_{p}_{model}.xml")
            if not os.path.exists(f):
                continue
            try:
                hyp = cells(f)
            except Exception as e:
                print(f"  !! {model} p{p}: {e}", file=sys.stderr)
                continue
            r, h = as_lines(gt_all[p], hyp)
            if not r.strip():
                continue
            ref_parts.append(r); hyp_parts.append(h)
            try:
                res = cer(r, h, unicode_ranges=RANGES, debug=False, return_char_stats=False)
                per_page[p] = res['CER']
            except ValueError:
                pass

        if not ref_parts:
            continue
        ref, hyp = "\n".join(ref_parts), "\n".join(hyp_parts)
        res = cer(ref, hyp, unicode_ranges=RANGES, debug=False, return_char_stats=True)
        report[model] = {
            'CER': res['CER'],
            'numCount': res['numCount'],
            'numSub': res['numSub'], 'numIns': res['numIns'], 'numDel': res['numDel'],
            'perPage': per_page,
            'blockStats': res['blockStats'],
            'confusionStats': sorted(res['confusionStats'],
                                     key=lambda d: -d['count'])[:40],
            'charStats': res.get('charStats', [])[:40],
        }
        print(f"{model:22s} CER={res['CER']:6.2f}%  chars={res['numCount']:6d}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cerberus_report.json')
    # pandas leaves NaN in empty-block ratios; bare NaN is not valid JSON and
    # JSON.parse() in the browser rejects it, so emit null instead.
    def clean(o):
        if isinstance(o, float) and o != o:
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    json.dump(clean(report), open(out, 'w'),
              ensure_ascii=False, indent=1, allow_nan=False)
    print(f"\nwrote {out} ({len(report)} models)")


if __name__ == '__main__':
    main()
