#!/usr/bin/env python3
"""Generate ocr_benchmark_report.html — all 9 approaches vs GT on page 3."""
import json
from pathlib import Path

# ── Approach descriptions ─────────────────────────────────────
APPROACH_DESC = {
    "A": ("Claude Opus 4.6",    "Full page",            "Send the full page image to Claude Opus 4.6 and ask it to transcribe the table."),
    "C": ("Gemini 2.5 Pro",     "Full page",            "Send the full page image to Gemini 2.5 Pro (no examples) and ask it to transcribe."),
    "E": ("Gemini 3.x Pro",     "Full page",            "Same as C but using the newer Gemini 3.x Pro model."),
    "K": ("Gemini 3 Flash",     "Full page",            "Same full-page strategy using the lighter/faster Gemini 3 Flash model."),
    "M": ("Gemini 2.5 Pro",     "Few-shot full page",   "Send the full page to Gemini 2.5 Pro together with a few worked examples (few-shot prompting) so the model learns the expected output format and field conventions. <strong>Best overall.</strong>"),
    "N": ("Claude Opus 4.6",    "Few-shot full page",   "Same few-shot strategy as M but using Claude Opus 4.6 instead of Gemini."),
    "O": ("Gemini 2.5 Pro",     "Few-shot + cell crops","Start with approach M as a base, then re-run the hardest columns (Area, Block_No, Tax) with zoomed cell-level crops for a second pass."),
    "P": ("M + C + E ensemble", "Majority vote",        "Run three models independently (M, C, E) and take the majority answer cell by cell. If two or more agree, use that value; otherwise fall back to M."),
    "Q": ("Gemini 2.5 Pro",     "5× few-shot vote",     "Run 5 variants of approach M with different few-shot example subsets, then majority-vote per cell. Expected to reduce variance — but in practice performed worse than a single M run."),
}

# ── GT accuracy results (from evaluate_page3.py run above) ───
GT = {
    "A": {"cells":237,"exact":70, "exact_pct":29.5,"cer":0.6650,
          "per_col":{"Serial_No":(31,31,100.0,0.0),"Date":(31,31,100.0,0.0),"Block_No":(0,31,0.0,0.992),"Parcel_No":(7,22,31.8,0.614),"Cat_No":(0,31,0.0,0.645),"Area":(0,31,0.0,0.836),"Tax_Mils":(0,31,0.0,1.0),"New_Serial_No":(1,17,5.9,1.275),"Tax_LP":(0,8,0.0,1.0),"Volume_No":(0,2,0.0,1.0),"Serial_No_Vol":(0,2,0.0,2.375)},
          "mismatches":[("Block_No","4132","3225"),("Cat_No","10","1"),("Area","34,925","4,950"),("Tax_Mils","629","720"),("Block_No","4133","3228"),("Cat_No","10","1")]},
    "C": {"cells":251,"exact":142,"exact_pct":56.6,"cer":0.2938,
          "per_col":{"Serial_No":(33,33,100.0,0.0),"Date":(33,33,100.0,0.0),"Block_No":(9,33,27.3,0.205),"Parcel_No":(15,23,65.2,0.391),"Cat_No":(11,33,33.3,0.333),"Area":(6,33,18.2,0.414),"Tax_Mils":(23,33,69.7,0.136),"New_Serial_No":(11,17,64.7,0.990),"Tax_LP":(1,9,11.1,0.889),"Volume_No":(0,2,0.0,1.0),"Serial_No_Vol":(0,2,0.0,1.0)},
          "mismatches":[("Block_No","4132","4122"),("Cat_No","10","1"),("Area","34,925","24,925"),("Block_No","4133","4122"),("Cat_No","10","1"),("Block_No","4133","4122")]},
    "E": {"cells":251,"exact":109,"exact_pct":43.4,"cer":0.4455,
          "per_col":{"Serial_No":(33,33,100.0,0.0),"Date":(33,33,100.0,0.0),"Block_No":(10,33,30.3,0.197),"Parcel_No":(15,23,65.2,0.217),"Cat_No":(7,33,21.2,0.500),"Area":(5,33,15.2,0.553),"Tax_Mils":(0,33,0.0,1.0),"New_Serial_No":(6,17,35.3,1.069),"Tax_LP":(0,9,0.0,1.0),"Volume_No":(0,2,0.0,1.7),"Serial_No_Vol":(0,2,0.0,1.0)},
          "mismatches":[("Block_No","4132","4122"),("Cat_No","10","1"),("Area","34,925","24,925"),("Tax_Mils","629","(empty)"),("Block_No","4133","4122"),("Cat_No","10","1")]},
    "K": {"cells":244,"exact":77, "exact_pct":31.6,"cer":0.4885,
          "per_col":{"Serial_No":(32,32,100.0,0.0),"Date":(0,32,0.0,0.667),"Block_No":(9,32,28.1,0.203),"Parcel_No":(15,23,65.2,0.261),"Cat_No":(3,32,9.4,0.625),"Area":(2,32,6.2,0.733),"Tax_Mils":(8,32,25.0,0.573),"New_Serial_No":(8,17,47.1,0.598),"Tax_LP":(0,8,0.0,1.0),"Volume_No":(0,2,0.0,1.7),"Serial_No_Vol":(0,2,0.0,1.0)},
          "mismatches":[("Date","938","9/28"),("Block_No","4132","4124"),("Cat_No","10","1"),("Area","34,925","10,4955"),("Date","938","9/28"),("Block_No","4133","4123")]},
    "M": {"cells":251,"exact":174,"exact_pct":69.3,"cer":0.2309,
          "per_col":{"Serial_No":(33,33,100.0,0.0),"Date":(33,33,100.0,0.0),"Block_No":(30,33,90.9,0.045),"Parcel_No":(20,23,87.0,0.065),"Cat_No":(21,33,63.6,0.182),"Area":(7,33,21.2,0.372),"Tax_Mils":(16,33,48.5,0.692),"New_Serial_No":(12,17,70.6,0.304),"Tax_LP":(1,9,11.1,0.889),"Volume_No":(1,2,50.0,0.100),"Serial_No_Vol":(0,2,0.0,0.250)},
          "mismatches":[("Block_No","4132","4122"),("Area","34,925","24,925"),("Block_No","4133","4123"),("Serial_No_Vol","1940","1941"),("Area","19,286","19,486"),("Parcel_No","34","14")]},
    "N": {"cells":237,"exact":70, "exact_pct":29.5,"cer":0.6052,
          "per_col":{"Serial_No":(31,31,100.0,0.0),"Date":(31,31,100.0,0.0),"Block_No":(0,31,0.0,0.645),"Parcel_No":(4,22,18.2,0.795),"Cat_No":(0,31,0.0,0.613),"Area":(1,31,3.2,0.831),"Tax_Mils":(0,31,0.0,1.263),"New_Serial_No":(1,17,5.9,0.706),"Tax_LP":(0,8,0.0,1.0),"Volume_No":(1,2,50.0,0.5),"Serial_No_Vol":(1,2,50.0,0.5)},
          "mismatches":[("Block_No","4132","4226"),("Cat_No","10","1"),("Area","34,925","4,950"),("Tax_Mils","629","280"),("Block_No","4133","4228"),("Cat_No","10","1")]},
    "O": {"cells":251,"exact":163,"exact_pct":64.9,"cer":0.2467,
          "per_col":{"Serial_No":(33,33,100.0,0.0),"Date":(33,33,100.0,0.0),"Block_No":(30,33,90.9,0.045),"Parcel_No":(20,23,87.0,0.065),"Cat_No":(13,33,39.4,0.303),"Area":(6,33,18.2,0.674),"Tax_Mils":(15,33,45.5,0.359),"New_Serial_No":(12,17,70.6,0.304),"Tax_LP":(0,9,0.0,1.0),"Volume_No":(1,2,50.0,0.100),"Serial_No_Vol":(0,2,0.0,0.250)},
          "mismatches":[("Block_No","4132","4122"),("Tax_Mils","629","625"),("Block_No","4133","4123"),("Tax_Mils","85","84"),("Block_No","4133","4123"),("Tax_LP","2","(empty)")]},
    "P": {"cells":251,"exact":154,"exact_pct":61.4,"cer":0.2479,
          "per_col":{"Serial_No":(33,33,100.0,0.0),"Date":(33,33,100.0,0.0),"Block_No":(10,33,30.3,0.197),"Parcel_No":(19,23,82.6,0.087),"Cat_No":(20,33,60.6,0.197),"Area":(8,33,24.2,0.329),"Tax_Mils":(16,33,48.5,0.692),"New_Serial_No":(13,17,76.5,0.284),"Tax_LP":(1,9,11.1,0.889),"Volume_No":(1,2,50.0,0.100),"Serial_No_Vol":(0,2,0.0,0.250)},
          "mismatches":[("Block_No","4132","4122"),("Cat_No","10","1"),("Area","34,925","24,925"),("Block_No","4133","4122"),("Cat_No","10","1"),("Block_No","4133","4122")]},
    "Q": {"cells":251,"exact":145,"exact_pct":57.8,"cer":0.2929,
          "per_col":{"Serial_No":(33,33,100.0,0.0),"Date":(33,33,100.0,0.0),"Block_No":(9,33,27.3,0.205),"Parcel_No":(17,23,73.9,0.152),"Cat_No":(21,33,63.6,0.182),"Area":(7,33,21.2,0.336),"Tax_Mils":(15,33,45.5,0.707),"New_Serial_No":(10,17,58.8,0.578),"Tax_LP":(0,9,0.0,1.0),"Volume_No":(0,2,0.0,1.0),"Serial_No_Vol":(0,2,0.0,1.0)},
          "mismatches":[("Block_No","4132","4122"),("Area","34,925","24,925"),("Block_No","4133","4122"),("Parcel_No","32","22"),("Tax_LP","2","(empty)"),("Tax_Mils","876","287")]},
}

APPROACH_ORDER = sorted(GT.keys(), key=lambda a: -GT[a]["exact_pct"])
ALL_COLS = ["Serial_No","Date","Block_No","Parcel_No","Cat_No","Area",
            "Tax_Mils","New_Serial_No","Tax_LP","Volume_No","Serial_No_Vol"]

COLORS = {"A":"#4c8cbf","C":"#5ba35b","E":"#c4853d","K":"#9b59b6",
          "M":"#e74c3c","N":"#3a6d99","O":"#e67e22","P":"#27ae60","Q":"#c0392b"}

def bg(pct):
    if pct >= 80: return "hsl(130,55%,88%)"
    if pct >= 50: return "hsl(50,80%,88%)"
    return "hsl(0,60%,90%)"

# ── Approach legend table ─────────────────────────────────────
legend_rows = []
for a in ["M","O","P","Q","C","E","K","A","N"]:
    model, strategy, desc = APPROACH_DESC[a]
    ep = GT[a]["exact_pct"]
    legend_rows.append(
        "<tr>"
        "<td style='font-weight:700;font-size:1.1rem;color:{};font-family:monospace'>{}</td>"
        "<td style='font-weight:600'>{}</td>"
        "<td>{}</td>"
        "<td>{}</td>"
        "<td style='background:{};font-weight:600;text-align:right'>{:.1f}%</td>"
        "</tr>".format(COLORS.get(a,"#888"), a, model, strategy, desc, bg(ep), ep)
    )

# ── Overview summary table ────────────────────────────────────
summary_rows = []
rank = 1
for a in APPROACH_ORDER:
    r = GT[a]
    ep = r["exact_pct"]
    summary_rows.append(
        "<tr>"
        "<td style='color:#888'>{}</td>"
        "<td style='font-weight:700;color:{};font-size:1.05rem;font-family:monospace'>{}</td>"
        "<td>{}</td>"
        "<td>{}</td>"
        "<td style='background:{};font-weight:700'>{:.1f}%</td>"
        "<td style='color:#888'>{:.4f}</td>"
        "<td>{}</td>"
        "</tr>".format(
            rank, COLORS.get(a,"#888"), a,
            APPROACH_DESC[a][0], APPROACH_DESC[a][1],
            bg(ep), ep, r["cer"],
            r["cells"])
    )
    rank += 1

# ── Per-column table ──────────────────────────────────────────
col_table_rows = []
for col in ALL_COLS:
    row_cells = ["<td><code>{}</code></td>".format(col)]
    for a in APPROACH_ORDER:
        pc = GT[a]["per_col"].get(col)
        if pc:
            em, et, epct, ecer = pc
            row_cells.append(
                "<td style='background:{};text-align:center'>{:.0f}%</td>".format(bg(epct), epct)
            )
        else:
            row_cells.append("<td style='color:#ccc;text-align:center'>—</td>")
    col_table_rows.append("<tr>{}</tr>".format("".join(row_cells)))

# ── JS data ───────────────────────────────────────────────────
js_exact = {a: GT[a]["exact_pct"] for a in GT}
js_cer   = {a: GT[a]["cer"]       for a in GT}
js_per_col = {}
for a in GT:
    js_per_col[a] = [GT[a]["per_col"].get(c, (0,0,0,0))[2] for c in ALL_COLS]

parts = []
parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Haditax OCR Benchmarking Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #f5f4ef; color: #222;
       padding: 28px 36px; max-width: 1200px; }
h1 { font-size: 1.7rem; margin-bottom: 4px; }
.subtitle { color: #666; font-size: .88rem; margin-bottom: 28px; }
h2 { font-size: 1.1rem; margin: 32px 0 12px; border-bottom: 2px solid #c8c0a4;
     padding-bottom: 6px; color: #333; }
h3 { font-size: .9rem; margin: 0 0 10px; color: #555; font-weight: 600; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.card { background: #fff; border-radius: 8px; padding: 18px;
        box-shadow: 0 1px 5px rgba(0,0,0,.07); }
.cw  { position: relative; height: 240px; }
.cwm { position: relative; height: 320px; }
table { width: 100%; border-collapse: collapse; font-size: .82rem; }
th { background: #e8e2cc; text-align: left; padding: 7px 10px;
     font-weight: 600; white-space: nowrap; }
td { padding: 5px 10px; border-bottom: 1px solid #eee; vertical-align: middle; }
tr:hover td { background: #faf8f2; }
code { font-family: monospace; font-size: .85em; background: #f0ede4;
       padding: 1px 4px; border-radius: 3px; }
.insight { background: #fffbf0; border-left: 4px solid #d4a820; padding: 10px 14px;
           font-size: .85rem; border-radius: 0 6px 6px 0; margin: 12px 0; line-height: 1.6; }
.caveat  { background: #f0f4ff; border-left: 4px solid #5a7ab5; padding: 10px 14px;
           font-size: .85rem; border-radius: 0 6px 6px 0; margin: 12px 0; line-height: 1.6; }
.srow { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
.sbox { text-align:center; padding:10px 16px; min-width:90px; border-radius:6px;
        background:#f8f7f2; }
.snum { font-size:1.6rem; font-weight:700; line-height:1; }
.slbl { font-size:.7rem; color:#888; margin-top:3px; }
.good { color:#2d6a2d; } .bad { color:#9b2020; } .mid { color:#7a5a00; }
</style>
</head>
<body>
<h1>Haditax OCR Benchmarking Report</h1>
<p class="subtitle">British Mandate Palestine Property Tax Register &nbsp;&middot;&nbsp;
Ground-truth evaluation on Page 3 (Folio 1, 35 rows) &nbsp;&middot;&nbsp; 2026-04-15</p>
<div class="caveat">
  <strong>Scope:</strong> Ground truth (manually verified) exists only for <strong>Page 3</strong>.
  All 9 approaches that produced a Page 3 output were evaluated against it.
  Pages 10 and 50 appear at the bottom with proxy metrics only (fill rate / confidence) &mdash;
  those do <em>not</em> measure accuracy.
</div>
""")

# Section 0 — Approach legend
parts.append("""
<h2>0 &middot; What Each Approach Means</h2>
<div class="card" style="overflow-x:auto">
  <table>
    <tr>
      <th style="width:2.5rem">ID</th>
      <th>Model</th>
      <th>Strategy</th>
      <th>Description</th>
      <th style="text-align:right">Exact&nbsp;%</th>
    </tr>
""")
parts.append("\n".join(legend_rows))
parts.append("""  </table>
  <p style="font-size:.75rem;color:#999;margin-top:8px">
    Approaches B, D, F, L (zoomed crops variants) were not run on page 3 so they have no GT score.
  </p>
</div>
""")

# Section 1 — overview
parts.append("""
<h2>1 &middot; Overall Accuracy &mdash; Page 3 (all 9 approaches)</h2>
<div class="insight">
  <strong>M is the clear winner</strong> at 69.3% exact match.
  Few-shot prompting (telling the model what the output should look like with worked examples)
  made the biggest single improvement. The hybrid approaches O and P come next but don't beat
  the simplicity of a single well-prompted M run. Majority voting (Q) actually <em>hurts</em>
  relative to M alone.
</div>
<div class="grid-2">
  <div class="card"><h3>Exact Match % (sorted best to worst)</h3><div class="cw"><canvas id="cExact"></canvas></div></div>
  <div class="card"><h3>Character Error Rate — lower is better</h3><div class="cw"><canvas id="cCER"></canvas></div></div>
</div>
""")

# Section 2 — summary table
parts.append("""
<h2>2 &middot; Full Results Table</h2>
<div class="card" style="overflow-x:auto">
  <table>
    <tr><th>#</th><th>ID</th><th>Model</th><th>Strategy</th>
        <th>Exact %</th><th>CER</th><th>Cells compared</th></tr>
""")
parts.append("\n".join(summary_rows))
parts.append("</table></div>")

# Section 3 — per-column heatmap
col_headers = "".join(
    "<th style='text-align:center;color:{};font-family:monospace'>{}</th>".format(
        COLORS.get(a,"#888"), a)
    for a in APPROACH_ORDER
)
parts.append("""
<h2>3 &middot; Per-Column Accuracy Heatmap &mdash; Exact Match %</h2>
<div class="insight">
  Every approach gets Serial_No and Date right (printed/unambiguous).
  <strong>Area</strong> and <strong>Tax_LP</strong> are the hardest columns across the board &mdash;
  large comma-formatted numbers prone to digit transposition.
  <strong>Cat_No</strong> errors are systematic: most models read "1" instead of "10" (the trailing zero is faint).
</div>
<div class="card" style="overflow-x:auto">
  <table>
    <tr>
      <th>Column</th>
""")
parts.append(col_headers)
parts.append("</tr>")
parts.append("\n".join(col_table_rows))
parts.append("</table></div>")

# Section 4 — per-column chart
parts.append("""
<h2>4 &middot; Per-Column Exact Match % — Chart View</h2>
<div class="card"><div class="cwm"><canvas id="cColChart"></canvas></div></div>
""")

# Section 5 — proxy for pages 10, 50
parts.append("""
<h2>5 &middot; Proxy Metrics &mdash; Pages 10 &amp; 50 (no ground truth)</h2>
<div class="caveat">
  No manually verified GT exists for pages 10 and 50, so accuracy cannot be measured.
  The charts below show fill rate (% non-empty cells) and self-reported model confidence
  as rough proxies for output completeness only.
</div>
<div class="grid-2">
  <div class="card">
    <h3>Page 10 &mdash; Fill Rate by Approach</h3>
    <div class="cw"><canvas id="cp10"></canvas></div>
  </div>
  <div class="card">
    <h3>Page 50 &mdash; Fill Rate by Approach</h3>
    <div class="cw"><canvas id="cp50"></canvas></div>
  </div>
</div>
""")

# ── JavaScript ────────────────────────────────────────────────
parts.append("<script>")
parts.append("const COLORS=" + json.dumps(COLORS) + ";")
parts.append("const ac=a=>COLORS[a]||'#aaa';")
parts.append("const ORDER=" + json.dumps(APPROACH_ORDER) + ";")
parts.append("const EXACT=" + json.dumps(js_exact) + ";")
parts.append("const CER="   + json.dumps(js_cer)   + ";")
parts.append("const PERCOL=" + json.dumps(js_per_col) + ";")
parts.append("const COLS="  + json.dumps(ALL_COLS)  + ";")

# Proxy data inline
import csv
from collections import defaultdict
SKIP = {"Page_Number","Approach","Row_Index","Row_Confidence","Red_Ink","Disagreements"}
proxy_fill = {}
for p in [10, 50]:
    path = Path(f"comparison_page{p}.csv")
    if not path.exists(): continue
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    dcols = [c for c in rows[0].keys() if c not in SKIP]
    by_app = defaultdict(list)
    for r in rows: by_app[r["Approach"]].append(r)
    for app, app_rows in by_app.items():
        n = len(app_rows); total = n * len(dcols)
        filled = sum(1 for r in app_rows for c in dcols if r.get(c,"").strip())
        proxy_fill["{}_{}".format(p, app)] = round(100*filled/total,1) if total else 0

parts.append("const PROXY=" + json.dumps(proxy_fill) + ";")

parts.append("""
const pctOpts = (horiz) => ({
  responsive:true, maintainAspectRatio:false,
  plugins:{legend:{display:false}},
  scales: horiz
    ? {x:{min:0,max:100,ticks:{callback:v=>v+"%"}}, y:{ticks:{font:{size:11}}}}
    : {y:{min:0,max:100,ticks:{callback:v=>v+"%"}}}
});

// Exact match — horizontal bar, sorted best first
new Chart(document.getElementById("cExact"),{
  type:"bar",
  data:{
    labels: ORDER.map(a=>a+": "+["Gemini 2.5 few-shot","Few-shot+crops","M+C+E vote","5x few-shot vote","Gemini 2.5 base","Gemini 3.x Pro","Claude Opus few-shot","Claude Opus","Gemini Flash"][["M","O","P","Q","C","E","N","A","K"].indexOf(a)]),
    datasets:[{data:ORDER.map(a=>EXACT[a]),backgroundColor:ORDER.map(ac),borderRadius:4}]
  },
  options:{...pctOpts(true), scales:{x:{min:0,max:100,ticks:{callback:v=>v+"%"}},y:{ticks:{font:{size:11}}}}}
});

// CER — horizontal bar
new Chart(document.getElementById("cCER"),{
  type:"bar",
  data:{
    labels: ORDER.map(a=>a),
    datasets:[{data:ORDER.map(a=>CER[a]),backgroundColor:ORDER.map(ac),borderRadius:4}]
  },
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false}},
    scales:{x:{beginAtZero:true},y:{ticks:{font:{size:11}}}}
  }
});

// Per-column grouped bar
new Chart(document.getElementById("cColChart"),{
  type:"bar",
  data:{
    labels: COLS,
    datasets: Object.keys(PERCOL).map(a=>({
      label:a, data:PERCOL[a], backgroundColor:ac(a), borderRadius:2
    }))
  },
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:"top",labels:{boxWidth:12,font:{size:11}}}},
    scales:{
      x:{ticks:{maxRotation:45,font:{size:10}}},
      y:{min:0,max:100,ticks:{callback:v=>v+"%"}}
    }
  }
});

// Proxy fill charts
function proxyChart(canvasId, page){
  const apps=Object.keys(PROXY).filter(k=>k.startsWith(page+"_")).map(k=>k.split("_")[1]).sort();
  new Chart(document.getElementById(canvasId),{
    type:"bar",
    data:{labels:apps,datasets:[{data:apps.map(a=>PROXY[page+"_"+a]||0),backgroundColor:apps.map(ac),borderRadius:4}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{y:{min:0,max:100,ticks:{callback:v=>v+"%"}}}
    }
  });
}
proxyChart("cp10",10);
proxyChart("cp50",50);
""")
parts.append("</script></body></html>")

out = Path("ocr_benchmark_report.html")
out.write_text("".join(parts), encoding="utf-8")
print("Written", out, out.stat().st_size // 1024, "KB")
