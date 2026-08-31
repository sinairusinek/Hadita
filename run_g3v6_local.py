"""Local CLI runner for v6 (dewarp-damage recovery) using the public google-genai SDK.

The v6 notebook (Hadita_Gemini3_to_PAGEXML_v6.ipynb) targets `client.interactions`,
which is not in the public PyPI SDK (see run_showcase_sample.py header for the same
issue on v5). This runner reuses v6's SYSTEM_INSTRUCTION + PROMPT + SectionRows
schema verbatim (exec'd from the notebook so they stay in sync) and drives them
through `client.models.generate_content` with `tools=[Tool(code_execution=...)]`
for the agentic-vision loop.

Usage:
  GOOGLE_API_KEY=... python run_g3v6_local.py 5
  GOOGLE_API_KEY=... python run_g3v6_local.py 3 4 5 6 9 10
  GOOGLE_API_KEY=... python run_g3v6_local.py 5 --no-raw           # v5-style single-image
  GOOGLE_API_KEY=... python run_g3v6_local.py 5 --thinking medium  # default: low

Per page writes:
  Hadita_{N}_G3v6.json   raw row dicts (LEFT_COLS order)
  Hadita_{N}_G3v6.xml    patched PAGE XML
  g3_runs_v6.csv         appended cost log
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import ValidationError

NB = Path("Hadita_Gemini3_to_PAGEXML_v6.ipynb")
PROCESSED_DIR = Path(".")
RAW_DIR = Path(".")
XML_DIR = Path("Transkribus upload/final")
RUNS_CSV = Path("g3_runs_v6.csv")

MODEL = "gemini-3-flash-preview"     # Gemini 3 family — same family as v5/v6 in Colab.
                                     # If Google later renames to gemini-3.5-flash on the public
                                     # API, override via --model.
DEFAULT_MAX_OUTPUT = 32768

PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "cached": 0.05},
    "gemini-3.5-flash":       {"input": 1.50, "output": 9.00, "cached": 0.15},
}

# THINKING level → thinking_budget tokens. Values are Google's published defaults.
# Adjust if the public API later exposes named levels.
THINKING_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}


def _cell_src(cell_idx: int) -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "".join(nb["cells"][cell_idx]["source"])


def _load_prompts_and_schema() -> dict:
    ns = {}
    exec(_cell_src(4), ns)   # LEFT_COLS, LedgerRow, SectionRows, _strict_schema, _to_sinai_row
    exec(_cell_src(5), ns)   # SYSTEM_INSTRUCTION, PROMPT
    # Forward refs (`list[LedgerRow]`) inside SectionRows need explicit rebuild
    # when classes were created in an exec'd namespace (no __module__ resolution).
    ns["SectionRows"].model_rebuild(_types_namespace=ns)
    ns["LedgerRow"].model_rebuild(_types_namespace=ns)
    needed = ("LEFT_COLS", "LedgerRow", "SectionRows", "_strict_schema",
              "_to_sinai_row", "SYSTEM_INSTRUCTION", "PROMPT")
    return {k: ns[k] for k in needed}


# ---------- PAGE XML patcher (copied from build_g3_notebook_v6.py cell 6) ----------
def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def patch_xml(xml_str: str, text_rows: list[dict], left_cols: list[str]) -> tuple[str, int, int]:
    n_patched, n_skipped = 0, 0
    for r_idx, row in enumerate(text_rows):
        for c_idx, col in enumerate(left_cols):
            text = (row.get(col, "") or "").strip()
            cell_id = f'id="cell_r{r_idx}_c{c_idx}"'
            cs = xml_str.find(cell_id)
            if cs == -1:
                n_skipped += 1
                continue
            nxt = xml_str.find("<TableCell", cs + len(cell_id))
            end = nxt if nxt != -1 else len(xml_str)
            us = xml_str.find("<Unicode>", cs, end)
            ue = xml_str.find("</Unicode>", cs, end)
            if us == -1 or ue == -1:
                n_skipped += 1
                continue
            cstart = us + len("<Unicode>")
            xml_str = xml_str[:cstart] + _escape_xml(text) + xml_str[ue:]
            n_patched += 1
    return xml_str, n_patched, n_skipped


def xml_grid_size(xml_str: str) -> tuple[int, int]:
    cells = set(re.findall(r"cell_r(\d+)_c(\d+)", xml_str))
    if not cells:
        return 0, 0
    return max(int(r) for r, _ in cells) + 1, max(int(c) for _, c in cells) + 1


def _estimate_cost(model: str, um) -> float:
    p = PRICING.get(model)
    if not p:
        return float("nan")
    inp = getattr(um, "prompt_token_count", 0) or 0
    out = getattr(um, "candidates_token_count", 0) or 0
    tho = getattr(um, "thoughts_token_count", 0) or 0
    cac = getattr(um, "cached_content_token_count", 0) or 0
    billable_input = max(inp - cac, 0)
    return (billable_input * p["input"] + cac * p["cached"] + (out + tho) * p["output"]) / 1e6


def _find_processed(page: int) -> Path | None:
    for parent in (Path("processed"), Path(".")):
        for stem in (f"Hadita-{page}Processed", f"Hadita_{page}Processed"):
            cand = parent / f"{stem}.jpg"
            if cand.exists():
                return cand
    return None


def _find_raw(page: int) -> Path | None:
    cand = RAW_DIR / f"000nvrj-432316TAX 1-85_page-{page:04d}.jpg"
    return cand if cand.exists() else None


def _find_xml(page: int) -> Path | None:
    cand = XML_DIR / f"Hadita_{page}.xml"
    return cand if cand.exists() else None


def _build_contents(processed_path: Path, raw_path: Path | None, prompt: str) -> list:
    """v6 input order: text labels around images, then prompt last."""
    contents: list = []
    contents.append("=== PROCESSED (IMAGE 1) ===")
    contents.append(types.Part.from_bytes(
        data=processed_path.read_bytes(), mime_type="image/jpeg"))
    if raw_path is not None:
        contents.append("=== RAW (IMAGE 2) ===")
        contents.append(types.Part.from_bytes(
            data=raw_path.read_bytes(), mime_type="image/jpeg"))
    contents.append(prompt)
    return contents


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_BARE_RE = re.compile(r"(\{[^{}]*\"rows\".*\}|\[\s*\{.*?\}\s*\])", re.DOTALL)


def _candidate_json_blobs(text: str) -> list[str]:
    """Yield JSON candidate substrings from a text part: fenced, then bare."""
    if not text:
        return []
    out = []
    if text.strip().startswith(("{", "[")):
        out.append(text.strip())
    for m in _FENCE_RE.finditer(text):
        out.append(m.group(1))
    for m in _BARE_RE.finditer(text):
        out.append(m.group(1))
    return out


def _extract_json(resp) -> str | None:
    """Pull the most-likely SectionRows JSON from a code_execution response.

    With agentic vision the response interleaves: text · executable_code ·
    code_execution_result · inline_data (model-generated crops) · ... · text.
    The final SectionRows JSON usually sits in the LAST text part, possibly
    wrapped in a ```json ... ``` fence.
    """
    out_text = getattr(resp, "text", None)
    for blob in _candidate_json_blobs(out_text or ""):
        return blob
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in reversed(getattr(content, "parts", None) or []):
            t = getattr(part, "text", None)
            for blob in _candidate_json_blobs(t or ""):
                return blob
    return None


def _count_code_execs(resp) -> int:
    n = 0
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in (getattr(content, "parts", None) or []):
            if getattr(part, "executable_code", None) is not None:
                n += 1
    return n


def run_page(page: int, *, use_raw: bool, thinking: str, model: str,
             max_output: int, ns: dict, use_schema: bool) -> dict | None:
    proc = _find_processed(page)
    xml_path = _find_xml(page)
    raw_path = _find_raw(page) if use_raw else None
    if not proc or not xml_path:
        miss = [n for n, v in (("processed image", proc), ("XML", xml_path)) if not v]
        print(f"  ⚠ page {page}: missing {' & '.join(miss)}; skipping")
        return None
    mode = "v6 (processed + raw)" if raw_path else ("v5-fallback" if use_raw else "v5-mode (raw disabled)")
    print(f"  page {page}: {mode}")
    if use_raw and raw_path is None:
        print(f"    (no raw scan at {RAW_DIR}/000nvrj-...page-{page:04d}.jpg)")

    contents = _build_contents(proc, raw_path, ns["PROMPT"])

    cfg_kwargs = dict(
        system_instruction=ns["SYSTEM_INSTRUCTION"],
        max_output_tokens=max_output,
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGETS[thinking]),
    )
    if use_schema:
        cfg_kwargs["response_mime_type"] = "application/json"
        # Use the strict JSON schema dict (same as v6 notebook), avoids
        # pydantic forward-ref rebuild issue in exec'd namespace.
        cfg_kwargs["response_schema"] = ns["_strict_schema"](ns["SectionRows"])
    config = types.GenerateContentConfig(**cfg_kwargs)

    client = genai.Client()
    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(model=model, contents=contents, config=config)
    except Exception as e:
        # If server rejects code_execution + response_schema combo, retry once
        # without the schema and validate client-side.
        if "code_execution" in str(e).lower() or "tools" in str(e).lower() or "schema" in str(e).lower():
            print(f"    ⚠ combo rejected: {e}")
            print(f"    retrying WITHOUT response_schema (validate JSON client-side)...")
            cfg_kwargs.pop("response_schema", None)
            cfg_kwargs.pop("response_mime_type", None)
            config = types.GenerateContentConfig(**cfg_kwargs)
            resp = client.models.generate_content(model=model, contents=contents, config=config)
        else:
            raise
    elapsed = round(time.perf_counter() - t0, 2)

    final_text = _extract_json(resp)
    if not final_text:
        print(f"    ✖ no JSON in response")
        return None
    try:
        # Reshape tolerance: when the API runs without a forced response_schema,
        # the model sometimes wraps rows under "SectionRows" or just emits a bare list.
        parsed = json.loads(final_text)
        # Find the rows list anywhere in the response. Models without a forced
        # response_schema variously emit: {"rows": [...]}; bare [...];
        # {"SectionRows": [...]}; or deeply-nested like
        # {"project": ..., "transcription": {"SectionRows": [...]}}.
        # Identify rows by shape: a list of dicts whose first dict has serial_no
        # (the marker field for LedgerRow).
        # Recurse through both dict values AND list items — some responses wrap
        # the LedgerRow list inside a container row (e.g. {"ledgers": [{"section_rows": [...]}]}).
        _ROW_MARKERS = {"serial_no", "block_no", "tax_mils", "ref_serial_no"}
        def _find_rows(node):
            if isinstance(node, list) and node and isinstance(node[0], dict) \
                    and _ROW_MARKERS & set(node[0].keys()):
                return node
            if isinstance(node, dict):
                for v in node.values():
                    found = _find_rows(v)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for v in node:
                    found = _find_rows(v)
                    if found is not None:
                        return found
            return None
        rows_list = _find_rows(parsed)
        if rows_list is None:
            raise ValueError(f"Could not find a SectionRows-shaped list in response: "
                             f"top-level keys={list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")
        parsed = {"rows": rows_list}
        section = ns["SectionRows"].model_validate(parsed)
    except (ValidationError, ValueError) as ve:
        print(f"    ✖ JSON validation failed: {ve!s:.200}")
        debug_path = Path(f"Hadita_{page}_G3v6_raw_response.txt")
        debug_path.write_text(final_text, encoding="utf-8")
        print(f"      raw response dumped to {debug_path}")
        return None

    rows = [ns["_to_sinai_row"](r) for r in section.rows]
    src_xml = xml_path.read_text(encoding="utf-8")
    patched, n_patched, n_skipped = patch_xml(src_xml, rows, ns["LEFT_COLS"])
    out_xml = Path(f"Hadita_{page}_G3v6.xml")
    out_json = Path(f"Hadita_{page}_G3v6.json")
    out_xml.write_text(patched, encoding="utf-8")
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    um = getattr(resp, "usage_metadata", None)
    cost = _estimate_cost(model, um) if um else float("nan")
    zoom = _count_code_execs(resp)
    xr, xc = xml_grid_size(src_xml)
    note = ""
    if len(rows) > xr:
        note = f"   ⚠ {len(rows)-xr} extra row(s) won't fit XML grid ({xr}x{xc})"
    elif len(rows) < xr:
        note = f"   ℹ {xr-len(rows)} XML row(s) overwritten empty"
    print(f"    → {len(rows)} rows · {zoom} code-exec rounds · {elapsed}s · ${cost:.4f}")
    print(f"      wrote {out_json.name} + {out_xml.name} (patched {n_patched} cells){note}")

    return {
        "page": page, "rows": len(rows), "elapsed_s": elapsed, "zoom_rounds": zoom,
        "cost_usd": cost, "out_xml": str(out_xml), "out_json": str(out_json),
        "usage": um, "n_patched": n_patched, "n_skipped": n_skipped,
        "mode": mode,
    }


def _append_cost_log(model: str, thinking: str, result: dict) -> None:
    is_new = not RUNS_CSV.exists()
    um = result.get("usage")
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["page", "model", "thinking", "mode", "rows", "zoom_rounds",
                        "elapsed_s", "input_tok", "cached_tok", "output_tok",
                        "thought_tok", "total_tok", "cost_usd"])
        w.writerow([
            result["page"], model, thinking, result["mode"], result["rows"],
            result["zoom_rounds"], result["elapsed_s"],
            getattr(um, "prompt_token_count", 0) if um else 0,
            getattr(um, "cached_content_token_count", 0) if um else 0,
            getattr(um, "candidates_token_count", 0) if um else 0,
            getattr(um, "thoughts_token_count", 0) if um else 0,
            getattr(um, "total_token_count", 0) if um else 0,
            round(result["cost_usd"], 6) if result["cost_usd"] == result["cost_usd"] else "",
        ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--no-raw", action="store_true",
                    help="Don't send the raw scan (v5-style single-image)")
    ap.add_argument("--thinking", choices=list(THINKING_BUDGETS), default="low")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-output", type=int, default=DEFAULT_MAX_OUTPUT)
    ap.add_argument("--schema", action="store_true",
                    help="Send response_schema. Default is off because Gemini's structured-"
                         "output endpoint rejects the strict schema's `additionalProperties: "
                         "false` keys; we validate JSON client-side instead.")
    args = ap.parse_args()

    ns = _load_prompts_and_schema()
    print(f"model={args.model}  thinking={args.thinking}  use_raw={not args.no_raw}  "
          f"schema={'on' if args.schema else 'off'}")
    print(f"system_instruction: {len(ns['SYSTEM_INSTRUCTION']):,} chars  ·  prompt: {len(ns['PROMPT']):,} chars")

    total_cost, ok = 0.0, 0
    for page in args.pages:
        print(f"\n══════════ PAGE {page} ══════════")
        r = run_page(page, use_raw=not args.no_raw, thinking=args.thinking,
                     model=args.model, max_output=args.max_output, ns=ns,
                     use_schema=args.schema)
        if r:
            _append_cost_log(args.model, args.thinking, r)
            total_cost += r["cost_usd"] if r["cost_usd"] == r["cost_usd"] else 0
            ok += 1
    print(f"\nDone: {ok}/{len(args.pages)} pages.  Total cost ≈ ${total_cost:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
