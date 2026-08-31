"""E1 runner: model A/B on the frozen v5 prompt, against the final2 (undamaged) corpus.

Sibling of run_g3v6_local.py — does NOT touch v5 artifacts. Differences:
  * images come from `Transkribus upload/final2/Hadita_{N}.jpeg` (undamaged deskewed crop)
  * XML geometry from the same final2 dir
  * single-image mode only (v6 dual-image is a rejected mode)
  * new models + pricing; own output suffix and cost log

Usage:
  python run_exp2608.py 3 4 5 6 9 10 --model gemini-3.7-flash --tag e1
Writes per page:
  exp2608/Hadita_{N}_{tag}.json / .xml, and appends g3_runs_exp2608.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

import run_g3v6_local as base   # reuse patch_xml, _extract_json, xml_grid_size, etc.

FINAL2 = Path("Transkribus upload/final2")
OUT_DIR = Path("exp2608")
RUNS_CSV = Path("g3_runs_exp2608.csv")

# $ per 1M tokens. Unknown models fall back to the 3.7-flash row (estimate only).
PRICING = {
    "gemini-3-flash-preview":        {"input": 0.50, "output": 3.00, "cached": 0.05},
    "gemini-3.7-flash":              {"input": 0.50, "output": 3.00, "cached": 0.05},
    "gemini-3.6-flash":              {"input": 0.50, "output": 3.00, "cached": 0.05},
    "gemini-3.5-flash":              {"input": 1.50, "output": 9.00, "cached": 0.15},
    "gemini-3.1-pro-preview":        {"input": 2.00, "output": 12.00, "cached": 0.20},
    "gemini-3.1-flash-lite-preview": {"input": 0.15, "output": 0.60, "cached": 0.02},
    "gemini-3.5-transcribe":         {"input": 0.50, "output": 3.00, "cached": 0.05},
}


def _price(model: str) -> dict:
    return PRICING.get(model, PRICING["gemini-3.7-flash"])


def clean_schema(s):
    """Strip keys the current API rejects (v6's _strict_schema emits additionalProperties)."""
    drop = ("additionalProperties", "additional_properties", "$schema", "title")
    if isinstance(s, dict):
        return {k: clean_schema(v) for k, v in s.items() if k not in drop}
    if isinstance(s, list):
        return [clean_schema(x) for x in s]
    return s


def to_sinai_rows(data: list, ns: dict) -> list[dict]:
    """Model rows (snake_case fields) -> LEFT_COLS-keyed dicts.

    Rows already keyed by LEFT_COLS pass through; snake_case rows are validated
    through LedgerRow so _to_sinai_row's getattr access works.
    """
    out = []
    for r in data:
        if isinstance(r, dict) and any(c in r for c in ns["LEFT_COLS"][:4]):
            out.append({c: (r.get(c, "") or "") for c in ns["LEFT_COLS"]})
            continue
        try:
            out.append(ns["_to_sinai_row"](ns["LedgerRow"].model_validate(r)))
        except Exception:
            out.append({c: "" for c in ns["LEFT_COLS"]})
    return out


def estimate_cost(model: str, um) -> float:
    p = _price(model)
    inp = getattr(um, "prompt_token_count", 0) or 0
    out = getattr(um, "candidates_token_count", 0) or 0
    tho = getattr(um, "thoughts_token_count", 0) or 0
    cac = getattr(um, "cached_content_token_count", 0) or 0
    return (max(inp - cac, 0) * p["input"] + cac * p["cached"]
            + (out + tho) * p["output"]) / 1e6


def run_page(page: int, *, model: str, thinking: str, tag: str,
             ns: dict, max_output: int, use_tools: bool) -> dict | None:
    img = FINAL2 / f"Hadita_{page}.jpeg"
    xml_path = FINAL2 / f"Hadita_{page}.xml"
    if not img.exists() or not xml_path.exists():
        print(f"  ! page {page}: missing image or xml; skipping")
        return None

    contents = [
        "=== PAGE IMAGE ===",
        types.Part.from_bytes(data=img.read_bytes(), mime_type="image/jpeg"),
        ns["PROMPT"],
    ]
    cfg_kwargs = dict(
        system_instruction=ns["SYSTEM_INSTRUCTION"],
        max_output_tokens=max_output,
    )
    if use_tools:
        # Newer models return MALFORMED_FUNCTION_CALL on the v6 prompt with
        # code_execution; structured output is the working path (E0 finding).
        cfg_kwargs["tools"] = [types.Tool(code_execution=types.ToolCodeExecution())]
    else:
        cfg_kwargs["response_mime_type"] = "application/json"
        cfg_kwargs["response_schema"] = clean_schema(ns["_strict_schema"](ns["SectionRows"]))
    if thinking != "none":
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=base.THINKING_BUDGETS[thinking])

    client = genai.Client()
    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs))
    except Exception as e:
        msg = str(e)
        print(f"    ! error: {msg[:200]}")
        if use_tools and ("tool" in msg.lower() or "code_execution" in msg.lower()):
            print("    retrying without code_execution tool ...")
            cfg_kwargs.pop("tools", None)
            resp = client.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs))
        else:
            return None
    elapsed = round(time.perf_counter() - t0, 2)

    blob = resp.text if not use_tools else base._extract_json(resp)
    if not blob:
        print(f"    x no JSON (finish={resp.candidates[0].finish_reason if resp.candidates else '?'})")
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        print(f"    x JSON decode failed: {e}")
        return None
    if isinstance(data, dict):
        for key in ("rows", "SectionRows", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        print("    x unexpected JSON shape")
        return None

    rows = to_sinai_rows(data, ns)

    OUT_DIR.mkdir(exist_ok=True)
    json_path = OUT_DIR / f"Hadita_{page}_{tag}.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    xml_str = xml_path.read_text(encoding="utf-8")
    patched, n_patched, n_skipped = base.patch_xml(xml_str, rows, ns["LEFT_COLS"])
    (OUT_DIR / f"Hadita_{page}_{tag}.xml").write_text(patched, encoding="utf-8")

    um = getattr(resp, "usage_metadata", None)
    cost = estimate_cost(model, um) if um else float("nan")
    n_rows_xml, n_cols_xml = base.xml_grid_size(xml_str)
    print(f"    ok {len(rows)} rows (grid {n_rows_xml}x{n_cols_xml}) "
          f"patched={n_patched} skipped={n_skipped} {elapsed}s ${cost:.4f}")

    rec = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), page=page, model=model,
               tag=tag, thinking=thinking, rows=len(rows), grid_rows=n_rows_xml,
               patched=n_patched, skipped=n_skipped, seconds=elapsed,
               prompt_tokens=getattr(um, "prompt_token_count", 0) if um else 0,
               output_tokens=getattr(um, "candidates_token_count", 0) if um else 0,
               thought_tokens=getattr(um, "thoughts_token_count", 0) if um else 0,
               cost_usd=round(cost, 5))
    new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rec))
        if new:
            w.writeheader()
        w.writerow(rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", type=int)
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--tag", default=None, help="output suffix (default: derived from model)")
    ap.add_argument("--thinking", default="low", choices=["none", "low", "medium", "high"])
    ap.add_argument("--max-output", type=int, default=32768)
    ap.add_argument("--tools", action="store_true",
                    help="use code_execution instead of structured output "
                         "(fails with MALFORMED_FUNCTION_CALL on 3.x models)")
    args = ap.parse_args()

    tag = args.tag or args.model.replace("gemini-", "g").replace("-preview", "").replace(".", "")
    ns = base._load_prompts_and_schema()
    print(f"model={args.model} tag={tag} thinking={args.thinking} pages={args.pages}")
    total = 0.0
    for p in args.pages:
        print(f"  page {p}:")
        rec = run_page(p, model=args.model, thinking=args.thinking, tag=tag,
                       ns=ns, max_output=args.max_output, use_tools=args.tools)
        if rec:
            total += rec["cost_usd"]
    print(f"total cost this run: ${total:.4f}")


if __name__ == "__main__":
    main()
