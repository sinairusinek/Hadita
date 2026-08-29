"""Pull Hadid01 (model 592509) per-page transcripts from Transkribus.

Reads `hadid01_tsids.json` at repo root:
  {"11": <tsId>, "12": <tsId>, "13": <tsId>}

For each page, looks up the transcript with that tsId in the page's
tsList and downloads it to `g3_results/Hadita_{N}_Hadid01.xml`.
Idempotent — skips pages whose file already exists.

Requires TRANSKRIBUS_USER / TRANSKRIBUS_PASS in env.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/sinairusinek/Documents/GitHub/Dybbuk/YiDraCor/code")
from transkribus.client import TrpClient

COL_ID = 2377415
DOC_ID = 15829823
OUT_DIR = Path("g3_results")
TSID_JSON = Path("hadid01_tsids.json")


def main() -> int:
    tsid_map_raw = json.loads(TSID_JSON.read_text())
    tsid_map = {k: v for k, v in tsid_map_raw.items()
                if not k.startswith("_") and v is not None}
    if not tsid_map:
        print(f"No tsIds populated in {TSID_JSON}; nothing to fetch.")
        return 1

    client = TrpClient.from_env()
    fd = client.fulldoc(COL_ID, DOC_ID)
    pages_by_nr = {int(p["pageNr"]): p for p in fd["pageList"]["pages"]}

    OUT_DIR.mkdir(exist_ok=True)
    for page_str, tsid in tsid_map.items():
        page_nr = int(page_str)
        out_path = OUT_DIR / f"Hadita_{page_nr}_Hadid01.xml"
        if out_path.exists():
            print(f"  page {page_nr}: {out_path.name} exists, skipping")
            continue
        page = pages_by_nr.get(page_nr)
        if not page:
            print(f"  page {page_nr}: not found in doc {DOC_ID}", file=sys.stderr)
            continue
        match = next((t for t in page["tsList"]["transcripts"]
                      if int(t.get("tsId", -1)) == int(tsid)), None)
        if not match:
            print(f"  page {page_nr}: tsId {tsid} not in tsList", file=sys.stderr)
            continue
        xml = client.fetch_transcript(match["url"])
        out_path.write_text(xml, encoding="utf-8")
        print(f"  page {page_nr}: wrote {out_path}  (tsId={tsid}, tool={match.get('toolName')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
