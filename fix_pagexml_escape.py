#!/usr/bin/env python3
"""Fix unescaped <, >, & inside <Unicode>...</Unicode> in PAGE XML files.

Re-encodes the inner text properly so the XML is well-formed for Transkribus.
"""
import re
import sys
from pathlib import Path

UNI_RE = re.compile(r"<Unicode>(.*?)</Unicode>", re.S)


def fix_text(t: str) -> str:
    # Unescape existing entities (idempotent if already correct)
    s = (t.replace("&amp;", "&").replace("&lt;", "<")
          .replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'"))
    # Re-escape XML-special chars
    return (s.replace("&", "&amp;").replace("<", "&lt;")
              .replace(">", "&gt;").replace('"', "&quot;"))


def fix_file(path: Path) -> int:
    txt = path.read_text(encoding="utf-8")
    n = 0
    def repl(m):
        nonlocal n
        n += 1
        return f"<Unicode>{fix_text(m.group(1))}</Unicode>"
    new_txt = UNI_RE.sub(repl, txt)
    if new_txt != txt:
        path.write_text(new_txt, encoding="utf-8")
    return n


if __name__ == "__main__":
    targets = [Path(a) for a in sys.argv[1:]] or [
        *Path("Transkribus upload").rglob("Hadita_*.xml")
    ]
    for p in targets:
        n = fix_file(p)
        print(f"{p}: {n} <Unicode> blocks normalized")
