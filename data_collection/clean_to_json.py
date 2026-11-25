"""Collapse whitespace in scraped remedy .txt files and write one {remedy: text} json.

    python data_collection/clean_to_json.py medicines_data medicines_cleaned.json
"""
import json
import re
import sys
from pathlib import Path


def main(src: str, dst: str) -> None:
    out = {}
    for p in sorted(Path(src).rglob("*.txt")):
        name = p.stem.replace("_", " ").replace("-", " ").lower()
        out[name] = re.sub(r"\s+", " ", p.read_text(encoding="utf-8")).strip()
    Path(dst).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} remedies -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
