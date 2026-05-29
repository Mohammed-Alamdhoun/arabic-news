#!/usr/bin/env python3
"""
fix_al_prefix.py
Copy preprocessed_final.jsonl and fix every occurrence of detached ال:

  "ال مستلزمات"  →  "المستلزمات"
  "ال يونيسيف"  →  "اليونيسيف"

Applied to:
  - String fields : body, title, normalized_text, processed_text
  - List fields   : tokens, filtered_tokens, lemmas
    (any standalone "ال" token is merged with the token that follows it)

Usage
-----
  python preprocessing/fix_al_prefix.py
"""

import json
import re
from pathlib import Path

PROJECT_DIR  = Path(__file__).parent.parent
INPUT_PATH   = PROJECT_DIR / "data" / "processed" / "preprocessed.jsonl"
OUTPUT_PATH  = PROJECT_DIR / "data" / "processed" / "preprocessed_al_fixed.jsonl"

# Match standalone ال followed by a space then an Arabic letter.
# Negative lookbehind (?<![ء-ي]) ensures the ا is NOT preceded by an Arabic
# letter, so words like خلال/إيصال/قال whose endings happen to spell ال
# are never touched — only a genuinely detached ال token is fixed.
_AL_IN_STR   = re.compile(r"(?<![ء-ي])ال ([ء-ي])")
_STRING_FIELDS = ("title", "body", "normalized_text", "processed_text")
_LIST_FIELDS   = ("tokens", "filtered_tokens", "lemmas")


def fix_string(text: str) -> str:
    """Collapse 'ال <arabic_letter>...' → 'ال<arabic_letter>...'"""
    return _AL_IN_STR.sub(r"ال\1", text)


def fix_list(tokens: list) -> list:
    """Merge any standalone 'ال' token with the token that follows it."""
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "ال" and i + 1 < len(tokens):
            result.append("ال" + str(tokens[i + 1]))
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def fix_record(record: dict) -> dict:
    for field in _STRING_FIELDS:
        if field in record and isinstance(record[field], str):
            record[field] = fix_string(record[field])
    for field in _LIST_FIELDS:
        if field in record and isinstance(record[field], list):
            record[field] = fix_list(record[field])
    return record


def main() -> None:
    total   = sum(1 for _ in open(INPUT_PATH, encoding="utf-8"))
    fixed   = 0
    written = 0

    print(f"Input  : {INPUT_PATH.name}  ({total:,} records)")
    print(f"Output : {OUTPUT_PATH.name}")

    with (
        open(INPUT_PATH,  encoding="utf-8") as fin,
        open(OUTPUT_PATH, "w", encoding="utf-8") as fout,
    ):
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # track whether this record needed a fix
            before = json.dumps(record, ensure_ascii=False)
            fix_record(record)
            if json.dumps(record, ensure_ascii=False) != before:
                fixed += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if i % 1000 == 0 or i == total:
                print(f"  {i:>6}/{total}  fixed={fixed}", end="\r")

    print()
    print(f"Done.  Written: {written:,}   Records fixed: {fixed:,}")
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
