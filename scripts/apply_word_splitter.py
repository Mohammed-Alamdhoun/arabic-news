#!/usr/bin/env python3
"""
apply_splitter.py

Apply the trained AraBERT word-boundary splitter to every article in
preprocessed_v2.jsonl and write a fixed copy to
data/preprocessed_v2_splitter_fixed.jsonl.

Fields processed per article:
  body              → merged words in the raw article body
  normalized_text   → merged words in the normalized version
  processed_text    → merged words in the stopword-filtered version
  tokens            → each token in the list is split if merged
  filtered_tokens   → same as tokens

Usage:
  python apply_splitter.py
  python apply_splitter.py --input  data/preprocessed_v2.jsonl \
                           --output data/preprocessed_v2_splitter_fixed.jsonl \
                           --min-token-len 7 \
                           --threshold 0.5
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"

INPUT_PATH  = DATA_DIR / "processed" / "preprocessed_al_fixed.jsonl"
OUTPUT_PATH = DATA_DIR / "processed" / "preprocessed_final.jsonl"


def fix_token_list(splitter, tokens: list[str]) -> list[str]:
    """Apply splitter to each token; if a token splits, expand to multiple tokens."""
    result = []
    for tok in tokens:
        fixed = splitter.split_token(tok)
        result.extend(fixed.split())
    return result


def fix_article(splitter, record: dict) -> dict:
    out = dict(record)

    if "body" in record and record["body"]:
        out["body"] = splitter.fix(record["body"])

    if "normalized_text" in record and record["normalized_text"]:
        out["normalized_text"] = splitter.fix(record["normalized_text"])

    if "processed_text" in record and record["processed_text"]:
        out["processed_text"] = splitter.fix(record["processed_text"])

    if "tokens" in record and record["tokens"]:
        out["tokens"] = fix_token_list(splitter, record["tokens"])

    if "filtered_tokens" in record and record["filtered_tokens"]:
        out["filtered_tokens"] = fix_token_list(splitter, record["filtered_tokens"])

    return out


def main():
    parser = argparse.ArgumentParser(description="Apply word-boundary splitter to corpus")
    parser.add_argument("--input",          default=str(INPUT_PATH))
    parser.add_argument("--output",         default=str(OUTPUT_PATH))
    parser.add_argument("--min-token-len",  type=int,   default=7)
    parser.add_argument("--threshold",      type=float, default=0.5)
    args = parser.parse_args()

    # Put src/ on sys.path so ``import arnlp`` works without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from arnlp.cleaning.word_splitter import WordBoundarySplitter, MODEL_DIR

    model_dir = str(MODEL_DIR)
    if not Path(model_dir).exists():
        print(f"[ERROR] Model not found at {model_dir}")
        print("        Run training first:  python arabert_word_splitter.py train")
        sys.exit(1)

    splitter = WordBoundarySplitter.load(
        model_dir,
        min_token_len=args.min_token_len,
        threshold=args.threshold,
    )

    input_path  = Path(args.input)
    output_path = Path(args.output)

    total = sum(1 for _ in open(input_path, encoding="utf-8"))
    print(f"Processing {total:,} articles  →  {output_path.name}")

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total, unit="article"):
            line = line.strip()
            if not line:
                continue
            start = line.find("{")
            if start == -1:
                continue
            try:
                record = json.loads(line[start:])
            except json.JSONDecodeError:
                continue

            fixed = fix_article(splitter, record)
            fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")

    print(f"\nDone.  Output saved to {output_path}")


if __name__ == "__main__":
    main()
