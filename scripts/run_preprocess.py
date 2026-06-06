#!/usr/bin/env python3
"""Stage-2 preprocessing CLI.

Reads ``data/raw/articles_*_fixed.jsonl`` (or any JSONL with ``title`` +
``body`` fields), runs the Arabic text pipeline, and writes one JSON
object per article to the chosen output path.

Examples::

    # 50-article smoke test from the middle of the corpus
    python scripts/run_preprocess.py --offset 5000 --limit 50

    # full corpus
    python scripts/run_preprocess.py --input data/raw/articles_2026_05_fixed.jsonl \
                                     --output data/processed/preprocessed.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Put src/ on sys.path so ``import arnlp`` works without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arnlp.preprocessing import ArabicTextPipeline, PipelineConfig
from arnlp.utils import read_jsonl, write_jsonl

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "data" / "raw"       / "articles_2026_05_fixed.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "preprocessed.jsonl"


def build_record(article: dict, doc) -> dict:
    return {
        "url":      article.get("url", ""),
        "category": article.get("category", ""),
        "title":    article.get("title", ""),
        "body":     article.get("body", ""),
        **doc.to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N records")
    ap.add_argument("--limit",  type=int, default=None,
                    help="cap the number of records processed")
    ap.add_argument("--report-every", type=int, default=500,
                    help="print a progress line every N articles (0 = silent)")
    args = ap.parse_args()

    if args.limit is not None and args.output == DEFAULT_OUTPUT:
        # don't overwrite the canonical output with a partial sample
        args.output = (
            PROJECT_ROOT / "data" / "processed"
            / f"preprocessed_sample_offset{args.offset}_n{args.limit}.jsonl"
        )

    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Slice  : offset={args.offset}  limit={args.limit}")

    pipe = ArabicTextPipeline(PipelineConfig())

    def records():
        t0 = time.time()
        n  = 0
        for art in read_jsonl(args.input, offset=args.offset, limit=args.limit):
            raw = f"{art.get('title') or ''} {art.get('body') or ''}".strip()
            yield build_record(art, pipe.process(raw))
            n += 1
            if args.report_every and n % args.report_every == 0:
                rate = n / (time.time() - t0)
                print(f"  {n:>7,} processed  ({rate:.1f}/s)")

    n_written = write_jsonl(args.output, records())
    print(f"\nDone. Wrote {n_written:,} records.")


if __name__ == "__main__":
    main()
