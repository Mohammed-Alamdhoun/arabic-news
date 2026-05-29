#!/usr/bin/env python3
"""Stage-4 CLI — encode preprocessed articles into dense vectors.

Reads ``data/processed/preprocessed*.jsonl`` (one preprocessed record
per line, with ``embedding_view`` / ``filtered_tokens`` / ``url`` /
``category`` fields), runs the chosen encoder, and writes:

    <out_dir>/<name>.npy         shape (N, dim), float32
    <out_dir>/<name>_ids.jsonl   row → {url, category}

Examples::

    # E5-large on a sample (auto-detect GPU)
    python scripts/run_embeddings.py \
        --input data/processed/preprocessed_sample_offset0_n10.jsonl

    # E5-large on the full corpus, force CPU
    python scripts/run_embeddings.py \
        --input data/processed/preprocessed.jsonl \
        --device cpu

    # fastText baseline
    python scripts/run_embeddings.py --encoder fasttext \
        --input data/processed/preprocessed.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# put src/ on sys.path so ``import arnlp`` works without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arnlp.embeddings import E5Encoder, FastTextEncoder
from arnlp.embeddings.io import save_embeddings
from arnlp.utils import read_jsonl

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed"  / "preprocessed.jsonl"
DEFAULT_OUT   = PROJECT_ROOT / "data" / "embeddings"
DEFAULT_FT    = PROJECT_ROOT / "models" / "fasttext"  / "cc.ar.300.bin"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input",    type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir",  type=Path, default=DEFAULT_OUT)
    ap.add_argument("--encoder",  choices=["e5", "fasttext"], default="e5")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device",   choices=["auto", "cpu", "cuda", "mps"],
                    default="auto", help="device for E5 (fastText is CPU-only)")
    ap.add_argument("--ft-model", type=Path, default=DEFAULT_FT,
                    help="path to fastText cc.ar.300.bin (downloads if missing)")
    return ap.parse_args()


def resolve_device(choice: str) -> str | None:
    """Map ``auto`` to the best available device for sentence-transformers."""
    if choice != "auto":
        return choice
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    print(f"Input    : {args.input}")
    print(f"Out dir  : {args.out_dir}")
    print(f"Encoder  : {args.encoder}")

    # Load preprocessed records into memory once (16k records fit easily)
    records = list(read_jsonl(args.input))
    print(f"Records  : {len(records):,}")

    if args.encoder == "e5":
        device = resolve_device(args.device)
        print(f"Device   : {device}")
        encoder = E5Encoder(batch_size=args.batch_size, device=device)
        # Encode the `embedding_view` field — keeps digits + decimals + Latin
        # so E5 has the semantic signal it expects.
        inputs = [r.get("embedding_view") or r.get("title", "") for r in records]
    else:
        encoder = FastTextEncoder(model_path=args.ft_model)
        # fastText takes token lists; use the post-stopword-filter view
        inputs = [r.get("filtered_tokens", []) for r in records]

    t0 = time.time()
    vectors = encoder.encode(inputs)
    elapsed = time.time() - t0

    ids = [
        {"row": i, "url": r.get("url", ""), "category": r.get("category", "")}
        for i, r in enumerate(records)
    ]
    npy_path, ids_path = save_embeddings(args.out_dir, encoder.name, vectors, ids)

    print(f"\nDone in {elapsed:.1f}s  →  {vectors.shape} {vectors.dtype}")
    print(f"  vectors : {npy_path}")
    print(f"  ids     : {ids_path}")
    # quick sanity print
    norms = (vectors ** 2).sum(axis=1) ** 0.5
    print(f"  mean L2 norm: {norms.mean():.4f}  (expect ≈1.0 for E5)")


if __name__ == "__main__":
    main()
