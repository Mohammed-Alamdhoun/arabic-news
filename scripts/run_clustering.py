#!/usr/bin/env python3
"""Stage-5 CLI — cluster the news subset into topics with BERTopic.

Loads the precomputed E5 embeddings + their id map, filters to the news
category (أخبار), runs BERTopic (UMAP → HDBSCAN) **without re-encoding**,
prints internal quality metrics, and writes::

    <out>/clusters.jsonl   one line per news article: {url, topic, prob}

Examples::

    # default: news subset, min_cluster_size=15
    python scripts/run_clustering.py

    # coarser topics + custom output
    python scripts/run_clustering.py --min-cluster-size 30 \
        --out data/clusters/clusters_mcs30.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# put src/ on sys.path so ``import arnlp`` works without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arnlp.clustering import (
    build_topic_model,
    cluster_quality,
    fit_topics,
    save_clusters,
)
from arnlp.embeddings.io import load_embeddings
from arnlp.utils.io import read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
DEFAULT_PREP = PROJECT_ROOT / "data" / "processed" / "preprocessed.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "clusters" / "clusters.jsonl"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emb-dir", type=Path, default=DEFAULT_EMB_DIR)
    ap.add_argument("--emb-name", default="e5_large")
    ap.add_argument("--preprocessed", type=Path, default=DEFAULT_PREP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--category", default="أخبار",
                    help="category to cluster (default: news)")
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load embeddings + id map (row-aligned with preprocessed.jsonl).
    vectors, ids = load_embeddings(args.emb_dir, args.emb_name)
    print(f"embeddings : {vectors.shape}  from {args.emb_dir}/{args.emb_name}.npy")

    # 2. Select the requested category subset by row.
    sel = [i for i, rec in enumerate(ids) if rec.get("category") == args.category]
    if not sel:
        sys.exit(f"no rows with category == {args.category!r}")
    emb = vectors[sel]
    urls = [ids[i]["url"] for i in sel]
    print(f"subset     : {len(sel)} articles in category {args.category!r}")

    # 3. Pull the clustering_view text for the same rows (join on url to be safe).
    view_by_url = {
        rec["url"]: rec.get("clustering_view") or rec.get("embedding_view") or rec.get("body", "")
        for rec in read_jsonl(args.preprocessed)
    }
    docs = [view_by_url.get(u, "") for u in urls]

    # 4. Fit BERTopic on the precomputed embeddings.
    model = build_topic_model(
        min_cluster_size=args.min_cluster_size,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
    )
    topics, probs = fit_topics(model, docs, emb)

    # 5. Report internal quality + a topic overview.
    q = cluster_quality(emb, topics)
    print("\n=== cluster quality ===")
    for k, v in q.items():
        print(f"  {k:18s}: {v:.4f}" if isinstance(v, float) else f"  {k:18s}: {v}")
    print("\n=== topics ===")
    print(model.get_topic_info().head(20).to_string(index=False))

    # 6. Persist assignments for downstream stages.
    n = save_clusters(args.out, urls, topics, probs)
    print(f"\nwrote {n} rows → {args.out}")


if __name__ == "__main__":
    main()
