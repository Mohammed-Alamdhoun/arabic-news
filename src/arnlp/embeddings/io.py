"""Persistence helpers for embedding artifacts.

We save embeddings as ``<name>.npy`` next to a ``<name>_ids.jsonl`` file
that records the row → (url, category) mapping so downstream code can
join back to article metadata without re-loading the full preprocessed
corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_embeddings(
    out_dir: str | Path,
    name: str,
    vectors: np.ndarray,
    ids: list[dict],
) -> tuple[Path, Path]:
    """Write ``<name>.npy`` and ``<name>_ids.jsonl`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{name}.npy"
    ids_path = out_dir / f"{name}_ids.jsonl"

    np.save(npy_path, vectors)
    with open(ids_path, "w", encoding="utf-8") as f:
        for row in ids:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return npy_path, ids_path


def load_embeddings(
    out_dir: str | Path, name: str
) -> tuple[np.ndarray, list[dict]]:
    out_dir = Path(out_dir)
    vectors = np.load(out_dir / f"{name}.npy")
    with open(out_dir / f"{name}_ids.jsonl", encoding="utf-8") as f:
        ids = [json.loads(line) for line in f if line.strip()]
    return vectors, ids
