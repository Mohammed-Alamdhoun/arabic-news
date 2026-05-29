"""Multilingual-E5-large encoder.

E5 has a specific input convention:
  * passages are prefixed with ``"passage: "``
  * queries  are prefixed with ``"query: "``

Forgetting the prefix silently degrades retrieval quality, so the
:class:`E5Encoder` injects it for you.

Output vectors are L2-normalized → cosine similarity == dot product
downstream (BERTopic, FAISS, nearest-neighbor sanity checks).
"""

from __future__ import annotations

import numpy as np

from arnlp.embeddings.base import BaseEncoder

MODEL_NAME = "intfloat/multilingual-e5-large"


class E5Encoder(BaseEncoder):
    name = "e5_large"
    dim = 1024
    input_kind = "text"

    def __init__(
        self,
        *,
        batch_size: int = 32,
        device: str | None = None,
        prefix: str = "passage: ",
        show_progress: bool = True,
    ) -> None:
        # Lazy import so the package imports even without torch installed.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME, device=device)
        self.batch_size    = batch_size
        self.prefix        = prefix
        self.show_progress = show_progress

    def encode(self, texts: list[str]) -> np.ndarray:
        prepared = [f"{self.prefix}{t}" for t in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query (uses the ``query:`` prefix, not ``passage:``)."""
        vec = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32)[0]
