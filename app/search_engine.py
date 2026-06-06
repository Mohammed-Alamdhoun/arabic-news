"""Semantic search over the Arabic news corpus (multilingual-E5 + cosine).

Data loads up-front; the ~2 GB E5 model loads lazily on the first search. The
article source is *streamed* keeping only the display fields, so it stays light
even when pointed at the 295 MB ``preprocessed.jsonl`` (whose row order matches
``e5_large.npy``, and whose ``body`` is the cleaned/word-fixed text we show).

Each result carries ``idx`` — its row in the (news) embedding matrix — so the
3D galaxy can light up exactly where a search lands.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

MODEL_NAME = "intfloat/multilingual-e5-large"
_DISPLAY_FIELDS = ("url", "category", "title", "body")


def _load_source(path: str) -> pd.DataFrame:
    """Stream a JSONL article/preprocessed file, keeping only display fields."""
    rows: list[list] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append([d.get(k, "") for k in _DISPLAY_FIELDS])
    return pd.DataFrame(rows, columns=list(_DISPLAY_FIELDS))


class SemanticSearchEngine:
    def __init__(
        self,
        articles_path,
        embeddings_path,
        clusters_path=None,
        topics_path=None,
        labels_path=None,
        news_only=False,
        model_name=MODEL_NAME,
        device=None,
    ):
        print("Loading articles (preprocessed body)...")
        self.df = _load_source(articles_path)

        print("Loading embeddings...")
        self.embeddings = np.load(embeddings_path)

        if clusters_path is not None:
            clusters_df = pd.read_json(clusters_path, lines=True)
            self.df = self.df.merge(clusters_df[["url", "topic", "prob"]], on="url", how="left")
        else:
            self.df["topic"] = None
            self.df["prob"] = None

        # Topic names: prefer the clean label map, fall back to BERTopic's Name.
        self.topic_name_map: dict = {}
        if topics_path is not None:
            try:
                tdf = pd.read_csv(topics_path)
                self.topic_name_map = {int(k): v for k, v in zip(tdf["Topic"], tdf["Name"])}
            except Exception:  # noqa: BLE001
                pass
        if labels_path is not None:
            try:
                import topic_labels

                self.topic_name_map = topic_labels.load_topic_labels(labels_path)
            except Exception:  # noqa: BLE001
                pass

        if news_only:
            mask = self.df["category"] == "أخبار"
            self.df = self.df[mask].reset_index(drop=True)
            self.embeddings = self.embeddings[mask.values]

        # The ~2 GB E5 model is lazy (see ``model``) so data + the 3D view are
        # instant; the model loads on the first search.
        self.model_name = model_name
        self.device = device
        self._model = None
        print(f"Ready: {len(self.df)} news articles (model loads on first search).")

    @property
    def model(self):
        """Lazily load (and cache) the SentenceTransformer E5 encoder."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def get_topic_name(self, topic) -> str:
        if topic is None or (isinstance(topic, float) and pd.isna(topic)):
            return "بدون موضوع"
        return self.topic_name_map.get(int(topic), f"موضوع {int(topic)}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = self.model.encode(f"query: {query}", normalize_embeddings=True)
        scores = self.embeddings @ query_vec  # cosine (both L2-normalized)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            topic = row.get("topic", None)
            results.append(
                {
                    "idx": int(idx),
                    "title": row["title"],
                    "body": row["body"],
                    "url": row["url"],
                    "category": row["category"],
                    "score": float(scores[idx]),
                    "topic": None if pd.isna(topic) else int(topic),
                    "topic_name": self.get_topic_name(topic),
                    "topic_probability": row.get("prob", None),
                }
            )
        return results
