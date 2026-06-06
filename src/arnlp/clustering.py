"""Stage 5 — topic clustering of the news subset (single-module edition).

BERTopic runs over the **precomputed** E5 embeddings on /embeddings/e5_large.npy, so we can skip the expensive text encoding step and cluster directly on the vectors.

Pipeline inside BERTopic:
    UMAP (cosine)  →  HDBSCAN (euclidean, density)  →  c-TF-IDF topic words

E5 vectors are L2-normalized, so cosine is the right UMAP metric. HDBSCAN
labels low-density points as topic ``-1`` (noise).

Public API:
    build_topic_model / fit_topics   — fit BERTopic on precomputed vectors
    build_umap / build_hdbscan       — the underlying reducer / clusterer
    cluster_quality / noise_fraction — internal geometry metrics
    per_topic_silhouette             — per-cluster silhouette (find weak topics)
    topic_coherence_npmi             — NPMI topic-word coherence (no gensim)
    topic_diversity                  — unique-word fraction across topics
    topic_words_from_model           — pull top-N words per topic
    save_clusters / load_clusters    — persist {url, topic, prob}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score

from arnlp.utils.io import read_jsonl, write_jsonl

__all__ = [
    "build_topic_model",
    "build_umap",
    "build_hdbscan",
    "fit_topics",
    "cluster_quality",
    "noise_fraction",
    "per_topic_silhouette",
    "topic_diversity",
    "topic_coherence_npmi",
    "topic_words_from_model",
    "save_clusters",
    "load_clusters",
]

# ===========================================================================
# Model — BERTopic over precomputed embeddings
# ===========================================================================

def _disable_triton() -> None:
    """Stop ``torch`` from importing ``triton`` while loading BERTopic.

    BERTopic pulls in ``torch._dynamo``, which probes for ``triton`` (a GPU
    codegen backend). On this CPU box the native ``triton`` import segfaults,
    so we register it as unimportable *before* the chain loads — ``torch``
    catches the ``ImportError`` and carries on. No effect on CPU clustering.
    """
    if "triton" not in sys.modules and "torch" not in sys.modules:
        sys.modules["triton"] = None  # type: ignore[assignment]


def build_umap(
    *,
    n_neighbors: int = 15,
    n_components: int = 5,
    min_dist: float = 0.0,
    random_state: int = 42,
):
    """UMAP reducer configured for L2-normalized embeddings (cosine)."""
    from umap import UMAP  # deferred: optional heavy dep (numba JIT); keeps metrics/io importable without it

    return UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )


def build_hdbscan(
    *,
    min_cluster_size: int = 15,
    min_samples: int | None = None,
):
    """HDBSCAN clusterer. ``min_cluster_size`` is the main knob for topic count."""
    from hdbscan import HDBSCAN  # deferred: optional heavy dep; keeps metrics/io importable without it

    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )


def build_topic_model(
    *,
    min_cluster_size: int = 15,
    n_neighbors: int = 15,
    n_components: int = 5,
    random_state: int = 42,
    min_topic_size: int | None = None,
    verbose: bool = True,
):
    """Assemble a :class:`bertopic.BERTopic` for precomputed embeddings.

    A whitespace-token ``CountVectorizer`` drives the topic-word labels; the
    documents we pass for representation should be the normalized
    ``clustering_view`` text. ``\\w`` is Unicode-aware so Arabic survives.
    """
    _disable_triton()
    from bertopic import BERTopic  # deferred: must load after _disable_triton()

    from arnlp.preprocessing import PLACEHOLDER_TOKENS, build_stopwords

    # The clustering_view keeps function words and the number placeholders
    # (NUM_TOKEN, ...). Both appear in nearly every article, so without
    # excluding them they dominate every topic's c-TF-IDF words. Drop the
    # Arabic stopword set plus the (lowercased) placeholder masks.
    stop_words = sorted(build_stopwords() | {t.lower() for t in PLACEHOLDER_TOKENS})
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b", stop_words=stop_words)

    return BERTopic(
        # "multilingual" (not the default "english") is critical: the English
        # path strips every non-[A-Za-z0-9] char from docs before building
        # topic words, which would erase all Arabic and leave only placeholders.
        language="multilingual",
        umap_model=build_umap(
            n_neighbors=n_neighbors,
            n_components=n_components,
            random_state=random_state,
        ),
        hdbscan_model=build_hdbscan(min_cluster_size=min_cluster_size),
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size or min_cluster_size,
        calculate_probabilities=False,
        verbose=verbose,
    )


def fit_topics(
    model,
    docs: list[str],
    embeddings: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """Fit ``model`` on ``docs`` using ``embeddings`` (no re-encoding).

    Returns ``(topics, probs)`` aligned to the input rows. ``probs`` is the
    HDBSCAN membership strength of each article to its assigned topic.
    """
    topics, probs = model.fit_transform(docs, embeddings=embeddings)
    probs = np.asarray(probs, dtype=float) if probs is not None else np.zeros(len(topics))
    return list(topics), probs


# ===========================================================================
# Metrics — internal cluster quality
# ===========================================================================
#
# Every article in the news subset shares the same category (أخبار), so
# external/category metrics don't apply — we score the partition with
# internal metrics on the embeddings. HDBSCAN noise points (topic ``-1``)
# are excluded before scoring.


def noise_fraction(topics: list[int] | np.ndarray) -> float:
    """Share of articles HDBSCAN left unclustered (topic ``-1``)."""
    topics = np.asarray(topics)
    return float(np.mean(topics == -1)) if topics.size else 0.0


def cluster_quality(
    embeddings: np.ndarray,
    topics: list[int] | np.ndarray,
) -> dict[str, float]:
    """Silhouette (cosine), Davies–Bouldin, noise fraction and topic count.

    Returns NaN for the two distance metrics when fewer than two non-noise
    clusters survive (they are undefined in that case).
    """
    topics = np.asarray(topics)
    mask = topics != -1
    kept = topics[mask]
    n_clusters = len(set(kept.tolist()))

    out: dict[str, float] = {
        "n_clusters": float(n_clusters),
        "n_clustered": float(mask.sum()),
        "noise_fraction": noise_fraction(topics),
    }
    if n_clusters < 2:
        out["silhouette_cosine"] = float("nan")
        out["davies_bouldin"] = float("nan")
        return out

    X = embeddings[mask]
    out["silhouette_cosine"] = float(silhouette_score(X, kept, metric="cosine"))
    out["davies_bouldin"] = float(davies_bouldin_score(X, kept))
    return out


def per_topic_silhouette(
    embeddings: np.ndarray,
    topics: list[int] | np.ndarray,
) -> dict[int, float]:
    """Mean cosine silhouette per topic — surfaces which clusters are weak.

    Noise (``-1``) is dropped. Returns ``{topic_id: mean_silhouette}``; a
    low (or negative) value flags a topic whose articles sit closer to a
    neighbouring topic than to their own.
    """
    from sklearn.metrics import silhouette_samples

    topics = np.asarray(topics)
    mask = topics != -1
    kept = topics[mask]
    if len(set(kept.tolist())) < 2:
        return {}
    samples = silhouette_samples(embeddings[mask], kept, metric="cosine")
    return {
        int(t): float(samples[kept == t].mean())
        for t in sorted(set(kept.tolist()))
    }


def topic_diversity(topic_words: list[list[str]], top_n: int = 10) -> float:
    """Fraction of *unique* words among the top-N words of every topic.

    1.0 means no word is shared across topics; low values mean topics
    repeat the same generic words (a sign of over-merging / weak topics).
    """
    words = [w for tw in topic_words for w in tw[:top_n]]
    return len(set(words)) / len(words) if words else 0.0


def topic_coherence_npmi(
    docs: list[str],
    topic_words: list[list[str]],
    top_n: int = 10,
    eps: float = 1e-12,
) -> tuple[float, list[float]]:
    """Document-level NPMI coherence (no gensim dependency).

    For each topic, average the pairwise NPMI of its top-N words over the
    document co-occurrence of ``docs`` (the ``clustering_view`` texts).
    NPMI ∈ [-1, 1]; higher = the topic's words genuinely co-occur. Returns
    ``(mean_over_topics, per_topic_scores)``.
    """
    vocab = sorted({w for tw in topic_words for w in tw[:top_n]})
    if not vocab:
        return float("nan"), []
    idx = {w: i for i, w in enumerate(vocab)}

    cv = CountVectorizer(vocabulary=vocab, binary=True, token_pattern=r"(?u)\b\w\w+\b")
    X = cv.transform(docs)              # (D, V) boolean doc-term matrix
    n_docs = X.shape[0]
    df = np.asarray(X.sum(axis=0)).ravel().astype(float)   # docs per word
    co = (X.T @ X).toarray().astype(float)                 # docs with both words
    p_w = df / n_docs

    per_topic: list[float] = []
    for tw in topic_words:
        words = [w for w in tw[:top_n] if w in idx]
        pairs: list[float] = []
        for a in range(len(words)):
            for b in range(a + 1, len(words)):
                i, j = idx[words[a]], idx[words[b]]
                p_ij = co[i, j] / n_docs
                if p_ij <= 0:
                    pairs.append(-1.0)        # never co-occur → minimally coherent
                    continue
                pmi = np.log(p_ij / (p_w[i] * p_w[j] + eps) + eps)
                pairs.append(pmi / (-np.log(p_ij + eps)))
        if pairs:
            per_topic.append(float(np.mean(pairs)))
    mean = float(np.mean(per_topic)) if per_topic else float("nan")
    return mean, per_topic


def topic_words_from_model(model, top_n: int = 10) -> list[list[str]]:
    """Extract the top-N representative words for each non-noise topic."""
    info = model.get_topic_info()
    out: list[list[str]] = []
    for t in info["Topic"]:
        if t == -1:
            continue
        out.append([w for w, _ in model.get_topic(t)][:top_n])
    return out


# ===========================================================================
# IO — persist cluster assignments
# ===========================================================================
#
# One line per news article so downstream stages (summarization, evaluation)
# can join back on ``url``: ``{"url": ..., "topic": 3, "prob": 0.87}``.

def save_clusters(
    path: str | Path,
    urls: list[str],
    topics: list[int] | np.ndarray,
    probs: list[float] | np.ndarray,
) -> int:
    """Write ``clusters.jsonl`` (``url`` → ``topic``/``prob``)."""
    topics = np.asarray(topics)
    probs = np.asarray(probs, dtype=float)
    rows = (
        {"url": u, "topic": int(t), "prob": round(float(p), 4)}
        for u, t, p in zip(urls, topics, probs)
    )
    return write_jsonl(path, rows)


def load_clusters(path: str | Path) -> list[dict]:
    """Read ``clusters.jsonl`` back into a list of records."""
    return list(read_jsonl(path))
