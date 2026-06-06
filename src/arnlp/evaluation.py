"""Stage 8 — evaluation metrics, focused on summarization.

The internal **clustering** metrics live in :mod:`arnlp.clustering`
(``cluster_quality``, ``per_topic_silhouette``, ``topic_coherence_npmi``,
``topic_diversity``, ``noise_fraction``). This module covers **summarization**.

The corpus has **no gold summaries**, so evaluation combines:

* **Relevance** — Arabic-normalized ROUGE-1/2/L of the summary against the
  article **title** (a weak, headline-style reference). Recall is the
  meaningful direction: how much of the headline the summary recovers.
* **Faithfulness** (reference-free) — ``grounding`` (share of summary content
  words found in the source), ``number_hallucination`` (numbers asserted that
  the source never mentions — a high-signal error), and ``novel_content_rate``.
* **Fluency / non-redundancy** (reference-free) — ``distinct_n`` and the derived
  ``redundancy`` (repeated bigrams within a summary).
* **Compression** — summary length as a fraction of the source.

Everything is dependency-light (no ``rouge_score``/``gensim``) and reuses the
Arabic normalizers from :mod:`arnlp.preprocessing`, so ROUGE folds hamza/alef
and digits the same way the rest of the pipeline does.

Public API::

    rouge_scores(predictions, references)      # ROUGE-1/2/L P/R/F1 (mean)
    grounding(summary, source)                 # content-word coverage ∈ [0,1]
    number_hallucination(summary, source)      # ungrounded-number rate ∈ [0,1]
    novel_content_rate(summary, source)        # abstractiveness ∈ [0,1]
    distinct_n(text, n) / redundancy(text)     # intra-summary repetition
    compression_ratio(summary, source)
    evaluate_summaries(sources, summaries, titles)  # aggregate everything
"""

from __future__ import annotations

import re
from collections import Counter

from arnlp.preprocessing import (
    PLACEHOLDER_TOKENS,
    build_stopwords,
    normalize_digits,
    normalize_for_embedding,
)

__all__ = [
    "rouge_scores",
    "rouge_pair",
    "grounding",
    "number_hallucination",
    "novel_content_rate",
    "distinct_n",
    "redundancy",
    "compression_ratio",
    "evaluate_summaries",
]

# Unicode word tokens (Arabic survives ``\w``); punctuation is dropped.
_WORD = re.compile(r"\w+", re.UNICODE)
# Numeric tokens after digit-folding: integers/decimals like 50, 2026, 3.5, 12,000.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

_STOPWORDS = build_stopwords() | {t.lower() for t in PLACEHOLDER_TOKENS}


# ===========================================================================
# Tokenization (shared, Arabic-normalized)
# ===========================================================================

def _tokens(text: str) -> list[str]:
    """Normalized word tokens (hamza/alef folded, diacritics stripped)."""
    return _WORD.findall(normalize_for_embedding(text or ""))


def _content_tokens(text: str) -> list[str]:
    """``_tokens`` with Arabic stopwords and placeholder masks removed."""
    return [t for t in _tokens(text) if t not in _STOPWORDS]


def _numbers(text: str) -> list[str]:
    """Digit sequences after folding Arabic-Indic digits to ASCII."""
    return _NUMBER.findall(normalize_digits(text or ""))


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


# ===========================================================================
# ROUGE (relevance vs. the title)
# ===========================================================================

def _prf(match: float, pred_total: int, ref_total: int) -> dict[str, float]:
    p = match / pred_total if pred_total else 0.0
    r = match / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"p": p, "r": r, "f": f}


def _rouge_n(pred: list[str], ref: list[str], n: int) -> dict[str, float]:
    pg, rg = _ngrams(pred, n), _ngrams(ref, n)
    match = sum((pg & rg).values())  # clipped overlap
    return _prf(float(match), max(0, len(pred) - n + 1), max(0, len(ref) - n + 1))


def _lcs(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (token level)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_pair(prediction: str, reference: str) -> dict[str, dict[str, float]]:
    """ROUGE-1/2/L (each as ``{p, r, f}``) for one summary/reference pair."""
    pred, ref = _tokens(prediction), _tokens(reference)
    lcs = _lcs(pred, ref)
    return {
        "rouge1": _rouge_n(pred, ref, 1),
        "rouge2": _rouge_n(pred, ref, 2),
        "rougeL": _prf(float(lcs), len(pred), len(ref)),
    }


def rouge_scores(
    predictions: list[str], references: list[str]
) -> dict[str, dict[str, float]]:
    """Mean ROUGE-1/2/L over many pairs. Returns ``{metric: {p, r, f}}``."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    keys = ("rouge1", "rouge2", "rougeL")
    acc = {k: {"p": 0.0, "r": 0.0, "f": 0.0} for k in keys}
    n = len(predictions) or 1
    for pred, ref in zip(predictions, references):
        scored = rouge_pair(pred, ref)
        for k in keys:
            for m in ("p", "r", "f"):
                acc[k][m] += scored[k][m]
    return {k: {m: acc[k][m] / n for m in ("p", "r", "f")} for k in keys}


# ===========================================================================
# Faithfulness (reference-free, vs. the source body)
# ===========================================================================

def grounding(summary: str, source: str) -> float:
    """Share of summary **content** words that also occur in the source.

    1.0 = every content word is anchored in the article; low values flag
    off-topic or hallucinated content. Abstractive paraphrase lowers this
    legitimately, so read it as a relative signal, not an absolute.
    """
    summ = _content_tokens(summary)
    if not summ:
        return 0.0
    src = set(_content_tokens(source))
    return sum(t in src for t in summ) / len(summ)


def number_hallucination(summary: str, source: str) -> float:
    """Fraction of the summary's numbers that never appear in the source.

    Numbers in a faithful news summary almost always come from the article, so
    an ungrounded number is a near-certain factual error — the sharpest
    reference-free faithfulness signal we have. Returns 0.0 when the summary
    states no numbers.
    """
    nums = _numbers(summary)
    if not nums:
        return 0.0
    src = set(_numbers(source))
    return sum(n not in src for n in nums) / len(nums)


def novel_content_rate(summary: str, source: str) -> float:
    """Share of summary content words **not** in the source (abstractiveness).

    Some novelty is expected from an abstractive model; very high novelty
    alongside low ``grounding`` is the hallucination danger zone.
    """
    summ = _content_tokens(summary)
    if not summ:
        return 0.0
    src = set(_content_tokens(source))
    return sum(t not in src for t in summ) / len(summ)


# ===========================================================================
# Redundancy / fluency (reference-free, within a summary)
# ===========================================================================

def distinct_n(text: str, n: int = 2) -> float:
    """Unique n-grams / total n-grams. 1.0 = no repetition; lower = repetitive."""
    grams = _ngrams(_tokens(text), n)
    total = sum(grams.values())
    return len(grams) / total if total else 1.0


def redundancy(text: str) -> float:
    """Repeated-bigram fraction within a summary (``1 - distinct_2``)."""
    return 1.0 - distinct_n(text, 2)


# ===========================================================================
# Length
# ===========================================================================

def compression_ratio(summary: str, source: str) -> float:
    """Summary words / source words (lower = more compression)."""
    s = len((source or "").split())
    return len((summary or "").split()) / s if s else 0.0


# ===========================================================================
# Aggregate
# ===========================================================================

def evaluate_summaries(
    sources: list[str],
    summaries: list[str],
    titles: list[str] | None = None,
) -> dict[str, float]:
    """Aggregate every metric over a batch into a flat ``{name: mean}`` dict.

    ``titles`` (optional) are the weak ROUGE references; omit them to get only
    the reference-free faithfulness/redundancy/compression metrics.
    """
    n = len(summaries) or 1
    out: dict[str, float] = {
        "n": float(len(summaries)),
        "summary_words": sum(len(s.split()) for s in summaries) / n,
        "grounding": sum(grounding(s, src) for s, src in zip(summaries, sources)) / n,
        "number_hallucination": sum(
            number_hallucination(s, src) for s, src in zip(summaries, sources)
        )
        / n,
        "novel_content_rate": sum(
            novel_content_rate(s, src) for s, src in zip(summaries, sources)
        )
        / n,
        "redundancy": sum(redundancy(s) for s in summaries) / n,
        "distinct_2": sum(distinct_n(s, 2) for s in summaries) / n,
        "compression_ratio": sum(
            compression_ratio(s, src) for s, src in zip(summaries, sources)
        )
        / n,
    }
    if titles is not None:
        r = rouge_scores(summaries, titles)
        out["rouge1_r"] = r["rouge1"]["r"]
        out["rouge1_f"] = r["rouge1"]["f"]
        out["rouge2_f"] = r["rouge2"]["f"]
        out["rougeL_f"] = r["rougeL"]["f"]
    return out
