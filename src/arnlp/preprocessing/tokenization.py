"""Whitespace tokenization + stopword filtering."""

from __future__ import annotations

from typing import Iterable

from arnlp.preprocessing.normalizers import PLACEHOLDER_TOKENS


def tokenize(text: str) -> list[str]:
    """Whitespace split. ``text`` is expected to already be normalized."""
    return text.split()


def filter_stopwords(
    tokens: Iterable[str],
    stopwords: frozenset[str],
    min_len: int = 2,
) -> list[str]:
    """Drop tokens that are stopwords or below the length floor.

    Placeholder tokens (``YEAR_TOKEN``, ...) are always preserved.
    """
    out: list[str] = []
    for t in tokens:
        if t in PLACEHOLDER_TOKENS:
            out.append(t)
            continue
        if t in stopwords:
            continue
        if len(t) <= min_len:
            continue
        out.append(t)
    return out
