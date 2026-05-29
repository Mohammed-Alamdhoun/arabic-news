"""Pluggable lemmatizer strategies.

``MLELemmatizer`` is fast and context-free — picks the corpus-wide most
likely analysis per token. Suitable for an MVP run.

``BERTLemmatizer`` is context-aware (CAMeL's BERT unfactored
disambiguator). Slower, but the right choice for this project's stated
ambiguity-handling goal. Not instantiated by default — wire it in via
``PipelineConfig.lemmatizer = "bert"``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from arnlp.preprocessing.normalizers import PLACEHOLDER_TOKENS, strip_diacritics


class BaseLemmatizer(ABC):
    """Interface for any lemmatizer the pipeline can use."""

    @abstractmethod
    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Return one bare (diacritic-stripped) lemma per input token.

        Placeholders must be preserved unchanged.
        """


class MLELemmatizer(BaseLemmatizer):
    """CAMeL ``MLEDisambiguator`` — fast, context-free."""

    def __init__(self) -> None:
        from camel_tools.disambig.mle import MLEDisambiguator
        self._mle = MLEDisambiguator.pretrained()

    def lemmatize(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []

        to_disambig: list[str] = []
        is_placeholder: list[bool] = []
        for t in tokens:
            if t in PLACEHOLDER_TOKENS:
                is_placeholder.append(True)
            else:
                is_placeholder.append(False)
                to_disambig.append(t)

        analyzed = iter(self._mle.disambiguate(to_disambig)) if to_disambig else iter([])

        out: list[str] = []
        for tok, ph in zip(tokens, is_placeholder):
            if ph:
                out.append(tok)
                continue
            d = next(analyzed)
            raw = d.analyses[0].analysis.get("lex", d.word) if d.analyses else d.word
            out.append(strip_diacritics(raw))
        return out


class BERTLemmatizer(BaseLemmatizer):
    """CAMeL ``BERTUnfactoredDisambiguator`` — context-aware (~15× slower).

    Use this when ambiguity matters: distinguishes "علي" (Ali, name) from
    "علي" (preposition), "الإمام" from "الأم" from "الآلام", etc.
    """

    def __init__(self) -> None:
        from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
        self._bert = BERTUnfactoredDisambiguator.pretrained("msa")

    def lemmatize(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []

        to_disambig: list[str] = []
        is_placeholder: list[bool] = []
        for t in tokens:
            if t in PLACEHOLDER_TOKENS:
                is_placeholder.append(True)
            else:
                is_placeholder.append(False)
                to_disambig.append(t)

        analyzed = iter(self._bert.disambiguate(to_disambig)) if to_disambig else iter([])

        out: list[str] = []
        for tok, ph in zip(tokens, is_placeholder):
            if ph:
                out.append(tok)
                continue
            d = next(analyzed)
            raw = d.analyses[0].analysis.get("lex", d.word) if d.analyses else d.word
            out.append(strip_diacritics(raw))
        return out


@lru_cache(maxsize=2)
def get_lemmatizer(kind: str) -> BaseLemmatizer:
    """Lazily build a lemmatizer; cached so we never load the model twice."""
    if kind == "mle":
        return MLELemmatizer()
    if kind == "bert":
        return BERTLemmatizer()
    raise ValueError(f"unknown lemmatizer kind: {kind!r}")


def filter_by_lemma(
    filtered_tokens: list[str],
    lemmas: list[str],
    stopwords: frozenset[str],
    min_len: int = 2,
) -> tuple[list[str], list[str]]:
    """Drop ``(token, lemma)`` pairs whose lemma is a stopword.

    Keeps the two arrays index-aligned so downstream code can zip them.
    """
    out_t: list[str] = []
    out_l: list[str] = []
    for tok, lem in zip(filtered_tokens, lemmas):
        if lem in PLACEHOLDER_TOKENS:
            out_t.append(tok); out_l.append(lem); continue
        if lem in stopwords or len(lem) <= min_len:
            continue
        out_t.append(tok); out_l.append(lem)
    return out_t, out_l
