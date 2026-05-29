"""Composable Arabic preprocessing pipeline (Stage 2).

Typical use::

    from arnlp.preprocessing import ArabicTextPipeline, PipelineConfig

    pipe = ArabicTextPipeline(PipelineConfig(lemmatizer="mle"))
    doc  = pipe.process(article_text)

    # multi-view output:
    doc.embedding_view   # → E5 / mT5 / search
    doc.clustering_view  # → BERTopic
    doc.lemmas           # 1:1 aligned with doc.filtered_tokens
    doc.dates            # list[DateHit]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal

from arnlp.preprocessing.lemmatization import (
    BaseLemmatizer, filter_by_lemma, get_lemmatizer,
)
from arnlp.preprocessing.normalizers import (
    normalize_digits, normalize_for_clustering, normalize_for_embedding,
)
from arnlp.preprocessing.numbers_dates import extract_dates, mask_numbers
from arnlp.preprocessing.stopwords import build_stopwords
from arnlp.preprocessing.tokenization import filter_stopwords, tokenize
from arnlp.preprocessing.types import PreprocessedDocument


@dataclass
class PipelineConfig:
    """Knobs for ``ArabicTextPipeline``.

    Defaults reflect the validated v2 behavior. Flip ``lemmatizer="bert"``
    to swap to the context-aware disambiguator (~15× slower).
    """
    enable_number_masking:    bool = True
    enable_date_extraction:   bool = True
    enable_lemmatization:     bool = True
    enable_post_lemma_filter: bool = True
    lemmatizer:               Literal["mle", "bert"] = "mle"
    min_token_len:            int = 2
    extra_stopwords:          frozenset[str] = field(default_factory=frozenset)


class ArabicTextPipeline:
    """Composes the preprocessing steps into a single ``process`` call.

    The lemmatizer is lazily constructed from ``config.lemmatizer`` on
    first use (so importing the pipeline is fast). Pass ``lemmatizer=``
    explicitly to bypass — useful in tests.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        lemmatizer: BaseLemmatizer | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._lemmatizer = lemmatizer
        self.stopwords = build_stopwords(extra=self.config.extra_stopwords)

    @property
    def lemmatizer(self) -> BaseLemmatizer:
        if self._lemmatizer is None:
            self._lemmatizer = get_lemmatizer(self.config.lemmatizer)
        return self._lemmatizer

    def process(self, text: str | None) -> PreprocessedDocument:
        text = text or ""

        digit_norm = normalize_digits(text)

        dates = (
            extract_dates(digit_norm)
            if self.config.enable_date_extraction
            else []
        )

        embedding_view = normalize_for_embedding(digit_norm)

        masked = mask_numbers(digit_norm) if self.config.enable_number_masking else digit_norm
        clustering_view = normalize_for_clustering(masked)

        tokens = tokenize(clustering_view)
        filtered = filter_stopwords(
            tokens, self.stopwords, min_len=self.config.min_token_len
        )

        if self.config.enable_lemmatization:
            lemmas = self.lemmatizer.lemmatize(filtered)
            if self.config.enable_post_lemma_filter:
                filtered, lemmas = filter_by_lemma(
                    filtered, lemmas, self.stopwords,
                    min_len=self.config.min_token_len,
                )
        else:
            lemmas = list(filtered)

        return PreprocessedDocument(
            embedding_view=embedding_view,
            clustering_view=clustering_view,
            tokens=tokens,
            filtered_tokens=filtered,
            lemmas=lemmas,
            dates=dates,
        )

    def process_batch(self, texts: Iterable[str | None]) -> Iterator[PreprocessedDocument]:
        for t in texts:
            yield self.process(t)
