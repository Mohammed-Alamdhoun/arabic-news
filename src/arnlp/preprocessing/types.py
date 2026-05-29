"""Shared dataclasses for the preprocessing layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DateKind = Literal["dmy", "my", "numeric", "year"]


@dataclass(frozen=True)
class DateHit:
    """A single date mention extracted from an article."""
    raw: str          # original substring from the source
    iso: str          # best-effort ISO 8601: YYYY[-MM[-DD]]
    kind: DateKind
    span: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class PreprocessedDocument:
    """All views produced for one article.

    The three text views serve three different downstream tasks:
      * ``embedding_view``  — for multilingual-E5 / mT5 / semantic search
      * ``clustering_view`` — for BERTopic class-TF-IDF
      * ``tokens`` / ``filtered_tokens`` / ``lemmas`` — derived from clustering view
    """
    embedding_view: str
    clustering_view: str
    tokens: list[str]
    filtered_tokens: list[str]
    lemmas: list[str]
    dates: list[DateHit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "embedding_view":  self.embedding_view,
            "clustering_view": self.clustering_view,
            "tokens":          self.tokens,
            "filtered_tokens": self.filtered_tokens,
            "lemmas":          self.lemmas,
            "dates": [
                {"raw": d.raw, "iso": d.iso, "kind": d.kind}
                for d in self.dates
            ],
        }
