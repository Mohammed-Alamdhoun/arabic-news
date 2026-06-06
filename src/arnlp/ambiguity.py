"""Stage 6 — ambiguity handling via embedding-based Word Sense Disambiguation.

Same approach as the search app (``app/search_engine.py``): encode text with
multilingual-E5 and rank by cosine similarity. Here, instead of ranking
*articles* against a query, we rank the candidate **senses** of an ambiguous
Arabic word against its surrounding context — the sense whose gloss sits
closest to the context in E5 space wins.

    context  ──encode_query──►  q          (1024-d, L2-normalized)
    senses   ──encode────────►  S          (n×1024, L2-normalized)
    score    = S · q                       (cosine, since both normalized)
    best     = argmax(score)

The encoder is injectable (any object exposing ``encode(list[str]) -> ndarray``
and, optionally, ``encode_query(str) -> ndarray``); it defaults to
:class:`arnlp.embeddings.E5Encoder`, so the disambiguator stays unit-testable
without downloading the model.

Example::

    from arnlp.ambiguity import EmbeddingDisambiguator, DEFAULT_SENSES

    wsd = EmbeddingDisambiguator()
    ranked = wsd.disambiguate_word(
        "عين",
        "تدفقت مياه العين الباردة من سفح الجبل",
    )
    print(ranked[0].sense.sense_id, round(ranked[0].score, 3))   # -> spring
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "Sense",
    "SenseScore",
    "EmbeddingDisambiguator",
    "DEFAULT_SENSES",
    "load_sense_inventory",
]


# ===========================================================================
# Data types
# ===========================================================================

@dataclass(frozen=True)
class Sense:
    """One candidate meaning of an ambiguous word.

    ``gloss`` is a short Arabic definition; ``examples`` are optional example
    usages. Both are embedded together to give E5 more signal about the sense.
    """
    word: str
    sense_id: str
    gloss: str
    examples: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """The string actually embedded for this sense (gloss + examples)."""
        return " ".join((self.gloss, *self.examples)).strip()


@dataclass(frozen=True)
class SenseScore:
    """A sense paired with its cosine similarity to the query context."""
    sense: Sense
    score: float


# ===========================================================================
# Disambiguator
# ===========================================================================

class EmbeddingDisambiguator:
    """Pick the contextually correct sense of a word using E5 + cosine.

    Parameters
    ----------
    encoder:
        Any encoder exposing ``encode(list[str]) -> (n, d) ndarray`` returning
        L2-normalized vectors. If it also exposes ``encode_query(str)`` that is
        used for the context (E5's ``query:`` prefix), otherwise ``encode`` is
        used as a fallback. Defaults to :class:`arnlp.embeddings.E5Encoder`.
    device:
        Forwarded to the default ``E5Encoder`` when ``encoder`` is not given.
    """

    def __init__(self, encoder=None, *, device: str | None = None) -> None:
        self._encoder = encoder
        self._device = device

    @property
    def encoder(self):
        """Lazily build the default E5 encoder on first use."""
        if self._encoder is None:
            from arnlp.embeddings import E5Encoder

            self._encoder = E5Encoder(device=self._device, show_progress=False)
        return self._encoder

    def _encode_context(self, context: str) -> np.ndarray:
        enc = self.encoder
        if hasattr(enc, "encode_query"):
            return np.asarray(enc.encode_query(context), dtype=np.float32)
        return np.asarray(enc.encode([context])[0], dtype=np.float32)

    def disambiguate(
        self,
        context: str,
        senses: list[Sense],
        *,
        top_k: int | None = None,
    ) -> list[SenseScore]:
        """Rank ``senses`` by cosine similarity of their gloss to ``context``.

        Returns a list of :class:`SenseScore` sorted best-first. ``top_k``
        trims the result; ``None`` returns all senses.
        """
        if not senses:
            return []

        sense_vecs = np.asarray(
            self.encoder.encode([s.text for s in senses]), dtype=np.float32
        )
        ctx_vec = self._encode_context(context)

        # Both sides are L2-normalized, so the dot product is the cosine.
        scores = sense_vecs @ ctx_vec
        order = np.argsort(scores)[::-1]

        ranked = [SenseScore(senses[i], float(scores[i])) for i in order]
        return ranked if top_k is None else ranked[:top_k]

    def best_sense(self, context: str, senses: list[Sense]) -> SenseScore | None:
        """Return the single best sense (or ``None`` if no senses given)."""
        ranked = self.disambiguate(context, senses, top_k=1)
        return ranked[0] if ranked else None

    def disambiguate_word(
        self,
        word: str,
        context: str,
        inventory: dict[str, list[Sense]] | None = None,
        *,
        top_k: int | None = None,
    ) -> list[SenseScore]:
        """Look ``word`` up in a sense inventory, then disambiguate by context.

        Falls back to :data:`DEFAULT_SENSES`. Returns ``[]`` for unknown words.
        """
        inventory = inventory if inventory is not None else DEFAULT_SENSES
        return self.disambiguate(context, inventory.get(word, []), top_k=top_k)


# ===========================================================================
# Sense inventory
# ===========================================================================

def _senses(word: str, *entries: tuple[str, str]) -> list[Sense]:
    return [Sense(word=word, sense_id=sid, gloss=gloss) for sid, gloss in entries]


#: A small built-in inventory of genuinely ambiguous Arabic words, each mapped
#: to its distinct senses (Arabic glosses). Extend it, or load your own with
#: :func:`load_sense_inventory`.
DEFAULT_SENSES: dict[str, list[Sense]] = {
    "عين": _senses(
        "عين",
        ("eye", "العضو الذي يبصر به الإنسان والحيوان"),
        ("spring", "نبع الماء الذي يتدفق من باطن الأرض"),
        ("spy", "الجاسوس أو المراقب الذي يرسل لتقصي الأخبار"),
        ("notable", "الشخص المهم أو الوجيه ذو المكانة"),
    ),
    "خال": _senses(
        "خال",
        ("uncle", "أخو الأم من القرابة"),
        ("mole", "الشامة أو النقطة السوداء على الجلد"),
        ("empty", "الفارغ الذي لا شيء فيه"),
    ),
    "ذهب": _senses(
        "ذهب",
        ("gold", "المعدن النفيس الأصفر الثمين"),
        ("went", "فعل الذهاب والانتقال من مكان إلى آخر"),
    ),
    "جناح": _senses(
        "جناح",
        ("wing", "جناح الطائر أو القسم الجانبي من المبنى"),
        ("sin", "الإثم أو الذنب والحرج"),
    ),
    "نواة": _senses(
        "نواة",
        ("nucleus", "مركز الذرة أو الخلية"),
        ("pit", "بذرة الثمرة الصلبة في داخلها"),
        ("core", "النواة الأساسية أو الأصل الذي يبنى عليه"),
    ),
}


def load_sense_inventory(path: str | Path) -> dict[str, list[Sense]]:
    """Load a sense inventory from JSON.

    Expected shape::

        {
          "عين": [
            {"sense_id": "eye",    "gloss": "...", "examples": ["..."]},
            {"sense_id": "spring", "gloss": "..."}
          ],
          ...
        }
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    inventory: dict[str, list[Sense]] = {}
    for word, entries in raw.items():
        inventory[word] = [
            Sense(
                word=word,
                sense_id=e["sense_id"],
                gloss=e["gloss"],
                examples=tuple(e.get("examples", ())),
            )
            for e in entries
        ]
    return inventory
