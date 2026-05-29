"""Encoder interface used by ``scripts/run_embeddings.py``.

Concrete encoders subclass :class:`BaseEncoder` and implement
``encode``. The CLI calls them uniformly so swapping E5 for fastText
(or anything else later) is a config switch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEncoder(ABC):
    """Common contract for document encoders."""

    name: str           # short identifier used in output file names
    dim:  int           # output vector dimension
    input_kind: str     # "text" (single string) or "tokens" (list[str])

    @abstractmethod
    def encode(self, items: list[Any]) -> np.ndarray:
        """Return a ``(len(items), dim)`` float32 array.

        For ``input_kind == "text"`` each item is a string; for
        ``"tokens"`` each item is a list of token strings.
        E5 vectors are expected to be L2-normalized so cosine similarity
        equals dot product downstream.
        """
