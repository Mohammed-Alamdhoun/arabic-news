"""Stage 4 — document embeddings.

Public API::

    from arnlp.embeddings import E5Encoder
    from arnlp.embeddings.io import save_embeddings, load_embeddings
"""

from arnlp.embeddings.base import BaseEncoder
from arnlp.embeddings.e5 import E5Encoder

__all__ = ["BaseEncoder", "E5Encoder"]
