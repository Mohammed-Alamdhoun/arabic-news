"""fastText baseline encoder (cc.ar.300).

Consumes the ``filtered_tokens`` field (a list of tokens). Each
document vector is the mean of its token vectors; OOV tokens
contribute the zero vector.

Loading the model is a ~4 GB read. The encoder downloads it on first
use if the file isn't present at ``model_path``.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from arnlp.embeddings.base import BaseEncoder

FT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.bin.gz"


def _download(dest: Path) -> Path:
    """Download + decompress the Arabic fastText binary if missing."""
    if dest.exists():
        return dest
    gz_path = dest.with_suffix(dest.suffix + ".gz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fastText] downloading {FT_URL} → {gz_path}")
    urllib.request.urlretrieve(FT_URL, gz_path)
    print(f"[fastText] decompressing → {dest}")
    with gzip.open(gz_path, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    gz_path.unlink()
    return dest


class FastTextEncoder(BaseEncoder):
    name = "fasttext_ar"
    dim = 300
    input_kind = "tokens"

    def __init__(self, model_path: str | Path) -> None:
        import fasttext

        self.model_path = _download(Path(model_path))
        print(f"[fastText] loading {self.model_path}")
        self._ft = fasttext.load_model(str(self.model_path))

    def encode(self, token_lists: list[list[str]]) -> np.ndarray:
        from tqdm import tqdm

        out = np.zeros((len(token_lists), self.dim), dtype=np.float32)
        for i, toks in enumerate(tqdm(token_lists, desc="fastText", unit="doc")):
            if not toks:
                continue
            vecs = np.stack([self._ft.get_word_vector(t) for t in toks])
            out[i] = vecs.mean(axis=0)
        return out
