"""Embedding models behind one small interface.

Descriptions are compared passage-to-passage (symmetric similarity), so no query/passage prefixes are added.
"""

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """An ONNX model run on CPU through fastembed. Downloaded on first use (~70-220 MB) and cached."""

    def __init__(self, model_name: str, dim: int):
        from fastembed import TextEmbedding  # imported lazily: tests that use HashEmbedder don't pay for onnxruntime

        self.name = model_name
        self.dim = dim
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = [v.tolist() for v in self._model.embed(texts, batch_size=32)]
        for v in vectors:
            if len(v) != self.dim:
                raise ValueError(f"{self.name} returned {len(v)} dimensions, schema expects {self.dim}")
        return [_unit(v) for v in vectors]


class HashEmbedder:
    """Hashed bag-of-words. Deterministic and instant: used by the tests, never by the running service."""

    def __init__(self, dim: int = 384):
        self.name = f"hash-bow-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for token in re.findall(r"\w+", text.lower()):
                h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
                v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
            out.append(_unit(v))
        return out


def _unit(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        # Cosine distance is undefined for a zero vector; nudge it so pgvector doesn't return NaN.
        v = [0.0] * len(v)
        v[0] = 1.0
        return v
    return [x / norm for x in v]


def build_embedder(model_name: str, dim: int) -> Embedder:
    if model_name.startswith("hash-bow"):
        return HashEmbedder(dim)
    return FastEmbedder(model_name, dim)
