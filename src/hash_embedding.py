from __future__ import annotations

import hashlib
from typing import List

import numpy as np
from llama_index.core.embeddings import BaseEmbedding


class HashEmbedding(BaseEmbedding):
    """
    Lightweight, offline embedding.
    Deterministic hashing -> fixed-size vector.
    Not semantic like transformer embeddings, but works for demo + Qdrant.
    """

    def __init__(self, dim: int = 384):
        super().__init__()
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _hash_to_vec(self, text: str) -> List[float]:
        # Tokenize very simply
        tokens = (text or "").lower().split()
        vec = np.zeros(self._dim, dtype=np.float32)

        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            # Use first 4 bytes as an integer -> index
            idx = int.from_bytes(h[:4], "little") % self._dim
            vec[idx] += 1.0

        # L2 normalize to make cosine meaningful
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._hash_to_vec(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._hash_to_vec(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._hash_to_vec(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._hash_to_vec(query)
