import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import (
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TTL,
)


@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: np.ndarray
    created_at: float
    ttl: int


class SemanticCache:
    """
    In-memory semantic cache.

    SentenceTransformer creates embeddings and cosine
    similarity is used to find semantically similar queries.

    Redis can be added later as a persistent storage layer.
    """

    def __init__(
        self,
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        ttl: int = CACHE_TTL,
    ):
        self.threshold = threshold
        self.ttl = ttl
        self.entries: List[CacheEntry] = []

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Install it using:\n"
                "pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    @staticmethod
    def cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a, b)
            / denominator
        )

    def _embed(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Generate an embedding for text.
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    def set(
        self,
        query: str,
        response: str,
    ) -> None:
        """
        Store a query-response pair.
        """
        embedding = self._embed(
            query
        )

        entry = CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            created_at=time.time(),
            ttl=self.ttl,
        )

        self.entries.append(
            entry
        )

    def get(
        self,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Search the cache using cosine similarity.

        Returns a cache hit when similarity >= threshold.
        """
        if not self.entries:
            return None

        query_embedding = self._embed(
            query
        )

        best_match = None
        best_similarity = -1.0

        current_time = time.time()

        for entry in self.entries:

            # Check TTL
            if (
                current_time
                - entry.created_at
                > entry.ttl
            ):
                continue

            similarity = self.cosine_similarity(
                query_embedding,
                entry.embedding,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if (
            best_match is not None
            and best_similarity
            >= self.threshold
        ):
            return {
                "response": best_match.response,
                "matched_query": best_match.query,
                "similarity": best_similarity,
                "cache_hit": True,
            }

        return None

    def clear(self) -> None:
        """
        Clear all cache entries.
        """
        self.entries.clear()

    def size(self) -> int:
        """
        Return number of cache entries.
        """
        return len(
            self.entries
        )


def exact_cache_key(
    query: str,
) -> str:
    """
    Create a deterministic SHA256 key.
    """
    normalized = (
        query.strip().lower()
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()