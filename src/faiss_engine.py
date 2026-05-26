import json
from pathlib import Path

import faiss
import numpy as np

from src.exceptions import IndexNotFoundError


class FaissEngine:
    """FAISS-based vector similarity search engine."""

    def __init__(self, index: faiss.Index, passage_ids: list[int]) -> None:
        self.index = index
        self.passage_ids = passage_ids
        self.dimension: int = index.d

    def search(
        self, query_vectors: np.ndarray, topk: int = 20
    ) -> list[tuple[int, float]]:
        """Search for similar vectors.

        Returns:
            List of (article_id, inner_product_score) sorted by score desc.
        """
        scores, indices = self.index.search(query_vectors.astype(np.float32), topk)
        results: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx == -1:
                continue
            results.append((self.passage_ids[int(idx)], float(score)))
        return results

    @classmethod
    def build_from_vectors(
        cls,
        vectors: np.ndarray,
        passage_ids: list[int],
        index_path: str,
    ) -> "FaissEngine":
        """Build a FAISS index from embedding vectors and save to disk."""
        vectors = vectors.astype(np.float32)
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, index_path)
        return cls(index, passage_ids)

    @classmethod
    def load(cls, index_path: str, passage_ids_path: str) -> "FaissEngine":
        """Load a FAISS index and passage_ids from disk."""
        if not Path(index_path).exists():
            raise IndexNotFoundError(f"FAISS index not found at {index_path}")
        if not Path(passage_ids_path).exists():
            raise IndexNotFoundError(f"Passage IDs file not found at {passage_ids_path}")
        index = faiss.read_index(str(index_path))
        with open(passage_ids_path, encoding="utf-8") as f:
            passage_ids = json.load(f)
        return cls(index, passage_ids)
