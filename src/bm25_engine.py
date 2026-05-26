import pickle
import warnings
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from src.exceptions import IndexNotFoundError

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="jieba._compat",
)


class BM25Engine:
    """Sparse retrieval engine based on BM25 + jieba tokenization."""

    def __init__(
        self,
        corpus: list[str],
        passage_ids: list[int] | None = None,
        tokenized_corpus: list[list[str]] | None = None,
    ) -> None:
        self.corpus = corpus
        self.passage_ids = passage_ids or list(range(len(corpus)))
        self.tokenized_corpus = tokenized_corpus or [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return list(jieba.cut(text))

    def search(self, query: str, topk: int = 20) -> list[tuple[int, float]]:
        """Search and return [(article_id, bm25_score), ...], sorted by score desc."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:topk]
        return [
            (self.passage_ids[int(idx)], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

    def save(self, path: str) -> None:
        """Persist the engine to disk via pickle."""
        data = {
            "corpus": self.corpus,
            "tokenized": self.tokenized_corpus,
            "passage_ids": self.passage_ids,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BM25Engine":
        """Restore the engine from a pickle file."""
        if not Path(path).exists():
            raise IndexNotFoundError(f"BM25 index not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            corpus=data["corpus"],
            passage_ids=data.get("passage_ids"),
            tokenized_corpus=data["tokenized"],
        )
