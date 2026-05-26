"""Consolidated mock classes for unit testing all modules."""

import numpy as np

from src.models import RerankedArticle, RetrievedArticle


class MockEmbeddingEngine:
    """Fake embedding engine outputting random normalized vectors."""

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        rng = np.random.default_rng(42)
        arr = rng.normal(size=(len(texts), self.dimension)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms


class MockBM25Engine:
    """Fake BM25 engine matching keywords in documents."""

    def __init__(
        self,
        corpus: list[str],
        passage_ids: list[int] | None = None,
    ) -> None:
        self.corpus = corpus
        self.passage_ids = passage_ids or list(range(len(corpus)))

    def search(self, query: str, topk: int = 20) -> list[tuple[int, float]]:
        words = set(query)
        scored = []
        for i, doc in enumerate(self.corpus):
            score = float(len(words & set(doc)))
            if score > 0:
                scored.append((self.passage_ids[i], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]


class MockFaissEngine:
    """Fake FAISS engine returning deterministic random results."""

    def __init__(self, passage_ids: list[int], dimension: int = 1024) -> None:
        self.passage_ids = passage_ids
        self.dimension = dimension

    def search(
        self, query_vectors: np.ndarray, topk: int = 20
    ) -> list[tuple[int, float]]:
        rng = np.random.default_rng(42)
        sampled = rng.choice(
            self.passage_ids,
            size=min(topk, len(self.passage_ids)),
            replace=False,
        )
        return [(int(pid), round(rng.random(), 4)) for pid in sampled]


class MockReranker:
    """Fake reranker using deterministic scoring based on article_id parity."""

    def rerank(
        self,
        query: str,
        articles: list[RetrievedArticle],
        topk: int = 5,
    ) -> list[RerankedArticle]:
        scored = []
        for art in articles:
            score = 0.9 if art.id % 2 == 0 else 0.3
            scored.append(
                RerankedArticle(
                    id=art.id,
                    title=art.title,
                    content=art.content,
                    relevance_score=score,
                    rrf_score=art.rrf_score,
                )
            )
        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored[:topk]


class MockLLMClient:
    """Fake LLM client returning preset responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.last_messages: list[dict[str, str]] | None = None
        self.last_model: str | None = None
        self.last_usage: dict[str, int] = {}

    def chat(self, **kwargs: object) -> str:
        self.last_messages = kwargs.get("messages")  # type: ignore[assignment]
        self.last_model = kwargs.get("model")  # type: ignore[assignment]
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        return resp
