
import numpy as np
import pytest

from src.models import Article, RetrievedArticle
from src.retriever import Retriever


class MockBM25Engine:
    def __init__(self, corpus: list[str], passage_ids: list[int] | None = None) -> None:
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


class MockEmbeddingEngine:
    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        rng = np.random.default_rng(hash(str(texts)) % (2**31))
        arr = rng.normal(size=(len(texts), self.dimension)).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms


class MockFaissEngine:
    def __init__(self, passage_ids: list[int], dimension: int = 1024) -> None:
        self.passage_ids = passage_ids
        self.dimension = dimension

    def search(
        self, query_vectors: np.ndarray, topk: int = 20
    ) -> list[tuple[int, float]]:
        rng = np.random.default_rng(42)
        sampled = rng.choice(self.passage_ids, size=min(topk, len(self.passage_ids)), replace=False)
        return [(int(pid), round(rng.random(), 4)) for pid in sampled]


class TestRetriever:
    @pytest.fixture
    def articles(self) -> dict[int, Article]:
        return {
            1: Article(id=1, content="为了惩罚犯罪，保护人民"),
            2: Article(id=2, content="中华人民共和国刑法的任务"),
            322: Article(id=322, content="违反国（边）境管理法规，偷越国（边）境"),
            232: Article(id=232, content="故意杀人的，处死刑"),
            264: Article(id=264, content="盗窃公私财物，数额较大的"),
        }

    @pytest.fixture
    def retriever(self, articles: dict[int, Article]) -> Retriever:
        corpus = [a.content for a in articles.values()]
        pids = list(articles.keys())
        bm25 = MockBM25Engine(corpus, pids)
        faiss = MockFaissEngine(pids)
        embedding = MockEmbeddingEngine()
        return Retriever(
            bm25_engine=bm25,
            faiss_engine=faiss,
            embedding_engine=embedding,
            articles=articles,
            rrf_k=60,
            rrf_topk=10,
        )

    def test_retrieve_returns_articles(self, retriever: Retriever) -> None:
        results = retriever.retrieve("故意杀人")
        assert len(results) > 0
        for r in results:
            assert isinstance(r, RetrievedArticle)
            assert r.id in retriever.articles

    def test_retrieved_article_has_source(self, retriever: Retriever) -> None:
        results = retriever.retrieve("test query")
        for r in results:
            assert r.source in ("bm25", "vector", "both")

    def test_rrf_fusion_formula_correct(self, retriever: Retriever) -> None:
        """Verify RRF formula: score = 1/(k+rank) summed across both lists."""
        bm25 = [(100, 10.0), (200, 5.0)]
        faiss = [(200, 8.0), (300, 6.0)]
        merged = retriever._rrf_fusion(bm25, faiss)

        # article 100: rank 1 in bm25 -> 1/(60+1) = 1/61 ≈ 0.01639
        # article 200: rank 2 in bm25 + rank 1 in faiss -> 1/(60+2) + 1/(60+1)
        # article 300: rank 2 in faiss -> 1/(60+2) = 1/62 ≈ 0.01613

        scores_by_id = {aid: score for aid, score, _src in merged}
        assert scores_by_id[100] == pytest.approx(1.0 / 61)
        assert scores_by_id[200] == pytest.approx(1.0 / 61 + 1.0 / 62)
        assert scores_by_id[300] == pytest.approx(1.0 / 62)

    def test_rrf_fusion_source_tracking(self, retriever: Retriever) -> None:
        bm25 = [(100, 10.0)]
        faiss = [(100, 8.0), (200, 6.0)]
        merged = retriever._rrf_fusion(bm25, faiss)

        sources_by_id = {aid: src for aid, _score, src in merged}
        assert sources_by_id[100] == "both"
        assert sources_by_id[200] == "vector"

    def test_rrf_topk_truncation(self, retriever: Retriever) -> None:
        """RRF results should not exceed rrf_topk."""
        many_bm25 = [(i, 1.0) for i in range(20)]
        many_faiss = [(i + 10, 1.0) for i in range(20)]
        merged = retriever._rrf_fusion(many_bm25, many_faiss)
        assert len(merged) <= retriever.rrf_topk

    def test_rrf_fusion_ordering(self, retriever: Retriever) -> None:
        """RRF results should be sorted by score descending."""
        bm25 = [(1, 10.0), (2, 5.0)]
        faiss = [(3, 8.0)]
        merged = retriever._rrf_fusion(bm25, faiss)
        scores = [s for _aid, s, _src in merged]
        assert scores == sorted(scores, reverse=True)

    def test_build_articles_missing_article_skipped(
        self, retriever: Retriever
    ) -> None:
        merged = [(999, 0.05, "bm25")]  # 999 not in articles
        results = retriever._build_articles(merged)
        assert len(results) == 0
