import pytest

from src.exceptions import ModelLoadError
from src.models import RerankedArticle, RetrievedArticle
from src.reranker import Reranker


@pytest.fixture
def sample_articles() -> list[RetrievedArticle]:
    return [
        RetrievedArticle(
            id=322,
            content="违反国（边）境管理法规，偷越国（边）境，情节严重的，处一年以下有期徒刑",
            rrf_score=0.0327,
            source="both",
        ),
        RetrievedArticle(
            id=232,
            content="故意杀人的，处死刑、无期徒刑或者十年以上有期徒刑",
            rrf_score=0.0163,
            source="bm25",
        ),
        RetrievedArticle(
            id=264,
            content="盗窃公私财物，数额较大的，处三年以下有期徒刑",
            rrf_score=0.0161,
            source="vector",
        ),
    ]


@pytest.fixture
def reranker() -> Reranker:
    try:
        return Reranker("BAAI/bge-reranker-base", device="cpu")
    except ModelLoadError:
        pytest.skip("Reranker model not available; set HF_ENDPOINT or check network")


class TestReranker:
    def test_rerank_empty_articles_returns_empty(self, reranker: Reranker) -> None:
        result = reranker.rerank("test query", [], topk=5)
        assert result == []

    def test_rerank_preserves_rrf_score(
        self, reranker: Reranker, sample_articles: list[RetrievedArticle]
    ) -> None:
        result = reranker.rerank("偷越国境罪怎么判？", sample_articles, topk=3)
        for r in result:
            assert isinstance(r, RerankedArticle)
            assert r.rrf_score >= 0

    def test_rerank_truncates_to_topk(
        self, reranker: Reranker, sample_articles: list[RetrievedArticle]
    ) -> None:
        result = reranker.rerank("test query", sample_articles, topk=2)
        assert len(result) <= 2

    def test_rerank_sorted_by_relevance_desc(
        self, reranker: Reranker, sample_articles: list[RetrievedArticle]
    ) -> None:
        result = reranker.rerank("偷越国境罪怎么判？", sample_articles, topk=3)
        scores = [r.relevance_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_returns_all_when_fewer_than_topk(
        self, reranker: Reranker, sample_articles: list[RetrievedArticle]
    ) -> None:
        result = reranker.rerank("test query", sample_articles, topk=10)
        assert len(result) == len(sample_articles)
