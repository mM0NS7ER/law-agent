"""Integration tests for the full RAG pipeline (all mocked)."""

import json

import pytest

from src.generator import Generator
from src.intent_classifier import IntentClassifier
from src.models import Article, RerankedArticle, RetrievedArticle
from src.reranker import Reranker
from src.retriever import Retriever
from tests.mocks import (
    MockBM25Engine,
    MockEmbeddingEngine,
    MockFaissEngine,
    MockLLMClient,
)


@pytest.fixture
def articles_map() -> dict[int, Article]:
    return {
        1: Article(id=1, title="第1条", content="为了惩罚犯罪，保护人民"),
        2: Article(id=2, title="第2条", content="中华人民共和国刑法的任务"),
        322: Article(id=322, title="第322条", content="违反国（边）境管理法规，偷越国（边）境"),
        232: Article(id=232, title="第232条", content="故意杀人的，处死刑"),
        264: Article(id=264, title="第264条", content="盗窃公私财物，数额较大的"),
    }


@pytest.fixture
def retriever(articles_map: dict[int, Article]) -> Retriever:
    corpus = [f"{a.title} {a.content}" for a in articles_map.values()]
    pids = list(articles_map.keys())
    return Retriever(
        bm25_engine=MockBM25Engine(corpus, pids),
        faiss_engine=MockFaissEngine(pids),
        embedding_engine=MockEmbeddingEngine(),
        articles=articles_map,
        rrf_k=60,
        rrf_topk=10,
    )


class TestFullPipeline:
    """End-to-end tests with all mocked dependencies."""

    def test_full_pipeline_criminal_law(
        self, retriever: Retriever, articles_map: dict[int, Article]
    ) -> None:
        """Test: intent → retrieve → rerank → generate (all mocked)."""
        # 1. Intent classification (criminal_law)
        mock_intent = MockLLMClient(
            [json.dumps({"label": "criminal_law", "confidence": 0.95})]
        )
        classifier = IntentClassifier(mock_intent, "qwen3", threshold=0.7)
        intent = classifier.classify("故意杀人罪怎么判？")
        assert intent.label == "criminal_law"
        assert intent.is_fallback is False

        # 2. Retrieval
        retrieved = retriever.retrieve("故意杀人罪怎么判？")
        assert len(retrieved) > 0
        for r in retrieved:
            assert isinstance(r, RetrievedArticle)
            assert r.source in ("bm25", "vector", "both")

        # 3. Reranking (real reranker needed, skip if not available)
        from src.exceptions import ModelLoadError

        try:
            reranker = Reranker("BAAI/bge-reranker-base", device="cpu")
        except ModelLoadError:
            # If reranker model unavailable, still verify pipeline up to this point
            assert len(retrieved) > 0
            return

        ranked = reranker.rerank("故意杀人罪怎么判？", retrieved, topk=3)
        assert len(ranked) <= 3
        for r in ranked:
            assert isinstance(r, RerankedArticle)

        # 4. Generation (mocked)
        mock_gen = MockLLMClient([
            json.dumps({
                "answer": "根据刑法第232条，故意杀人的，处死刑、无期徒刑或者十年以上有期徒刑。",
                "citations": [{"article_id": 232, "article_title": "第232条"}],
            })
        ])
        generator = Generator(mock_gen, "deepseek-chat")
        result = generator.generate("故意杀人罪怎么判？", ranked)
        assert result.answer != ""
        assert len(result.citations) > 0

    def test_out_of_scope_intercepted(self) -> None:
        """Test: out_of_scope question is correctly intercepted."""
        mock = MockLLMClient(
            [json.dumps({"label": "out_of_scope", "confidence": 0.9})]
        )
        classifier = IntentClassifier(mock, "qwen3", threshold=0.7)
        intent = classifier.classify("今天天气怎么样？")
        assert intent.label == "out_of_scope"

    def test_generation_failure_still_shows_articles(
        self, retriever: Retriever
    ) -> None:
        """Test: when generation fails, retrieved articles are still available."""
        # Retrieve articles first (pipeline up to retrieval)
        retrieved = retriever.retrieve("test query")
        assert len(retrieved) > 0

        # Simulate generation failure
        mock_gen = MockLLMClient(["bad json"] * 4)
        generator = Generator(mock_gen, "deepseek-chat", max_retries=3)

        from src.exceptions import GenerationError

        with pytest.raises(GenerationError):
            # Create minimal reranked articles from retrieved
            ranked = [
                RerankedArticle(
                    id=r.id, title=r.title, content=r.content,
                    relevance_score=0.5, rrf_score=r.rrf_score,
                )
                for r in retrieved[:3]
            ]
            generator.generate("test query", ranked)

        # Articles should still be available even though generation failed
        assert len(retrieved) > 0


class TestIntegrationEdgeCases:
    """Edge case integration tests."""

    def test_empty_corpus_graceful(self) -> None:
        """Test that empty corpus doesn't crash the system."""
        empty_retriever = Retriever(
            bm25_engine=MockBM25Engine([], []),
            faiss_engine=MockFaissEngine([]),
            embedding_engine=MockEmbeddingEngine(),
            articles={},
            rrf_k=60,
            rrf_topk=10,
        )
        results = empty_retriever.retrieve("test query")
        assert results == []

    def test_single_article_corpus(self) -> None:
        """Test with a single article corpus."""
        articles = {1: Article(id=1, title="第1条", content="为了惩罚犯罪")}
        retriever = Retriever(
            bm25_engine=MockBM25Engine(["第1条 为了惩罚犯罪"], [1]),
            faiss_engine=MockFaissEngine([1]),
            embedding_engine=MockEmbeddingEngine(),
            articles=articles,
            rrf_k=60,
            rrf_topk=10,
        )
        results = retriever.retrieve("test query")
        assert len(results) <= 1

    def test_api_fallback_pipeline(self, retriever: Retriever) -> None:
        """Test pipeline with API fallback (intent classifier fails)."""
        class FailingLLM:
            def chat(self, **kwargs: object) -> str:
                raise RuntimeError("API down")

        classifier = IntentClassifier(FailingLLM(), "qwen3", threshold=0.7)
        intent = classifier.classify("故意杀人罪怎么判？")
        assert intent.is_fallback is True
        assert intent.label == "criminal_law"

        # Pipeline should still proceed with retrieval
        retrieved = retriever.retrieve("故意杀人罪怎么判？")
        assert len(retrieved) > 0
