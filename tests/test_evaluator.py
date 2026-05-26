import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.evaluator import Evaluator
from src.models import (
    AnnotatedItem,
    Citation,
    GenerationResult,
    RerankedArticle,
    RetrievedArticle,
)


class MockRetriever:
    def retrieve(self, query: str) -> list[RetrievedArticle]:
        return []


class MockReranker:
    def rerank(
        self, query: str, articles: list[RetrievedArticle], topk: int = 5
    ) -> list[RerankedArticle]:
        return []


class MockGenerator:
    def generate(
        self, query: str, articles: list[RerankedArticle], session_id: str | None = None
    ) -> GenerationResult:
        return GenerationResult(
            answer="根据刑法第1条...",
            citations=[Citation(article_id=1, article_title="第1条")],
        )


class MockEmbeddingEngine:
    dimension: int = 1024

    def encode(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        return np.zeros((len(texts), 1024), dtype=np.float32)


class MockLLMClient:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [
            json.dumps({"correctness": 0.9, "completeness": 0.85})
        ]
        self.call_count = 0

    def chat(self, **kwargs) -> str:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp


class TestEvaluatorRetrieval:
    """Test retrieval metric calculations with hand-verified data."""

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = MockGenerator()
        embedding = MockEmbeddingEngine()
        llm = MockLLMClient()
        return Evaluator(retriever, reranker, generator, embedding, llm)

    @pytest.fixture
    def annotated_path(self, tmp_path: Path) -> str:
        """Create test data with known ground truth for hand verification.

        For item 1: ground truth "第322条" appears at rank 1 (0-indexed).
            Recall@1 = 1/1 = 1.0
            Recall@5 = 1/1 = 1.0
            MRR = 1/1 = 1.0
            NDCG@5: DCG = 1/log2(2) = 1.0, IDCG = 1/log2(2) = 1.0, NDCG = 1.0

        For item 2: ground truth "第264条" appears at rank 3 (0-indexed).
            Recall@1 = 0/1 = 0.0
            Recall@5 = 1/1 = 1.0
            MRR = 1/3 = 0.333...
            NDCG@5: DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.0, NDCG = 0.5
        """
        data = [
            AnnotatedItem(
                question="偷越国境罪怎么判？",
                answer="根据第322条...",
                ground_truth_articles=["第322条"],
                retrieved_candidates=[
                    {"id": 322, "title": "第322条", "rrf_score": 0.05},
                    {"id": 232, "title": "第232条", "rrf_score": 0.03},
                    {"id": 1, "title": "第1条", "rrf_score": 0.02},
                ],
            ),
            AnnotatedItem(
                question="盗窃罪立案标准？",
                answer="根据第264条...",
                ground_truth_articles=["第264条"],
                retrieved_candidates=[
                    {"id": 1, "title": "第1条", "rrf_score": 0.05},
                    {"id": 2, "title": "第2条", "rrf_score": 0.04},
                    {"id": 264, "title": "第264条", "rrf_score": 0.03},
                ],
            ),
        ]
        path = tmp_path / "test_eval_annotated.json"
        path.write_text(
            json.dumps([item.model_dump() for item in data], ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)

    def test_evaluate_retrieval_metrics(
        self, evaluator: Evaluator, annotated_path: str
    ) -> None:
        metrics = evaluator.evaluate_retrieval(annotated_path)

        # Item 1: recall_at_1=1.0, recall_at_5=1.0
        # Item 2: recall_at_1=0.0, recall_at_5=1.0
        # Average: recall_at_1=0.5, recall_at_5=1.0
        assert metrics.recall_at_1 == pytest.approx(0.5)
        assert metrics.recall_at_5 == pytest.approx(1.0)

        # MRR: (1/1 + 1/3) / 2 = (1 + 0.3333) / 2 ≈ 0.6667
        assert metrics.mrr == pytest.approx(0.6667, abs=1e-3)

    def test_ndcg_calculation(self, evaluator: Evaluator) -> None:
        """Verify NDCG formula matches hand calculation."""
        ranked = ["第2条", "第1条", "第322条", "第264条"]
        gt_set = {"第322条", "第264条"}

        # Formula: 1/log2(i+2) where i is 0-indexed position.
        # pos 2: "第322条" -> 1/log2(4) = 0.5
        # pos 3: "第264条" -> 1/log2(5) = 0.4307
        # DCG@4 = 0.5 + 0.4307 = 0.9307
        dcg = 1.0 / math.log2(4) + 1.0 / math.log2(5)
        # Ideal: both relevant at pos 0 and 1
        # IDCG@4 = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected_ndcg = dcg / idcg

        result = Evaluator._ndcg(ranked, gt_set, 4)
        assert result == pytest.approx(expected_ndcg)

    def test_ndcg_empty_gt(self, evaluator: Evaluator) -> None:
        result = Evaluator._ndcg(["第1条"], set(), 5)
        assert result == 0.0

    def test_ndcg_k_greater_than_list(self, evaluator: Evaluator) -> None:
        result = Evaluator._ndcg(["第322条"], {"第322条"}, 10)
        assert result == 1.0  # perfect ranking

    def test_evaluate_retrieval_empty_data(self, evaluator: Evaluator, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ZeroDivisionError):
            evaluator.evaluate_retrieval(str(path))


class TestEvaluatorGeneration:
    @pytest.fixture
    def evaluator(self) -> Evaluator:
        retriever = MockRetriever()
        reranker = MockReranker()
        generator = MockGenerator()
        embedding = MockEmbeddingEngine()
        llm = MockLLMClient()
        return Evaluator(retriever, reranker, generator, embedding, llm)

    def test_judge_single(self, evaluator: Evaluator) -> None:
        correctness, completeness = evaluator._judge_single(
            "test question?", "generated answer", "reference answer"
        )
        assert 0.0 <= correctness <= 1.0
        assert 0.0 <= completeness <= 1.0

    def test_evaluate_generation(
        self, evaluator: Evaluator, tmp_path: Path
    ) -> None:
        test_path = tmp_path / "test_gen.json"
        test_data = [
            {"question": "q1?", "answer": "a1"},
            {"question": "q2?", "answer": "a2"},
        ]
        test_path.write_text(json.dumps(test_data, ensure_ascii=False), encoding="utf-8")
        output_path = str(tmp_path / "gen_metrics.json")

        metrics = evaluator.evaluate_generation(str(test_path), output_path)

        assert metrics.correctness_score > 0
        assert metrics.completeness_score > 0
        assert Path(output_path).exists()
