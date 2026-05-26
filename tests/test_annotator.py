import json
from pathlib import Path

import numpy as np
import pytest

from src.annotator import Annotator
from src.models import Article, RetrievedArticle


class MockRetriever:
    def __init__(self) -> None:
        self.articles = {
            1: Article(id=1, title="第1条", content="为了惩罚犯罪"),
            2: Article(id=2, title="第2条", content="中华人民共和国刑法的任务"),
        }

    def retrieve(self, query: str) -> list[RetrievedArticle]:
        return [
            RetrievedArticle(
                id=1, title="第1条", content="为了惩罚犯罪",
                rrf_score=0.05, source="both",
            ),
            RetrievedArticle(
                id=2, title="第2条", content="中华人民共和国刑法的任务",
                rrf_score=0.03, source="bm25",
            ),
        ]


class MockEmbeddingEngine:
    dimension: int = 1024

    def encode(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        return np.zeros((len(texts), 1024), dtype=np.float32)


class TestAnnotator:
    @pytest.fixture
    def test_data_path(self, tmp_path: Path) -> str:
        data = [
            {"question": "什么是盗窃罪？", "answer": "盗窃罪的答案是..."},
            {"question": "正当防卫的构成要件？", "answer": "正当防卫需满足..."},
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return str(path)

    @pytest.fixture
    def annotator(self) -> Annotator:
        retriever = MockRetriever()
        embedding = MockEmbeddingEngine()
        return Annotator(retriever, embedding)

    def test_annotate_output_format(
        self, annotator: Annotator, test_data_path: str, tmp_path: Path
    ) -> None:
        output_path = str(tmp_path / "test_annotated.json")
        annotator.annotate(test_data_path, output_path, topk=10)

        with open(output_path, encoding="utf-8") as f:
            result = json.load(f)

        assert len(result) == 2
        for item in result:
            assert "question" in item
            assert "answer" in item
            assert "ground_truth_articles" in item
            assert "retrieved_candidates" in item
            assert isinstance(item["ground_truth_articles"], list)
            assert isinstance(item["retrieved_candidates"], list)

    def test_annotate_ground_truth_empty(
        self, annotator: Annotator, test_data_path: str, tmp_path: Path
    ) -> None:
        output_path = str(tmp_path / "test_annotated2.json")
        annotator.annotate(test_data_path, output_path, topk=5)

        with open(output_path, encoding="utf-8") as f:
            result = json.load(f)

        for item in result:
            assert item["ground_truth_articles"] == []

    def test_annotate_candidates_format(
        self, annotator: Annotator, test_data_path: str, tmp_path: Path
    ) -> None:
        output_path = str(tmp_path / "test_annotated3.json")
        annotator.annotate(test_data_path, output_path, topk=10)

        with open(output_path, encoding="utf-8") as f:
            result = json.load(f)

        for item in result:
            for candidate in item["retrieved_candidates"]:
                assert "id" in candidate
                assert "title" in candidate
                assert "rrf_score" in candidate
