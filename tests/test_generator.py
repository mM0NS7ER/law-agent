import json

import pytest

from src.exceptions import GenerationError
from src.generator import Generator
from src.models import RerankedArticle


class MockLLMClient:
    """Test fake LLM client that returns preset responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.last_usage: dict[str, int] = {}

    def chat(self, **kwargs: object) -> str:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50}
        return resp


@pytest.fixture
def sample_articles() -> list[RerankedArticle]:
    return [
        RerankedArticle(
            id=322,
            content="违反国（边）境管理法规，偷越国（边）境，情节严重的，处一年以下有期徒刑",
            relevance_score=0.92,
            rrf_score=0.0327,
        ),
        RerankedArticle(
            id=232,
            content="故意杀人的，处死刑、无期徒刑或者十年以上有期徒刑",
            relevance_score=0.15,
            rrf_score=0.0163,
        ),
    ]


class TestGenerator:
    def test_build_context(self, sample_articles: list[RerankedArticle]) -> None:
        ctx = Generator._build_context(sample_articles)
        assert "【322】" in ctx
        assert "【232】" in ctx
        assert "\n\n" in ctx

    def test_generate_success(self, sample_articles: list[RerankedArticle]) -> None:
        response = json.dumps({
            "answer": "根据刑法第322条，偷越国境情节严重的，处一年以下有期徒刑。",
            "citations": [{"article_id": 322, "article_title": "第322条"}],
        })
        mock = MockLLMClient([response])
        generator = Generator(mock, "deepseek-chat")
        result = generator.generate("偷越国境罪怎么判？", sample_articles)
        assert result.answer.startswith("根据刑法第322条")
        assert len(result.citations) == 1
        assert result.citations[0].article_id == 322

    def test_generate_with_empty_citations(
        self, sample_articles: list[RerankedArticle]
    ) -> None:
        response = json.dumps({
            "answer": "现有知识库未涵盖足够信息。",
            "citations": [],
        })
        mock = MockLLMClient([response])
        generator = Generator(mock, "deepseek-chat")
        result = generator.generate("test query?", sample_articles)
        assert len(result.citations) == 0

    def test_generate_json_parse_retry(
        self, sample_articles: list[RerankedArticle]
    ) -> None:
        # First response is bad JSON, second is valid JSON
        responses = [
            "not valid json",
            json.dumps({
                "answer": "根据相关法条...",
                "citations": [{"article_id": 1, "article_title": "第1条"}],
            }),
        ]
        mock = MockLLMClient(responses)
        generator = Generator(mock, "deepseek-chat", max_retries=3)
        result = generator.generate("test query?", sample_articles)
        assert result.answer == "根据相关法条..."
        assert mock.call_count == 2

    def test_generate_all_retries_exhausted(
        self, sample_articles: list[RerankedArticle]
    ) -> None:
        mock = MockLLMClient(["bad json"] * 4)  # 3 retries + 1 extra
        generator = Generator(mock, "deepseek-chat", max_retries=3)
        with pytest.raises(GenerationError, match="Failed to parse"):
            generator.generate("test query?", sample_articles)
        assert mock.call_count == 3

    def test_parse_response(self) -> None:
        raw = json.dumps({
            "answer": "test answer",
            "citations": [{"article_id": 1, "article_title": "第1条"}],
        })
        result = Generator._parse_response(raw)
        assert result.answer == "test answer"
        assert result.citations[0].article_id == 1

    def test_parse_response_missing_answer(self) -> None:
        raw = json.dumps({"citations": []})
        result = Generator._parse_response(raw)
        assert result.answer == ""
