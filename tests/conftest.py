"""Shared pytest fixtures for all test modules."""

import json

import pytest

from src.models import Article


@pytest.fixture
def sample_articles() -> list[Article]:
    """Return a small set of test legal articles."""
    return [
        Article(id=1, content="为了惩罚犯罪，保护人民"),
        Article(id=2, content="中华人民共和国刑法的任务"),
        Article(
            id=322,
            content="违反国（边）境管理法规，偷越国（边）境，情节严重的，处一年以下有期徒刑",
        ),
        Article(id=232, content="故意杀人的，处死刑、无期徒刑或者十年以上有期徒刑"),
        Article(id=264, content="盗窃公私财物，数额较大的，处三年以下有期徒刑"),
    ]


@pytest.fixture
def sample_articles_dict(sample_articles: list[Article]) -> dict[int, Article]:
    """Return articles as a dict keyed by id."""
    return {a.id: a for a in sample_articles}


@pytest.fixture
def mock_config():
    """Return an in-memory AppConfig with fake API keys."""
    from src.config import AppConfig

    return AppConfig(
        deepseek_api_key="fake-deepseek-key",
        qwen_api_key="fake-qwen-key",
        embedding_model="BAAI/bge-small-zh-v1.5",
        log_dir="logs",
        log_level="INFO",
    )


@pytest.fixture
def mock_llm_client():
    """Return a MockLLMClient preset for criminal_law classification."""
    from tests.mocks import MockLLMClient

    return MockLLMClient(
        responses=[json.dumps({"label": "criminal_law", "confidence": 0.95})]
    )
