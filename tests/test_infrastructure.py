from pathlib import Path

import pytest

from src.config import get_config
from src.exceptions import (
    ConfigError,
    GenerationError,
    IndexNotFoundError,
    IntentClassificationError,
    JsonParseError,
    LawAgentError,
    ModelLoadError,
)
from src.logger import SessionFilter, get_logger
from src.models import (
    AnnotatedItem,
    Article,
    Citation,
    GenerationMetrics,
    GenerationResult,
    IntentResult,
    RerankedArticle,
    RetrievalMetrics,
    RetrievedArticle,
)
from src.prompt_loader import PromptLoader


class TestExceptions:
    def test_exception_hierarchy(self) -> None:
        assert issubclass(ConfigError, LawAgentError)
        assert issubclass(ModelLoadError, LawAgentError)
        assert issubclass(IndexNotFoundError, LawAgentError)
        assert issubclass(IntentClassificationError, LawAgentError)
        assert issubclass(GenerationError, LawAgentError)
        assert issubclass(JsonParseError, LawAgentError)

    def test_exception_message(self) -> None:
        e = ConfigError("missing key")
        assert str(e) == "missing key"
        assert isinstance(e, LawAgentError)


class TestModels:
    def test_article_creation(self) -> None:
        a = Article(id=1, title="第1条", content="test content")
        assert a.id == 1
        assert a.title == "第1条"
        assert a.metadata == {}

    def test_intent_result(self) -> None:
        r = IntentResult(label="criminal_law", confidence=0.95)
        assert r.label == "criminal_law"
        assert r.is_fallback is False

        r2 = IntentResult(label="out_of_scope", confidence=0.0, is_fallback=True)
        assert r2.is_fallback is True

    def test_intent_result_confidence_clamped(self) -> None:
        with pytest.raises(ValueError):
            IntentResult(label="criminal_law", confidence=1.5)

    def test_retrieved_article(self) -> None:
        a = RetrievedArticle(
            id=322, title="第322条", content="test", rrf_score=0.05, source="both"
        )
        assert a.source == "both"

    def test_reranked_article(self) -> None:
        a = RerankedArticle(
            id=322,
            title="第322条",
            content="test",
            relevance_score=0.92,
            rrf_score=0.05,
        )
        assert a.relevance_score == 0.92

    def test_generation_result(self) -> None:
        r = GenerationResult(
            answer="根据刑法第322条...",
            citations=[Citation(article_id=322, article_title="第322条")],
        )
        assert len(r.citations) == 1
        assert r.citations[0].article_id == 322

    def test_annotated_item(self) -> None:
        item = AnnotatedItem(
            question="test?",
            answer="test answer",
            ground_truth_articles=["第322条"],
            retrieved_candidates=[{"id": 322, "title": "第322条", "rrf_score": 0.05}],
        )
        assert item.ground_truth_articles == ["第322条"]

    def test_retrieval_metrics(self) -> None:
        m = RetrievalMetrics(
            recall_at_1=0.5,
            recall_at_5=0.8,
            recall_at_10=0.9,
            mrr=0.6,
            ndcg_at_5=0.7,
            ndcg_at_10=0.75,
        )
        assert m.recall_at_5 == 0.8

    def test_generation_metrics(self) -> None:
        m = GenerationMetrics(
            correctness_score=0.9,
            completeness_score=0.85,
            avg_prompt_tokens=500,
            avg_completion_tokens=200,
        )
        assert m.correctness_score == 0.9


class TestConfig:
    def test_config_singleton(self) -> None:
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_config_defaults(self) -> None:
        config = get_config()
        assert config.embedding_model == "BAAI/bge-large-zh-v1.5"
        assert config.reranker_model == "BAAI/bge-reranker-base"
        assert config.bm25_topk == 20
        assert config.rerank_topk == 5
        assert config.intent_threshold == 0.7
        assert config.generation_max_retries == 3
        assert config.log_level == "INFO"

    def test_config_has_api_keys(self) -> None:
        config = get_config()
        assert len(config.deepseek_api_key) > 0
        assert len(config.qwen_api_key) > 0


class TestLogger:
    def test_get_logger_returns_logger(self) -> None:
        logger = get_logger("test_module")
        assert isinstance(logger, __import__("logging").Logger)

    def test_get_logger_same_instance(self) -> None:
        logger1 = get_logger("test_module2")
        logger2 = get_logger("test_module2")
        assert logger1 is logger2

    def test_session_filter(self) -> None:
        sf = SessionFilter("abc123")
        assert sf.session_id == "abc123"

    def test_session_filter_default(self) -> None:
        sf = SessionFilter()
        assert sf.session_id == "N/A"

    def test_log_file_created(self, tmp_path: Path) -> None:
        log_dir = str(tmp_path / "logs")
        logger = get_logger("test_file", log_dir=log_dir)
        logger.info("test message")
        # Should have created a log file
        log_files = list(Path(log_dir).glob("app_*.log"))
        assert len(log_files) == 1


class TestPromptLoader:
    def test_load_intent_classification(self) -> None:
        prompt = PromptLoader.load("intent_classification", query="什么是盗窃罪？")
        assert "什么是盗窃罪？" in prompt
        assert "{{query}}" not in prompt

    def test_load_answer_generation(self) -> None:
        prompt = PromptLoader.load(
            "answer_generation",
            context="第1条 为了惩罚犯罪...",
            query="什么是盗窃罪？",
        )
        assert "第1条 为了惩罚犯罪..." in prompt
        assert "什么是盗窃罪？" in prompt

    def test_load_llm_as_judge(self) -> None:
        prompt = PromptLoader.load(
            "llm_as_judge",
            question="test?",
            reference="ref answer",
            generated="gen answer",
        )
        assert "test?" in prompt
        assert "ref answer" in prompt
        assert "gen answer" in prompt

    def test_cache_reuse(self) -> None:
        p1 = PromptLoader.load("intent_classification", query="q1")
        p2 = PromptLoader.load("intent_classification", query="q2")
        assert p1 != p2  # different substitution
        # Both should use the cached template
        assert "intent_classification" in PromptLoader._cache

    def test_missing_template(self) -> None:
        with pytest.raises(FileNotFoundError):
            PromptLoader.load("nonexistent_template")

    def test_unresolved_placeholder_raises(self) -> None:
        # Use a template that still has {{unresolved}} after substitution
        # We need a temporary template file for this test
        tmp_dir = Path("prompts")
        tmp_file = tmp_dir / "_test_unresolved.txt"
        tmp_file.write_text("{{var1}} {{var2}}", encoding="utf-8")
        PromptLoader._cache["_test_unresolved"] = tmp_file.read_text(encoding="utf-8")
        with pytest.raises(ValueError, match="Unresolved placeholders"):
            PromptLoader.load("_test_unresolved", var1="resolved")
        tmp_file.unlink()
        PromptLoader._cache.pop("_test_unresolved", None)
