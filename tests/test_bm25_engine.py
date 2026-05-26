from pathlib import Path

import pytest

from src.bm25_engine import BM25Engine
from src.exceptions import IndexNotFoundError


class TestBM25Engine:
    @pytest.fixture
    def sample_corpus(self) -> list[str]:
        return [
            "第1条 为了惩罚犯罪，保护人民，根据宪法，制定本法",
            "第2条 中华人民共和国刑法的任务，是用刑罚同一切犯罪行为作斗争",
            "第322条 违反国（边）境管理法规，偷越国（边）境，情节严重的处一年以下有期徒刑",
            "第232条 故意杀人的，处死刑、无期徒刑或者十年以上有期徒刑",
            "第264条 盗窃公私财物，数额较大的，处三年以下有期徒刑",
        ]

    @pytest.fixture
    def sample_passage_ids(self) -> list[int]:
        return [1, 2, 322, 232, 264]

    @pytest.fixture
    def engine(self, sample_corpus: list[str], sample_passage_ids: list[int]) -> BM25Engine:
        return BM25Engine(sample_corpus, passage_ids=sample_passage_ids)

    def test_tokenize(self) -> None:
        tokens = BM25Engine._tokenize("故意杀人罪")
        assert len(tokens) > 0
        # jieba should segment the text into multiple tokens
        assert any("故意" in t for t in tokens) or any("杀人" in t for t in tokens)

    def test_search_returns_article_ids(
        self, engine: BM25Engine
    ) -> None:
        results = engine.search("故意杀人", topk=3)
        assert len(results) > 0
        article_ids = [r[0] for r in results]
        # Should find article 232 about intentional homicide
        assert 232 in article_ids

    def test_search_returns_empty_for_no_match(
        self, engine: BM25Engine
    ) -> None:
        results = engine.search("宇宙飞船星际旅行")
        assert len(results) == 0

    def test_search_respects_topk(
        self, engine: BM25Engine
    ) -> None:
        results = engine.search("刑法", topk=2)
        assert len(results) <= 2

    def test_save_and_load(
        self, engine: BM25Engine, tmp_path: Path
    ) -> None:
        save_path = tmp_path / "bm25_test.pkl"
        engine.save(str(save_path))
        assert save_path.exists()

        loaded = BM25Engine.load(str(save_path))
        assert loaded.corpus == engine.corpus
        assert loaded.passage_ids == engine.passage_ids

        # Verify search still works after load
        results = loaded.search("故意杀人", topk=3)
        assert len(results) > 0

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IndexNotFoundError):
            BM25Engine.load(str(tmp_path / "nonexistent.pkl"))

    def test_default_passage_ids(self, sample_corpus: list[str]) -> None:
        engine = BM25Engine(sample_corpus)
        assert engine.passage_ids == list(range(len(sample_corpus)))

    def test_tokenized_corpus_injection(
        self, sample_corpus: list[str], sample_passage_ids: list[int]
    ) -> None:
        tokenized = [BM25Engine._tokenize(doc) for doc in sample_corpus]
        engine = BM25Engine(
            sample_corpus,
            passage_ids=sample_passage_ids,
            tokenized_corpus=tokenized,
        )
        results = engine.search("盗窃", topk=3)
        assert len(results) > 0

    def test_score_is_positive_for_matches(
        self, engine: BM25Engine
    ) -> None:
        results = engine.search("故意杀人", topk=5)
        for _, score in results:
            assert score > 0
