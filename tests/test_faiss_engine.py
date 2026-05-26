import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from src.exceptions import IndexNotFoundError
from src.faiss_engine import FaissEngine


class TestFaissEngine:
    @pytest.fixture
    def passage_ids(self) -> list[int]:
        return [1, 2, 322, 232, 264]

    @pytest.fixture
    def vectors(self) -> np.ndarray:
        """Create 5 random normalized vectors of dimension 128."""
        rng = np.random.default_rng(42)
        v = rng.normal(size=(5, 128)).astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / norms

    @pytest.fixture
    def engine(self, vectors: np.ndarray, passage_ids: list[int]) -> FaissEngine:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return FaissEngine(index, passage_ids)

    def test_dimension(self, engine: FaissEngine) -> None:
        assert engine.dimension == 128

    def test_search_self_returns_high_score(
        self, engine: FaissEngine, vectors: np.ndarray
    ) -> None:
        query = vectors[0:1]
        results = engine.search(query, topk=3)
        assert len(results) > 0
        # The first result should be the vector itself with score ≈ 1.0
        article_id, score = results[0]
        assert article_id == 1
        assert score > 0.99

    def test_search_respects_topk(
        self, engine: FaissEngine, vectors: np.ndarray
    ) -> None:
        results = engine.search(vectors[:1], topk=2)
        assert len(results) == 2

    def test_build_from_vectors(
        self, vectors: np.ndarray, passage_ids: list[int], tmp_path: Path
    ) -> None:
        index_path = str(tmp_path / "test_faiss.bin")
        engine = FaissEngine.build_from_vectors(vectors, passage_ids, index_path)
        assert Path(index_path).exists()
        assert engine.dimension == 128
        assert engine.passage_ids == passage_ids

    def test_save_and_load(
        self, vectors: np.ndarray, passage_ids: list[int], tmp_path: Path
    ) -> None:
        index_path = str(tmp_path / "test_faiss2.bin")
        passage_path = str(tmp_path / "test_passage_ids.json")

        # Build and save
        engine = FaissEngine.build_from_vectors(vectors, passage_ids, index_path)
        # Save passage_ids separately (as build_index.py does)
        with open(passage_path, "w", encoding="utf-8") as f:
            json.dump(passage_ids, f)

        # Load and verify
        loaded = FaissEngine.load(index_path, passage_path)
        assert loaded.dimension == engine.dimension
        assert loaded.passage_ids == engine.passage_ids

        # Search with loaded engine
        query = vectors[2:3]
        results = loaded.search(query, topk=1)
        assert results[0][0] == 322  # should match itself

    def test_load_nonexistent_index(self, tmp_path: Path) -> None:
        with pytest.raises(IndexNotFoundError):
            FaissEngine.load(
                str(tmp_path / "no_index.bin"),
                str(tmp_path / "no_passage.json"),
            )

    def test_load_nonexistent_passage_ids(
        self, vectors: np.ndarray, passage_ids: list[int], tmp_path: Path
    ) -> None:
        index_path = str(tmp_path / "test_faiss3.bin")
        FaissEngine.build_from_vectors(vectors, passage_ids, index_path)
        with pytest.raises(IndexNotFoundError):
            FaissEngine.load(index_path, str(tmp_path / "no_passage.json"))

    def test_search_with_fewer_results_than_topk(
        self, vectors: np.ndarray
    ) -> None:
        # Build a tiny index with only 2 vectors
        small_vectors = vectors[:2]
        small_ids = [10, 20]
        index = faiss.IndexFlatIP(small_vectors.shape[1])
        index.add(small_vectors)
        engine = FaissEngine(index, small_ids)

        # Request more than available
        results = engine.search(small_vectors[:1], topk=10)
        assert len(results) == 2
