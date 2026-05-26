import numpy as np
import pytest

from src.embedding_engine import EmbeddingEngine
from src.exceptions import ModelLoadError


class TestEmbeddingEngine:
    def test_load_nonexistent_model_raises(self) -> None:
        with pytest.raises(ModelLoadError):
            EmbeddingEngine("nonexistent/model-name-that-does-not-exist-xyz")

    @pytest.mark.slow
    def test_encode_shape_and_normalization(self) -> None:
        """Test with a real small model to verify output shape and normalization."""
        try:
            engine = EmbeddingEngine("BAAI/bge-small-zh-v1.5", device="cpu")
        except ModelLoadError:
            pytest.skip("Model not available for download; set HF_ENDPOINT or check network")

        texts = ["故意杀人罪如何量刑？", "盗窃罪的立案标准是什么？", "正当防卫的构成要件"]
        embeddings = engine.encode(texts)
        assert embeddings.shape == (3, engine.dimension)
        assert embeddings.dtype == np.float32

        # Verify L2 normalization (norm ≈ 1.0 for each vector)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    @pytest.mark.slow
    def test_encode_single_text(self) -> None:
        try:
            engine = EmbeddingEngine("BAAI/bge-small-zh-v1.5", device="cpu")
        except ModelLoadError:
            pytest.skip("Model not available for download")

        embeddings = engine.encode(["测试"])
        assert embeddings.shape == (1, engine.dimension)

    @pytest.mark.slow
    def test_encode_with_progress(self) -> None:
        try:
            engine = EmbeddingEngine("BAAI/bge-small-zh-v1.5", device="cpu")
        except ModelLoadError:
            pytest.skip("Model not available for download")

        texts = ["文本一", "文本二", "文本三"] * 3
        embeddings = engine.encode(texts, show_progress=True)
        assert embeddings.shape == (9, engine.dimension)
