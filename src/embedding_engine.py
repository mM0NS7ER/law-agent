import numpy as np
from sentence_transformers import SentenceTransformer

from src.exceptions import ModelLoadError


class EmbeddingEngine:
    """Load BGE-large-zh-v1.5 and provide text vectorization."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load embedding model '{model_name}'. "
                f"Ensure HF_ENDPOINT is set for mirror access. Error: {e}"
            ) from e
        get_dimension = getattr(self.model, "get_embedding_dimension", None)
        if callable(get_dimension):
            self.dimension = int(get_dimension())
        else:
            self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into normalized float32 embeddings.

        Returns:
            np.ndarray of shape (len(texts), dimension), L2-normalized.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        result: np.ndarray = embeddings.astype(np.float32)
        return result
