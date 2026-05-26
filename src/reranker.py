from sentence_transformers import CrossEncoder

from src.exceptions import ModelLoadError
from src.models import RerankedArticle, RetrievedArticle


class Reranker:
    """Cross-Encoder based reranker using bge-reranker-base."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            self.model = CrossEncoder(model_name, device=device)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load reranker model '{model_name}'. "
                f"Ensure HF_ENDPOINT is set for mirror access. Error: {e}"
            ) from e

    def rerank(
        self,
        query: str,
        articles: list[RetrievedArticle],
        topk: int = 5,
    ) -> list[RerankedArticle]:
        """Rerank retrieved articles by relevance to the query."""
        if not articles:
            return []

        pairs = [(query, f"{a.title} {a.content}") for a in articles]
        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)

        scored_articles = [
            RerankedArticle(
                id=art.id,
                title=art.title,
                content=art.content,
                relevance_score=float(score),
                rrf_score=art.rrf_score,
            )
            for art, score in zip(articles, scores, strict=True)
        ]
        scored_articles.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_articles[:topk]
