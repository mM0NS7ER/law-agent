from collections import defaultdict
from typing import Literal

from src.bm25_engine import BM25Engine
from src.embedding_engine import EmbeddingEngine
from src.faiss_engine import FaissEngine
from src.models import Article, RetrievedArticle


class Retriever:
    """Dual-path retrieval with BM25 + FAISS, fused via RRF."""

    def __init__(
        self,
        bm25_engine: BM25Engine,
        faiss_engine: FaissEngine,
        embedding_engine: EmbeddingEngine,
        articles: dict[int, Article],
        rrf_k: int = 60,
        rrf_topk: int = 20,
    ) -> None:
        self.bm25 = bm25_engine
        self.faiss = faiss_engine
        self.embedding = embedding_engine
        self.articles = articles
        self.rrf_k = rrf_k
        self.rrf_topk = rrf_topk

    def retrieve(self, query: str) -> list[RetrievedArticle]:
        """Execute the full retrieval pipeline."""
        bm25_results = self.bm25.search(query, topk=self.rrf_topk)
        query_vec = self.embedding.encode([query])
        faiss_results = self.faiss.search(query_vec, topk=self.rrf_topk)
        merged = self._rrf_fusion(bm25_results, faiss_results)
        return self._build_articles(merged)

    def _rrf_fusion(
        self,
        bm25_results: list[tuple[int, float]],
        faiss_results: list[tuple[int, float]],
    ) -> list[tuple[int, float, Literal["bm25", "vector", "both"]]]:
        """Fuse BM25 and FAISS results using Reciprocal Rank Fusion.

        Returns list of (article_id, rrf_score, source) sorted by rrf_score desc.
        """
        rrf_scores: dict[int, float] = defaultdict(float)
        sources: dict[int, set[str]] = defaultdict(set)

        for rank, (article_id, _score) in enumerate(bm25_results, start=1):
            rrf_scores[article_id] += 1.0 / (self.rrf_k + rank)
            sources[article_id].add("bm25")

        for rank, (article_id, _score) in enumerate(faiss_results, start=1):
            rrf_scores[article_id] += 1.0 / (self.rrf_k + rank)
            sources[article_id].add("vector")

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        result: list[tuple[int, float, Literal["bm25", "vector", "both"]]] = []
        for article_id, rrf_score in sorted_results[: self.rrf_topk]:
            source_set = sources[article_id]
            source: Literal["bm25", "vector", "both"]
            if "bm25" in source_set and "vector" in source_set:
                source = "both"
            elif "bm25" in source_set:
                source = "bm25"
            else:
                source = "vector"
            result.append((article_id, rrf_score, source))

        return result

    def _build_articles(
        self,
        merged: list[tuple[int, float, Literal["bm25", "vector", "both"]]],
    ) -> list[RetrievedArticle]:
        """Convert RRF fusion results into RetrievedArticle objects."""
        results: list[RetrievedArticle] = []
        for article_id, rrf_score, source in merged:
            article = self.articles.get(article_id)
            if article is None:
                continue
            results.append(
                RetrievedArticle(
                    id=article.id,
                    title=article.title,
                    content=article.content,
                    rrf_score=rrf_score,
                    source=source,
                )
            )
        return results
