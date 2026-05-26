from typing import Any, Literal

from pydantic import BaseModel, Field


class Article(BaseModel):
    """A legal article from the knowledge base."""

    id: int
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntentResult(BaseModel):
    """Intent classification result."""

    label: Literal["criminal_law", "out_of_scope"]
    confidence: float = Field(ge=0.0, le=1.0)
    is_fallback: bool = False


class RetrievedArticle(BaseModel):
    """Retrieval result after RRF fusion."""

    id: int
    title: str
    content: str
    rrf_score: float
    source: Literal["bm25", "vector", "both"]


class RerankedArticle(BaseModel):
    """Reranked article with relevance score."""

    id: int
    title: str
    content: str
    relevance_score: float = Field(description="Reranker output 0~1 score")
    rrf_score: float = Field(description="Original RRF score, kept for analysis")


class Citation(BaseModel):
    """Citation to a legal article."""

    article_id: int
    article_title: str


class GenerationResult(BaseModel):
    """Generated answer with citations."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AnnotatedItem(BaseModel):
    """Semi-automated annotation entry."""

    question: str
    answer: str
    ground_truth_articles: list[str] = Field(default_factory=list)
    retrieved_candidates: list[dict[str, Any]] = Field(default_factory=list)


class RetrievalMetrics(BaseModel):
    """Retrieval evaluation metrics."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float


class GenerationMetrics(BaseModel):
    """Generation evaluation metrics."""

    correctness_score: float = Field(description="LLM-as-Judge correctness score 0~1")
    completeness_score: float = Field(description="LLM-as-Judge completeness score 0~1")
    avg_prompt_tokens: int
    avg_completion_tokens: int
