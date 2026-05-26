import json
import math
from pathlib import Path

from src.embedding_engine import EmbeddingEngine
from src.generator import Generator
from src.llm_client import LLMClient
from src.models import AnnotatedItem, GenerationMetrics, RetrievalMetrics
from src.prompt_loader import PromptLoader
from src.reranker import Reranker
from src.retriever import Retriever


class Evaluator:
    """Evaluation module for retrieval and generation quality."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        embedding_engine: EmbeddingEngine,
        llm_client: LLMClient,
        judge_model: str = "deepseek-chat",
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.embedding_engine = embedding_engine
        self.llm_judge = llm_client
        self.judge_model = judge_model

    # ========== Retrieval Metrics ==========

    def evaluate_retrieval(self, annotated_path: str) -> RetrievalMetrics:
        """Compute retrieval metrics from annotated test data."""
        with open(annotated_path, encoding="utf-8-sig") as f:
            data = [AnnotatedItem(**item) for item in json.load(f)]

        recalls: dict[int, list[float]] = {1: [], 5: [], 10: []}
        rr_list: list[float] = []
        ndcg_5_list: list[float] = []
        ndcg_10_list: list[float] = []

        for item in data:
            gt_set = set(item.ground_truth_articles)
            candidates = item.retrieved_candidates
            titles = [c["title"] for c in candidates]

            for k in [1, 5, 10]:
                hits = len(gt_set & set(titles[:k]))
                denominator = len(gt_set) if gt_set else 1
                recalls[k].append(hits / denominator)

            # MRR
            rr = 0.0
            for rank, title in enumerate(titles, start=1):
                if title in gt_set:
                    rr = 1.0 / rank
                    break
            rr_list.append(rr)

            # NDCG
            ndcg_5_list.append(self._ndcg(titles, gt_set, 5))
            ndcg_10_list.append(self._ndcg(titles, gt_set, 10))

        n = len(data)
        return RetrievalMetrics(
            recall_at_1=sum(recalls[1]) / n,
            recall_at_5=sum(recalls[5]) / n,
            recall_at_10=sum(recalls[10]) / n,
            mrr=sum(rr_list) / n,
            ndcg_at_5=sum(ndcg_5_list) / n,
            ndcg_at_10=sum(ndcg_10_list) / n,
        )

    @staticmethod
    def _ndcg(ranked: list[str], gt_set: set[str], k: int) -> float:
        """Compute NDCG@k."""
        dcg = sum(
            1.0 / math.log2(i + 2) for i, t in enumerate(ranked[:k]) if t in gt_set
        )
        ideal_hits = min(len(gt_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0

    # ========== Generation Metrics (LLM-as-Judge) ==========

    def evaluate_generation(
        self, test_path: str, output_path: str
    ) -> GenerationMetrics:
        """Run full RAG pipeline on the test set and evaluate with LLM-as-Judge."""
        with open(test_path, encoding="utf-8-sig") as f:
            test_data = json.load(f)

        correctness_scores: list[float] = []
        completeness_scores: list[float] = []
        prompt_tokens_list: list[int] = []
        completion_tokens_list: list[int] = []

        for item in test_data:
            query = item["question"]
            reference = item.get("answer", "")

            # Full RAG pipeline
            retrieved = self.retriever.retrieve(query)
            ranked = self.reranker.rerank(query, retrieved, topk=5)
            try:
                result = self.generator.generate(query, ranked)
            except Exception:
                continue

            prompt_tokens_list.append(result.prompt_tokens)
            completion_tokens_list.append(result.completion_tokens)

            correctness, completeness = self._judge_single(
                query, result.answer, reference
            )
            correctness_scores.append(correctness)
            completeness_scores.append(completeness)

        n = max(len(correctness_scores), 1)
        metrics = GenerationMetrics(
            correctness_score=sum(correctness_scores) / n,
            completeness_score=sum(completeness_scores) / n,
            avg_prompt_tokens=int(sum(prompt_tokens_list) / n) if prompt_tokens_list else 0,
            avg_completion_tokens=int(sum(completion_tokens_list) / n) if completion_tokens_list else 0,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics.model_dump(), f, ensure_ascii=False, indent=2)

        return metrics

    def _judge_single(
        self, question: str, generated: str, reference: str
    ) -> tuple[float, float]:
        """Use LLM-as-Judge to score correctness and completeness."""
        prompt = PromptLoader.load(
            "llm_as_judge",
            question=question,
            reference=reference,
            generated=generated,
        )
        raw = self.llm_judge.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.judge_model,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(raw)
        return float(data["correctness"]), float(data["completeness"])
