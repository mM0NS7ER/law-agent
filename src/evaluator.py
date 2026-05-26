import json
import math
import logging
from pathlib import Path

from src.embedding_engine import EmbeddingEngine
from src.generator import Generator
from src.llm_client import LLMClient
from src.models import AnnotatedItem, GenerationMetrics, JudgeResult, RetrievalMetrics
from src.prompt_loader import PromptLoader
from src.reranker import Reranker
from src.retriever import Retriever

logger = logging.getLogger(__name__)


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
        judge_max_retries: int = 3,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.embedding_engine = embedding_engine
        self.llm_judge = llm_client
        self.judge_model = judge_model
        self.judge_max_retries = judge_max_retries

    # ========== Retrieval Metrics ==========

    def evaluate_retrieval(self, annotated_path: str) -> RetrievalMetrics:
        """Compute retrieval metrics from annotated test data."""
        with open(annotated_path, encoding="utf-8-sig") as f:
            data = [AnnotatedItem(**item) for item in json.load(f)]

        recalls: dict[int, list[float]] = {1: [], 5: [], 10: []}
        rr_list: list[float] = []
        ndcg_5_list: list[float] = []

        for item in data:
            gt_set = set(item.ground_truth_articles)

            retrieved = self.retriever.retrieve(item.question)
            candidate_ids = [int(r.id) for r in retrieved]

            for k in [1, 5, 10]:
                hits = len(gt_set & set(candidate_ids[:k]))
                denominator = len(gt_set) if gt_set else 1
                recalls[k].append(hits / denominator)

            # MRR (mean reciprocal rank)
            rr = 0.0
            for rank, article_id in enumerate(candidate_ids, start=1):
                if article_id in gt_set:
                    rr = 1.0 / rank
                    break
            rr_list.append(rr)

            # NDCG using retrieved ranking vs. ideal based on ground truth
            ndcg_5_list.append(self._ndcg(candidate_ids, gt_set, 5))

        n = len(data)
        return RetrievalMetrics(
            recall_at_1=sum(recalls[1]) / n,
            recall_at_5=sum(recalls[5]) / n,
            recall_at_10=sum(recalls[10]) / n,
            mrr=sum(rr_list) / n,
            ndcg_at_5=sum(ndcg_5_list) / n,
        )

    @staticmethod
    def _ndcg(ranked: list[int], gt_set: set[int], k: int) -> float:
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
        """Run full RAG pipeline on the test set and evaluate with LLM-as-Judge.

        Saves per-sample JudgeResult in the output file alongside aggregate metrics.
        """
        with open(test_path, encoding="utf-8-sig") as f:
            test_data = json.load(f)

        correctness_scores: list[float] = []
        completeness_scores: list[float] = []
        prompt_tokens_list: list[int] = []
        completion_tokens_list: list[int] = []
        samples: list[JudgeResult] = []

        total = len(test_data)
        logger.info("Starting end-to-end evaluation on %d samples with judge model %s",
                     total, self.judge_model)

        for idx, item in enumerate(test_data, start=1):
            query = item["question"]
            reference = item.get("answer", "")

            # Full RAG pipeline
            retrieved = self.retriever.retrieve(query)
            ranked = self.reranker.rerank(query, retrieved, topk=5)
            try:
                result = self.generator.generate(query, ranked)
            except Exception as exc:
                logger.warning("[%d/%d] Generation failed for: %s — %s",
                               idx, total, query[:60], exc)
                continue

            prompt_tokens_list.append(result.prompt_tokens)
            completion_tokens_list.append(result.completion_tokens)

            correctness, completeness, error = self._judge_single(
                query, result.answer, reference
            )

            if error:
                logger.warning("[%d/%d] Judge failed: %s", idx, total, error)
            else:
                correctness_scores.append(correctness)
                completeness_scores.append(completeness)
                logger.info("[%d/%d] correctness=%.2f completeness=%.2f",
                            idx, total, correctness, completeness)

            samples.append(JudgeResult(
                question=query,
                reference=reference,
                generated=result.answer,
                correctness=correctness,
                completeness=completeness,
                error=error,
            ))

        n = max(len(correctness_scores), 1)
        metrics = GenerationMetrics(
            correctness_score=sum(correctness_scores) / n,
            completeness_score=sum(completeness_scores) / n,
            avg_prompt_tokens=int(sum(prompt_tokens_list) / n) if prompt_tokens_list else 0,
            avg_completion_tokens=int(sum(completion_tokens_list) / n) if completion_tokens_list else 0,
            sample_count=len(samples),
            samples=samples,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics.model_dump(), f, ensure_ascii=False, indent=2)

        logger.info("Evaluation complete. Results saved to %s", output_path)
        return metrics

    def _judge_single(
        self, question: str, generated: str, reference: str
    ) -> tuple[float, float, str | None]:
        """Use LLM-as-Judge to score correctness and completeness.

        Retries on JSON parse failure up to judge_max_retries.
        Returns (correctness, completeness, error_message).
        error_message is None on success.
        """
        prompt = PromptLoader.load(
            "llm_as_judge",
            question=question,
            reference=reference,
            generated=generated,
        )

        last_error: str | None = None
        for attempt in range(1, self.judge_max_retries + 1):
            try:
                raw = self.llm_judge.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.judge_model,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                data = json.loads(raw)
                correctness = float(data["correctness"])
                completeness = float(data["completeness"])
                # Clamp to valid range
                correctness = max(0.0, min(1.0, correctness))
                completeness = max(0.0, min(1.0, completeness))
                return correctness, completeness, None
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                last_error = str(e)
                logger.debug("Judge attempt %d/%d failed: %s",
                             attempt, self.judge_max_retries, last_error)
                continue
            except Exception as e:
                last_error = str(e)
                logger.debug("Judge attempt %d/%d unexpected error: %s",
                             attempt, self.judge_max_retries, last_error)
                continue

        return 0.0, 0.0, last_error