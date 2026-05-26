import json
from pathlib import Path

from src.embedding_engine import EmbeddingEngine
from src.models import AnnotatedItem
from src.retriever import Retriever


class Annotator:
    """Semi-automated annotation tool for test dataset.

    Reuses the Retriever to generate candidate article lists for each question,
    skipping Reranker and Generator to reduce latency.
    """

    def __init__(self, retriever: Retriever, embedding_engine: EmbeddingEngine) -> None:
        self.retriever = retriever
        self.embedding_engine = embedding_engine

    def annotate(
        self,
        test_path: str,
        output_path: str,
        topk: int = 10,
    ) -> None:
        """Read test.json, retrieve candidates for each question, write annotated output."""
        with open(test_path, encoding="utf-8-sig") as f:
            test_data = json.load(f)

        results: list[AnnotatedItem] = []
        total = len(test_data)
        for i, item in enumerate(test_data):
            query = item["question"]
            retrieved = self.retriever.retrieve(query)
            candidates = [
                {"id": r.id, "title": r.title, "rrf_score": round(r.rrf_score, 4)}
                for r in retrieved[:topk]
            ]
            results.append(
                AnnotatedItem(
                    question=query,
                    answer=item.get("answer", ""),
                    ground_truth_articles=[],
                    retrieved_candidates=candidates,
                )
            )
            if (i + 1) % 10 == 0 or i + 1 == total:
                print(f"Annotated {i + 1}/{total} questions...")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.model_dump() for r in results],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Annotation complete. Output written to {output_path}")
