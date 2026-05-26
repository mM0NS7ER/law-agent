"""Build a lightweight annotation file for manual review.

Usage:
    python build_annotation_file.py --input data/test.json --output data/test_annotated.json

The generated file keeps only the fields needed for manual checking:
    question, answer, ground_truth_articles
"""

from __future__ import annotations

import argparse
import json
import os
import re
import random
from pathlib import Path

from src.bm25_engine import BM25Engine
from src.embedding_engine import EmbeddingEngine
from src.faiss_engine import FaissEngine
from src.models import Article
from src.retriever import Retriever


GROUND_TRUTH_ARTICLES_RE = re.compile(
    r'("ground_truth_articles": )\[.*?\]',
    re.DOTALL,
)


def _load_retriever(
    corpus_path: str,
    bm25_index_path: str,
    faiss_index_path: str,
    passage_ids_path: str,
    embedding_model: str,
    rrf_k: int,
    rrf_topk: int,
) -> Retriever:
    """Load local retrieval modules used to prefill annotation candidates."""
    with open(corpus_path, encoding="utf-8") as f:
        raw_articles = [json.loads(line) for line in f]

    articles_map: dict[int, Article] = {
        item["id"]: Article(
            id=item["id"],
            title=item.get("title", f"第{item['id']}条"),
            content=item["content"],
            metadata=item.get("metadata", {}),
        )
        for item in raw_articles
    }

    bm25 = BM25Engine.load(bm25_index_path)
    faiss = FaissEngine.load(faiss_index_path, passage_ids_path)
    embedding = EmbeddingEngine(embedding_model)

    return Retriever(
        bm25_engine=bm25,
        faiss_engine=faiss,
        embedding_engine=embedding,
        articles=articles_map,
        rrf_k=rrf_k,
        rrf_topk=rrf_topk,
    )


def build_annotation_file(
    input_path: str,
    output_path: str,
    retriever: Retriever,
    topk: int,
) -> None:
    """Convert a test question file into a manual annotation template."""
    with open(input_path, encoding="utf-8-sig") as f:
        test_data = json.load(f)

    annotated_data = []
    total = len(test_data)

    for index, item in enumerate(test_data, start=1):
        retrieved = retriever.retrieve(item.get("question", ""))
        max_count = min(3, topk, len(retrieved))
        min_count = min(1, max_count)
        selected_count = random.randint(min_count, max_count) if max_count > 0 else 0
        article_ids = [article.id for article in retrieved[:selected_count]]

        annotated_data.append(
            {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "ground_truth_articles": article_ids,
            }
        )

        if index % 10 == 0 or index == total:
            print(f"Annotated {index}/{total} items...")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        serialized = json.dumps(annotated_data, ensure_ascii=False, indent=2)
        serialized = GROUND_TRUTH_ARTICLES_RE.sub(
            lambda match: f'{match.group(1)}{json.dumps(json.loads(match.group(0).split(": ", 1)[1]), ensure_ascii=False)}',
            serialized,
        )
        f.write(serialized)

    print(f"Annotation template written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manual annotation file")
    parser.add_argument(
        "--input",
        default="data/test_small.json",
        help="Path to the source test file",
    )
    parser.add_argument(
        "--output",
        default="data/test_small_annotated.json",
        help="Path to the output annotation file",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="How many retrieved articles to prefill into ground_truth_articles",
    )
    parser.add_argument(
        "--corpus",
        default="data/crimina_law_china.jsonl",
        help="Path to the law corpus JSONL",
    )
    parser.add_argument(
        "--bm25-index",
        default="vector_store/bm25_index.pkl",
        help="Path to the BM25 index",
    )
    parser.add_argument(
        "--faiss-index",
        default="vector_store/faiss_index.bin",
        help="Path to the FAISS index",
    )
    parser.add_argument(
        "--passage-ids",
        default="vector_store/passage_ids.json",
        help="Path to the passage ID mapping",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),
        help="Embedding model name used by the retriever",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF smoothing constant",
    )
    parser.add_argument(
        "--rrf-topk",
        type=int,
        default=20,
        help="How many fused results to keep in the retriever",
    )
    args = parser.parse_args()

    retriever = _load_retriever(
        corpus_path=args.corpus,
        bm25_index_path=args.bm25_index,
        faiss_index_path=args.faiss_index,
        passage_ids_path=args.passage_ids,
        embedding_model=args.embedding_model,
        rrf_k=args.rrf_k,
        rrf_topk=args.rrf_topk,
    )

    build_annotation_file(args.input, args.output, retriever, args.topk)


if __name__ == "__main__":
    main()