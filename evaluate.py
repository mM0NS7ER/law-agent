"""Evaluation CLI for retrieval and end-to-end generation metrics.

Usage:
    python evaluate.py --mode retrieval
    python evaluate.py --mode end2end
"""
import argparse
import json

from src.bm25_engine import BM25Engine
from src.config import AppConfig, get_config
from src.embedding_engine import EmbeddingEngine
from src.evaluator import Evaluator
from src.faiss_engine import FaissEngine
from src.generator import Generator
from src.llm_client import LLMClient
from src.logger import get_logger
from src.models import Article
from src.reranker import Reranker
from src.retriever import Retriever


def _load_modules(
    config: AppConfig,
) -> tuple[Retriever, Reranker, Generator, EmbeddingEngine, LLMClient]:
    """Load all modules needed for evaluation."""
    # Load articles
    with open(config.corpus_path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]

    articles_map = {
        a["id"]: Article(
            id=a["id"],
            title=a["title"],
            content=a["content"],
            metadata=a.get("metadata", {}),
        )
        for a in raw
    }

    bm25 = BM25Engine.load(config.bm25_index_path)
    faiss = FaissEngine.load(config.faiss_index_path, config.passage_ids_path)
    embedding = EmbeddingEngine(config.embedding_model)

    retriever = Retriever(
        bm25, faiss, embedding,
        articles=articles_map,
        rrf_k=config.rrf_k,
        rrf_topk=config.rrf_topk,
    )

    reranker = Reranker(config.reranker_model)

    deepseek_client = LLMClient(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        timeout=config.deepseek_timeout,
    )
    generator = Generator(
        deepseek_client, config.deepseek_model,
        max_retries=config.generation_max_retries,
    )

    return retriever, reranker, generator, embedding, deepseek_client


def main() -> None:
    config = get_config()
    logger = get_logger("evaluate", log_dir=config.log_dir, level=config.log_level)

    parser = argparse.ArgumentParser(description="Law Agent Evaluation CLI")
    parser.add_argument(
        "--mode", choices=["retrieval", "end2end"], required=True,
        help="Evaluation mode",
    )
    parser.add_argument(
        "--annotated", default="data/test_annotated.json",
        help="Path to annotated test data (for retrieval mode)",
    )
    parser.add_argument(
        "--test", default="data/test.json",
        help="Path to test questions (for end2end mode)",
    )
    parser.add_argument(
        "--output", default="data/eval_result.json",
        help="Path to output evaluation results",
    )
    args = parser.parse_args()

    logger.info("Loading modules for %s evaluation...", args.mode)
    retriever, reranker, generator, embedding_engine, llm_client = _load_modules(config)

    evaluator = Evaluator(retriever, reranker, generator, embedding_engine, llm_client, judge_model=config.deepseek_model)

    if args.mode == "retrieval":
        logger.info("Running retrieval evaluation on %s", args.annotated)
        retrieval_metrics = evaluator.evaluate_retrieval(args.annotated)
        print(retrieval_metrics.model_dump_json(indent=2))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(retrieval_metrics.model_dump(), f, ensure_ascii=False, indent=2)
    elif args.mode == "end2end":
        logger.info("Running end-to-end evaluation")
        gen_metrics = evaluator.evaluate_generation(args.test, args.output)
        print(gen_metrics.model_dump_json(indent=2))

    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
