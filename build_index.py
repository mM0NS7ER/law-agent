"""Build FAISS and BM25 indices for the law knowledge base.

Usage: python build_index.py
"""
import json

from src.bm25_engine import BM25Engine
from src.config import get_config
from src.embedding_engine import EmbeddingEngine
from src.faiss_engine import FaissEngine
from src.logger import get_logger


def main() -> None:
    config = get_config()
    logger = get_logger("build_index", log_dir=config.log_dir, level=config.log_level)
    logger.info("Starting index build...")

    # 1. Load corpus
    articles = []
    corpus = []
    with open(config.corpus_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            articles.append(item)
            # The source corpus no longer contains a `title` field — use `content` only.
            corpus.append(item.get("content", ""))

    passage_ids = [a["id"] for a in articles]
    logger.info(f"Loaded {len(articles)} articles from {config.corpus_path}")

    # 2. Build BM25 index
    logger.info("Building BM25 index...")
    bm25 = BM25Engine(corpus, passage_ids=passage_ids)
    bm25.save(config.bm25_index_path)
    logger.info(f"BM25 index saved to {config.bm25_index_path}")

    # 3. Build FAISS index
    logger.info(f"Loading embedding model: {config.embedding_model}")
    embedding = EmbeddingEngine(config.embedding_model)
    logger.info(f"Encoding {len(corpus)} passages (dim={embedding.dimension})...")
    vectors = embedding.encode(corpus, show_progress=True)
    logger.info(f"Encoded vectors shape: {vectors.shape}")

    FaissEngine.build_from_vectors(vectors, passage_ids, config.faiss_index_path)
    logger.info(f"FAISS index saved to {config.faiss_index_path}")

    # Save passage_ids mapping
    with open(config.passage_ids_path, "w", encoding="utf-8") as f:
        json.dump(passage_ids, f)
    logger.info(f"Passage IDs saved to {config.passage_ids_path}")

    print("\n索引构建完成！")


if __name__ == "__main__":
    main()
