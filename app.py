"""Streamlit web interface for the Criminal Law Q&A Agent.

Usage: streamlit run app.py
"""
import json
import uuid

import streamlit as st

from src.bm25_engine import BM25Engine
from src.config import get_config
from src.embedding_engine import EmbeddingEngine
from src.exceptions import GenerationError
from src.faiss_engine import FaissEngine
from src.generator import Generator
from src.intent_classifier import IntentClassifier
from src.llm_client import LLMClient
from src.logger import get_logger
from src.models import Article, GenerationResult, RerankedArticle
from src.reranker import Reranker
from src.retriever import Retriever


def _show_citations(articles: list[RerankedArticle]) -> None:
    """Display retrieved articles as expandable cards."""
    if not articles:
        return
    st.subheader("相关法条")
    for art in articles:
        with st.expander(f"{art.title} (相关度: {art.relevance_score:.3f})"):
            st.write(art.content)


def _show_citations_from_result(result: GenerationResult) -> None:
    """Display citations from the generation result."""
    if result.citations:
        titles = ", ".join(c.article_title for c in result.citations)
        st.caption(f"引用法条：{titles}")


def main() -> None:
    st.set_page_config(page_title="刑法知识问答", layout="wide")
    st.title("刑法知识问答 Agent")

    config = get_config()
    logger = get_logger("app", log_dir=config.log_dir, level=config.log_level)

    @st.cache_resource
    def load_local_modules() -> tuple[
        EmbeddingEngine, Retriever, Reranker,
    ]:
        """Load and cache local models (expensive to reload)."""
        # Load articles
        with open(config.corpus_path, encoding="utf-8") as f:
            raw = [json.loads(line) for line in f]

        articles_map: dict[int, Article] = {}
        for a in raw:
            articles_map[a["id"]] = Article(
                id=a["id"],
                title=a["title"],
                content=a["content"],
                metadata=a.get("metadata", {}),
            )

        # Load engines
        embedding = EmbeddingEngine(config.embedding_model)
        bm25 = BM25Engine.load(config.bm25_index_path)
        faiss = FaissEngine.load(config.faiss_index_path, config.passage_ids_path)
        reranker = Reranker(config.reranker_model)

        retriever = Retriever(
            bm25_engine=bm25,
            faiss_engine=faiss,
            embedding_engine=embedding,
            articles=articles_map,
            rrf_k=config.rrf_k,
            rrf_topk=config.rrf_topk,
        )
        return embedding, retriever, reranker

    # Check indices exist
    import os
    if not os.path.exists(config.bm25_index_path):
        st.error("BM25 索引未找到，请先运行 `python build_index.py` 构建索引。")
        return

    embedding_engine, retriever, reranker = load_local_modules()

    # API clients (lightweight, create per-session)
    qwen_client = LLMClient(
        api_key=config.qwen_api_key,
        base_url=config.qwen_base_url,
        timeout=config.qwen_timeout,
    )
    deepseek_client = LLMClient(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        timeout=config.deepseek_timeout,
    )

    intent_clf = IntentClassifier(
        qwen_client, config.qwen_model,
        threshold=config.intent_threshold,
    )
    generator = Generator(
        deepseek_client, config.deepseek_model,
        max_retries=config.generation_max_retries,
    )

    query = st.text_input("请输入您的刑法相关问题：", placeholder="例如：故意杀人罪如何量刑？")
    if st.button("提交") and query:
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"New query: {query}")

        with st.spinner("分析中..."):
            # 1. Intent classification
            intent = intent_clf.classify(query, session_id=session_id)
            if intent.label == "out_of_scope":
                st.warning(
                    "您好，我目前仅能回答与《中华人民共和国刑法》"
                    "及其修正案、司法解释、相关案例有关的问题。请您重新提问。"
                )
                return

            # 2. Retrieval
            retrieved = retriever.retrieve(query)
            # 3. Rerank
            ranked = reranker.rerank(query, retrieved, topk=config.rerank_topk)
            # 4. Generate
            try:
                result = generator.generate(query, ranked, session_id=session_id)
            except GenerationError as e:
                st.error(f"回答生成失败，请稍后重试。错误详情：{e}")
                _show_citations(ranked)
                return

        # Display answer
        st.markdown("### 回答")
        st.markdown(result.answer)
        _show_citations_from_result(result)
        _show_citations(ranked)


if __name__ == "__main__":
    main()
