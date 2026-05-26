"""Streamlit web interface for the Criminal Law Q&A Agent.

Usage: streamlit run app.py
"""

import json
import os

import streamlit as st

from src.bm25_engine import BM25Engine
from src.config import get_config
from src.embedding_engine import EmbeddingEngine
from src.faiss_engine import FaissEngine
from src.generator import Generator
from src.intent_classifier import IntentClassifier
from src.llm_client import LLMClient
from src.logger import get_logger
from src.models import Article
from src.reranker import Reranker
from src.retriever import Retriever
from src.session_manager import SessionManager

# ---------------------------------------------------------------------------
# session state keys
# ---------------------------------------------------------------------------
KEY_SESSION_ID = "current_session_id"
KEY_MESSAGES = "session_messages"

TITLE_MAX_CHARS = 30



# ---------------------------------------------------------------------------
# sidebar — session list & management
# ---------------------------------------------------------------------------

def render_sidebar(sm: SessionManager) -> None:
    st.sidebar.title("会话")

    if st.sidebar.button("＋ 新建会话", use_container_width=True):
        st.session_state[KEY_SESSION_ID] = None
        st.session_state[KEY_MESSAGES] = []
        st.rerun()

    st.sidebar.divider()

    sessions = sm.list_sessions()
    for s in sessions:
        sid = s["id"]
        cols = st.sidebar.columns([6, 1])
        label = s.get("title", sid[:8])
        if cols[0].button(label, key=f"load_{sid}", use_container_width=True):
            _load_session(sm, sid)

        if cols[1].button("🗑", key=f"del_{sid}"):
            _delete_session(sm, sid)
            st.rerun()


def _load_session(sm: SessionManager, session_id: str) -> None:
    data = sm.load_session(session_id)
    if data is None:
        st.sidebar.error("会话不存在")
        return
    st.session_state[KEY_SESSION_ID] = session_id
    st.session_state[KEY_MESSAGES] = data.get("messages", [])


def _delete_session(sm: SessionManager, session_id: str) -> None:
    sm.delete_session(session_id)
    if st.session_state.get(KEY_SESSION_ID) == session_id:
        st.session_state[KEY_SESSION_ID] = None
        st.session_state[KEY_MESSAGES] = []


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="刑法知识问答", layout="wide")
    st.title("刑法知识问答 Agent")

    config = get_config()
    get_logger("app", log_dir=config.log_dir, level=config.log_level)
    sm = SessionManager()

    # init session state
    if KEY_SESSION_ID not in st.session_state:
        st.session_state[KEY_SESSION_ID] = None
    if KEY_MESSAGES not in st.session_state:
        st.session_state[KEY_MESSAGES] = []

    @st.cache_resource
    def load_local_modules() -> tuple[EmbeddingEngine, Retriever, Reranker]:
        with open(config.corpus_path, encoding="utf-8") as f:
            raw = [json.loads(line) for line in f]

        articles_map: dict[int, Article] = {}
        for a in raw:
            articles_map[a["id"]] = Article(
                id=a["id"],
                content=a["content"],
                metadata=a.get("metadata", {}),
            )

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

    if not os.path.exists(config.bm25_index_path):
        st.error("BM25 索引未找到，请先运行 `python build_index.py` 构建索引。")
        return

    render_sidebar(sm)

    embedding_engine, retriever, reranker = load_local_modules()

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
        qwen_client, config.qwen_model, threshold=config.intent_threshold
    )
    generator = Generator(
        deepseek_client, config.deepseek_model,
        max_retries=config.generation_max_retries,
    )

    # ----- display chat history -----
    for msg in st.session_state[KEY_MESSAGES]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ----- input -----
    query = st.chat_input("请输入您的刑法相关问题...")

    if query:
        session_id = st.session_state[KEY_SESSION_ID]

        # first message in a new session → create session now
        if session_id is None:
            title = query[:TITLE_MAX_CHARS] + ("..." if len(query) > TITLE_MAX_CHARS else "")
            session_id = sm.create_session(title)
            st.session_state[KEY_SESSION_ID] = session_id

        # create run
        run_id, run_path = sm.create_run_dir(session_id)

        # Load session data
        session_data = sm.load_session(session_id)
        if session_data is None:
            st.error("会话数据丢失")
            return

        # Append user message
        session_data["messages"].append({"role": "user", "content": query})
        sm.save_session(session_data)
        st.session_state[KEY_MESSAGES] = session_data["messages"]

        # Trace: user question
        sm.append_trace(run_path, {"event": "user_query", "question": query})

        # Show user message
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("分析中..."):
                # 1. Intent classification
                intent = intent_clf.classify(query, session_id=session_id)
                sm.append_trace(run_path, {
                    "event": "intent",
                    "label": intent.label,
                    "confidence": intent.confidence,
                })

                if intent.label == "out_of_scope":
                    answer = (
                        "您好，我目前仅能回答与《中华人民共和国刑法》"
                        "及其修正案、司法解释、相关案例有关的问题。请您重新提问。"
                    )
                    st.warning(answer)
                    sm.append_trace(run_path, {"event": "out_of_scope"})
                    sm.write_report(run_path, {
                        "question": query,
                        "intent": intent.label,
                        "intent_confidence": intent.confidence,
                        "answer": answer,
                        "status": "out_of_scope",
                    })
                    session_data["messages"].append({"role": "assistant", "content": answer})
                    sm.save_session(session_data)
                    st.session_state[KEY_MESSAGES] = session_data["messages"]
                    st.rerun()

                # 2. Retrieval
                retrieved = retriever.retrieve(query)
                sm.append_trace(run_path, {
                    "event": "retrieval",
                    "count": len(retrieved),
                })

                # 3. Rerank
                ranked = reranker.rerank(query, retrieved, topk=config.rerank_topk)
                sm.append_trace(run_path, {
                    "event": "rerank",
                    "topk": len(ranked),
                    "articles": [{"id": a.id, "score": a.relevance_score} for a in ranked],
                })

            # 4. Generate with streaming (outside spinner for incremental output)
            try:
                answer = st.write_stream(generator.generate_stream(query, ranked))
            except Exception as e:
                err_msg = f"回答生成失败，请稍后重试。错误详情：{e}"
                st.error(err_msg)
                sm.append_trace(run_path, {"event": "generation_error", "error": str(e)})
                sm.write_report(run_path, {
                    "question": query,
                    "intent": intent.label,
                    "intent_confidence": intent.confidence,
                    "retrieval_count": len(retrieved),
                    "rerank_topk": len(ranked),
                    "error": str(e),
                    "status": "error",
                })
                session_data["messages"].append({"role": "assistant", "content": err_msg})
                sm.save_session(session_data)
                st.session_state[KEY_MESSAGES] = session_data["messages"]
                st.rerun()

            sm.append_trace(run_path, {"event": "generation_complete"})
            sm.write_report(run_path, {
                "question": query,
                "intent": intent.label,
                "intent_confidence": intent.confidence,
                "retrieval_count": len(retrieved),
                "rerank_topk": len(ranked),
                "rerank_articles": [{"id": a.id, "score": a.relevance_score} for a in ranked],
                "answer": answer,
                "citations": [f"第{a.id}条" for a in ranked],
                "status": "success",
            })

        # Save assistant message
        session_data = sm.load_session(session_id)
        if session_data is not None:
            session_data["messages"].append({"role": "assistant", "content": answer})
            sm.save_session(session_data)
            st.session_state[KEY_MESSAGES] = session_data["messages"]

        st.rerun()


if __name__ == "__main__":
    main()
