
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_dir: str = "logs"
    log_level: str = "INFO"

    # Model paths / names
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Model cache
    model_cache_dir: str = "data/models"

    # Data and index paths
    corpus_path: str = "data/crimina_law_china.jsonl"
    faiss_index_path: str = "vector_store/faiss_index.bin"
    passage_ids_path: str = "vector_store/passage_ids.json"
    bm25_index_path: str = "vector_store/bm25_index.pkl"

    # Retrieval parameters
    bm25_topk: int = 20
    vector_topk: int = 20
    rrf_k: int = 60
    rrf_topk: int = 20
    rerank_topk: int = 5

    # DeepSeek API
    deepseek_api_key: str = Field(..., validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_timeout: int = 30

    # Qwen API
    qwen_api_key: str = Field(..., validation_alias="QWEN_API_KEY")
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3.6-plus"
    qwen_timeout: int = 10

    # Intent classification
    intent_threshold: float = 0.7

    # Generation
    generation_max_retries: int = 3


_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Lazy-load singleton config. Reads .env on first call."""
    global _config
    if _config is None:
        _config = AppConfig()  # type: ignore[call-arg]
        # Set model cache directory before any SentenceTransformer / CrossEncoder is loaded
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.abspath(_config.model_cache_dir)
    return _config
