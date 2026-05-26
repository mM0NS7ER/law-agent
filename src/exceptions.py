class LawAgentError(Exception):
    """Root exception for all law-agent errors."""


class ConfigError(LawAgentError):
    """Configuration loading failed."""


class ModelLoadError(LawAgentError):
    """Local model (Embedding/Reranker) loading failed."""


class IndexNotFoundError(LawAgentError):
    """FAISS/BM25 index file missing and cannot be auto-rebuilt."""


class IntentClassificationError(LawAgentError):
    """Intent classification API call failed (non-fallback scenario, for log marking)."""


class GenerationError(LawAgentError):
    """Answer generation API call failed or JSON parsing ultimately failed."""


class JsonParseError(LawAgentError):
    """LLM output JSON parsing failed (retryable)."""
