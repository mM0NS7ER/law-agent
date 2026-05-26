import json

from pydantic import ValidationError

from src.exceptions import GenerationError, JsonParseError
from src.llm_client import LLMClient
from src.models import Citation, GenerationResult, RerankedArticle
from src.prompt_loader import PromptLoader


class Generator:
    """Generate answers using DeepSeek API with structured JSON output."""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        prompt_template_name: str = "answer_generation",
        max_retries: int = 3,
    ) -> None:
        self.llm = llm_client
        self.model = model
        self.prompt_template = prompt_template_name
        self.max_retries = max_retries

    def generate(
        self,
        query: str,
        articles: list[RerankedArticle],
        session_id: str | None = None,
    ) -> GenerationResult:
        """Generate an answer based on retrieved and reranked articles.

        Retries up to max_retries on JSON parse failure.
        Raises GenerationError if all retries are exhausted.
        """
        context = self._build_context(articles)
        prompt = PromptLoader.load(self.prompt_template, context=context, query=query)

        last_error: Exception | None = None
        for _attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                result = self._parse_response(raw)
                result.prompt_tokens = self.llm.last_usage.get("prompt_tokens", 0)
                result.completion_tokens = self.llm.last_usage.get("completion_tokens", 0)
                return result
            except (json.JSONDecodeError, ValidationError, KeyError, JsonParseError) as e:
                last_error = e
                continue

        raise GenerationError(
            f"Failed to parse generation response after {self.max_retries} retries: {last_error}"
        )

    @staticmethod
    def _build_context(articles: list[RerankedArticle]) -> str:
        parts = [f"【{a.id}】{a.content}" for a in articles]
        return "\n\n".join(parts)

    @staticmethod
    def _parse_response(raw: str) -> GenerationResult:
        data = json.loads(raw)
        answer = data.get("answer", "")
        citations_raw = data.get("citations", [])
        if not isinstance(citations_raw, list):
            raise JsonParseError("citations field is not a list")
        citations = [Citation(**c) for c in citations_raw]
        return GenerationResult(answer=answer, citations=citations)
