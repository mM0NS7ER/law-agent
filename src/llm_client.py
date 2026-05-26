from typing import Any

from openai import OpenAI


class LLMClient:
    """Generic LLM client compatible with OpenAI SDK format.

    Supports DeepSeek, Qwen, and any OpenAI-compatible API.
    Inject MockLLMClient for testing.
    """

    def __init__(self, api_key: str, base_url: str, timeout: int = 30) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.timeout = timeout
        self.last_usage: dict[str, int] = {}

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call ChatCompletion API and return the response text.

        Stores token usage in self.last_usage after each call.
        Raises exceptions directly; callers handle fallback/retry.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        resp = self.client.chat.completions.create(**kwargs)
        if resp.usage:
            self.last_usage = {
                "prompt_tokens": resp.usage.prompt_tokens or 0,
                "completion_tokens": resp.usage.completion_tokens or 0,
            }
        return resp.choices[0].message.content or ""
