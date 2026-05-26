
from src.llm_client import LLMClient


class MockLLMClient:
    """Test fake LLM client that returns preset responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.last_messages: list[dict[str, str]] = []
        self.last_model: str = ""
        self.last_temperature: float = 0.0

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        response_format: dict | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.last_messages = messages
        self.last_model = model
        self.last_temperature = temperature
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp


class TestLLMClient:
    def test_llm_client_init(self) -> None:
        client = LLMClient(api_key="test-key", base_url="https://test.api/v1")
        assert client.timeout == 30

    def test_llm_client_custom_timeout(self) -> None:
        client = LLMClient(api_key="test-key", base_url="https://test.api/v1", timeout=10)
        assert client.timeout == 10
