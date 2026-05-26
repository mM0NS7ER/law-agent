import json

import pytest

from src.intent_classifier import IntentClassifier


class MockLLMClient:
    """Test fake LLM client that returns preset responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    def chat(self, **kwargs) -> str:
        resp = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return resp


class TestIntentClassifier:
    @pytest.fixture
    def classifier_criminal(self) -> IntentClassifier:
        mock = MockLLMClient(
            [json.dumps({"label": "criminal_law", "confidence": 0.95})]
        )
        return IntentClassifier(mock, "qwen3", threshold=0.7)

    @pytest.fixture
    def classifier_out_of_scope(self) -> IntentClassifier:
        mock = MockLLMClient(
            [json.dumps({"label": "out_of_scope", "confidence": 0.9})]
        )
        return IntentClassifier(mock, "qwen3", threshold=0.7)

    @pytest.fixture
    def classifier_low_confidence(self) -> IntentClassifier:
        mock = MockLLMClient(
            [json.dumps({"label": "criminal_law", "confidence": 0.3})]
        )
        return IntentClassifier(mock, "qwen3", threshold=0.7)

    @pytest.fixture
    def classifier_api_error(self) -> IntentClassifier:
        class FailingMock:
            def chat(self, **kwargs) -> str:
                raise RuntimeError("API timeout")

        return IntentClassifier(FailingMock(), "qwen3", threshold=0.7)

    def test_classify_criminal_law(self, classifier_criminal: IntentClassifier) -> None:
        result = classifier_criminal.classify("故意杀人罪怎么判？")
        assert result.label == "criminal_law"
        assert result.confidence == 0.95
        assert result.is_fallback is False

    def test_classify_out_of_scope(
        self, classifier_out_of_scope: IntentClassifier
    ) -> None:
        result = classifier_out_of_scope.classify("今天天气怎么样？")
        assert result.label == "out_of_scope"
        assert result.is_fallback is False

    def test_classify_low_confidence_forces_out_of_scope(
        self, classifier_low_confidence: IntentClassifier
    ) -> None:
        result = classifier_low_confidence.classify("some question")
        assert result.label == "out_of_scope"
        assert result.confidence == 0.3

    def test_classify_api_error_fallback(
        self, classifier_api_error: IntentClassifier
    ) -> None:
        result = classifier_api_error.classify("故意杀人罪怎么判？")
        assert result.label == "criminal_law"
        assert result.confidence == 0.0
        assert result.is_fallback is True

    def test_classify_invalid_json_fallback(self) -> None:
        mock = MockLLMClient(["not valid json}"])
        classifier = IntentClassifier(mock, "qwen3")
        result = classifier.classify("test query")
        assert result.is_fallback is True
        assert result.label == "criminal_law"
