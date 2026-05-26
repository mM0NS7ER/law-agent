import json

from src.llm_client import LLMClient
from src.logger import get_logger
from src.models import IntentResult
from src.prompt_loader import PromptLoader


class IntentClassifier:
    """Classify user queries as criminal_law or out_of_scope via Qwen API."""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        prompt_template_name: str = "intent_classification",
        threshold: float = 0.7,
    ) -> None:
        self.llm = llm_client
        self.model = model
        self.prompt_template = prompt_template_name
        self.threshold = threshold

    def classify(self, query: str, session_id: str | None = None) -> IntentResult:
        """Classify the query and return an IntentResult.

        On API failure, falls back to criminal_law (better to allow
        a non-law question than to block a real law question).
        """
        prompt = PromptLoader.load(self.prompt_template, query=query)

        try:
            raw = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            data = json.loads(raw)
            label = data.get("label", "out_of_scope")
            confidence = float(data.get("confidence", 0.0))

            if confidence < self.threshold:
                label = "out_of_scope"

            return IntentResult(label=label, confidence=confidence, is_fallback=False)

        except Exception:
            logger = get_logger("intent_classifier")
            logger.warning(
                "Intent classification API failed, falling back to criminal_law",
                extra={"session_id": session_id or "N/A"},
            )
            return IntentResult(
                label="criminal_law",
                confidence=0.0,
                is_fallback=True,
            )
