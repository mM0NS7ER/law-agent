import re
from pathlib import Path


class PromptLoader:
    """Load prompt templates from the prompts/ directory with {{variable}} substitution."""

    _cache: dict[str, str] = {}
    _prompts_dir: Path = Path("prompts")

    @classmethod
    def load(cls, name: str, **kwargs: str) -> str:
        """Load prompts/{name}.txt and replace {{variable}} placeholders.

        Templates are cached in memory after first load.
        """
        cache_key = name
        if cache_key not in cls._cache:
            file_path = cls._prompts_dir / f"{name}.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {file_path}")
            cls._cache[cache_key] = file_path.read_text(encoding="utf-8")

        template = cls._cache[cache_key]
        for key, value in kwargs.items():
            template = template.replace("{{" + key + "}}", value)

        # Validate no unresolved placeholders remain
        unresolved = re.findall(r"\{\{(\w+)\}\}", template)
        if unresolved:
            raise ValueError(
                f"Unresolved placeholders in prompt '{name}': {unresolved}"
            )

        return template
