from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from voicechatai.domain.ports.llm_port import LLMGenerationError, LLMPort


@dataclass(slots=True)
class GeminiLLM(LLMPort):
    """Google Gemini chat completion adapter implementing the LLM port."""

    model: str = field(default_factory=lambda: os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash"))
    api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from google import genai  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("google-genai package is required for GeminiLLM.") from exc

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for GeminiLLM.")

        self._client = genai.Client(api_key=self.api_key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        try:
            from google.genai import types  # type: ignore[import]

            response = self._client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
            return (response.text or "").strip()
        except Exception as exc:
            raise LLMGenerationError(f"LLM generation failed: {exc}") from exc
