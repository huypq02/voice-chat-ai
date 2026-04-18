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
    _genai: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("google-generativeai package is required for GeminiLLM.") from exc

        genai.configure(api_key=self.api_key)
        self._genai = genai

    def generate(self, system_prompt: str, user_message: str) -> str:
        try:
            model = self._genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
            )
            response = model.generate_content(user_message)
            return (response.text or "").strip()
        except Exception as exc:
            raise LLMGenerationError(f"LLM generation failed: {exc}") from exc
