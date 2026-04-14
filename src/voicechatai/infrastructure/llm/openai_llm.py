from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from voicechatai.domain.ports.llm_port import LLMGenerationError, LLMPort


@dataclass(slots=True)
class OpenAILLM(LLMPort):
    """OpenAI chat completion adapter implementing the LLM port."""

    model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    api_key: str | None = os.getenv("OPENAI_API_KEY")
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is not None:
            self._client = self.client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for OpenAILLM.") from exc

        self._client = OpenAI(api_key=self.api_key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise LLMGenerationError(f"LLM generation failed: {exc}") from exc
