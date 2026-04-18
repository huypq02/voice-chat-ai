from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from voicechatai.domain.ports.embedder_port import EmbedderPort


@dataclass(slots=True)
class GeminiEmbedder(EmbedderPort):
    """Google Gemini embedding adapter implementing the embedder port."""

    model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"))
    api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    _genai: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("google-generativeai package is required for GeminiEmbedder.") from exc

        genai.configure(api_key=self.api_key)
        self._genai = genai

    def embed_text(self, text: str) -> list[float]:
        content = text.strip()
        if not content:
            return []

        result = self._genai.embed_content(model=self.model, content=content)
        vector: list[float] = result.get("embedding") or []
        if not vector:
            raise RuntimeError("Gemini embedding provider returned an empty vector.")
        return vector
