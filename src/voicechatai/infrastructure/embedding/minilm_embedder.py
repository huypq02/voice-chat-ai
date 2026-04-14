from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from voicechatai.domain.ports.embedder_port import EmbedderPort


@dataclass(slots=True)
class MiniLMEmbedder(EmbedderPort):
    """Local sentence-transformers embedder using all-MiniLM-L6-v2."""

    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model: Any | None = None

    def __post_init__(self) -> None:
        self._model = self.model

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers package is required for MiniLMEmbedder.") from exc

        self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        content = text.strip()
        if not content:
            return []

        model = self._ensure_model()
        vector = model.encode(content, convert_to_numpy=True, normalize_embeddings=True)
        return vector.tolist()
