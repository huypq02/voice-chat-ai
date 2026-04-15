from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from voicechatai.domain.ports.embedder_port import EmbedderPort


@dataclass(slots=True)
class OpenAIEmbedder(EmbedderPort):
	"""OpenAI embedding adapter implementing the embedder port."""

	model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
	api_key: str | None = os.getenv("OPENAI_API_KEY")
	client: Any | None = None

	def __post_init__(self) -> None:
		if self.client is not None:
			self._client = self.client
			return

		try:
			from openai import OpenAI
		except ImportError as exc:
			raise RuntimeError("openai package is required for OpenAIEmbedder.") from exc

		self._client = OpenAI(api_key=self.api_key)

	def embed_text(self, text: str) -> list[float]:
		content = text.strip()
		if not content:
			return []

		response = self._client.embeddings.create(model=self.model, input=content)
		if not response.data:
			raise RuntimeError("Embedding provider returned an empty vector.")

		return list(response.data[0].embedding)
