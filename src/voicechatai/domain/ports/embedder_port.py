from __future__ import annotations

from abc import ABC, abstractmethod


class EmbedderPort(ABC):
	"""Port for text-to-vector embedding providers."""

	@abstractmethod
	def embed_text(self, text: str) -> list[float]:
		raise NotImplementedError


