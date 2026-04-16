from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class VectorSearchMatch:
	item_id: str
	score: float
	payload: dict[str, str] = field(default_factory=dict)


class VectorStorePort(ABC):
	"""Port for vector-search storage engines (e.g., Chroma)."""

	@abstractmethod
	def query(self, vector: list[float], top_k: int) -> list[VectorSearchMatch]:
		raise NotImplementedError

	@abstractmethod
	def upsert(
		self,
		ids: list[str],
		documents: list[str],
		metadatas: list[dict[str, str]],
		embeddings: list[list[float]],
	) -> int:
		"""Insert or update vectorized records and return number of indexed rows."""
		raise NotImplementedError


