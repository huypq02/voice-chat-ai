from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from voicechatai.domain.ports.vector_store_port import VectorSearchMatch, VectorStorePort
from voicechatai.infrastructure.vector_store._store_utils import _build_payload


@dataclass(slots=True)
class ChromaStore(VectorStorePort):
	"""ChromaDB adapter implementing the vector store port."""

	collection_name: str = os.getenv("CHROMA_COLLECTION", "faqs")
	persist_directory: str = os.getenv("CHROMA_DB_PATH", "chroma_db")
	client: Any | None = None

	def __post_init__(self) -> None:
		if self.client is None:
			try:
				import chromadb
			except ImportError as exc:
				raise RuntimeError("chromadb package is required for ChromaStore.") from exc

			self.client = chromadb.PersistentClient(path=self.persist_directory)

		self._collection = self.client.get_or_create_collection(name=self.collection_name)

	def query(self, vector: list[float], top_k: int) -> list[VectorSearchMatch]:
		if not vector or top_k <= 0:
			return []

		result = self._collection.query(
			query_embeddings=[vector],
			n_results=top_k,
			include=["metadatas", "distances", "documents"],
		)

		ids = (result.get("ids") or [[]])[0]
		metadatas = (result.get("metadatas") or [[]])[0]
		distances = (result.get("distances") or [[]])[0]
		documents = (result.get("documents") or [[]])[0]

		matches: list[VectorSearchMatch] = []
		for idx, item_id in enumerate(ids):
			metadata_raw = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
			doc_text = documents[idx] if idx < len(documents) and documents[idx] else ""
			distance = float(distances[idx]) if idx < len(distances) else 1.0
			score = max(0.0, 1.0 - distance)

			payload = _build_payload(metadata_raw, doc_text)
			matches.append(VectorSearchMatch(item_id=str(item_id), score=score, payload=payload))

		return matches

	def upsert(
		self,
		ids: list[str],
		documents: list[str],
		metadatas: list[dict[str, str]],
		embeddings: list[list[float]],
	) -> int:
		if not ids:
			return 0

		self._collection.upsert(
			ids=ids,
			documents=documents,
			metadatas=metadatas,
			embeddings=embeddings,
		)
		return len(ids)
