from __future__ import annotations

from dataclasses import dataclass

from voicechatai.domain.entities.faq_entry import FAQEntry
from voicechatai.domain.ports.embedder_port import EmbedderPort
from voicechatai.domain.ports.vector_store_port import VectorStorePort


@dataclass(slots=True)
class RetrievalService:
	"""Fetches FAQ candidates from vector search for a text query."""

	embedder: EmbedderPort
	vector_store: VectorStorePort
	top_k: int = 5

	def retrieve_faq_candidates(self, query_text: str) -> list[FAQEntry]:
		text = query_text.strip()
		if not text:
			return []

		vector = self.embedder.embed_text(text)
		matches = self.vector_store.query(vector=vector, top_k=self.top_k)

		candidates: list[FAQEntry] = []
		for match in matches:
			payload = match.payload
			candidates.append(
				FAQEntry(
					faq_id=match.item_id,
					question=payload.get("question", ""),
					answer=payload.get("answer", ""),
					score=match.score,
					metadata={k: v for k, v in payload.items() if k not in {"question", "answer"}},
				)
			)

		return candidates


