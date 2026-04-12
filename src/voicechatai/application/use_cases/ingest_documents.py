from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voicechatai.domain.ports.embedder_port import EmbedderPort
from voicechatai.domain.ports.vector_store_port import VectorStorePort


DEFAULT_FAQ_SAMPLES: list[dict[str, str]] = [
	{
		"id": "faq-001",
		"question": "How do I reset my password?",
		"answer": "Open Settings, select Security, then choose Reset Password.",
	},
	{
		"id": "faq-002",
		"question": "How can I get a refund?",
		"answer": "Go to Billing, open the transaction, and submit a refund request.",
	},
	{
		"id": "faq-003",
		"question": "How long does shipping take?",
		"answer": "Standard shipping takes 3-5 business days.",
	},
	{
		"id": "faq-004",
		"question": "How do I contact support?",
		"answer": "Use the in-app Help Center or email support@example.com.",
	},
	{
		"id": "faq-005",
		"question": "Can I change my subscription plan?",
		"answer": "Yes. Open Account > Subscription and choose Change Plan.",
	},
]


@dataclass(slots=True)
class FAQSampleIndexer:
	"""Indexes FAQ entries into vector storage for retrieval testing."""

	embedder: EmbedderPort
	vector_store: VectorStorePort
	last_index_preview: list[dict[str, str]] | None = None

	def index_from_json(self, faq_json_path: str) -> int:
		rows = self._load_faq_rows(faq_json_path)
		self.last_index_preview = []
		ids: list[str] = []
		documents: list[str] = []
		metadatas: list[dict[str, str]] = []
		embeddings: list[list[float]] = []

		for row in rows:
			faq_id = str(row.get("id") or "").strip()
			question = str(row.get("question") or "").strip()
			answer = str(row.get("answer") or "").strip()
			if not faq_id or not question or not answer:
				continue

			text = f"{question}\n{answer}"
			embedding = self.embedder.embed_text(text)
			if not embedding:
				continue

			ids.append(faq_id)
			documents.append(answer)
			metadatas.append({"question": question, "answer": answer})
			embeddings.append(embedding)
			self.last_index_preview.append({"id": faq_id, "question": question, "answer": answer})

		return self.vector_store.upsert(
			ids=ids,
			documents=documents,
			metadatas=metadatas,
			embeddings=embeddings,
		)

	@staticmethod
	def _load_faq_rows(faq_json_path: str) -> list[dict[str, str]]:
		path = Path(faq_json_path)
		if not path.exists():
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_text(json.dumps(DEFAULT_FAQ_SAMPLES, indent=2), encoding="utf-8")
			return list(DEFAULT_FAQ_SAMPLES)

		content = path.read_text(encoding="utf-8").strip()
		if not content:
			path.write_text(json.dumps(DEFAULT_FAQ_SAMPLES, indent=2), encoding="utf-8")
			return list(DEFAULT_FAQ_SAMPLES)

		data = json.loads(content)
		if not isinstance(data, list) or not data:
			path.write_text(json.dumps(DEFAULT_FAQ_SAMPLES, indent=2), encoding="utf-8")
			return list(DEFAULT_FAQ_SAMPLES)

		return [row for row in data if isinstance(row, dict)]


