import json
import tempfile
import unittest
from pathlib import Path

from voicechatai.application.use_cases.ingest_documents import FAQSampleIndexer


class StubEmbedder:
    def embed_text(self, text: str) -> list[float]:
        return [float(len(text))]


class StubVectorStore:
    def __init__(self) -> None:
        self.last_upsert: dict | None = None

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str]],
        embeddings: list[list[float]],
    ) -> int:
        self.last_upsert = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
        }
        return len(ids)


class FAQSampleIndexerTests(unittest.TestCase):
    def test_index_from_empty_json_uses_default_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            faq_path = Path(tmpdir) / "faqs.json"
            faq_path.write_text("[]", encoding="utf-8")

            store = StubVectorStore()
            indexer = FAQSampleIndexer(embedder=StubEmbedder(), vector_store=store)

            count = indexer.index_from_json(str(faq_path))

            self.assertGreaterEqual(count, 5)
            self.assertIsNotNone(store.last_upsert)
            self.assertGreaterEqual(len(store.last_upsert["ids"]), 5)

    def test_index_from_json_ignores_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            faq_path = Path(tmpdir) / "faqs.json"
            faq_path.write_text(
                json.dumps(
                    [
                        {"id": "ok-1", "question": "q1", "answer": "a1"},
                        {"id": "bad-1", "question": "", "answer": "a2"},
                        {"id": "bad-2", "answer": "a3"},
                    ]
                ),
                encoding="utf-8",
            )

            store = StubVectorStore()
            indexer = FAQSampleIndexer(embedder=StubEmbedder(), vector_store=store)

            count = indexer.index_from_json(str(faq_path))

            self.assertEqual(count, 1)
            self.assertEqual(store.last_upsert["ids"], ["ok-1"])


if __name__ == "__main__":
    unittest.main()
