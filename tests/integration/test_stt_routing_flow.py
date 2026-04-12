import base64
import unittest

from fastapi.testclient import TestClient

from voicechatai.application.services.rag_service import RAGDecision
from voicechatai.domain.entities.faq_entry import FAQEntry
from voicechatai.interfaces.api.app import create_app


class StubProcessVoiceQuery:
    def execute(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            raise RuntimeError("no audio")
        return "how do i get a refund"


class StubRAGService:
    def evaluate_faq_decision(self, transcript: str) -> RAGDecision:
        return RAGDecision(
            decision="hit",
            top_candidate=FAQEntry(
                faq_id="faq-002",
                question="How can I get a refund?",
                answer="Go to Billing, open the transaction, and submit a refund request.",
                score=0.91,
            ),
            top1_score=0.91,
            top2_score=0.72,
            margin=0.19,
            effective_hit_threshold=0.82,
        )


class StubRAGServiceClarify:
    def evaluate_faq_decision(self, transcript: str) -> RAGDecision:
        return RAGDecision(
            decision="clarify",
            clarification_candidates=[
                FAQEntry(
                    faq_id="faq-010",
                    question="How can I request a refund?",
                    answer="refund answer",
                    score=0.75,
                ),
                FAQEntry(
                    faq_id="faq-011",
                    question="How can I track a refund status?",
                    answer="status answer",
                    score=0.72,
                ),
            ],
            top1_score=0.75,
            top2_score=0.72,
            margin=0.03,
            effective_hit_threshold=0.82,
        )


class StubRAGServiceFallback:
    def evaluate_faq_decision(self, transcript: str) -> RAGDecision:
        return RAGDecision(
            decision="fallback",
            top1_score=0.41,
            top2_score=0.0,
            margin=0.41,
            effective_hit_threshold=0.82,
        )


class StubFAQIndexer:
    def __init__(self) -> None:
        self.path_received: str | None = None
        self.last_index_preview = [
            {
                "id": "faq-001",
                "question": "Why is the sky blue?",
                "answer": "Blue light spreads out the most in the air.",
            }
        ]

    def index_from_json(self, faq_json_path: str) -> int:
        self.path_received = faq_json_path
        return 5


class STTRoutingIntegrationTests(unittest.TestCase):
    def test_transcribe_and_route_returns_transcript_and_faq_decision(self) -> None:
        app = create_app()
        app.state.process_voice_query = StubProcessVoiceQuery()
        app.state.stt_boot_error = None
        app.state.rag_service = StubRAGService()
        app.state.rag_boot_error = None

        client = TestClient(app)
        payload = {"audio_bytes_b64": base64.b64encode(b"audio-bytes").decode("utf-8")}

        response = client.post("/transcribe-and-route", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["statuscode"], 200)
        self.assertEqual(data["transcript"], "how do i get a refund")
        self.assertEqual(data["faq_decision"]["decision"], "hit")
        self.assertIn("refund", data["faq_decision"]["answer"].lower())

    def test_transcribe_and_route_returns_clarify_with_candidate_questions(self) -> None:
        app = create_app()
        app.state.process_voice_query = StubProcessVoiceQuery()
        app.state.stt_boot_error = None
        app.state.rag_service = StubRAGServiceClarify()
        app.state.rag_boot_error = None

        client = TestClient(app)
        payload = {"audio_bytes_b64": base64.b64encode(b"audio-bytes").decode("utf-8")}

        response = client.post("/transcribe-and-route", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["statuscode"], 200)
        self.assertEqual(data["faq_decision"]["decision"], "clarify")
        self.assertGreaterEqual(len(data["faq_decision"]["clarification_questions"]), 2)
        self.assertIn("refund", data["faq_decision"]["clarification_questions"][0].lower())

    def test_transcribe_and_route_returns_fallback_when_no_faq_hit(self) -> None:
        app = create_app()
        app.state.process_voice_query = StubProcessVoiceQuery()
        app.state.stt_boot_error = None
        app.state.rag_service = StubRAGServiceFallback()
        app.state.rag_boot_error = None

        client = TestClient(app)
        payload = {"audio_bytes_b64": base64.b64encode(b"audio-bytes").decode("utf-8")}

        response = client.post("/transcribe-and-route", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["statuscode"], 200)
        self.assertEqual(data["faq_decision"]["decision"], "fallback")
        self.assertIsNone(data["faq_decision"]["answer"])

    def test_index_faq_samples_indexes_expected_rows(self) -> None:
        app = create_app()
        app.state.faq_indexer = StubFAQIndexer()

        client = TestClient(app)
        response = client.post(
            "/index-faq-samples",
            json={"faq_json_path": "data/faqs/faqs.json"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["statuscode"], 200)
        self.assertEqual(data["indexed_count"], 5)
        self.assertEqual(data["preview"][0]["id"], "faq-001")


if __name__ == "__main__":
    unittest.main()
