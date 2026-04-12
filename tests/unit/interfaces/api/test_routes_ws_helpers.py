import base64
import unittest

from voicechatai.application.services.rag_service import RAGDecision
from voicechatai.domain.entities.faq_entry import FAQEntry
from voicechatai.interfaces.api.routes import _decode_audio_chunk_payload, _is_pcm16_silence, _to_faq_decision_response


class WebSocketRouteHelperTests(unittest.TestCase):
    def test_decode_audio_chunk_payload_returns_audio_bytes(self) -> None:
        payload = {"audio_b64": base64.b64encode(b"abc123").decode("utf-8")}

        audio, error = _decode_audio_chunk_payload(payload)

        self.assertEqual(audio, b"abc123")
        self.assertIsNone(error)

    def test_decode_audio_chunk_payload_rejects_missing_audio(self) -> None:
        audio, error = _decode_audio_chunk_payload({})

        self.assertIsNone(audio)
        self.assertIn("audio_b64", error or "")

    def test_is_pcm16_silence_true_for_zeroed_samples(self) -> None:
        audio = b"\x00\x00" * 160

        self.assertTrue(_is_pcm16_silence(audio))

    def test_is_pcm16_silence_false_for_loud_samples(self) -> None:
        # 3000 amplitude in little-endian signed 16-bit.
        sample = (3000).to_bytes(2, byteorder="little", signed=True)
        audio = sample * 160

        self.assertFalse(_is_pcm16_silence(audio))

    def test_to_faq_decision_response_includes_answer_for_clarify(self) -> None:
        decision = RAGDecision(
            decision="clarify",
            top_candidate=FAQEntry(
                faq_id="faq-1",
                question="How can I get a refund?",
                answer="Open Billing and submit a refund request.",
                score=0.79,
            ),
            clarification_candidates=[
                FAQEntry(
                    faq_id="faq-1",
                    question="How can I get a refund?",
                    answer="Open Billing and submit a refund request.",
                    score=0.79,
                )
            ],
            top1_score=0.79,
            top2_score=0.74,
            margin=0.05,
            effective_hit_threshold=0.82,
        )

        response = _to_faq_decision_response(decision)

        self.assertEqual(response.decision, "clarify")
        self.assertEqual(response.answer, "Open Billing and submit a refund request.")
        self.assertEqual(response.clarification_questions, ["How can I get a refund?"])


if __name__ == "__main__":
    unittest.main()
