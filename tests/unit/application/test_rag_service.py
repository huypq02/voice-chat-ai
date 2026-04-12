import unittest

from voicechatai.application.services.rag_service import RAGService
from voicechatai.domain.entities.faq_entry import FAQEntry


class StubRetrievalService:
    def __init__(self, candidates: list[FAQEntry]):
        self.candidates = candidates
        self.calls: list[str] = []

    def retrieve_faq_candidates(self, query_text: str) -> list[FAQEntry]:
        self.calls.append(query_text)
        return self.candidates


class RAGServiceDecisionTests(unittest.TestCase):
    def test_hit_when_score_and_margin_meet_policy(self) -> None:
        retrieval = StubRetrievalService(
            candidates=[
                FAQEntry(faq_id="1", question="q1", answer="a1", score=0.86),
                FAQEntry(faq_id="2", question="q2", answer="a2", score=0.70),
            ]
        )
        service = RAGService(retrieval_service=retrieval)

        decision = service.evaluate_faq_decision("how to reset password")

        self.assertEqual(decision.decision, "hit")
        self.assertEqual(decision.top_candidate.answer, "a1")

    def test_clarify_when_score_in_ambiguity_band(self) -> None:
        retrieval = StubRetrievalService(
            candidates=[
                FAQEntry(faq_id="1", question="refund policy", answer="a1", score=0.74),
                FAQEntry(faq_id="2", question="refund status", answer="a2", score=0.71),
            ]
        )
        service = RAGService(retrieval_service=retrieval)

        decision = service.evaluate_faq_decision("refund")

        self.assertEqual(decision.decision, "clarify")
        self.assertEqual(len(decision.clarification_candidates), 2)

    def test_fallback_when_top_score_below_ambiguity_threshold(self) -> None:
        retrieval = StubRetrievalService(
            candidates=[FAQEntry(faq_id="1", question="q1", answer="a1", score=0.42)]
        )
        service = RAGService(retrieval_service=retrieval)

        decision = service.evaluate_faq_decision("unrelated question")

        self.assertEqual(decision.decision, "fallback")

    def test_short_query_applies_stricter_hit_threshold(self) -> None:
        retrieval = StubRetrievalService(
            candidates=[
                FAQEntry(faq_id="1", question="q1", answer="a1", score=0.83),
                FAQEntry(faq_id="2", question="q2", answer="a2", score=0.70),
            ]
        )
        service = RAGService(retrieval_service=retrieval)

        decision = service.evaluate_faq_decision("refund?")

        self.assertEqual(decision.effective_hit_threshold, 0.79)
        self.assertEqual(decision.decision, "hit")

    def test_low_margin_blocks_hit_and_routes_to_clarify(self) -> None:
        retrieval = StubRetrievalService(
            candidates=[
                FAQEntry(faq_id="1", question="q1", answer="a1", score=0.90),
                FAQEntry(faq_id="2", question="q2", answer="a2", score=0.87),
            ]
        )
        service = RAGService(retrieval_service=retrieval)

        decision = service.evaluate_faq_decision("reset account password")

        self.assertAlmostEqual(decision.margin, 0.03, places=6)
        self.assertEqual(decision.decision, "clarify")

    def test_fallback_when_no_candidates(self) -> None:
        retrieval = StubRetrievalService(candidates=[])
        service = RAGService(retrieval_service=retrieval)

        decision = service.evaluate_faq_decision("anything")

        self.assertEqual(decision.decision, "fallback")

    def test_minilm_style_high_confidence_faq_question_is_hit(self) -> None:
        retrieval = StubRetrievalService(
            candidates=[
                FAQEntry(
                    faq_id="faq-sky",
                    question="Why is the sky blue?",
                    answer="Blue light spreads out the most in the air.",
                    score=0.762,
                ),
                FAQEntry(
                    faq_id="faq-water",
                    question="What is the water cycle?",
                    answer="Water turns into vapor, forms clouds, and falls as rain.",
                    score=0.0,
                ),
            ]
        )
        service = RAGService(retrieval_service=retrieval)

        decision = service.evaluate_faq_decision("I have some question, why is the sky blue?")

        self.assertEqual(decision.decision, "hit")
        self.assertEqual(decision.top_candidate.faq_id, "faq-sky")


if __name__ == "__main__":
    unittest.main()
