"""
Quick check: does the LLM correctly self-report intent?

Run from project root:
    python check_intent.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

from voicechatai.application.services.intent_service import IntentService
from voicechatai.application.services.llm_answer_service import LLMAnswerService
from voicechatai.infrastructure.llm.gemini_llm import GeminiLLM

llm = GeminiLLM()
svc = LLMAnswerService(llm=llm)
intent_svc = IntentService()

CASES = [
    {
        "label": "CONTINUE — normal question",
        "query": "Why is the sky blue?",
        "history": [],
    },
    {
        "label": "REPLAY — filler words only",
        "query": "umm... uh yeah",
        "history": [
            ("assistant", "Why is the sky blue? Sunlight scatters blue light more than other colors."),
            ("user", "uh huh"),
        ],
    },
    {
        "label": "STOP — repeated non-answer after question",
        "query": "I don't know",
        "history": [
            ("assistant", "Can you tell me what subject you need help with?"),
            ("user", "hmm"),
            ("assistant", "Sure! Which topic would you like to explore?"),
            ("user", "I don't know"),
            ("assistant", "No problem — what are you curious about today?"),
        ],
    },
]

print("=" * 60)
for case in CASES:
    print(f"\n▶ {case['label']}")
    print(f"  query   : {case['query']}")
    raw = svc.generate_answer_with_intent(
        query=case["query"],
        top_candidate=None,
        history=case["history"],
    )
    answer, intent = intent_svc.parse(raw)
    print(f"  raw LLM : {raw!r}")
    print(f"  intent  : {intent}")
    print(f"  answer  : {answer}")
    print("-" * 60)
