from __future__ import annotations

import re

_INTENT_PATTERN = re.compile(r"<intent>(STOP|REPLAY|CONTINUE)</intent>", re.IGNORECASE)

SILENCE_REPLAY_THRESHOLD = 2


class IntentService:
    def should_replay_on_silence(self, silence_streak: int) -> bool:
        return silence_streak >= SILENCE_REPLAY_THRESHOLD

    def parse(self, llm_response: str) -> tuple[str, str]:
        """Strip <intent> tag from LLM response. Returns (clean_text, intent)."""
        match = _INTENT_PATTERN.search(llm_response)
        intent = match.group(1).upper() if match else "CONTINUE"
        clean_text = _INTENT_PATTERN.sub("", llm_response).strip()
        return clean_text, intent
