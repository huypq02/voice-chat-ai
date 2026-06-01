from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationState:
    turn_count: int = 0
    silence_streak: int = 0
    history: list[tuple[str, str]] = field(default_factory=list)  # (role, text)

    def record_user_turn(self, text: str) -> None:
        self.history.append(("user", text))
        self.turn_count += 1
        self.silence_streak = 0

    def record_assistant_turn(self, text: str) -> None:
        self.history.append(("assistant", text))

    def record_silence(self) -> None:
        self.silence_streak += 1
        self.turn_count += 1

    def reset_silence(self) -> None:
        self.silence_streak = 0
