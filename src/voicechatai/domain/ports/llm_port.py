from __future__ import annotations

from abc import ABC, abstractmethod


class LLMGenerationError(Exception):
    """Raised when the LLM fails to generate a response."""


class LLMPort(ABC):
    """Port for large language model text generation."""

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        raise NotImplementedError
