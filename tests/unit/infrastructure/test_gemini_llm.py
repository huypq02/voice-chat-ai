from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from voicechatai.domain.ports.llm_port import LLMGenerationError
from voicechatai.infrastructure.llm.gemini_llm import GeminiLLM


def _make_mock_genai(response_text: str = "answer") -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    return mock_genai


class GeminiLLMConstructorTests(unittest.TestCase):
    def test_raises_runtime_error_when_package_missing(self) -> None:
        with patch.dict("sys.modules", {"google.generativeai": None}):
            with self.assertRaises(RuntimeError):
                GeminiLLM(api_key="key")

    def test_accepts_api_key_from_constructor(self) -> None:
        mock_genai = _make_mock_genai()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            llm = GeminiLLM(api_key="explicit-key")
        self.assertEqual(llm.api_key, "explicit-key")

    def test_accepts_api_key_from_env(self) -> None:
        mock_genai = _make_mock_genai()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}):
                llm = GeminiLLM()
        self.assertEqual(llm.api_key, "env-key")

    def test_default_model(self) -> None:
        mock_genai = _make_mock_genai()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            llm = GeminiLLM(api_key="key")
        self.assertEqual(llm.model, "gemini-1.5-flash")

    def test_custom_model_from_env(self) -> None:
        mock_genai = _make_mock_genai()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch.dict("os.environ", {"GEMINI_LLM_MODEL": "gemini-1.5-pro"}):
                llm = GeminiLLM(api_key="key")
        self.assertEqual(llm.model, "gemini-1.5-pro")


class GeminiLLMGenerateTests(unittest.TestCase):
    def _build_llm(self, mock_genai: MagicMock) -> GeminiLLM:
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            return GeminiLLM(api_key="key")

    def test_generate_returns_stripped_text(self) -> None:
        mock_genai = _make_mock_genai("  hello world  ")
        llm = self._build_llm(mock_genai)
        result = llm.generate(system_prompt="You are helpful.", user_message="Hi")
        self.assertEqual(result, "hello world")

    def test_generate_passes_system_instruction(self) -> None:
        mock_genai = _make_mock_genai("ok")
        llm = self._build_llm(mock_genai)
        llm.generate(system_prompt="sys", user_message="msg")
        mock_genai.GenerativeModel.assert_called_once_with(
            model_name=llm.model,
            system_instruction="sys",
        )

    def test_generate_passes_user_message_to_content(self) -> None:
        mock_genai = _make_mock_genai("ok")
        llm = self._build_llm(mock_genai)
        llm.generate(system_prompt="sys", user_message="hello")
        mock_genai.GenerativeModel.return_value.generate_content.assert_called_once_with("hello")

    def test_generate_raises_llm_generation_error_on_failure(self) -> None:
        mock_genai = _make_mock_genai()
        mock_genai.GenerativeModel.return_value.generate_content.side_effect = RuntimeError("api down")
        llm = self._build_llm(mock_genai)
        with self.assertRaises(LLMGenerationError):
            llm.generate(system_prompt="sys", user_message="msg")

    def test_generate_handles_none_text(self) -> None:
        mock_genai = _make_mock_genai(None)
        llm = self._build_llm(mock_genai)
        result = llm.generate(system_prompt="sys", user_message="msg")
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
