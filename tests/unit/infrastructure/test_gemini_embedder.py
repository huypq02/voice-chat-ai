from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from voicechatai.infrastructure.embedding.gemini_embedder import GeminiEmbedder


_SENTINEL = object()


def _make_mock_genai(embedding: list[float] | None = _SENTINEL) -> MagicMock:  # type: ignore[assignment]
    mock_genai = MagicMock()
    value = [0.1, 0.2, 0.3] if embedding is _SENTINEL else embedding
    mock_genai.embed_content.return_value = {"embedding": value}
    return mock_genai


class GeminiEmbedderConstructorTests(unittest.TestCase):
    def test_raises_runtime_error_when_package_missing(self) -> None:
        with patch.dict("sys.modules", {"google.generativeai": None}):
            with self.assertRaises(RuntimeError):
                GeminiEmbedder(api_key="key")

    def test_accepts_api_key_from_constructor(self) -> None:
        mock_genai = _make_mock_genai()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            embedder = GeminiEmbedder(api_key="explicit-key")
        self.assertEqual(embedder.api_key, "explicit-key")

    def test_accepts_api_key_from_env(self) -> None:
        mock_genai = _make_mock_genai()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}):
                embedder = GeminiEmbedder()
        self.assertEqual(embedder.api_key, "env-key")

    def test_default_model(self) -> None:
        mock_genai = _make_mock_genai()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            embedder = GeminiEmbedder(api_key="key")
        self.assertEqual(embedder.model, "models/text-embedding-004")

    def test_custom_model_from_env(self) -> None:
        mock_genai = _make_mock_genai()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            with patch.dict("os.environ", {"GEMINI_EMBEDDING_MODEL": "models/embedding-001"}):
                embedder = GeminiEmbedder(api_key="key")
        self.assertEqual(embedder.model, "models/embedding-001")


class GeminiEmbedderEmbedTextTests(unittest.TestCase):
    def _build_embedder(self, mock_genai: MagicMock) -> GeminiEmbedder:
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            return GeminiEmbedder(api_key="key")

    def test_returns_empty_list_for_blank_input(self) -> None:
        mock_genai = _make_mock_genai()
        embedder = self._build_embedder(mock_genai)
        self.assertEqual(embedder.embed_text("   "), [])
        mock_genai.embed_content.assert_not_called()

    def test_returns_vector_on_success(self) -> None:
        mock_genai = _make_mock_genai([0.1, 0.2, 0.3])
        embedder = self._build_embedder(mock_genai)
        result = embedder.embed_text("hello")
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_passes_correct_model_and_content(self) -> None:
        mock_genai = _make_mock_genai()
        embedder = self._build_embedder(mock_genai)
        embedder.embed_text("test input")
        mock_genai.embed_content.assert_called_once_with(
            model="models/text-embedding-004",
            content="test input",
        )

    def test_strips_whitespace_before_embedding(self) -> None:
        mock_genai = _make_mock_genai()
        embedder = self._build_embedder(mock_genai)
        embedder.embed_text("  padded  ")
        mock_genai.embed_content.assert_called_once_with(
            model="models/text-embedding-004",
            content="padded",
        )

    def test_raises_when_embedding_is_empty(self) -> None:
        mock_genai = _make_mock_genai([])
        embedder = self._build_embedder(mock_genai)
        with self.assertRaises(RuntimeError):
            embedder.embed_text("hello")

    def test_raises_when_embedding_is_none(self) -> None:
        mock_genai = _make_mock_genai(None)
        embedder = self._build_embedder(mock_genai)
        with self.assertRaises(RuntimeError):
            embedder.embed_text("hello")

    def test_propagates_api_error(self) -> None:
        mock_genai = _make_mock_genai()
        mock_genai.embed_content.side_effect = RuntimeError("api error")
        embedder = self._build_embedder(mock_genai)
        with self.assertRaises(RuntimeError):
            embedder.embed_text("hello")


if __name__ == "__main__":
    unittest.main()
