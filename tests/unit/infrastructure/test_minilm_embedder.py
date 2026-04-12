import unittest

from voicechatai.infrastructure.embedding.minilm_embedder import MiniLMEmbedder


class StubVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class StubSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, bool]] = []

    def encode(self, text: str, convert_to_numpy: bool, normalize_embeddings: bool) -> StubVector:
        self.calls.append((text, convert_to_numpy, normalize_embeddings))
        return StubVector([0.1, 0.2, 0.3])


class MiniLMEmbedderTests(unittest.TestCase):
    def test_embed_text_returns_vector_from_local_model(self) -> None:
        model = StubSentenceTransformer()
        embedder = MiniLMEmbedder(model=model)

        vector = embedder.embed_text("hello world")

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        self.assertEqual(model.calls, [("hello world", True, True)])

    def test_embed_text_returns_empty_for_blank_content(self) -> None:
        model = StubSentenceTransformer()
        embedder = MiniLMEmbedder(model=model)

        vector = embedder.embed_text("   ")

        self.assertEqual(vector, [])
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
