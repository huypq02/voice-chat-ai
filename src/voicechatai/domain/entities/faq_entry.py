from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FAQEntry:
	"""Single FAQ candidate returned by retrieval."""

	faq_id: str
	question: str
	answer: str
	score: float = 0.0
	metadata: dict[str, str] = field(default_factory=dict)


