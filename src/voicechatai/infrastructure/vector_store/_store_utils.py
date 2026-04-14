from __future__ import annotations

from typing import Any


def _build_payload(metadata: dict[str, Any], document: str) -> dict[str, str]:
    payload: dict[str, str] = {str(k): str(v) for k, v in metadata.items()}
    if "answer" not in payload and document:
        payload["answer"] = document
    return payload
