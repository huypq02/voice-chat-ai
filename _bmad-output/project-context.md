---
project_name: "voice-chat-ai"
user_name: "Hit"
date: "2026-04-11"
sections_completed: ["technology_stack", "critical_implementation_rules"]
existing_patterns_found: 12
---

# Project Context for AI Agents

This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss.

---

## Technology Stack & Versions

- **Python** ≥ 3.11 — required; uses `X | Y` union syntax and `@dataclass(slots=True)`
- **FastAPI** 0.135.3 — API layer; `src/voicechatai/interfaces/api/`
- **uvicorn** 0.44.0 — ASGI server
- **openai-whisper** 20250625 — local STT (`WhisperSTT`)
- **sentence-transformers** ≥ 3.0 — local embedding, model `all-MiniLM-L6-v2` (`MiniLMEmbedder`)
- **chromadb** ≥ 0.5 — vector store, persisted at `chroma_db/` (`ChromaStore`)
- **openai** ≥ 1.30.0 — embeddings (`OpenAIEmbedder`) + LLM chat completions (`OpenAILLM`, default `gpt-4o-mini`)
- **elevenlabs** — TTS synthesis (`ElevenLabsTTS`) ⚠️ not yet in `pyproject.toml` — must be added before use
- **psycopg[binary]** ≥ 3.1 + **pgvector** ≥ 0.3 — optional PostgreSQL vector store (`PgVectorStore`)
- **numpy** — PCM16 audio decoding in `WhisperSTT`
- Architecture style: Clean Architecture — `domain → application → infrastructure/interfaces`
- Project layout root: `src/voicechatai/`

**Required environment variables (full pipeline):**

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | Yes | — |
| `OPENAI_LLM_MODEL` | No | `gpt-4o-mini` |
| `ELEVENLABS_API_KEY` | Yes (TTS) | — |
| `ELEVENLABS_VOICE_ID` | No | `21m00Tcm4TlvDq8ikWAM` (Rachel) |
| `ELEVENLABS_MODEL_ID` | No | `eleven_monolingual_v1` |
| `WHISPER_MODEL` | No | `base` |
| `CHROMA_COLLECTION` | No | `faqs` |
| `CHROMA_DB_PATH` | No | `chroma_db` |

## Critical Implementation Rules

### Architecture Boundary Rules

- Respect dependency direction exactly: `interfaces -> application -> domain`.
- Infrastructure implements domain ports; domain and application must not import infrastructure or interfaces.
- Keep orchestration logic inside use cases/services under `application/`, not in API route handlers.
- Keep domain entities and ports framework-agnostic (no FastAPI/OpenAI/Chroma imports in domain).

### Code Organization Rules

- Place business entities only in `src/voicechatai/domain/entities/`.
- Place abstractions only in `src/voicechatai/domain/ports/`.
- Place adapter implementations only in `src/voicechatai/infrastructure/`.
- Place delivery concerns (HTTP/WebSocket/presenter mapping) only in `src/voicechatai/interfaces/`.

### Pipeline Behavior Rules

- Keep the canonical flow: STT -> embedding -> retrieval -> threshold -> FAQ hit or RAG miss -> TTS.
- Apply threshold logic in retrieval/application service, not in API controllers.
- Keep prompt templates externalized under `prompts/` (do not hardcode large prompt text in code).
- Treat FAQ hit path and RAG miss path as separate explicit branches for testing.

### Data and Storage Rules

- FAQ source of truth is `data/faqs/faqs.json`.
- Raw ingest sources belong in `data/documents/`.
- Local Chroma runtime data should remain in `chroma_db/` and be treated as generated runtime state.

### Testing Rules

- Mirror architecture in tests: `tests/unit/domain`, `tests/unit/application`, `tests/unit/infrastructure`, plus `tests/integration`.
- Prefer unit tests for business logic in domain/application before integration tests.
- Cover both retrieval branches: threshold hit and threshold miss.

### Configuration and Secrets Rules

- Keep provider/model configuration in environment variables; do not hardcode secrets.
- Keep `.env.example` as the only committed environment template.
- Never commit real credentials or provider tokens.

### Quality and Workflow Rules

- Preserve existing public module boundaries and naming conventions when adding code.
- Keep comments concise and only for non-obvious logic.
- Avoid over-abstraction for one-off constants unless reused.
- Document major architectural changes in `docs/` before implementation expansion.

### Critical Donot-Miss Rules

- Do not bypass ports by calling infrastructure adapters directly from domain/application internals.
- Do not place orchestration in `main.py` or route files beyond wiring and request mapping.
- Do not mix retrieval policy, prompt policy, and transport logic in the same class.
- Do not assume exact package versions without adding them to project metadata first.
