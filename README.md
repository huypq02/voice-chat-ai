# Voice Chat AI

A real-time voice assistant pipeline: Speech-to-Text → semantic retrieval (RAG) → LLM fallback → Text-to-Speech. Built with Clean Architecture so every provider (STT, LLM, TTS, Embedding, Vector DB) is swappable without touching business logic.

---

## Pipeline

### Full pipeline — `POST /voice-chat`

```
User Audio (base64 WAV/MP3)
   │
   ▼
┌──────────────────────┐
│  STT                 │  WhisperSTT (local) — batch
│  Speech → Text       │
└──────────┬───────────┘
           │ transcript
           ▼
┌──────────────────────┐
│  Embedding Model     │  OpenAI / Gemini / MiniLM (local)
│  Text → Vector       │
└──────────┬───────────┘
           │ vector
           ▼
┌──────────────────────┐
│  Vector Store Query  │  ChromaDB (local) or PostgreSQL pgvector
│  cosine top-k=5      │
└──────────┬───────────┘
           │ [FAQEntry, score]
           ▼
┌──────────────────────┐
│  RAGService          │  hit=0.76 · clarify=0.65 · margin=0.06
│  Decision            │
└────────┬─────────────┘
         │
   ┌─────┼──────────────┐
   │     │              │
  HIT  CLARIFY        MISS
   │     │              │
   ▼     ▼              ▼
FAQ   Clarif.        LLM call
Answer  Prompt    (OpenAI gpt-4o-mini or Gemini gemini-1.5-flash)
   │     │         + RAG context via system prompt
   └─────┴──────────────┘
         │ answer text
         ▼
┌──────────────────────┐
│  TTS                 │  ElevenLabs (MP3) or Gemini TTS (WAV)
│  Text → Audio        │
└──────────┬───────────┘
           │
           ▼
        Response
{ transcript, answer, audio_b64, decision, top1_score, clarification_questions }
```

### Streaming pipeline — `GET /ws/transcribe`

Client streams binary PCM16 (16 kHz mono) → server yields `{text, is_final}` JSON events.
Provider selected via `?provider=deepgram|assemblyai` query param.

---

## Provider Matrix

| Stage | Providers | Default |
|-------|-----------|---------|
| STT (batch) | OpenAI Whisper (local) | `base` model |
| STT (stream) | Deepgram nova-2, AssemblyAI | — |
| Embedding | OpenAI `text-embedding-3-small`, Gemini `text-embedding-004`, MiniLM L6-v2 (local) | OpenAI |
| Vector DB | ChromaDB (local), PostgreSQL pgvector | ChromaDB |
| LLM | OpenAI `gpt-4o-mini`, Gemini `gemini-1.5-flash` | OpenAI |
| TTS | ElevenLabs `eleven_monolingual_v1`, Gemini `gemini-2.5-flash-preview-tts` | ElevenLabs |

---

## Project Structure

Clean Architecture — dependencies only point **inward**. Domain has zero external deps; Infrastructure implements Domain ports.

```
voice-chat-ai/
├── src/voicechatai/
│   ├── domain/                         # Layer 1 — pure business rules, no external deps
│   │   ├── entities/
│   │   │   ├── faq_entry.py            # FAQEntry(faq_id, question, answer, score, metadata)
│   │   │   ├── message.py
│   │   │   └── document.py
│   │   └── ports/                      # Abstract interfaces (dependency inversion)
│   │       ├── stt_port.py             # transcribe(bytes) → str
│   │       ├── tts_port.py             # synthesize(str) → bytes
│   │       ├── llm_port.py             # generate(system_prompt, user_message) → str
│   │       ├── embedder_port.py        # embed_text(str) → list[float]
│   │       ├── vector_store_port.py    # query(vector, top_k), upsert(...)
│   │       └── realtime_stt_port.py    # async streaming interface
│   │
│   ├── application/                    # Layer 2 — use cases, depends on domain only
│   │   ├── use_cases/
│   │   │   └── process_voice_chat.py   # Main pipeline orchestrator
│   │   └── services/
│   │       ├── retrieval_service.py    # embed + vector store query
│   │       ├── rag_service.py          # hit / clarify / fallback decision policy
│   │       └── llm_answer_service.py   # RAG context assembly + LLM call
│   │
│   ├── infrastructure/                 # Layer 3 — concrete adapters (implements ports)
│   │   ├── stt/
│   │   │   ├── whisper_stt.py          # local openai-whisper (batch)
│   │   │   ├── deepgram_stt.py         # Deepgram streaming (deepgram-sdk v3)
│   │   │   └── assemblyai_stt.py       # AssemblyAI streaming
│   │   ├── tts/
│   │   │   ├── elevenlabs_tts.py       # ElevenLabs MP3
│   │   │   └── gemini_tts.py           # Gemini WAV/PCM
│   │   ├── llm/
│   │   │   ├── openai_llm.py
│   │   │   └── gemini_llm.py
│   │   ├── embedding/
│   │   │   ├── openai_embedder.py
│   │   │   ├── gemini_embedder.py
│   │   │   └── minilm_embedder.py      # local sentence-transformers (384-dim)
│   │   └── vector_store/
│   │       ├── chroma_store.py         # ChromaDB, cosine similarity
│   │       └── pgvector_store.py       # PostgreSQL + pgvector
│   │
│   └── interfaces/                     # Layer 4 — delivery (FastAPI)
│       ├── api/
│       │   ├── app.py                  # FastAPI factory + DI wiring
│       │   ├── routes.py               # POST /voice-chat, GET /health
│       │   ├── websocket_routes.py     # GET /ws/transcribe
│       │   └── static/index.html       # Web UI (full pipeline + WS tester)
│       └── presenters/
│           └── response.py             # VoiceChatResponse Pydantic schema
│
├── data/faqs/faqs.json                 # Sample FAQ entries (ingested at startup)
├── chroma_db/                          # Persisted ChromaDB vector store
├── prompts/rag_system.txt              # RAG system prompt
├── docs/                               # Architecture docs and execution plans
├── tests/
├── main.py                             # Entry point (Uvicorn)
├── pyproject.toml
└── .env.example
```

---

## Getting Started

**Prerequisites:** Python ≥ 3.11, [uv](https://github.com/astral-sh/uv), ffmpeg (for local Whisper)

```bash
# Windows
winget install Gyan.FFmpeg

# macOS
brew install ffmpeg
```

```bash
git clone https://github.com/your-org/voice-chat-ai.git
cd voice-chat-ai

uv sync

cp .env.example .env
# fill in API keys (see Configuration below)

uv run python main.py
# server starts at http://localhost:8000
# open http://localhost:8000 for the Web UI
```

---

## API

### `POST /voice-chat`

**Request**
```json
{ "audio_bytes_b64": "<base64-encoded WAV or MP3>" }
```

**Response**
```json
{
  "statuscode": 200,
  "transcript": "What time does the park open?",
  "answer": "The park opens at 9 AM.",
  "audio_b64": "<base64 MP3>",
  "decision": "hit",
  "top1_score": 0.91,
  "clarification_questions": []
}
```

`decision` values: `"hit"` · `"clarify"` · `"fallback"`

### `GET /ws/transcribe?provider=deepgram|assemblyai`

Stream binary PCM16 audio (16 kHz, mono). Receive JSON events:
```json
{ "type": "partial_transcript", "payload": { "text": "hello", "is_final": false } }
{ "type": "final_transcript",   "payload": { "text": "hello world", "is_final": true } }
```

### `GET /health`

```json
{ "status": "ok", "service": "voice-chat-ai" }
```

---

## Configuration

### App
| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | `127.0.0.1` | Bind address |
| `APP_PORT` | `8000` | Bind port |
| `APP_DEV` | `false` | Enable dev/reload mode |

### Whisper (STT)
| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Model size: `tiny` `base` `small` `medium` `large` |
| `WHISPER_LANGUAGE` | — | Optional language hint (e.g. `en`) |
| `WHISPER_TASK` | `transcribe` | `transcribe` or `translate` |
| `WHISPER_FP16` | `false` | Enable fp16 on CUDA |
| `WHISPER_AUDIO_SUFFIX` | `.wav` | Temp file suffix for audio bytes |

### OpenAI
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for OpenAI STT / LLM / Embedding |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | |

### Gemini
| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Required for Gemini LLM / Embedding / TTS |
| `GEMINI_LLM_MODEL` | `gemini-1.5-flash` | |
| `GEMINI_EMBEDDING_MODEL` | `models/text-embedding-004` | |
| `GEMINI_TTS_MODEL` | `gemini-2.5-flash-preview-tts` | |
| `GEMINI_TTS_VOICE` | `Kore` | Voices: `Kore` `Charon` `Fenrir` `Aoede` `Puck` |

### ElevenLabs (TTS)
| Variable | Default | Description |
|----------|---------|-------------|
| `ELEVENLABS_API_KEY` | — | |
| `ELEVENLABS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | Rachel voice |
| `ELEVENLABS_MODEL_ID` | `eleven_monolingual_v1` | |

### ChromaDB
| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_COLLECTION` | `faqs` | Collection name |
| `CHROMA_DB_PATH` | `chroma_db` | Local persistence directory |

### PgVector
| Variable | Default | Description |
|----------|---------|-------------|
| `PGVECTOR_CONN` | — | PostgreSQL connection string |
| `PGVECTOR_TABLE` | `vectors` | |
| `PGVECTOR_DIMENSIONS` | `384` | Must match embedding model output |

### Deepgram (streaming STT)
| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPGRAM_API_KEY` | — | |
| `DEEPGRAM_MODEL` | `nova-2` | |
| `DEEPGRAM_LANGUAGE` | `en-US` | |
| `DEEPGRAM_SAMPLE_RATE` | `16000` | |

### AssemblyAI (streaming STT)
| Variable | Default | Description |
|----------|---------|-------------|
| `ASSEMBLYAI_API_KEY` | — | |
| `ASSEMBLYAI_SAMPLE_RATE` | `16000` | |

---

## RAG Decision Thresholds

Configured in `src/voicechatai/application/services/rag_service.py`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `hit_threshold` | `0.76` | Score ≥ 0.76 → return FAQ answer directly |
| `ambiguity_threshold` | `0.65` | Score 0.65–0.76 with low margin → return clarification questions |
| `min_margin` | `0.06` | Min gap between top-2 scores to avoid ambiguity |
| `short_query_hit_boost` | `0.03` | Boost applied to queries ≤ 3 tokens |
| `clarification_top_n` | `2` | Number of candidate questions surfaced on clarify |

---

## Tech Stack

- **Runtime:** Python ≥ 3.11, [uv](https://github.com/astral-sh/uv)
- **API:** FastAPI ≥ 0.109, Uvicorn (ASGI)
- **STT:** openai-whisper, deepgram-sdk ≥ 3.0, assemblyai ≥ 0.17
- **Embedding:** openai ≥ 1.30, google-generativeai ≥ 0.8, sentence-transformers ≥ 3.0
- **Vector DB:** chromadb ≥ 0.5, psycopg[binary] ≥ 3.1, pgvector ≥ 0.3
- **LLM / TTS:** openai, google-genai ≥ 1.0, elevenlabs

---

## License

MIT. See `LICENSE`.
