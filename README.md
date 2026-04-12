# Voice Chat AI

A real-time voice assistant pipeline combining Speech-to-Text, semantic retrieval, RAG, LLM, and Text-to-Speech.

---

## Pipeline Overview

```
User Voice
   │
   ▼
┌──────────────────────┐
│  STT                 │  (e.g. Whisper, Deepgram)
│  Speech → Text       │
└──────────┬───────────┘
           │
           ▼
        Raw Text
           │
           ▼
┌──────────────────────┐
│  Embedding Model     │  (e.g. text-embedding-3-small, BGE)
│  Text → Vector       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Chroma Query        │  top-k nearest neighbours
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Threshold Check     │  similarity score ≥ τ ?
└────────┬─────────────┘
         │
   ┌─────┴──────┐
   │            │
  HIT          MISS
   │            │
   ▼            ▼
FAQ Answer   ┌──────────────────────┐
(cached)     │  RAG + LLM           │  retrieve context → prompt → generate
             │  (e.g. GPT-4o, LLaMA)│
             └──────────┬───────────┘
   │                    │
   └─────────┬──────────┘
             │
             ▼
      Answer Text
             │
             ▼
┌──────────────────────┐
│  TTS                 │  (e.g. ElevenLabs, OpenAI TTS, Coqui)
│  Text → Audio        │
└──────────┬───────────┘
           │
           ▼
        Client
   (audio stream / text)
```

---

## Component Breakdown

| Stage                | Description                                                 | Example Implementations           |
| -------------------- | ----------------------------------------------------------- | --------------------------------- |
| **STT**              | Transcribes user audio to text                              | Whisper, Deepgram, Azure Speech   |
| **Embedding**        | Encodes text into a dense vector                            | `text-embedding-3-small`, BGE, E5 |
| **Chroma Query**     | Retrieves top-k semantically similar documents              | ChromaDB (local or cloud)         |
| **Threshold Check**  | Decides FAQ hit or RAG fallback based on similarity score τ | Configurable per use case         |
| **FAQ (HIT)**        | Returns a pre-cached answer directly — low latency          | JSON / vector store metadata      |
| **RAG + LLM (MISS)** | Builds a context-augmented prompt and generates a response  | GPT-4o, LLaMA 3, Mistral          |
| **TTS**              | Synthesises the answer text into speech                     | ElevenLabs, OpenAI TTS, Coqui     |

---

## Flow Summary

1. **User speaks** → audio captured on the client.
2. **STT** transcribes the audio to raw text.
3. **Embedding model** converts the text to a vector.
4. **ChromaDB** performs a top-k similarity search against the knowledge base.
5. **Threshold check**:
   - **HIT** (score ≥ τ): a cached FAQ answer is returned immediately.
   - **MISS** (score < τ): retrieved chunks are injected into an LLM prompt (RAG) and a fresh answer is generated.
6. **TTS** converts the final answer text to audio.
7. **Client** receives and plays the audio (or displays the text).

---

## Project Structure

Clean Architecture with four concentric layers — dependencies only point **inward**.

```
voice-chat-ai/
├── src/
│   └── voicechatai/
│       ├── domain/                      # Layer 1 - Enterprise rules (no external deps)
│       │   ├── entities/
│       │   │   ├── message.py           # Conversation / turn entity
│       │   │   ├── faq_entry.py         # FAQ entry entity
│       │   │   └── document.py          # Document chunk entity
│       │   └── ports/                   # Abstract interfaces (dependency inversion)
│       │       ├── stt_port.py          # STT interface
│       │       ├── tts_port.py          # TTS interface
│       │       ├── llm_port.py          # LLM interface
│       │       ├── embedder_port.py     # Embedding interface
│       │       └── vector_store_port.py # Vector store interface
│       │
│       ├── application/                 # Layer 2 - Use cases (depends on domain only)
│       │   ├── use_cases/
│       │   │   ├── process_voice_query.py # Main STT→Embed→Retrieve→LLM→TTS orchestration
│       │   │   └── ingest_documents.py  # Document ingestion pipeline
│       │   └── services/
│       │       ├── retrieval_service.py # Chroma top-k query + threshold check
│       │       └── rag_service.py       # RAG context assembly + prompt building
│       │
│       ├── infrastructure/              # Layer 3 - Concrete adapters (implements ports)
│       │   ├── stt/
│       │   │   └── whisper_stt.py       # Whisper → STTPort
│       │   ├── tts/
│       │   │   └── elevenlabs_tts.py    # ElevenLabs → TTSPort
│       │   ├── llm/
│       │   │   └── openai_llm.py        # OpenAI → LLMPort
│       │   ├── embedding/
│       │   │   └── openai_embedder.py   # OpenAI → EmbedderPort
│       │   └── vector_store/
│       │       └── chroma_store.py      # ChromaDB → VectorStorePort
│       │
│       └── interfaces/                  # Layer 4 - Delivery (FastAPI, WebSocket)
│           ├── api/
│           │   ├── app.py               # FastAPI app factory + DI wiring
│           │   └── routes.py            # HTTP / WebSocket endpoints
│           └── presenters/
│               └── response.py          # Domain output → API response schema
│
├── data/
│   ├── faqs/
│   │   └── faqs.json                    # Pre-cached FAQ answers
│   └── documents/                       # Raw documents for ingestion
├── chroma_db/                           # Persisted ChromaDB vector store
├── prompts/
│   └── rag_system.txt                   # System prompt template for RAG
├── tests/
│   ├── unit/
│   │   ├── domain/
│   │   ├── application/
│   │   └── infrastructure/
│   └── integration/
├── main.py                              # Entry point
├── pyproject.toml                       # uv / PEP 517 project config
└── .env.example                         # Environment variable template
```

### Dependency Rule

```
interfaces → application → domain
infrastructure → domain (implements ports)
```

Nothing in `domain` or `application` imports from `infrastructure` or `interfaces`. Concrete implementations are injected at startup in `src/voicechatai/interfaces/api/app.py`.

---

## Getting Started

```bash
# clone the repo
git clone https://github.com/your-org/voice-chat-ai.git
cd voice-chat-ai

# create virtual environment and install dependencies
uv sync

# install ffmpeg so local Whisper decoding works
# Windows example: winget install Gyan.FFmpeg

# configure environment
cp .env.example .env
# fill in API keys (STT, LLM, TTS, embedding provider)

# run
uv run python main.py
```

---

## Configuration

| Variable               | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `STT_PROVIDER`         | STT backend (`whisper`, `deepgram`, …)                |
| `WHISPER_MODEL`        | Local Whisper model name (`tiny`, `base`, `small`, …) |
| `WHISPER_LANGUAGE`     | Optional language hint for Whisper                    |
| `WHISPER_TASK`         | Whisper mode (`transcribe` or `translate`)            |
| `WHISPER_FP16`         | Enable fp16 inference when supported                  |
| `WHISPER_AUDIO_SUFFIX` | Temporary file suffix used for raw audio bytes        |
| `EMBEDDING_MODEL`      | Embedding model name                                  |
| `CHROMA_COLLECTION`    | ChromaDB collection to query                          |
| `FAQ_THRESHOLD`        | Similarity score τ for FAQ hit (0–1)                  |
| `LLM_MODEL`            | LLM used for RAG generation                           |
| `TTS_PROVIDER`         | TTS backend (`elevenlabs`, `openai`, `coqui`, …)      |

The STT adapter in `src/voicechatai/infrastructure/stt/whisper_stt.py` now uses the `openai-whisper` package locally. It loads the configured model once at startup, writes incoming audio bytes to a temporary file, and returns the trimmed Whisper transcript through the existing `STTPort` contract.

---

## License

MIT. See `LICENSE`.

---

## Execution Plan

AI-first project plan with phases, sprints, deadlines, quality gates, and KPI targets:

- `docs/ai-first-execution-plan.md`
