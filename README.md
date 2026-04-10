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

## Getting Started

```bash
# clone the repo
git clone https://github.com/your-org/voice-chat-ai.git
cd voice-chat-ai

# create virtual environment and install dependencies
uv sync

# configure environment
cp .env.example .env
# fill in API keys (STT, LLM, TTS, embedding provider)

# run
uv run python main.py
```

---

## Configuration

| Variable            | Description                                      |
| ------------------- | ------------------------------------------------ |
| `STT_PROVIDER`      | STT backend (`whisper`, `deepgram`, …)           |
| `EMBEDDING_MODEL`   | Embedding model name                             |
| `CHROMA_COLLECTION` | ChromaDB collection to query                     |
| `FAQ_THRESHOLD`     | Similarity score τ for FAQ hit (0–1)             |
| `LLM_MODEL`         | LLM used for RAG generation                      |
| `TTS_PROVIDER`      | TTS backend (`elevenlabs`, `openai`, `coqui`, …) |

---

## License

MIT. See `LICENSE`.
