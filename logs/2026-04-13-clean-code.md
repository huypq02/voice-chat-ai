# Clean Code Session — 2026-04-13

## Summary
Code review and cleanup of the `feature/faq-flow` branch.
All 51 unit tests pass after changes.

---

## Files Changed

### `src/voicechatai/infrastructure/stt/whisper_stt.py`
- Removed dead methods: `_ensure_ffmpeg_available()`, `_patch_whisper_audio_runner()` (removed from `__init__` in prior commit but left as orphans)
- Removed unused imports: `shutil`, `subprocess`, `from tempfile import NamedTemporaryFile`
- Removed unused attribute: `self._sample_rate` and `sample_rate` constructor param
- Scoped `audio_path` variable to the `else` branch where it is used, with its own `try/finally` cleanup (was a misleading outer `None` binding that leaked through the PCM path)
- Consolidated to single `import tempfile` (was imported twice)

### `src/voicechatai/infrastructure/vector_store/pgvector_store.py`
- Extracted `_try_register_vector()` module-level helper — eliminates duplicated `register_vector` call in `__post_init__` branches
- Fixed `except (ImportError, Exception)` → `except Exception` (ImportError was redundant)
- Removed `table = self.table_name` local aliases in `_ensure_schema`, `query`, `upsert` — inlined `self.table_name`
- `upsert` loop → `cur.executemany()` — single DB round-trip instead of N; `zip()` replaces fragile index-bounds checks
- Added length assertion for mismatched input lists — was silently writing empty/corrupt rows, now raises `ValueError`
- `query` uses CTE so distance is computed once by Postgres; vector parameter sent once instead of twice

### `src/voicechatai/infrastructure/vector_store/_store_utils.py` *(new file)*
- Extracted `_build_payload(metadata, document)` helper shared by `ChromaStore` and `PgVectorStore`
- Eliminates duplicated metadata-to-payload conversion logic across both adapters

### `src/voicechatai/infrastructure/vector_store/chroma_store.py`
- Uses `_build_payload()` from shared util — 3 lines collapsed to 1

### `tests/unit/infrastructure/test_whisper_stt.py`
- Added `_make_adapter(fake_module, **kwargs)` helper — 7 copy-pasted `with patch(...)` blocks reduced to a single definition
- Removed stale `test_constructor_raises_controlled_error_when_ffmpeg_is_missing` (tested dead `_ensure_ffmpeg_available` code)
- Updated `test_constructor_raises_controlled_error_when_whisper_is_missing` — removed irrelevant `shutil.which` patch

### `tests/unit/infrastructure/test_pgvector_store.py`
- Added `executemany()` to `StubCursor`
- Fixed param index assertion for CTE query: `params[1]` (not `params[2]`) since vector is now passed once
- Replaced `test_upsert_uses_empty_doc_when_documents_shorter_than_ids` with `test_upsert_rejects_ragged_input_lists` to match new contract

---

## Issues Found (Not Fixed — Out of Scope for Cleanup)

These were identified in the code review but left for separate work:

| Issue | Severity | Location |
|---|---|---|
| `MiniLMEmbedder.__post_init__` broken lazy init | HIGH | `infrastructure/embedding/minilm_embedder.py` |
| `OpenAIEmbedder._client` never initialized when `client=None` | HIGH | `infrastructure/embedding/openai_embedder.py` |
| `llm_port.py`, `tts_port.py`, `openai_llm.py`, `elevenlabs_tts.py` — empty stubs | HIGH | Multiple |
| Zero logging throughout entire codebase | MEDIUM | All |
| No size limit on base64 audio decode in routes | MEDIUM | `interfaces/api/routes.py:261` |
| No `OPENAI_API_KEY` presence validation at startup | MEDIUM | `infrastructure/embedding/openai_embedder.py` |
| Naive tokenization in RAGService (`text.split()`) | MEDIUM | `application/services/rag_service.py:91` |
| No end-to-end WebSocket integration test | MEDIUM | `tests/` |
| No tests for `ChromaStore`, `OpenAIEmbedder` | MEDIUM | `tests/` |
| `conn_string` / `conn` mutable after construction on `PgVectorStore` | LOW | `infrastructure/vector_store/pgvector_store.py` |
| `_ensure_schema` runs DDL on every `PgVectorStore` init (lock contention) | LOW | `infrastructure/vector_store/pgvector_store.py` |
