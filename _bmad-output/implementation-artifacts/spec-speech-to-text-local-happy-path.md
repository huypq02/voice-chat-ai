---
title: "Implement Speech-To-Text local happy path"
type: "feature"
created: "2026-04-12"
status: "done"
baseline_commit: "4dad63d363e811701e8a8ed2877391a299ec0f3b"
context: ["_bmad-output/project-context.md", "docs/ai-first-execution-plan.md"]
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The repository defines a Speech-To-Text stage in the product pipeline, but the Speech-To-Text port, adapter, and orchestration surface are still empty. Sprint 1 requires a local happy path so the team can demonstrate the voice pipeline structure without depending on production-grade providers.

**Approach:** Implement a local-only Speech-To-Text slice that accepts raw audio bytes at the port boundary, returns plain transcript text, and uses a fake or deterministic Whisper-shaped adapter behind the infrastructure boundary. Wire the Speech-To-Text dependency into the application layer only as far as needed to support the local happy path, while explicitly excluding public API exposure and advanced Speech-To-Text capabilities.

## Boundaries & Constraints

**Always:** Respect Clean Architecture boundaries from `_bmad-output/project-context.md`; keep Speech-To-Text abstractions in `domain/ports`, adapter logic in `infrastructure/stt`, and orchestration in `application/use_cases`; keep this slice local-happy-path only; use raw bytes as input and plain transcript text as output; add unit tests for both happy and basic failure scenarios.

**Ask First:** Any change that introduces a public HTTP/WebSocket endpoint, a real hosted provider dependency, a richer Speech-To-Text result object, or a different input contract than raw bytes.

**Never:** Do not add streaming transcription, speaker diarization, multilingual auto-detection, or production deployment concerns; do not bypass the port by calling the adapter directly from delivery code; do not couple Speech-To-Text implementation details to retrieval, prompt, or Text-To-Speech logic.

## I/O & Edge-Case Matrix

| Scenario        | Input / State                                               | Expected Output / Behavior                           | Error Handling                             |
| --------------- | ----------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------ |
| HAPPY_PATH      | Non-empty raw audio bytes passed to Speech-To-Text use path | Plain transcript text is returned                    | N/A                                        |
| EMPTY_AUDIO     | Empty bytes passed to Speech-To-Text path                   | No transcript is produced                            | Raise a domain-safe validation error       |
| ADAPTER_FAILURE | Adapter receives bytes but simulated transcription fails    | Application path does not continue with invalid text | Raise a controlled application-level error |

</frozen-after-approval>

## Code Map

- `src/voicechatai/domain/ports/stt_port.py` -- Speech-To-Text abstraction; currently empty and must define the local contract.
- `src/voicechatai/infrastructure/stt/whisper_stt.py` -- Whisper-shaped local adapter implementation; currently empty.
- `src/voicechatai/application/use_cases/process_voice_query.py` -- orchestration entry point for local voice happy path; currently scaffold-only.
- `tests/unit/infrastructure/` -- adapter-level tests for deterministic transcription and failure behavior.
- `tests/unit/application/` -- use-case tests covering Speech-To-Text wiring and error handling.

## Tasks & Acceptance

**Execution:**

- [x] `src/voicechatai/domain/ports/stt_port.py` -- define the Speech-To-Text interface using raw audio bytes input and plain string transcript output -- establishes the stable local contract for Sprint 1.
- [x] `src/voicechatai/infrastructure/stt/whisper_stt.py` -- implement a local fake or deterministic Whisper-shaped adapter that satisfies the port and validates empty input -- delivers the local happy path without external provider dependency.
- [x] `src/voicechatai/application/use_cases/process_voice_query.py` -- wire Speech-To-Text invocation into the use-case path at the minimum level needed for local orchestration -- ensures the adapter participates through the application boundary rather than ad hoc calls.
- [x] `tests/unit/infrastructure/test_whisper_stt.py` -- verify successful transcription behavior and empty-audio failure behavior -- locks the adapter contract.
- [x] `tests/unit/application/test_process_voice_query_stt.py` -- verify the use case invokes the Speech-To-Text port and propagates controlled failures correctly -- protects orchestration behavior.

**Acceptance Criteria:**

- Given non-empty raw audio bytes, when the Sprint 1 local Speech-To-Text path is executed, then plain transcript text is returned through the port contract.
- Given empty raw audio bytes, when transcription is requested, then the system rejects the input with a controlled error instead of producing a transcript.
- Given a Speech-To-Text adapter failure, when the application use case runs, then the failure is surfaced predictably and no downstream pipeline stage is invoked from invalid transcript state.
- Given the implementation is complete, when the code is reviewed, then Speech-To-Text logic remains inside the domain port, infrastructure adapter, and application use case boundaries with no direct delivery-layer coupling.

## Spec Change Log

## Design Notes

This slice is intentionally narrow. The adapter should look like a real Whisper-backed implementation from the outside, but remain deterministic and local-first so Sprint 1 can validate orchestration before provider integration. If future work needs a richer transcription payload, that should be a separate intent because it changes the port contract.

## Verification

**Commands:**

- `$env:PYTHONPATH='src'; C:/Users/huypq/AppData/Local/Programs/Python/Python311/python.exe -m unittest tests.unit.infrastructure.test_whisper_stt tests.unit.application.test_process_voice_query_stt` -- expected: 6 tests pass

**Manual checks (if no CLI):**

- Inspect `src/voicechatai/domain/ports/stt_port.py` and confirm the contract is raw bytes in, plain string out.
- Inspect `src/voicechatai/infrastructure/stt/whisper_stt.py` and confirm no external hosted provider dependency is required for the local happy path.
- Inspect `src/voicechatai/application/use_cases/process_voice_query.py` and confirm orchestration goes through the Speech-To-Text port rather than infrastructure direct calls.
- Inspect the new unit tests and confirm they cover happy path, empty input, and adapter failure behavior.

## Suggested Review Order

**Contract and validation boundary**

- Start at the port contract that defines the Sprint 1 STT slice.
  [`stt_port.py:6`](../../src/voicechatai/domain/ports/stt_port.py#L6)

- Review adapter constructor safeguards that prevent invalid local test configuration.
  [`whisper_stt.py:9`](../../src/voicechatai/infrastructure/stt/whisper_stt.py#L9)

- Check runtime input validation for raw bytes and controlled local failure behavior.
  [`whisper_stt.py:19`](../../src/voicechatai/infrastructure/stt/whisper_stt.py#L19)

**Application orchestration**

- Confirm the use case crosses the boundary only through the STT port.
  [`process_voice_query.py:8`](../../src/voicechatai/application/use_cases/process_voice_query.py#L8)

- Verify blank transcripts are rejected before any downstream stage could run.
  [`process_voice_query.py:22`](../../src/voicechatai/application/use_cases/process_voice_query.py#L22)

**Supporting tests**

- Review adapter tests for happy path, invalid config, invalid input, and simulated failure.
  [`test_whisper_stt.py:8`](../../tests/unit/infrastructure/test_whisper_stt.py#L8)

- Review use-case tests for port invocation and controlled error propagation.
  [`test_process_voice_query_stt.py:32`](../../tests/unit/application/test_process_voice_query_stt.py#L32)
