---
stepsCompleted: [1, 2]
inputDocuments: []
session_topic: "Whether to use WebSocket streaming STT vs HTTP chunked upload for real-time voice chat"
session_goals: "A decision framework + recommended architecture for MVP now and scale later"
selected_approach: "ai-recommended"
techniques_used: ["decision-matrix", "progressive-architecture"]
ideas_generated:
  ["websocket-first-hybrid-mvp", "event-schema-stability", "scale-triggers"]
context_file: ""
session_continued: true
continuation_date: "2026-04-12"
---

# Brainstorming Session Results

**Facilitator:** Hit
**Date:** 2026-04-11

## Session Overview

**Topic:** Whether to use WebSocket streaming STT vs HTTP chunked upload for real-time voice chat
**Goals:** A decision framework + recommended architecture for MVP now and scale later

### Session Setup

Session resumed on 2026-04-12 and initialized for decision-focused architecture brainstorming.
Scope includes transport strategy, latency/cost/complexity trade-offs, MVP recommendation, and scale-path evolution guardrails.

## Technique Selection

**Approach:** AI-Recommended Techniques
**Selected Techniques:**

- Decision Matrix: Weighted criteria to compare WebSocket streaming STT vs HTTP chunked upload.
- Progressive Architecture: MVP-now design plus scale-later migration triggers and guardrails.

## Brainstorm Outcomes

### Decision Framework

Weighted criteria for transport selection:

- Latency (25%)
- Turn-taking quality and barge-in support (20%)
- Engineering complexity (15%)
- Infra cost at scale (15%)
- Reliability on weak networks (10%)
- Observability and debugging (10%)
- Vendor portability (5%)

### Recommendation

Use a **WebSocket-first hybrid** strategy for MVP:

- Client to gateway via WebSocket for low-latency UX and partial transcripts.
- Internal STT processing can remain micro-batch/short-window to reduce operational complexity.

### MVP Architecture (Now)

1. WebSocket gateway for audio streaming, session management, heartbeats, and reconnect handling.
2. STT adapter service behind existing STT port; emit partial/final transcript events.
3. Orchestrator triggers retrieval/RAG on final transcript and returns response events.
4. Telemetry for latency, reconnect rate, and STT failure reasons.

### Scale Architecture (Later)

1. Split gateway and STT workers.
2. Add message bus between gateway and STT workers.
3. Partition by session_id for ordering guarantees.
4. Autoscale on concurrent sessions, queue depth, and p95 latency.

### Migration Triggers

- p95 first partial transcript > 700ms
- Concurrent sessions > 300 per region
- Reconnect/error rate > 2%
- GPU utilization > 70% with queue growth

### Guardrails

- Keep event schema stable from MVP onward.
- Keep STT port transport-agnostic.
- Enforce per-session backpressure.
- Use sequence numbers for replay/reconnect.
