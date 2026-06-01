---
stepsCompleted: [1, 2, 3]
inputDocuments: []
session_topic: 'Conversation intent classifier: STOP / REPLAY / CONTINUE for voice chat AI'
session_goals: 'Architecture decisions, signal design, edge case handling, integration patterns'
selected_approach: 'progressive-flow'
techniques_used: ['What If Scenarios', 'Morphological Analysis', 'First Principles Thinking', 'Decision Tree Mapping']
ideas_generated: []
context_file: ''
---

# Brainstorming Session Results

**Facilitator:** Hit
**Date:** 2026-06-01

## Session Overview

**Topic:** Conversation intent classifier — `STOP` / `REPLAY` / `CONTINUE` — for a voice chat AI pipeline

**Goals:**
- Architecture decisions (rule-based vs. LLM vs. hybrid)
- Signal design (what features/inputs drive each intent)
- Edge case handling
- Integration patterns into the existing STT → LLM → TTS pipeline

### Intent Definitions

| Intent | Trigger Condition |
|--------|------------------|
| `STOP` | LLM asked a specific question AND Q&A exchange exceeded 3 turns |
| `REPLAY` | User silence occurred 2+ times, OR user spoke something with no semantic meaning |
| `CONTINUE` | Normal conversation — keep the pipeline running |

---

## Phase 1: Expansive Exploration — What If Scenarios

### Ideas Generated

**[Signal #1]: Double Silence = Dead End**
_Concept:_ Client silent → LLM fills gap → client silent again → STOP. Second silence is the confirming signal: user has checked out, not just paused. LLM response is used as a stimulus probe.
_Novelty:_ Not one silence but the silence-after-response pattern. The LLM's own output becomes a test probe.

**[Signal #2]: User Engagement Score (UES)**
_Concept:_ Rolling float (0.0–1.0) computed per turn from AssemblyAI signals: speech confidence, silence ratio, staleness, response length. Degrades gracefully across turns rather than firing on hard thresholds.
_Novelty:_ Continuous signal avoids cliff edges. Classifier input = `[current_UES, UES_delta, UES_streak]` — trend matters more than snapshot.

**[Signal #3]: UES as Learned Feature Vector**
_Concept:_ Feed `[current_UES, UES_delta, UES_streak, turn_count, last_llm_was_question]` into classifier — thresholds learned from data, not hardcoded.
_Novelty:_ Shifts from rule-engineering to data-driven boundary learning.

**[Signal #4]: Two-Tier Classifier**
_Concept:_ Tier 1 = deterministic fast-path (<1ms) handles obvious cases (silence_streak ≥ 2 → REPLAY, turn_count > 3 + Q asked → STOP). Tier 2 = LLM only on AMBIGUOUS cases.
_Novelty:_ LLM latency paid ~20% of the time. Fast-path removes cost from the happy path entirely.

### Key Breakthrough
AssemblyAI exposes behavioral silence signals (duration, frequency, speech confidence) but NOT acoustic ones (energy level, ambient noise). The classifier must work from behavioral patterns, not audio physics.

---

## Phase 2: Pattern Recognition — Morphological Analysis

| Axis | Decision |
|------|----------|
| Input signals | Conversation state only (`turn_count`, `qa_depth`, `silence_streak`, `last_llm_was_question`) |
| Architecture | LLM self-reports intent via `<intent>` tag inline with every response |
| When fires | After every user turn (always running) |
| Action on intent | Immediate — no buffering |
| REPLAY execution | Pre-recorded audio prompt ("Are you still there?") |
| STOP execution | LLM closing statement → session close |

### Key Insight
The entire classifier collapses to **system prompt + output parsing**. No separate model, no training data, no ML pipeline. The LLM tracks its own conversation state and signals intent inline.

---

### Open Question (deferred)
Semantic emptiness detection ("umm", fillers, talking to someone else) — transcript alone may not be enough; may need audio energy or a dedicated filler-word model.

---

## Phase 3: First Principles Thinking

**Core validation:** LLM has full conversation history and can correctly judge short answers ("yes", "no") in context. Self-reported intent via `<intent>` tag is viable.

**Minimum required information confirmed:**
- Conversation history (already in LLM context)
- Turn count (trackable in state)
- Whether LLM last output was a question (detectable from LLM's own output)
- Whether user speech was meaningful (LLM judges from transcript)

---

## Phase 4: Action Planning — Decision Tree Mapping

### Pipeline Flow

```
User speaks
    │
    ▼
STT (AssemblyAI) → transcript
    │
    ├─ no transcript + silence_streak >= 2 ──────────────→ REPLAY
    │                                                       (play pre-recorded audio)
    ▼
LLM receives:
  - conversation history
  - transcript
  - system prompt with intent rules
    │
    ▼
LLM outputs:
  [natural response]
  <intent>STOP | REPLAY | CONTINUE</intent>
    │
    ├─ parse <intent> tag
    │
    ├─ CONTINUE ──→ TTS → play response → wait for next user turn
    │
    ├─ REPLAY ────→ play pre-recorded "Are you still there?" → reset silence_streak
    │
    └─ STOP ──────→ LLM closing statement (2nd call) → TTS → close session
```

### Session State (in-memory per WebSocket session)

```python
session_state = {
    "turn_count": 0,
    "silence_streak": 0,
    "qa_since_last_question": 0,
}
```

### System Prompt

```
You are a voice assistant. After every response, you MUST append an intent tag on its own line.

## Intent Rules

**STOP** — end the session if ANY of these are true:
- You asked the user a direct question, and the user has not given a meaningful answer
  after 3 or more exchanges
- The user's last message was silence or completely meaningless AND this has happened
  2 or more times in a row

**REPLAY** — prompt the user to re-engage if:
- The user's last message contained only filler words (e.g. "umm", "uh", "yeah yeah",
  "okay okay") with no substantive content
- The user's message was off-topic or clearly addressed to someone else

**CONTINUE** — everything else. Normal conversation.

## Output Format

Always end your response with exactly this on the last line:
<intent>CONTINUE</intent>

Replace CONTINUE with STOP or REPLAY when the rules above apply.

## Examples

User: "uh... yeah..."
→ <intent>REPLAY</intent>

User: "yes" (after you asked a yes/no question)
→ <intent>CONTINUE</intent>

User gives no meaningful answer to your question for the 4th time
→ <intent>STOP</intent>
```

### Implementation Notes

1. **Silence fast-path** — `silence_streak >= 2` fires REPLAY before LLM is called. Saves latency and cost.
2. **STOP closing** — when intent is STOP, make a second LLM call: `"Generate a warm closing statement and say goodbye."` Then TTS → close session.
3. **Intent parsing** — strip `<intent>...</intent>` from LLM output before passing text to TTS.

---

## Session Summary

**Architecture decision:** LLM self-reports intent via `<intent>` tag — no separate classifier model needed.

**Signal stack:**
- Layer 1 (pre-LLM): `silence_streak` counter → fast REPLAY path
- Layer 2 (LLM): semantic judgment → STOP / REPLAY / CONTINUE inline

**What to build next:**
1. Add `silence_streak` tracking to WebSocket session state
2. Inject system prompt into existing LLM call
3. Parse `<intent>` tag from LLM response before TTS
4. Add REPLAY pre-recorded audio asset
5. Add STOP second LLM call + session close logic
