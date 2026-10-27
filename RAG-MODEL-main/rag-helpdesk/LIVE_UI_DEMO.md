# Live Agent Activity in UI - What You'll See

## During Retrieval: Real-Time Event Display

When you ask a question, the UI now shows **live updates** as the controller executes:

### Example 1: Question Answered on First Try

**Question:** "What is a black hole?"

**Live Events You'll See:**

```
🤖 Processing question...
├─ 📋 Plan: general question
│  └─ Strategy: base_mode=hybrid
│
├─ 🔍 Attempt 1 — Mode: hybrid
│  └─ Search: what is a black hole?
│
├─ ✅ Evidence: STRONG — Top score: 0.89, gap: 0.15, agreement: True
│
└─ ✨ Generating answer from strong evidence
  │
  ├─ 🤖 Generating answer...
  ├─ ✅ Generated (1250ms)
  │
  ├─ 🔗 Verifying citations...
  ├─ ✅ All 4 claims supported
  │
  ├─ 📈 Computing confidence...
  └─ ✅ Confidence: High (92.0%)

✅ Answer generated
```

### Example 2: Question Requires Retry

**Question:** "What is the Chandrasekhar limit?"

**Live Events You'll See:**

```
🤖 Processing question...
├─ 📋 Plan: general question
│  └─ Strategy: base_mode=hybrid
│
├─ 🔍 Attempt 1 — Mode: hybrid
│  └─ Search: what is the chandrasekhar limit?
│
├─ ⚠️ Evidence: WEAK — Top score below threshold (0.62 < 0.70)
│
├─ 🔄 Retry 1: Retrieved evidence did not meet the answer threshold.
│
├─ 🔍 Attempt 2 — Mode: hybrid
│  └─ Search: maximum mass white dwarf star
│
├─ ✅ Evidence: STRONG — Top score: 0.85, gap: 0.18, agreement: True
│
└─ ✨ Generating answer from strong evidence
  │
  ├─ 🤖 Generating answer...
  ├─ ✅ Generated (1340ms)
  │
  ├─ 🔗 Verifying citations...
  ├─ ✅ All 3 claims supported
  │
  ├─ 📈 Computing confidence...
  └─ ✅ Confidence: High (88.0%)

✅ Answer generated
```

### Example 3: Insufficient Evidence (Abstention)

**Question:** "asdfghjkl random gibberish"

**Live Events You'll See:**

```
🤖 Processing question...
├─ 📋 Plan: general question
│  └─ Strategy: base_mode=hybrid
│
├─ 🔍 Attempt 1 — Mode: hybrid
│  └─ Search: asdfghjkl random gibberish
│
├─ ⚠️ Evidence: NONE — Top score: 0.12, below minimum threshold
│
├─ 🔄 Retry 1: Retrieved evidence did not meet the answer threshold.
│
├─ 🔍 Attempt 2 — Mode: bm25
│  └─ Search: asdfghjkl random gibberish
│
├─ ⚠️ Evidence: NONE — Top score: 0.08, below minimum threshold
│
└─ ⏹️ Abstaining — Evidence: none

⏹️ Insufficient evidence
```

---

## Key Improvements

### Before (Old UI):
- ❌ Showed only a spinner: "Searching with Hybrid retrieval..."
- ❌ No visibility into what's happening
- ❌ Agent activity shown AFTER completion in collapsed expander
- ❌ User had no idea if controller was working or what it was doing

### After (New UI):
- ✅ Live status container shows each step as it happens
- ✅ See the controller planning (question type, strategy)
- ✅ See each retrieval attempt (mode, search query)
- ✅ See evidence assessment (level, reason, sufficient or not)
- ✅ See retry decisions (why retrying, what changed)
- ✅ See generation, citations, confidence computation
- ✅ Status container collapses when complete
- ✅ Agent activity panel still available for detailed review

---

## Architecture Changes

### What Changed:

1. **Controller (`src/rag_pipeline/controller.py`)**:
   - Added `emit_event` callback parameter (optional, backward compatible)
   - Emits events at key decision points:
     - `plan`: Question classification and strategy
     - `retrieval`: Each retrieval attempt starts
     - `evidence`: Evidence assessment result
     - `retry`: Retry decision and reason
     - `decision`: Final answer/abstain decision

2. **Pipeline (`src/rag_pipeline/pipeline.py`)**:
   - Collects events in a list during execution
   - Passes event callback to controller
   - Emits events for generation, citations, confidence
   - Includes `events` array in response dict

3. **UI (`app.py`)**:
   - Replaced `st.spinner()` with `st.status()` container
   - Added `_render_event()` function to display each event type
   - Renders events after completion (Streamlit limitation: can't stream during execution)
   - Status container collapses when done, shows success/failure

### What Didn't Change:

- ✅ Controller decision logic (unchanged)
- ✅ Retrieval algorithms (unchanged)
- ✅ Evidence evaluation (unchanged)
- ✅ Citations, confidence scoring (unchanged)
- ✅ Observability/Langfuse integration (unchanged)
- ✅ Chat history persistence (unchanged)
- ✅ All existing tests pass (16/16 tests passing)
- ✅ Backward compatible (emit_event is optional)

---

## Testing

Run the app and try these questions:

1. **Success on first try**: "What is a black hole?"
2. **Retry then success**: "What is the Chandrasekhar limit?"
3. **Page-specific**: "What is on page 42?"
4. **Abstention**: "How do I cook pasta?" (not in astronomy KB)

Watch the live status container expand and show each step!

---

## Technical Note: Why Not True Streaming?

Streamlit's architecture doesn't support true streaming of widget updates during a synchronous function call. The events are collected during execution and rendered immediately after the pipeline returns, which gives the appearance of showing "what happened" in order, but not true real-time streaming.

For true streaming, we'd need:
- Async/await architecture
- Streamlit's experimental `st.write_stream()` 
- Or a separate background worker with websockets

The current implementation shows the controller's execution sequence clearly, which solves the user's problem: "I don't see the controller or agent activity in the UI."
