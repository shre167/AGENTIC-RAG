# Current RAG Architecture Analysis

## Executive Summary

**KEY FINDING:** The controller DOES execute and control retrieval, BUT the UI does NOT show live agent activity. The `agent_activity` data is built DURING execution by the controller, but it is only rendered AFTER the complete response is assembled. There are NO live events or streaming progress indicators during question processing.

The architecture is hybrid: the controller drives retrieval (planning, attempting, retrying, evaluating), but the pipeline still drives generation, citations, confidence scoring, and response assembly. The UI renders agent activity as a static post-execution summary.

---

## 1. Entry Point

**File:** `app.py`

**Function:** Inline code in the main page rendering (around line 1470-1600)

**Line:** 1552

**What happens:**
1. User types question in `st.text_input` (line ~1640)
2. User presses Enter or clicks "→" button (line ~1665)
3. Streamlit form submission triggers
4. The question is displayed in chat (line ~1545)
5. `pipeline.answer_question(question, retrieval_mode=selected_mode)` is called (line 1552)
6. After the complete response returns, the answer is displayed (line ~1575)
7. `_render_answer_extras(meta)` is called to show agent activity panel (line ~1577)

**Critical gap:** The entire question-answering process happens inside the `st.spinner()` context. The user sees only "Searching with [mode] retrieval..." until the complete response is ready. No live progress.

---

## 2. Current Execution Flow

### Step-by-Step Trace

#### 1. **User enters question** (app.py:~1640)
   - Handler: Streamlit form with `st.text_input` and `st.form_submit_button`
   - What it does: Captures user input and triggers form submission on Enter or button click

#### 2. **Pipeline.answer_question() called** (app.py:1552)
   - What it does: Entry point wrapper that adds observability and records the observation
   - Delegates to: `_answer_question_impl(question, retrieval_mode)`
   - After completion: Calls `observe_pipeline_answer()` and `record_observation()`
   - Returns complete response dict with all metadata

#### 3. **Pipeline._answer_question_impl() called** (pipeline.py:1452)
   The main orchestrator. Key stages:

   - **Validation** (line 1527): Check if question is empty
   - **Greeting check** (line 1538): Detect greetings using `is_greeting()`
   - **Multi-intent** (line 1546): Split multi-part questions
   - **KB validation** (line 1553): Ensure knowledge base exists
   - **Vector store check** (line 1564): Get collection count
   - **Query preparation** (line 1579): Call `prepare_search_query()`
   - **CONTROLLER CALL** (line 1642): `AgentController(self).run_retrieval_phase(question, retrieval_mode)`
   - **Extract results** (line 1644-1688): Unpack controller's AgentRetrievalPhase object
   - **Fallback handling** (line 1691): If no candidates, try keyword fallback
   - **Evidence re-check** (line 1762): Re-evaluate if needed
   - **Runtime evaluation** (line 1773): Call `evaluate_retrieval()` for 7 criteria
   - **Abstention path** (line 1780): Return early if `evidence.should_answer` is False
   - **Token optimization** (line 1876): Call `optimize_context()` to trim candidates
   - **Context building** (line 1907): Call `_build_context()` to format prompt
   - **Generation** (line 1918): Call `_ask_llm()` with question + context
   - **Citations** (line 1933): Call `generate_claim_citations()`
   - **Confidence** (line 1956): Call `evaluate_confidence()`
   - **Response assembly** (line 1972): Build final dict with answer, sources, trace, agent_activity

#### 4. **AgentController.run_retrieval_phase() executes** (controller.py:228)
   What it does:
   - **Line 229:** Extract page_info from question using `extract_page_query_info()`
   - **Line 230:** Extract chapter_info using `extract_chapter_query_info()`
   - **Line 231:** Classify question type: page_specific, chapter_specific, multi_intent, or general
   - **Line 232-237:** Build retrieval plan string describing the strategy
   - **Line 248:** Initialize empty AgentRetrievalPhase result container
   - **Line 250-253:** Enter retry loop (MAX_RETRIEVAL_ATTEMPTS = 2)
   - **Line 254-266:** For attempt 0: use initial mode and filters; for attempt 1+: call `_retry_mode_and_filters()`
   - **Line 268:** **INVOKE TOOL:** Call `execute_retrieval_attempt()` (the actual retrieval)
   - **Line 270-285:** Check if fingerprint is duplicate; if so, skip and record
   - **Line 287:** Add fingerprint to seen set
   - **Line 289:** Call `pipeline._rerank()` on fused results
   - **Line 290-296:** Call `pipeline._evaluate_evidence()` to get final_candidates and Evidence
   - **Line 298-311:** **Page-specific guard:** If user asked for specific page and it wasn't found, force abstention
   - **Line 313:** Determine if evidence is sufficient
   - **Line 315-329:** Build AgentAttemptRecord with all metrics and decisions
   - **Line 330-331:** Append record to phase.attempts list
   - **Line 333-340:** If sufficient, store as `best` and break; else continue
   - **Line 342-352:** If at max attempts, stop
   - **Line 354-390:** After loop: determine final_decision ("answer" or "abstain"), select best attempt, populate phase object
   - **Line 392-399:** Build timings_ms dict
   - **Line 401-431:** Build agent_activity dict from attempts list
   - **Line 433:** Return AgentRetrievalPhase

   **What controller returns:**
   An `AgentRetrievalPhase` object containing:
   - `question_type`, `retrieval_plan`, `final_decision`
   - `attempts[]`: list of AgentAttemptRecord objects (one per retrieval attempt)
   - `agent_activity{}`: dict representation of all attempts and decisions
   - `attempt_result`, `fused`, `final_candidates`, `evidence`: the best retrieval payload
   - `timings_ms{}`

   **Whether pipeline uses the results:** YES

   In `pipeline.py` lines 1644-1688, the pipeline extracts:
   - `agent_activity` dict (line 1647) → included in final response
   - `attempt_result` (line 1651) → used to get candidates, embeddings, raw results
   - `candidates`, `query_embedding`, `bm25_query_tokens` (lines 1656-1662)
   - `raw_dense_results`, `raw_bm25_results`, `rrf_results_snapshot`, `rrf_calculations` (lines 1663-1670)
   - `fused`, `reranked`, `final_candidates`, `evidence` (lines 1671-1674)

### Complete Flow Diagram

```
User Question
    ↓
app.py:1640 — st.text_input / form submission
    ↓
app.py:1552 — pipeline.answer_question(question, retrieval_mode)
    ↓
pipeline.py:558 — answer_question() wrapper (adds observability)
    ↓
pipeline.py:1452 — _answer_question_impl(question, retrieval_mode)
    ↓
pipeline.py:1527 — Validate question
    ↓
pipeline.py:1538 — Check if greeting
    ↓
pipeline.py:1546 — Check if multi-intent (may split and recurse)
    ↓
pipeline.py:1553 — Validate knowledge base exists
    ↓
pipeline.py:1564 — Check vector store collection count
    ↓
pipeline.py:1579 — prepare_search_query(question) → search_text
    ↓
pipeline.py:1642 — AgentController(self).run_retrieval_phase(question, retrieval_mode)
    ↓
    controller.py:229-433 — CONTROLLER EXECUTION
        ↓
        extract_page_query_info() and extract_chapter_query_info()
        ↓
        classify question type
        ↓
        build retrieval plan
        ↓
        FOR attempt in range(MAX_RETRIEVAL_ATTEMPTS):
            ↓
            determine mode and filters (initial or retry)
            ↓
            execute_retrieval_attempt() — retrieval_runner.py:98
                ↓
                Dense retrieval (if mode is vector or hybrid)
                ↓
                BM25 retrieval (if mode is bm25 or hybrid)
                ↓
                RRF fusion (if mode is hybrid)
                ↓
                Return RetrievalAttemptResult with candidates, fused, metrics
            ↓
            Check fingerprint for duplicate
            ↓
            pipeline._rerank(fused)
            ↓
            pipeline._evaluate_evidence(fused, reranked, search_text) → final_candidates, Evidence
            ↓
            Page-specific guard (force abstention if page not found)
            ↓
            Record AgentAttemptRecord
            ↓
            IF evidence sufficient: store as best and BREAK
            ↓
            ELSE: continue to next attempt
        ↓
        Build agent_activity dict
        ↓
        Return AgentRetrievalPhase
    ↓
pipeline.py:1647 — Extract agent_activity dict
pipeline.py:1651-1674 — Extract attempt_result, candidates, fused, evidence
    ↓
pipeline.py:1691 — IF no candidates: try keyword fallback
    ↓
pipeline.py:1762 — IF evidence is None: re-evaluate
    ↓
pipeline.py:1773 — evaluate_retrieval() for 7 criteria
    ↓
pipeline.py:1780 — IF not evidence.should_answer: return abstention response with agent_activity
    ↓
pipeline.py:1876 — optimize_context(final_candidates) → trim to token budget
    ↓
pipeline.py:1907 — build_context(final_candidates) → context string
    ↓
pipeline.py:1918 — ask_llm(question, context) → raw_answer
    ↓
pipeline.py:1933 — generate_claim_citations(raw_answer, final_candidates) → citations, citation_coverage
    ↓
pipeline.py:1956 — evaluate_confidence(evaluation, citations, coverage, evidence_level) → confidence_result
    ↓
pipeline.py:1972 — Assemble final response dict with:
    - answer, raw_answer, sources, retrieved_chunks
    - evidence, evaluation, citations, citation_coverage, confidence
    - kb_state, response_type, retrieval_mode
    - agent_activity ← from controller
    - trace{} with full execution details
    ↓
pipeline.py:558 — answer_question() adds observation metadata
    ↓
app.py:1575 — Display answer in chat
    ↓
app.py:1577 — _render_answer_extras(meta) → render agent_activity panel
    ↓
app.py:901 — _render_agent_activity(agent_activity)
    - Renders expandable "🤖 Agent Activity" panel
    - Shows question_type, final_decision, attempt_count, retrieval_mode
    - For each attempt: mode, tools, search text, evidence level, reason, retry info
    - Shows timings_ms
    ↓
User sees complete answer + agent activity panel
```

---

## 3. Controller Analysis

### Current Role

**The controller DOES drive retrieval.**

Evidence:
- The controller is responsible for planning the retrieval strategy (page 232-237 in controller.py)
- The controller invokes `execute_retrieval_attempt()` which performs the actual dense/BM25/hybrid retrieval (line 268)
- The controller evaluates evidence after each attempt (line 290-296)
- The controller makes retry decisions based on evidence sufficiency (line 333-340)
- The controller determines the final decision: "answer" or "abstain" (line 356-372)

**The controller DOES make decisions.**

Evidence:
- Question type classification (general, page_specific, chapter_specific, multi_intent)
- Retrieval mode selection for retries (hybrid → bm25, vector → hybrid, etc.)
- Retry logic based on evidence level and page match status
- Duplicate attempt detection via fingerprinting
- Final answer/abstain decision

**The controller's results ARE used by the pipeline.**

Evidence:
- `agent_activity` dict is included in the final response (pipeline.py:1647, 1972)
- `candidates`, `fused`, `final_candidates`, `evidence` are extracted and used (lines 1651-1674)
- The pipeline does NOT re-run retrieval; it uses the controller's results directly
- Abstention responses include agent_activity (line 1802, 1845)

### Key Finding

**The controller truly controls retrieval execution.** It is NOT a post-hoc reporter or simulator.

However:
1. The controller does NOT control generation (pipeline still calls `ask_llm()`)
2. The controller does NOT control citations (pipeline calls `generate_claim_citations()`)
3. The controller does NOT control confidence scoring (pipeline calls `evaluate_confidence()`)
4. The controller does NOT control context optimization (pipeline calls `optimize_context()`)

### Evidence

- **In pipeline.py:1642**, the controller is invoked: `AgentController(self).run_retrieval_phase(question, retrieval_mode)`
- **In pipeline.py:1647-1674**, the pipeline extracts and uses the controller's results
- **In pipeline.py:1918**, the pipeline (not controller) calls generation: `raw_answer = self._ask_llm(question, context, trace=generation_trace)`
- **The agent_activity dict is built DURING execution:** controller.py:401-431 constructs it from the attempts list
- **Events are NOT emitted live:** The agent_activity dict is constructed in memory and only returned at the end

---

## 4. Tool Inventory

### Existing Functions

| Function | File | Line | Purpose | Should Be Tool? | Currently Called By |
|----------|------|------|---------|----------------|---------------------|
| `execute_retrieval_attempt` | retrieval_runner.py | 98 | Execute dense/BM25/hybrid retrieval with filters | **YES** (already is) | Controller (line 268) |
| `dense_retrieve` | retrieval.py | N/A | Query vector store with embedding | **YES** | `execute_retrieval_attempt` |
| `bm25_retrieve` | retrieval.py | N/A | Query BM25 index with tokens | **YES** | `execute_retrieval_attempt` |
| `fuse_rrf` | retrieval.py | N/A | Reciprocal Rank Fusion of dense + BM25 | **YES** | `execute_retrieval_attempt` |
| `page_retrieve` | retrieval.py | N/A | Metadata lookup for specific pages | **YES** | `execute_retrieval_attempt` (line 123) |
| `chapter_retrieve` | retrieval.py | N/A | Metadata lookup for specific chapters | **YES** | `execute_retrieval_attempt` (line 130) |
| `evaluate_evidence` | evidence.py | N/A | Assess evidence quality and decide whether to answer | **YES** (already is) | Controller (line 290) |
| `rerank` | retrieval.py | N/A | CrossEncoder reranking (currently disabled) | **YES** | Pipeline via controller (line 289) |
| `optimize_context` | token_optimizer.py | N/A | Trim candidates to token budget | **YES** | Pipeline (line 1876) |
| `build_context` | context.py | N/A | Format candidates into LLM context string | **YES** | Pipeline (line 1907) |
| `ask_llm` | generation.py | N/A | Generate answer from question + context | **YES** (already is) | Pipeline (line 1918) |
| `generate_claim_citations` | citations.py | N/A | Verify claims against evidence | **YES** | Pipeline (line 1933) |
| `evaluate_confidence` | confidence.py | N/A | Compute evidence-backed confidence score | **YES** | Pipeline (line 1956) |
| `prepare_search_query` | query.py | N/A | Normalize and expand query text | **YES** | Pipeline (line 1579) |
| `extract_page_query_info` | query.py | N/A | Detect page-specific questions | **YES** | Controller (line 229) |
| `extract_chapter_query_info` | query.py | N/A | Detect chapter-specific questions | **YES** | Controller (line 230) |
| `is_greeting` | query.py | N/A | Detect greeting messages | No (simple classifier) | Pipeline (line 1538) |
| `is_multi_intent_query` | query.py | N/A | Detect multi-part questions | No (simple classifier) | Pipeline (line 1546) |

### Functions Already Acting as Tools

- `execute_retrieval_attempt` ✓
- `evaluate_evidence` ✓
- `ask_llm` ✓

### Functions That Should Be Tools But Aren't

- `optimize_context` — currently called directly by pipeline
- `build_context` — currently called directly by pipeline
- `generate_claim_citations` — currently called directly by pipeline
- `evaluate_confidence` — currently called directly by pipeline

---

## 5. UI Integration

### Current Agent Activity Display

**Where rendered:** app.py:901

**Function:** `_render_agent_activity(agent_activity: dict)`

**Data source:** `response['agent_activity']` (from controller.py:401-431)

**Built:** DURING execution (controller builds it in memory while executing)

**Live updates:** NO

The UI rendering sequence:
1. User submits question (app.py:~1640)
2. `st.spinner("Searching with [mode] retrieval...")` displays (app.py:1544)
3. Entire pipeline executes synchronously inside spinner context (app.py:1552)
4. After response returns, spinner disappears
5. Answer is displayed (app.py:1575)
6. `_render_answer_extras(meta)` is called (app.py:1577)
7. Inside extras, `_render_agent_activity()` is called (app.py:901)
8. Agent activity panel renders as expandable section

**The panel shows:**
- Question type (general, page_specific, chapter_specific, multi_intent)
- Final decision (answer ✅ or abstain ✗)
- Attempt count (1 or 2)
- Active retrieval mode
- Retrieval plan summary
- For each attempt:
  - Mode, tools called, search text
  - Candidate count, fused count, latency
  - Evidence level, sufficiency, reason
  - Retry reason, changes from previous
  - Page/chapter match status
- Timings: controller_total_ms, last_retrieval_ms

### Gap Analysis

**Events emitted during execution:** NO

The agent_activity dict is built in memory during controller execution, but it is not streamed or emitted as events. The UI has no way to observe progress until the complete response returns.

**UI shows live progress:** NO

The spinner shows only generic text: "Searching with [mode] retrieval..." There are no intermediate status updates like:
- "Planning retrieval strategy..."
- "Executing vector search..."
- "Evaluating evidence..."
- "Insufficient evidence, retrying with BM25..."
- "Generating answer..."

**What's missing:**
1. Event emission from controller during execution
2. Streamlit status container or placeholder for live updates
3. Tool execution events (when each tool is called)
4. Evidence assessment events (when evaluation completes)
5. Retry events (when controller decides to retry)
6. Generation events (when LLM is called)

---

## 6. Proposed New Flow

### What Needs to Change

#### 1. **Controller changes:**

**Add event emitter to controller:**
- Pass a callback function to `run_retrieval_phase()` that emits events
- Emit events at key points:
  - "Planning retrieval for [question_type] question"
  - "Executing [mode] retrieval (attempt 1/2)"
  - "Retrieved [N] candidates, evaluating evidence..."
  - "Evidence level: [level] — [reason]"
  - "Retrying with [new_mode] — [reason]"
  - "Final decision: [answer/abstain]"

**Example signature:**
```python
def run_retrieval_phase(
    self,
    question: str,
    retrieval_mode: str,
    event_callback: Callable[[str, dict], None] | None = None,
) -> AgentRetrievalPhase:
    if event_callback:
        event_callback("plan", {"question_type": question_type, "plan": retrieval_plan})
    # ...
    for attempt_index in range(MAX_RETRIEVAL_ATTEMPTS):
        if event_callback:
            event_callback("retrieval_start", {"attempt": attempt_index + 1, "mode": mode})
        # ...
```

#### 2. **Pipeline changes:**

**Add event handling to _answer_question_impl:**
- Accept optional event_callback parameter
- Pass it to controller
- Emit events for generation, citations, confidence
- Example:
  ```python
  if event_callback:
      event_callback("generation_start", {})
  raw_answer = self._ask_llm(question, context)
  if event_callback:
      event_callback("generation_complete", {"tokens": len(raw_answer)})
  ```

**Expose event-aware answer_question variant:**
```python
def answer_question_streaming(
    self,
    question: str,
    retrieval_mode: str = "hybrid",
    event_callback: Callable[[str, dict], None] | None = None,
) -> dict[str, Any]:
    # Same as answer_question but passes event_callback through
```

#### 3. **UI changes:**

**Replace spinner with status container:**
```python
status_container = st.status("Processing question...", expanded=True)

def ui_event_handler(event_type: str, event_data: dict):
    if event_type == "plan":
        status_container.update(label=f"Planning: {event_data['question_type']}")
        status_container.write(f"Strategy: {event_data['plan']}")
    elif event_type == "retrieval_start":
        status_container.write(f"🔍 Attempt {event_data['attempt']}: {event_data['mode']} retrieval")
    elif event_type == "evidence_assessment":
        status_container.write(f"📊 Evidence: {event_data['level']} — {event_data['reason']}")
    # ...

response = pipeline.answer_question_streaming(
    question,
    retrieval_mode=selected_mode,
    event_callback=ui_event_handler,
)

status_container.update(label="Complete", state="complete", expanded=False)
```

### New Execution Flow Diagram

```
User Question
    ↓
UI: Create status container, set to "Processing..."
    ↓
UI: Call pipeline.answer_question_streaming(question, mode, event_callback=ui_handler)
    ↓
Pipeline: _answer_question_impl(question, mode, event_callback)
    ↓
Pipeline: Emit event("validate", {}) → UI updates: "Validating question"
    ↓
Pipeline: Emit event("prepare", {}) → UI updates: "Preparing search query"
    ↓
Pipeline: controller.run_retrieval_phase(question, mode, event_callback)
    ↓
Controller: Emit event("plan", {type, plan}) → UI updates: "Planning: page_specific"
    ↓
Controller: FOR attempt in [1, 2]:
    ↓
    Controller: Emit event("retrieval_start", {attempt, mode}) → UI: "🔍 Attempt 1: hybrid"
    ↓
    Controller: execute_retrieval_attempt()
        ↓
        Retrieval Runner: Emit event("tool_call", {"tool": "dense_retrieve"}) → UI: "  • dense_retrieve"
        ↓
        Retrieval Runner: Emit event("tool_call", {"tool": "bm25_retrieve"}) → UI: "  • bm25_retrieve"
        ↓
        Retrieval Runner: Emit event("tool_call", {"tool": "rrf_fusion"}) → UI: "  • rrf_fusion"
        ↓
        Return candidates
    ↓
    Controller: Emit event("evidence_start", {}) → UI: "📊 Evaluating evidence..."
    ↓
    Controller: evaluate_evidence()
    ↓
    Controller: Emit event("evidence_result", {level, reason}) → UI: "Evidence: weak — insufficient..."
    ↓
    Controller: IF insufficient and attempt < max:
        Emit event("retry_decision", {reason, changes}) → UI: "⚠️ Retrying with bm25..."
    ↓
    Controller: ELSE IF sufficient:
        Emit event("retrieval_complete", {decision: "answer"}) → UI: "✅ Sufficient evidence"
        BREAK
    ↓
Controller: Return AgentRetrievalPhase
    ↓
Pipeline: IF abstain:
    Emit event("abstain", {reason}) → UI: "❌ Abstaining — no evidence"
    Return response
    ↓
Pipeline: Emit event("generation_start", {}) → UI: "🤖 Generating answer..."
    ↓
Pipeline: ask_llm(question, context)
    ↓
Pipeline: Emit event("generation_complete", {tokens}) → UI: "✅ Generated 250 tokens"
    ↓
Pipeline: Emit event("citations_start", {}) → UI: "🔗 Verifying citations..."
    ↓
Pipeline: generate_claim_citations()
    ↓
Pipeline: Emit event("citations_complete", {supported, unsupported}) → UI: "✅ 5/5 claims supported"
    ↓
Pipeline: Emit event("confidence_assessment", {}) → UI: "📈 Computing confidence..."
    ↓
Pipeline: evaluate_confidence()
    ↓
Pipeline: Emit event("complete", {}) → UI: Status becomes "Complete" and collapses
    ↓
UI: Display answer
    ↓
UI: Display agent_activity panel (expandable, shows full details)
```

---

## 7. Functions to Refactor

### High Priority

#### 1. **controller.py:AgentController.run_retrieval_phase** (line 228)
   - **Current:** Builds agent_activity dict in memory, no events
   - **Change to:** Accept optional `event_callback` parameter, emit events at key points
   - **Reason:** Enable live UI updates during retrieval planning and execution

#### 2. **retrieval_runner.py:execute_retrieval_attempt** (line 98)
   - **Current:** Silent execution, no events
   - **Change to:** Accept optional `event_callback`, emit tool_call events
   - **Reason:** Show which tools (dense_retrieve, bm25_retrieve, rrf_fusion, page_retrieve) are executing

#### 3. **pipeline.py:_answer_question_impl** (line 1452)
   - **Current:** Calls controller synchronously, no events
   - **Change to:** Accept optional `event_callback`, pass to controller, emit events for generation/citations/confidence
   - **Reason:** Enable complete live execution trace in UI

#### 4. **pipeline.py:answer_question** (line 558)
   - **Current:** Wrapper with observability, not streaming-aware
   - **Change to:** Add `answer_question_streaming()` variant that accepts and passes event_callback
   - **Reason:** Public API for streaming execution events

#### 5. **app.py: mission control question handler** (line ~1540-1580)
   - **Current:** Uses `st.spinner()`, synchronous call
   - **Change to:** Use `st.status()` container, call `answer_question_streaming()`, handle events
   - **Reason:** Display live agent activity as it happens

### Medium Priority

#### 6. **generation.py:ask_llm** (estimated)
   - **Current:** Silent generation
   - **Change to:** Accept event_callback, emit start/complete events
   - **Reason:** Show when generation is happening

#### 7. **citations.py:generate_claim_citations** (estimated)
   - **Current:** Silent citation verification
   - **Change to:** Accept event_callback, emit progress events
   - **Reason:** Show citation verification progress

#### 8. **confidence.py:evaluate_confidence** (estimated)
   - **Current:** Silent confidence computation
   - **Change to:** Accept event_callback, emit result event
   - **Reason:** Show confidence assessment step

---

## 8. Risk Assessment

### Per Change

#### 1. **Add event_callback to controller**
   - **Risk:** LOW
   - **Reason:** Optional parameter, backward compatible, only affects execution when callback is provided
   - **Mitigation:** Make callback optional with default None, add null checks before calling

#### 2. **Add event_callback to retrieval_runner**
   - **Risk:** LOW
   - **Reason:** Optional parameter, called only by controller
   - **Mitigation:** Same as above

#### 3. **Add streaming variant to pipeline**
   - **Risk:** LOW
   - **Reason:** New function alongside existing one, doesn't change answer_question()
   - **Mitigation:** Keep answer_question() unchanged for backward compatibility, add answer_question_streaming() as new API

#### 4. **Change UI from spinner to status container**
   - **Risk:** MEDIUM
   - **Reason:** Changes user-facing behavior, requires Streamlit ≥1.39 for st.status()
   - **Mitigation:** Check Streamlit version, fall back to spinner if st.status() unavailable

#### 5. **Event callback threading/concurrency**
   - **Risk:** MEDIUM
   - **Reason:** If event_callback calls Streamlit functions from worker threads (e.g., concurrent dense+BM25), may cause errors
   - **Mitigation:** Document that event_callback must be thread-safe, or collect events in queue and flush from main thread

#### 6. **Observability (Langfuse) integration**
   - **Risk:** LOW
   - **Reason:** Observability layer wraps answer_question(), not _answer_question_impl()
   - **Mitigation:** Ensure observe_pipeline_answer() still works with streaming variant

#### 7. **Chat history persistence**
   - **Risk:** LOW
   - **Reason:** History is saved after response completes, not during streaming
   - **Mitigation:** No change needed

---

## 9. Backward Compatibility

### Observability (Langfuse)
**Impact:** NONE

The observability layer in `pipeline.py:558` wraps `answer_question()`, which delegates to `_answer_question_impl()`. Event callbacks are internal to `_answer_question_impl()` and don't affect the wrapper. Langfuse traces will continue to work.

### Multi-intent handling
**Impact:** NONE

Multi-intent splitting happens before controller invocation (line 1546), so event callbacks don't affect it.

### Chat history
**Impact:** NONE

Chat history is persisted after the complete response is assembled (app.py:1586-1593), so streaming events don't affect persistence.

### Evaluation scripts
**Impact:** NONE if we keep answer_question() unchanged

If we add `answer_question_streaming()` as a separate method, existing evaluation scripts that call `answer_question()` will continue to work without modification.

**Recommendation:** Keep `answer_question()` as-is, add `answer_question_streaming()` as a new method.

---

## 10. Key Insight

**The controller TRULY drives retrieval execution.** It is NOT a post-hoc reporter.

However, the system has a **visibility gap**, not an architecture gap:

1. ✅ The controller plans retrieval strategies
2. ✅ The controller invokes retrieval tools (`execute_retrieval_attempt`)
3. ✅ The controller evaluates evidence
4. ✅ The controller makes retry decisions
5. ✅ The controller determines answer/abstain
6. ✅ The pipeline uses the controller's results

**But:**

7. ❌ Events are NOT emitted during execution
8. ❌ The UI does NOT show live progress
9. ❌ Users see only a spinner until completion
10. ❌ Agent activity is shown AFTER execution, not DURING

**The fix is NOT to rebuild the controller or change its decision logic.** The controller is already making real decisions and driving execution.

**The fix IS to add event emission and live UI rendering:**
- Add `event_callback` parameter to controller, retrieval_runner, and pipeline
- Emit events at key execution points
- Replace `st.spinner()` with `st.status()` in the UI
- Render events as they arrive

**This is a 1-2 day refactor, not a rewrite.**

---

## 11. Verification Checklist

To verify the controller truly drives execution (not just reports on it):

✅ **Does the controller invoke retrieval?**
   - YES: controller.py:268 calls `execute_retrieval_attempt()`

✅ **Does the pipeline re-run retrieval after the controller returns?**
   - NO: pipeline.py:1651-1674 extracts controller results and uses them directly

✅ **Does the controller make decisions that change execution?**
   - YES: controller.py:333-340 breaks loop if sufficient, continues if not
   - YES: controller.py:254-266 selects retry mode and filters

✅ **Is agent_activity built during execution or after?**
   - DURING: controller.py:401-431 builds it from attempts list while executing

✅ **Does the final response include controller decisions?**
   - YES: pipeline.py:1647 extracts agent_activity, line 1972 includes it in response

✅ **Can the controller abstain?**
   - YES: controller.py:373 sets final_decision to "abstain"
   - YES: pipeline.py:1780-1867 returns early with abstention message

✅ **Does the UI display agent activity?**
   - YES: app.py:901 renders `_render_agent_activity()` expandable panel

❌ **Does the UI display LIVE agent activity?**
   - NO: Activity is displayed AFTER response completes
   - NO: There are no event emissions or streaming updates

---

## Conclusion

The current architecture is **functionally correct** but **observability-incomplete**.

- The controller drives retrieval ✓
- The controller makes decisions ✓
- The pipeline uses controller results ✓
- Agent activity is built during execution ✓
- Agent activity is displayed in the UI ✓

**Missing:** Live event emission and streaming UI updates.

**Solution:** Add event callbacks to controller/pipeline, replace spinner with status container in UI.

**Estimated effort:** 8-16 hours of implementation + testing.
