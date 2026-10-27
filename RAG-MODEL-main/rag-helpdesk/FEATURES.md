# RAG Helpdesk — Feature & Page Documentation

> A complete guide to every page, what it does, and how it connects to the RAG (Retrieval-Augmented Generation) pipeline underneath.

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. Instead of relying on an LLM's baked-in training data, the system:

1. **Ingests** your documents → splits them into chunks → stores them as vectors (numbers the model can compare mathematically)
2. When you ask a question, **retrieves** the most relevant chunks from that vector store
3. **Generates** an answer by feeding those chunks as context to the LLM

This means answers are grounded in *your* documents — not hallucinated from memory.

---

## Navigation Overview

```
Database & History     → Chat History, Database
Agentic RAG Control    → Agentic RAG Control  (main Q&A page)
Retrieval              → Retrieval Lab
Knowledge Base         → Ingestion Pipeline
Evaluation             → Observation & Evaluation, Evaluation Dashboard, Query Comparison
System                 → System Architecture
```

---

## Pages

---

### 1. Agentic RAG Control

**What it is:** The main question-answering interface.

**What you do here:**
- Type a question in the composer at the bottom
- Hit the → button or press Enter
- Get a grounded answer with sources, citations, and confidence score

**How RAG connects:**
When you submit a question, the full pipeline runs:
```
Your query
  → Query preparation (normalize, extract aliases)
  → Embedding (convert query to a vector)
  → Dense retrieval (find nearest vectors in Chroma)
  → BM25 retrieval (keyword match)
  → RRF fusion (combine and re-rank both lists)
  → Evidence gate (is the evidence strong enough to answer?)
  → Context selection (pick best chunks within token budget)
  → LLM generation (Gemini reads chunks + question → writes answer)
  → Citation verification (claims checked against source chunks)
  → Answer displayed with sources
```

**Retrieval mode selector** — you can switch between three modes:
| Mode | What it uses | Best for |
|------|-------------|----------|
| Vector / Dense | Chroma vector similarity only | Semantic / conceptual questions |
| BM25 | Keyword matching only | Exact terms, names, titles |
| Hybrid (default) | Both + RRF fusion | Best general accuracy |

**Answer extras shown below each reply:**
- Retrieval mode chip (e.g. "Hybrid — Dense + BM25 + RRF")
- Evidence strength badge (None / Weak / Moderate / Strong)
- Evidence-Backed Confidence score (0–100)
- Sources list (book, chapter, pages)
- Citation coverage (what % of LLM claims are grounded in retrieved text)

**Page/Chapter queries:**
Ask things like *"What's on page 60?"* or *"Summarize Chapter 3"* — the system detects this, runs a targeted page/chapter filter against chunk metadata, and returns that specific content.

**System Diagnostics expander:**
Shows live stats: Chroma collection name, total vectors, embedding dimension, BM25 status, embedding model, KB version, index consistency.

---

### 2. Chat History

**What it is:** A persistent log of every question asked and every answer given.

**Features:**
- Shows Q&A pairs in reverse-chronological order (newest first)
- Each entry is an expander showing the full question, full answer, and answer metadata
- Search bar to filter questions/answers by keyword
- Per-message delete button (🗑️) — removes just that Q&A pair, not the whole history
- "Clear All History" button at the top

**How it works technically:**
History is saved to `vectorstore/chat_history.json`. Each entry carries a timestamp, the message role (user / assistant), the full content, and the metadata dict from that query. The file persists across app restarts.

**Why this helps understand RAG:**
You can go back and compare how different questions were answered, notice patterns in which retrieval mode produced better results, and see exactly what sources were cited for each answer.

---

### 3. Database (Index & Chunk Inspector)

**What it is:** A live inspector of everything stored in the vector database and BM25 index.

**Four tabs:**

#### Book Hierarchy & Explorer
- Shows all ingested documents organized by book → chapter → chunk
- Each book displays total chunks, chapters, and page range
- Drill down into individual chapters and read the exact chunk text
- Understand how your document was actually split before it was indexed

#### Vector Chroma DB
- Shows every chunk stored in Chroma with: Chunk ID, Book, Chapter, Chapter Title, Section, Pages, Token count, Character count, Preview text
- Filterable by book and chapter
- Expandable to see full chunk text + all metadata fields
- Live metrics: total indexed chunks, embedding dimension, collection name

**Why this matters for RAG:** Retrieval quality is only as good as the chunks. This tab lets you verify that chapters are split correctly, pages are labeled, and no content is missing.

#### BM25 Lexical Index
- Shows all chunks indexed for keyword search
- Includes BM25-specific stats: vocabulary size, total terms, average document length
- Token frequency data for inspecting what keywords are searchable

**Why this matters:** BM25 is your keyword fallback. If the vector search misses something, BM25 catches it through exact term overlap. Knowing what's in the BM25 index helps explain why some queries retrieve better than others.

#### Document Files
- Lists all raw source files in `knowledge_base/`
- Shows file name, size, type (PDF / TXT / DOCX / MD)
- Quick check that your source documents are actually present and readable

---

### 4. Retrieval Lab

**What it is:** A step-by-step breakdown of exactly what happened during the most recent query. Read-only — no retrieval runs here.

**Sections rendered for each query:**

#### Section A — Query
- The original query text as submitted
- Prepared/normalized version (what was actually sent to the retrievers)
- Whether query normalization or alias expansion happened
- BM25 token list (how the query was tokenized for keyword search)
- Multi-intent detection flag (did the system detect multiple sub-questions?)
- Greeting detection flag

#### Section B — Retrieval Mode
- Which mode ran (Vector / BM25 / Hybrid)
- Dense / BM25 / RRF enabled flags
- Counts: candidates found, fused count, dense results, BM25 results
- Query embedding dimension, top-k settings

#### Retrieval Flow Diagram
- A visual node graph showing which components actually fired:
  - Query → Query Preparation → Dense Search / BM25 Search → RRF Fusion → Evidence Gate → Context Selection → Generation → Answer

#### Section C — Dense Retrieval Results
- Table of all chunks returned by vector search
- Columns: rank, filename, chapter, pages, cosine distance, relevance score, text preview

#### Section D — BM25 Results
- Table of all chunks returned by keyword search
- Rank, filename, chapter, BM25 score, text preview

#### Section E — RRF Fusion
- How the two ranked lists were combined using Reciprocal Rank Fusion
- RRF score formula: `1 / (k + rank)` where k=60
- Final merged ranking after fusion

#### Section F — Evidence Gate
- Evidence level: None / Weak / Moderate / Strong
- Top similarity score from retrieved chunks
- Whether the system decided to answer or abstain
- Reason for the decision

#### Section G — Retrieval Breakdown
- All candidate chunks that made it through the evidence gate
- Annotated with whether they ended up in the LLM context or were cut

#### Section H — Context Selection (Token Optimizer)
- How many chunks went in vs came out
- Input chars / output chars
- Which chunks were removed and why (token budget, source diversity cap)

#### Section I — Generation
- LLM model used
- Number of messages in the prompt, context length
- Response preview
- Generation status (success / error)

#### Section J — Grounding Chain
- Post-generation audit: each claim in the answer is matched back to the evidence chunk it came from
- Columns: Query → Evidence chunk → LLM claim → Citation marker → Verdict

#### Raw Trace (expandable)
- Full JSON of the entire trace, sanitized (secrets stripped)

**Why this page is the core RAG learning tool:**
Every black-box step of RAG is made visible here. You can see *why* an answer was good or bad — was it a retrieval miss? Evidence too weak? Context cut too aggressively? LLM rewording? This page answers all of that.

---

### 5. Ingestion Pipeline

**What it is:** Shows how documents enter the system, with a live status board, educational simulation, and real ingestion controls.

**Three tabs:**

#### Ingestion Architecture
- Visual diagram of the ingestion flow:
  - Raw document → Document Loader → Chunker → Embedder → Chroma vector store + BM25 index
- Shows the real config values (chunk target tokens, max tokens, embedding model)

#### Educational Simulation
- Step-by-step animated walkthrough of how one document is processed
- Each stage lights up sequentially: Load → Parse → OCR → Chunk → Embed → Index
- Purely illustrative — no real documents are processed

#### Real Ingestion
- Upload a PDF, TXT, DOCX, or MD file
- Triggers real incremental ingestion — the file is chunked and embedded
- Progress bar + ingestion health panel after completion:
  - Total pages processed
  - OCR pages (scanned/image pages)
  - Chunks created
  - Skipped chunks (empty or too short)
  - Ingestion time in ms

**Manage Knowledge Base section:**
- See all files currently in `knowledge_base/`
- Delete a file — stale chunks are automatically purged from Chroma and BM25 on the next status check

**Why this matters for understanding RAG:**
Ingestion is where retrieval quality is determined. Bad chunking → bad retrieval → bad answers. This page lets you see your chunk strategy (token sizes) and verify each document was processed correctly.

---

### 6. Observation & Evaluation

**What it is:** An analytics dashboard over all recorded queries. Every query you run is automatically saved as an "observation" with scores, latency, and metadata.

**Six tabs:**

#### Overview
- Total queries run
- Median end-to-end latency
- Average retrieval strength score
- Latency bar chart (one bar per query)
- Retrieval mode distribution (how often each mode was used)
- Top 5 slowest queries table

#### DeepEval Metrics
- Per-query scores across LLM-judge metrics:
  - Faithfulness (does the answer stick to the retrieved evidence?)
  - Answer Relevancy (is the answer actually relevant to the question?)
  - Contextual Precision / Recall / Relevancy
  - Answer Correctness
  - Groundedness, Citation Correctness/Completeness
  - Noise Sensitivity
- Select a query from the dropdown to drill into its scores

#### Citation Analysis
- Citation coverage per query (% of claims that have a grounded source)
- Which sources were cited most
- Unsupported claims list

#### Langfuse Traces
- If Langfuse tracing is enabled, links to the full distributed trace on Langfuse cloud
- Useful for latency waterfall analysis across each pipeline stage

#### Automatic Observations
- Heuristic scores that don't require an LLM judge:
  - Retrieval health (did strong evidence come back?)
  - Retriever agreement (did Dense and BM25 agree on the top chunks?)
  - Direct answer support (did the answer reference the evidence?)

#### Query Drill-Down
- Select any past query
- See its full observation record: retrieval mode, evidence level, all scores, timestamps

---

### 7. Evaluation Dashboard

**What it is:** A compact aggregate table of all observations.

Shows one row per query with: query text, latency, retrieval health score, citation coverage. Quick overview of system health at a glance.

---

### 8. Query Comparison

**What it is:** Side-by-side comparison of two specific queries from your history.

**How it works:**
- Select "First query" and "Second query" from dropdowns
- A table shows every metric for both queries side by side:
  - Latency, Retrieval Strength, Retriever Agreement, Evidence Support, Grounding score, Citation Coverage
- "What changed?" section highlights every metric that differs between the two

**Why this matters:**
This is how you validate that switching retrieval modes, adding new documents, or changing chunk sizes actually improves results. Compare the same question asked with Vector vs Hybrid, or before/after adding a document.

---

### 9. System Architecture

**What it is:** A static visual diagram of every component in the deployed system, labeled with live config values.

**Three planes shown:**

#### Knowledge Plane — Ingestion
Documents → Document Loader → Chunker (with real token sizes) → Embedder (shows actual model) → Chroma vector store + BM25 index (shows real chunk count)

#### Query Plane — Retrieval
User Query → Query Preparation → Dense Retrieval (Chroma) + BM25 Retrieval → RRF Fusion → Evidence Gate → Context Selection (Token Optimizer) → LLM Generation → Answer with Citations

#### Evaluation Plane
Answer → Citation Verifier → Confidence Scorer → DeepEval Metrics → Observation Store

**Why this matters:**
Most RAG tutorials show a simplified two-box diagram (retrieve → generate). This page shows the real complexity: dual retrieval paths, fusion, evidence gating, token budgeting, citation verification, confidence scoring, and evaluation — all as a real system.

---

## Core RAG Concepts Explained via This App

| Concept | Where to see it |
|---------|----------------|
| What chunks look like | Database → Chroma DB tab |
| How chunking works | Ingestion Pipeline → Architecture tab |
| Dense vs keyword retrieval | Retrieval Lab → Sections C and D |
| What RRF fusion does | Retrieval Lab → Section E |
| Why some queries get no answer | Retrieval Lab → Section F (Evidence Gate) |
| How context is trimmed before LLM | Retrieval Lab → Section H |
| How citations are verified | Retrieval Lab → Section J |
| How quality is measured | Observation & Evaluation → DeepEval tab |
| Comparing retrieval modes | Query Comparison page |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `BAAI/bge-small-en-v1.5` (local, 384-dim) |
| Vector Store | Chroma (SQLite backend) |
| Keyword Search | BM25 (custom implementation) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| LLM | Gemini (via OpenAI-compatible endpoint) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| UI | Streamlit |
| Tracing | Langfuse |
| Evaluation | DeepEval |
