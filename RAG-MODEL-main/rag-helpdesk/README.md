# Astronomy RAG System: End-to-End Walkthrough

This guide explains what the application currently does, from adding a source
file to returning an answer. It is written as presentation notes and separates
the implemented behavior from optional features and limitations.

## 1. What the System Is

This is a retrieval-augmented generation (RAG) assistant. It does not ask the
language model to answer from memory alone. It first searches the project's
knowledge base, selects relevant passages, then asks the model to answer using
those passages and cite them.

The main pieces are:
- Streamlit interface: [app.py](app.py)
- Configuration and environment loading: [src/config.py](src/config.py)
- File extraction and chunking: [src/document_loader.py](src/document_loader.py)
- Embedding API/local model adapter: [src/embeddings.py](src/embeddings.py)
- Backend selector: [src/unified_vector_store.py](src/unified_vector_store.py)
- PostgreSQL implementation selected in this environment: [src/postgres_vector_store_simple.py](src/postgres_vector_store_simple.py)
- Lexical index: [src/bm25_retriever.py](src/bm25_retriever.py)
- Question answering orchestration: [src/rag_pipeline/pipeline.py](src/rag_pipeline/pipeline.py)
- Query and response inspection: [src/rag_pipeline/observatory.py](src/rag_pipeline/observatory.py)

## 2. Current Runtime Snapshot

These are the settings and counts checked in the local environment on
2026-10-09. Secrets are intentionally not included.

| Item | Current value |
|---|---|
| Vector backend | PostgreSQL (`USE_POSTGRES=true`) |
| PostgreSQL table | `public.chunks` |
| Rows in PostgreSQL | 890 chunks |
| Stored embedding dimension | 1024 |
| Embedding model | `amazon.titan-embed-text-v2:0` |
| Chat model | `openai.gpt-5-mini` |
| Image-analysis flag | Enabled; PDF ingestion only |
| BM25 | Separate persistent local index under `vectorstore/` |
| Cross-encoder reranker | Disabled by design |

The old Chroma files are still present locally. With PostgreSQL selected, they
are not the active dense vector backend. Older observation logs or older UI
screenshots may still say “Chroma”; those can describe an earlier run and are
not proof of the currently selected backend.

## 3. Adding and Ingesting a File

1. A source file is placed or uploaded into `knowledge_base/`. The loader
supports TXT, Markdown, DOCX, and PDF inputs.
2. `load_document()` or `load_documents()` detects the type and extracts text.
   PDFs are read page by page, preserving page numbers. DOCX headings and
   tables are extracted. PDF pages with weak text can use the configured OCR
   fallback when Tesseract is available.
3. Text is cleaned and organized into paragraphs and structure. The loader
   detects chapter and section headings where it can, filters common front
   matter/TOC noise, and carries file, page, chapter, section, and image
   information forward as metadata.
4. `chunk_documents()` groups extracted pages by source document, builds
   structure-aware chunks, splits oversized content, and merges certain tiny
   neighboring chunks when appropriate.
5. Each chunk receives a stable `chunk_id`, an order value `chunk_index`, and
   links to its previous and next chunk. The ingestion pipeline creates the
   stored ID from the filename and chunk ID, so the same unchanged file can be
   recognized on a later ingest.
6. An ingestion manifest tracks file hashes and chunk IDs. If a file hash has
   not changed, the file can be skipped. When a file changes, its old records
   are replaced and stale chunk IDs are removed.
7. The embedding service converts the chunk text to a numeric vector in
   batches. For the current configuration, the stored vectors have 1024
   numbers each.
8. New text, vectors, IDs, and metadata are written through
   `UnifiedVectorStore` to PostgreSQL. The same canonical chunks are also used
   to build or refresh BM25.

The command-line bulk ingestion entry point is `python ingest.py`. The UI's
single-file flow calls `RAGPipeline.ingest_file()`. Both use the pipeline and
the configured vector-store selector, but optional image analysis is wired in
the single-file ingestion method; do not assume every bulk-ingestion path
performs PDF image captioning without checking that path.

## 4. What One PostgreSQL Chunk Contains

Each row in `chunks` stores:

- `id`: primary key used to find or replace the chunk
- `document_text`: the searchable chunk text
- `embedding_json`: the numeric embedding serialized as JSON
- Common searchable columns such as filename, book title, chapter, section,
  page range, token count, and image flag
- `metadata_json`: the full original metadata, including `chunk_id`,
  `chunk_index`, `prev_chunk_id`, `next_chunk_id`, file type, and optional
  `image_description`

This implementation uses ordinary PostgreSQL; it does **not** use the
pgvector extension. For similarity search, it loads stored embeddings and
calculates cosine similarity in Python. That is straightforward and supports
the current embedding dimensions, but it does more work as the collection
grows than a database-side vector index would.

The migration script copies existing Chroma IDs, texts, metadata, and vectors
without regenerating embeddings and does not delete the Chroma source. The live
database count is currently 890; migration should be re-run only deliberately,
not as part of normal startup.

## 5. Chunking: The Important Setting Correction

The low-level chunker currently reads these defaults from
`src/document_loader.py`:

- Target chunk size: 350 tokens (`CHUNK_TARGET_TOKENS`)
- Maximum chunk size: 450 tokens (`CHUNK_MAX_TOKENS`)
- Minimum size used by small-chunk merging/statistics: 80 tokens (`CHUNK_MIN_TOKENS`)
- Overlap when splitting an oversized paragraph: 40 tokens
  (`OVERSIZED_PARAGRAPH_OVERLAP`)

Important: `src/config.py` also declares `CHUNK_SIZE=800` and
`CHUNK_OVERLAP=200`, and callers pass those values to `chunk_documents()`.
However, the current public `chunk_documents()` signature accepts extra
keyword arguments and does not pass them into `_chunk_book()`. The low-level
token targets above therefore control actual chunk creation today. Do not
describe 800/200 as the active chunking behavior. Wiring the settings through
is a separate code fix if those values are meant to control chunk sizes.

The chunk index is useful for ordering chunks from a chunking run. It is not a
permanent database-wide row number: it can restart when files are chunked in
separate ingestion calls. Use `id` or `chunk_id` when you need a unique,
stable identifier. Chapter totals are counts grouped by book and chapter;
they are not the same thing as chunk indices.

## 6. Optional PDF Image Understanding

When image analysis is enabled and an ingested PDF page contains embedded
images:

1. PyMuPDF extracts up to three usable images from that page.
2. The configured vision-capable chat model receives each image with a short
   factual description prompt and optional chapter/section context.
3. A successful caption is stored as `image_description` metadata and appended
   to the text as `[Visual content on this page: ...]`.
4. Dense and BM25 indexing then sees the caption as searchable text; retrieved
   context can include it, allowing answers to use information from a diagram
   or chart even when that information was not present in extracted text.

This is not live visual question-answering over the original PDF at chat time.
It is an ingestion-time conversion of selected PDF visuals into text. It is
optional and failures are logged while text ingestion continues. It does not
run for TXT, Markdown, or DOCX, and scanned-page OCR is a separate feature from
image captioning.

## 7. Asking a Question

The normal question path in `RAGPipeline._answer_question_impl()` is:

1. **Validate input.** Empty questions get a prompt; simple greetings get a
   direct greeting and do not search the knowledge base.
2. **Split some multi-part questions.** A conservative rule-based check may
   split a question on conjunctions or multiple question marks and answer the
   parts separately.
3. **Prepare the query.** Text is normalized, aliases may be expanded, and
   the original question is retained for the final answer.
4. **Choose retrieval mode.** The interface supports vector-only, BM25-only,
   and hybrid. Hybrid is the default when no mode is supplied.
5. **Dense retrieval.** In vector or hybrid mode, the question is embedded
   using the same configured embedding model, then compared to stored vectors.
   PostgreSQL's current store calculates cosine similarities in Python.
6. **BM25 retrieval.** In BM25 or hybrid mode, the question is tokenized and
   matched against the persisted lexical index. BM25 is useful for exact
   words, names, and identifiers.
7. **Merge candidate records.** Dense and BM25 results are matched by
   filename plus chunk ID so a chunk found by both methods remains one
   candidate with both retrieval signals.
8. **Fuse in hybrid mode.** Reciprocal Rank Fusion (RRF) combines ranks, not
   raw distances and BM25 scores. With `RRF_K=60`, each retriever contributes
   approximately `1 / (60 + rank)`. RRF scores are ranking values, not
   probabilities or confidence percentages.
9. **Check evidence.** The evidence gate considers the selected mode, rank
   signals, direct/lexical text support, score separation, and retriever
   agreement. If evidence is too weak, the pipeline abstains instead of
   generating a knowledge-base answer.
10. **Remove duplicates and control source balance.** Duplicate IDs/text are
    removed and one source is limited to at most three selected chunks.
11. **Limit context.** The current maximum is five chunks, with an 18,000
    character cap. This keeps the prompt bounded and avoids sending the whole
    database to the language model.
12. **Build source-labelled context.** The selected passages are labelled
    with source information. If image descriptions exist, they are included.
13. **Generate.** The configured chat model is instructed to answer only from
    the supplied context, say when it is insufficient, cite factual claims,
    and list only source filenames present in that context.
14. **Ground claims after generation.** The pipeline compares answer claims
    with selected evidence, records supported/partial/unsupported citations,
    and computes coverage and an evidence-backed confidence summary.
15. **Return and record.** The UI receives the answer, sources, citations,
    confidence, retrieval evaluation, and trace. Query observations are
    appended to local JSONL storage; optional Langfuse/evaluation integrations
    may add observability when enabled/configured.

## 8. What the User Sees

- **Chat/assistant view:** answer, source list, citations, and confidence
- **Database / Chunk Inspector:** live vector rows, book/chapter hierarchy,
  chunk text, metadata, and vector-store diagnostics
- **Ingestion view:** add or replace a source file and see ingest counts
- **BM25 diagnostics:** lexical index contents and synchronization information
- **Retrieval Lab / Observations:** query trace showing which candidates came
  from Dense, BM25, RRF, context selection, and answer generation

The PostgreSQL-backed explorer uses the vector store's `get()` and `delete()`
compatibility methods. Its labels now use the selected backend instead of
calling PostgreSQL “Chroma.”

## 9. Persistence and Synchronization

| Data | Where it persists | Role |
|---|---|---|
| Dense text, vectors, chunk metadata | PostgreSQL `chunks` table | Dense search and live chunk inspection |
| BM25 token/chunk index | `vectorstore/bm25_index.pkl` | Lexical search |
| Per-file hashes and chunk IDs | `vectorstore/ingestion_manifest.json` | Skip unchanged files and replace stale chunks |
| KB version/state | `vectorstore/kb_state.json` | Version and consistency status |
| Query traces/observations | `vectorstore/query_observations.jsonl` | Debugging and evaluation history |
| Uploaded/source documents | `knowledge_base/` | Authoritative source files for re-ingestion |

The vector database and BM25 are distinct indexes. A healthy system should
keep them synchronized to the same source chunks. The app has status and
consistency checks, but an old status/trace record is a snapshot from the time
it was written, not a live report.

## 10. What Is Not Currently Enabled or Guaranteed

- The cross-encoder reranker is intentionally disabled; ranking is Dense,
  BM25, and RRF, followed by the evidence gate and context selection.
- PostgreSQL is plain SQL storage with JSON embeddings, not pgvector. Cosine
  ranking happens in Python and will be the main scaling bottleneck.
- `CHUNK_SIZE=800` and `CHUNK_OVERLAP=200` are declared/passed but currently
  ignored by the public chunking function; the 350/450/80 defaults govern
  actual chunking.
- `chunk_index` should not be presented as a unique database-wide number.
  `chunk_id`/row `id` is the unique identifier.
- Image understanding is optional, PDF-only, and runs during supported
  ingestion. It creates text descriptions; it does not preserve image pixels
  for later multimodal retrieval.
- A retrieval score or a high confidence label is not a mathematical
  probability that every generated statement is true. The evidence and
  citation checks are safeguards, not a substitute for reviewing sources.

## 11. Short Presentation Script

> “This project is an astronomy knowledge assistant built with retrieval-
> augmented generation. When we add a document, the loader extracts and cleans
> its text, preserves page and section information, and divides the content
> into searchable chunks. Each chunk is embedded and stored with its metadata
> in PostgreSQL. A separate BM25 index handles exact keyword matches.
>
> When someone asks a question, the system prepares the query and searches the
> vector index, BM25, or both. In hybrid mode, it combines their rankings with
> Reciprocal Rank Fusion, checks whether the evidence is strong enough, and
> sends only a small, source-labelled set of chunks to the language model.
> The model is instructed to stay within that evidence. The system then checks
> claims against the selected passages and returns citations, sources, and an
> evidence-backed confidence summary.
>
> PostgreSQL currently contains 890 chunks with 1024-dimensional embeddings.
> The optional image feature captions images in PDF pages during ingestion,
> making visual information searchable as text. One detail we are correcting
> is the chunk-size configuration: the active chunker currently uses a 350
> token target and 450 token maximum; the separately declared 800/200 settings
> are not wired through yet.”

## 12. Useful One-Line Answers

- **What is RAG?** Search first, then generate an answer grounded in the
  selected source passages.
- **Why have both Dense and BM25?** Dense search finds semantic matches; BM25
  catches exact terminology and identifiers.
- **What does RRF do?** It combines the rank positions from Dense and BM25; it
  does not turn them into probabilities.
- **Why can a chunk be numbered 268?** It is its order in a chunking run, not
  “chapter 268” and not necessarily a permanent database-wide number.
- **Where are new chunks stored?** PostgreSQL when `USE_POSTGRES=true`; BM25
  is persisted separately as a local index.
- **What does image analysis add?** A text caption for supported PDF visuals,
  so retrieval can find facts conveyed by those visuals.
- **What happens when evidence is weak?** The evidence gate can return an
  abstention instead of asking the model to invent an answer.
