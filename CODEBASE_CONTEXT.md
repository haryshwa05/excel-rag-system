# Excel RAG — Complete Codebase Context Document

> **Purpose**: This document gives full context on the Excel RAG application — architecture, data flow, every file, design decisions, and known issues — so an LLM assistant can help debug, extend, or refactor it without needing to read the actual files.

---

## 1. What This App Does

A **Retrieval-Augmented Generation (RAG)** application that lets users upload Excel/CSV files and ask natural language questions about their data. The app parses the spreadsheet into chunks, embeds them into vectors, stores them in a vector database, and retrieves relevant chunks to answer questions via a streaming LLM.

**Example flow**:
1. User uploads `MOCK_DATA.csv` (1000 rows, columns: id, first_name, last_name, email, ip_address, device_type, app_usage, screen_time, etc.)
2. User asks: "How many people use laptops?"
3. System retrieves the pre-computed summary chunk (which has `device_type` value counts: `laptop=342, phone=329, tablet=329`) + relevant data chunks
4. LLM reads the context and responds: "342 people use laptops"

---

## 2. Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Frontend** | Next.js (App Router) | 16.2.1 |
| **React** | React | 19.2.4 |
| **Styling** | Tailwind CSS v4 | ^4 (via `@import "tailwindcss"` + `@theme`) |
| **Icons** | lucide-react | ^1.6.0 |
| **Backend** | FastAPI | >=0.109.0 |
| **ASGI Server** | uvicorn | >=0.27.0 |
| **Vector DB** | ChromaDB | >=0.4.22 (local persistent) |
| **Embeddings** | HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) | >=2.3.0 |
| **LLM** | Groq (free tier, `llama-3.3-70b-versatile`) | via OpenAI-compatible API |
| **Data Processing** | pandas + openpyxl | >=2.1.0 / >=3.1.0 |
| **Config** | Pydantic BaseSettings + `.env` | >=2.5.0 |
| **Language** | Python 3.11+ (backend), TypeScript 5 (frontend) | |

---

## 3. Directory Structure

```
excel-rag/
├── backend/
│   ├── .env                          # API keys, provider selection
│   ├── main.py                       # FastAPI app entry point
│   ├── requirements.txt              # Python dependencies
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                 # HTTP endpoints (upload, query, files, health)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                 # Pydantic Settings (all config in one place)
│   │   ├── embedder.py               # Embedding abstraction (OpenAI, HuggingFace, Local)
│   │   └── llm.py                    # LLM abstraction (OpenAI, Anthropic, Grok, Gemini, Groq, Local)
│   ├── db/
│   │   ├── __init__.py
│   │   └── chroma_client.py          # ChromaDB singleton client + collection helpers
│   ├── generation/
│   │   ├── __init__.py
│   │   └── generator.py              # Orchestrates search → context → LLM streaming
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py                 # File parsing + chunking + summary generation
│   │   └── pipeline.py               # Full ingestion pipeline with progress callbacks
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                # Pydantic request/response models
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── searcher.py               # Hybrid search (semantic + keyword) + result formatting
│   └── data/
│       ├── chroma/                   # ChromaDB persistent storage (gitignored)
│       └── uploads/                  # Uploaded files saved here
│
└── frontend/
    ├── package.json                  # npm dependencies
    ├── next.config.ts                # API proxy rewrites + React compiler
    ├── tsconfig.json                 # TypeScript config
    ├── postcss.config.mjs            # Tailwind PostCSS
    ├── eslint.config.mjs             # ESLint flat config
    ├── app/
    │   ├── globals.css               # Design tokens, animations, Tailwind @theme
    │   ├── layout.tsx                # Root layout (fonts, metadata)
    │   └── page.tsx                  # Main page: sidebar + chat + upload modal
    ├── components/
    │   ├── ChatBox.tsx               # Chat input + streaming message display
    │   ├── FileSidebar.tsx           # File list sidebar with delete
    │   ├── FileUploader.tsx          # Drag-and-drop upload with SSE progress
    │   └── MessageList.tsx           # Message bubbles (user + assistant)
    └── lib/
        └── api.ts                    # API client: upload, query, files, SSE reader
```

---

## 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            FRONTEND (Next.js 16)                        │
│                                                                         │
│  page.tsx ──→ FileSidebar.tsx    (file list, select, delete)            │
│           ──→ FileUploader.tsx   (drag-drop, SSE progress)              │
│           ──→ ChatBox.tsx        (input, streaming tokens)              │
│                  └→ MessageList.tsx (render user/assistant bubbles)      │
│                                                                         │
│  lib/api.ts ──→ fetch() + manual SSE parser ──→ /api/* (proxied)       │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │  Next.js rewrites /api/* → localhost:8000/api/*
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKEND (FastAPI)                                │
│                                                                         │
│  main.py ──→ CORS middleware + router mount (/api prefix)               │
│                                                                         │
│  api/routes.py                                                          │
│    POST /api/upload    → SSE progress stream                            │
│    POST /api/query     → SSE token stream                               │
│    POST /api/query/full→ JSON response                                  │
│    GET  /api/files     → file list                                      │
│    DELETE /api/files/:id → delete file                                  │
│    GET  /api/health    → health check                                   │
│                                                                         │
│  ┌──── INGESTION PIPELINE ────┐  ┌──── QUERY PIPELINE ────────────┐    │
│  │                            │  │                                 │    │
│  │  parser.py                 │  │  searcher.py                    │    │
│  │    parse_file()            │  │    search()                     │    │
│  │    _parse_csv()            │  │    _search_collection() [semantic]   │
│  │    _parse_excel()          │  │    _keyword_search()    [keyword]    │
│  │    _dataframe_to_chunks()  │  │    _merge_results()     [hybrid]    │
│  │    _build_summary_chunk()  │  │    format_results_as_context()  │    │
│  │         │                  │  │         │                       │    │
│  │         ▼                  │  │         ▼                       │    │
│  │  pipeline.py               │  │  generator.py                   │    │
│  │    ingest_file()           │  │    generate_answer()             │    │
│  │    (parse→embed→store)     │  │    (search→format→stream LLM)   │    │
│  └────────────────────────────┘  └─────────────────────────────────┘    │
│                                                                         │
│  ┌──── PROVIDERS ─────────────┐  ┌──── STORAGE ──────────────────┐     │
│  │  embedder.py               │  │  chroma_client.py              │     │
│  │    HuggingFaceEmbedder     │  │    get_chroma_client() [singleton]   │
│  │    OpenAIEmbedder          │  │    get_collection()            │     │
│  │    get_embedder() [factory]│  │    get_all_collections()       │     │
│  │                            │  │    delete_collection()         │     │
│  │  llm.py                    │  │                                │     │
│  │    GroqLLM (active)        │  │  ChromaDB (PersistentClient)   │     │
│  │    OpenAILLM               │  │    ./data/chroma/              │     │
│  │    AnthropicLLM            │  └────────────────────────────────┘     │
│  │    GrokLLM                 │                                         │
│  │    GeminiLLM               │  ┌──── CONFIG ───────────────────┐     │
│  │    get_llm() [factory]     │  │  config.py                    │     │
│  └────────────────────────────┘  │    Settings (Pydantic)        │     │
│                                  │    reads .env at startup       │     │
│                                  └────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Flow — File Upload

```
User drops file → FileUploader.tsx
    │
    ▼
fetch POST /api/upload (multipart form)
    │
    ▼ (Next.js proxy)
    │
routes.py: upload_file()
    ├── Validate file type (.csv/.xlsx/.xls) and size (<100MB)
    ├── Save to ./data/uploads/{file_id}_{filename}
    ├── Start SSE stream via asyncio Queue
    │
    ▼
pipeline.py: ingest_file()
    │
    ├── Step 1: PARSE (asyncio.to_thread → parser.py)
    │   ├── _parse_csv() or _parse_excel()
    │   │   └── _dataframe_to_chunks()
    │   │       ├── Read rows in pandas batches of 1000 (RAM management)
    │   │       ├── Split into RAG chunks of 20 rows each, 3-row overlap
    │   │       ├── Prepend column headers to each chunk
    │   │       └── Attach metadata: file_id, file_name, sheet, row_start, row_end
    │   │
    │   └── _build_summary_chunk()
    │       ├── Read entire DataFrame once
    │       ├── For each column:
    │       │   ├── If ≤50 unique values: ALL value counts (e.g., laptop=342, phone=329)
    │       │   ├── If >50 unique values: top 10 most common with counts
    │       │   └── If >50% numeric: min, max, mean, median
    │       ├── Add 3 sample rows
    │       └── Insert as first chunk with sheet_name="summary"
    │
    ├── Step 2: EMBED (batches of 100 chunks)
    │   └── embedder.py: HuggingFaceEmbedder.embed_texts()
    │       └── sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
    │       └── Runs in asyncio.to_thread to avoid blocking event loop
    │
    ├── Step 3: STORE in ChromaDB
    │   └── chroma_client.py: get_or_create_collection("file_{file_id}")
    │   └── collection.add(ids, embeddings, documents, metadatas)
    │
    └── Step 4: Return result via SSE → frontend shows completion
```

### Chunk Format Example

```
Columns: id | first_name | last_name | email | ip_address | device_type
Row 2: 1 | John | Smith | john@example.com | 192.168.1.1 | laptop
Row 3: 2 | Sarah | Jones | sarah@example.com | 10.0.0.1 | phone
... (20 rows per chunk)
```

### Summary Chunk Format Example

```
=== DATASET SUMMARY for MOCK_DATA.csv ===
Total rows: 1000
Total columns: 8
Column names: id | first_name | last_name | email | ip_address | device_type | app_usage | screen_time
Sheets: default

=== COLUMN STATISTICS (computed over ALL rows) ===

  Column: id
    Non-empty values: 1000/1000
    Unique values: 1000
    Top 10 values: 1=1, 2=1, 3=1, 4=1, 5=1, 6=1, 7=1, 8=1, 9=1, 10=1
    Numeric stats: min=1.00, max=1000.00, mean=500.50, median=500.50

  Column: device_type
    Non-empty values: 1000/1000
    Unique values: 3
    Value counts: laptop=342, phone=329, tablet=329

  Column: app_usage
    Non-empty values: 1000/1000
    Unique values: 5
    Value counts: social_media=215, gaming=210, productivity=198, streaming=195, education=182

=== SAMPLE ROWS (first 3) ===
Columns: id | first_name | last_name | email | ...
Row 2: 1 | John | Smith | john@example.com | ...
Row 3: 2 | Sarah | Jones | sarah@example.com | ...
Row 4: 3 | Mike | Brown | mike@example.com | ...
```

---

## 6. Data Flow — Query (Ask a Question)

```
User types question → ChatBox.tsx
    │
    ▼
queryStream() in lib/api.ts
    ├── fetch POST /api/query { question, file_id, chat_history }
    ├── Read SSE stream via ReadableStream
    ├── Parse each "data: {...}\n\n" event
    └── Call onToken(token) for each token → append to message bubble
    │
    ▼ (Next.js proxy → FastAPI)
    │
routes.py: query_streaming()
    │
    ▼
generator.py: generate_answer()
    │
    ├── Step 1: SEARCH (retrieval/searcher.py)
    │   │
    │   ├── Embed the question → 384-dim vector
    │   │
    │   ├── SEMANTIC SEARCH: collection.query(query_embeddings, n_results=5)
    │   │   └── ChromaDB compares query vector vs all chunk vectors
    │   │   └── Returns top-5 by cosine similarity
    │   │   └── Always injects the summary chunk if not already present
    │   │
    │   ├── KEYWORD SEARCH: _keyword_search()
    │   │   └── Fetch all non-summary docs from collection
    │   │   └── Case-insensitive substring match for extracted keywords
    │   │   └── Score: 0.8 + 0.05 per keyword matched (max 4)
    │   │   └── Return top-5
    │   │
    │   └── MERGE: _merge_results()
    │       └── Keyword results first (exact matches prioritized)
    │       └── Then semantic results (meaning-based)
    │       └── Deduplicate by first 100 chars of text
    │       └── Cap at top_k + 1 = 6 results (5 data + 1 summary)
    │
    ├── Step 2: FORMAT CONTEXT (searcher.py: format_results_as_context)
    │   └── Each result becomes:
    │       "--- Source: file.csv | Sheet: summary | Rows: 0-1000 (relevance: 1.00) ---"
    │       + chunk text
    │
    ├── Step 3: BUILD PROMPT
    │   └── System prompt (DEFAULT_RAG_PROMPT) with {context} injected
    │   └── Instructions: use summary for aggregates, chunks for specifics
    │   └── User message = question (with chat history prepended if any)
    │
    └── Step 4: STREAM LLM RESPONSE
        └── llm.py: GroqLLM.stream_answer()
        └── AsyncOpenAI client → api.groq.com/openai/v1
        └── model: llama-3.3-70b-versatile, temperature: 0.1
        └── Yields tokens one at a time
        └── Each token → SSE event → frontend appends to bubble
```

---

## 7. Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | `embedder.py`, `llm.py` | Swap providers (OpenAI ↔ HuggingFace ↔ Groq) via config without changing code |
| **Factory** | `get_embedder()`, `get_llm()` | Single function returns the right provider based on `.env` |
| **Singleton** | `chroma_client.py` | One ChromaDB client shared across all requests (avoids SQLite locking) |
| **Callback/Observer** | `pipeline.py` progress callbacks | Pipeline reports progress without knowing how it's displayed (SSE, logs, etc.) |
| **Dependency Inversion** | `ProgressCallback` type alias | Pipeline depends on an abstraction, not on SSE directly |
| **ABC/Interface** | `BaseLLM`, `BaseEmbedder` | Enforces contract: every provider must implement `stream_answer()` / `embed_texts()` |
| **Async Generator** | LLM streaming | Yields tokens one at a time for real-time UI updates |

---

## 8. Key Configuration (config.py + .env)

```python
# Active providers
EMBEDDING_PROVIDER=huggingface    # Free, local, no API key
LLM_PROVIDER=groq                 # Free tier via Groq

# Groq API
GROQ_API_KEY=gsk_...              # Get at console.groq.com/keys
groq_llm_model = "llama-3.3-70b-versatile"
groq_base_url = "https://api.groq.com/openai/v1"

# Chunking
chunk_size = 20          # 20 rows per RAG chunk
chunk_overlap = 3        # 3-row overlap between chunks
pandas_chunk_size = 1000 # RAM management: read 1000 rows at a time

# Retrieval
top_k = 5               # Return 5 most relevant chunks (+ 1 summary = 6 total)

# Embedding
embedding_batch_size = 100
huggingface_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
```

---

## 9. Frontend Details

### Next.js Config (`next.config.ts`)
- **API Proxy**: All `/api/*` requests are rewritten to `http://localhost:8000/api/*` — the frontend never talks to the backend directly, avoiding CORS issues in dev.
- **React Compiler**: Enabled (`reactCompiler: true`) for automatic memoization.

### Styling (`globals.css`)
- **Tailwind v4** with `@import "tailwindcss"` and `@theme inline` block for design tokens
- Dark theme with CSS custom properties: `--color-surface-0` through `--color-surface-4`, ink colors, accent (indigo/violet)
- Custom animations: `fade-up`, `fade-in`, `slide-in`, `dot-bounce`, `spin-slow`, `pulse-ring`, `cursor-blink`
- Glass morphism utility: `.glass` class

### Component Hierarchy
```
page.tsx (Home)
├── useBackendStatus() — polls /api/files every 15s, shows green/red dot
├── FileSidebar — lists uploaded files, select active, delete
├── FileUploader — drag-drop zone, SSE progress with 4-stage stepper
├── ChatBox — textarea input, streaming phases (searching → generating)
│   └── MessageList — renders Message[] as user/assistant bubbles
│       ├── UserMessage — gradient bubble, right-aligned
│       └── AssistantMessage — dark bubble, left-aligned, typing cursor
└── Upload Modal — overlay with FileUploader
```

### SSE Parsing (`lib/api.ts`)
- **Manual SSE parser** (not EventSource, because EventSource only supports GET)
- Uses `fetch()` + `ReadableStream` + `TextDecoder`
- Buffer pattern: accumulates text, splits on `\n\n`, extracts `data: ` prefix, parses JSON
- `onEvent` callback is called outside the JSON try/catch so errors from the callback propagate

### State Management
- All state via React `useState`/`useRef` hooks (no external state library)
- **Immutable updates** throughout (critical for React 19 Strict Mode):
  ```tsx
  // Correct: create new array with spread
  setMessages(prev => [...prev.slice(0, -1), { ...last, content: last.content + token }]);
  ```

---

## 10. Backend Details

### Entry Point (`main.py`)
- Creates FastAPI app with CORS middleware (`allow_origins=["*"]`)
- Mounts router under `/api` prefix
- Startup event creates data directories and logs config

### API Endpoints (`routes.py`)

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| POST | `/api/upload` | Upload file, stream ingestion progress | SSE stream |
| POST | `/api/query` | Ask question, stream answer tokens | SSE stream |
| POST | `/api/query/full` | Ask question, get full JSON answer | JSON |
| GET | `/api/files` | List all uploaded files | JSON |
| DELETE | `/api/files/{id}` | Delete file and its vectors | JSON |
| GET | `/api/health` | Health check | JSON |

### Ingestion Pipeline (`pipeline.py`)
- Async function with progress callbacks
- Uses `asyncio.to_thread()` for synchronous pandas operations
- Creates one ChromaDB collection per file: `file_{file_id}`
- Embeds in batches of 100 chunks

### Parser (`parser.py`)
- Two-level chunking: pandas (1000 rows for RAM) → RAG (20 rows for semantics)
- Header prepending: every chunk starts with `Columns: col1 | col2 | ...`
- Row numbering: absolute (matches spreadsheet), not relative to chunk
- **Summary chunk**: Pre-computed statistics for ALL rows. For categorical columns (≤50 unique), includes complete value counts. For high-cardinality columns (>50 unique), includes top-10 with counts. For numeric columns, includes min/max/mean/median.

### Hybrid Search (`searcher.py`)
1. **Semantic search**: Embed query → ChromaDB cosine similarity → top-K chunks
2. **Keyword search**: Extract non-stop-words → case-insensitive substring match on all docs → score by match count
3. **Merge**: Keyword results first (exact match priority) → semantic results → deduplicate → cap at `top_k + 1`
4. **Summary injection**: Always includes the summary chunk (fetched separately if not in results)

### LLM System Prompt (`llm.py`)
```
You are a helpful data analyst. Answer questions using ONLY the provided context.

The context contains two types of information:
- A DATASET SUMMARY with pre-computed statistics over ALL rows
- Individual DATA CHUNKS showing specific rows

Rules:
1. For aggregate/count questions, use the DATASET SUMMARY statistics directly.
2. For specific lookups, use the individual data chunks.
3. Cite rows/columns when referencing specific data.
4. If the context truly doesn't contain relevant information, say so briefly.
5. Be concise and direct.
```

### ChromaDB Client (`chroma_client.py`)
- **Singleton**: Module-level `_client` variable, created once via `get_chroma_client()`
- Persistent storage at `./data/chroma/`
- Handles both old ChromaDB API (returns Collection objects) and new API (returns strings) for `list_collections()`

---

## 11. Known Issues & Constraints

### Token Limits (Groq Free Tier)
- `llama-3.3-70b-versatile` has a 12,000 TPM (tokens per minute) limit on Groq's free tier
- With `top_k=5` + summary chunk + system prompt, typical request is ~3,000-4,000 tokens
- If the summary chunk is very large (many columns with many categorical values), it could exceed limits
- Mitigation: cap `top_k` low, use concise summary format

### Summary Chunk Limitations
- The summary reads the entire file into memory once during parsing (separate from chunked reading)
- For files with 100+ columns or columns with ~50 unique values each, the summary could be very long
- Value counts for columns with exactly 50 unique values all get listed (could be trimmed)

### Keyword Search Performance
- Currently fetches ALL documents from the collection for keyword matching
- For files with 10,000+ chunks, this could be slow
- ChromaDB's built-in `$contains` filter is case-sensitive and inconsistent across versions, hence Python-side matching

### ChromaDB Version Compatibility
- `list_collections()` returns Collection objects in 0.4.x but strings in 0.5.x+
- Code handles both via `isinstance(item, str)` checks
- `$ne` filter for excluding summary chunks may not work in all ChromaDB versions (fallback fetches all docs)

### Frontend Polling
- `page.tsx` polls `/api/files` every 15 seconds to check backend status
- This means the "Backend offline" banner can take up to 15 seconds to update

### File Re-upload Required After Code Changes
- Changing `chunk_size`, `chunk_overlap`, or `_build_summary_chunk` logic requires re-uploading files
- Old ChromaDB data persists across server restarts and doesn't auto-update
- Must manually delete `./data/chroma/` directory (while server is stopped) and re-upload

---

## 12. How to Run

### Backend
```bash
cd backend
pip install -r requirements.txt
# Edit .env with your GROQ_API_KEY
uvicorn main:app --reload
# Runs on http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

### Environment Variables (.env)
```env
EMBEDDING_PROVIDER=huggingface
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
```

---

## 13. File-by-File Reference

### Backend Files

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 133 | FastAPI app, CORS, router mount, startup |
| `api/routes.py` | 377 | All HTTP endpoints, SSE streaming |
| `core/config.py` | 152 | Pydantic Settings, all config centralized |
| `core/embedder.py` | 311 | BaseEmbedder ABC + OpenAI/HuggingFace/Local implementations |
| `core/llm.py` | 514 | BaseLLM ABC + OpenAI/Anthropic/Grok/Gemini/Groq/Local + system prompt |
| `db/chroma_client.py` | 191 | ChromaDB singleton, collection CRUD helpers |
| `generation/generator.py` | 257 | RAG orchestration: search → format → stream LLM |
| `ingestion/parser.py` | 491 | File parsing, chunking, summary chunk generation |
| `ingestion/pipeline.py` | 296 | Full ingestion pipeline with progress callbacks |
| `models/schemas.py` | 81 | Pydantic models: QueryRequest, FileInfo, QueryResponse, DeleteResponse |
| `retrieval/searcher.py` | 457 | Hybrid search (semantic + keyword), result formatting |

### Frontend Files

| File | Lines | Purpose |
|------|-------|---------|
| `app/page.tsx` | 221 | Main page: sidebar, chat, upload modal, backend status |
| `app/layout.tsx` | 33 | Root layout: fonts, metadata |
| `app/globals.css` | 140 | Design tokens, animations, Tailwind v4 theme |
| `components/ChatBox.tsx` | 220 | Chat input, streaming phases, stop button |
| `components/FileUploader.tsx` | 217 | Drag-drop upload, SSE progress, 4-stage stepper |
| `components/FileSidebar.tsx` | 155 | File list, active selection, delete with confirm |
| `components/MessageList.tsx` | 190 | Message rendering: user/assistant bubbles, sources, typing cursor |
| `lib/api.ts` | 254 | API client: uploadFile, queryStream, getFiles, deleteFile, SSE parser |
| `next.config.ts` | 15 | API proxy rewrites, React compiler |
| `tsconfig.json` | 34 | TypeScript config with `@/*` path alias |
| `package.json` | 28 | Dependencies: next, react, lucide-react, tailwindcss |

---

## 14. Potential Improvements

1. **Reranker**: Add a cross-encoder reranker between retrieval and generation to improve result quality
2. **Caching**: Cache embeddings and summary chunks to avoid re-computing on identical files
3. **Streaming upload progress**: Show per-row parsing progress for large files
4. **Multi-file search**: Currently queries one file at a time (file_id required); could support cross-file questions
5. **Chat history persistence**: Messages are lost on page refresh; could use localStorage or a database
6. **Authentication**: No auth currently; anyone can upload/delete/query
7. **Rate limiting**: No rate limiting on API endpoints
8. **Better chunking**: Could use semantic chunking (split by topic changes) instead of fixed-size rows
9. **Evaluation**: RAGAS integration for measuring retrieval and generation quality
10. **Local LLM**: Ollama integration for fully offline operation
