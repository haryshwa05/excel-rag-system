"""
Ingestion Pipeline — orchestrates the full upload-to-searchable flow.

WHAT THIS FILE DOES:
Coordinates: file → parse → chunk → embed (in batches) → store in ChromaDB.
Reports progress at every step so the frontend can show a progress bar.

THE FULL FLOW:
    User uploads sales.xlsx
        │
        ▼
    1. Parse file (parser.py) → 500 chunks with metadata
        │  progress: "Parsing file... 10%"
        ▼
    2. Batch embed (embedder.py) → 500 vectors
        │  Batch 1/5: chunks 0-99   → 100 vectors  → progress: "Embedding... 30%"
        │  Batch 2/5: chunks 100-199 → 100 vectors  → progress: "Embedding... 40%"
        │  Batch 3/5: chunks 200-299 → 100 vectors  → progress: "Embedding... 50%"
        │  Batch 4/5: chunks 300-399 → 100 vectors  → progress: "Embedding... 60%"
        │  Batch 5/5: chunks 400-499 → 100 vectors  → progress: "Embedding... 70%"
        ▼
    3. Store in ChromaDB → 500 entries (vector + text + metadata each)
        │  progress: "Storing... 90%"
        ▼
    4. Done! → progress: "Complete! 100%"

WHY BATCHING MATTERS FOR EMBEDDINGS:
    - OpenAI charges per token. One call with 100 texts is the same cost as
      100 calls with 1 text — but 50x faster because you eliminate 99 network
      round trips (~100ms each).
    - If a batch fails (network error), you only retry 100 chunks, not all 500.
    - Progress updates happen per batch, so the user sees movement.

WHAT YOU'RE LEARNING:
- Pipeline orchestration: coordinating multiple async/sync steps
- asyncio.to_thread: bridging sync code (pandas) into async context
- Callback pattern: decoupling progress reporting from business logic
- Batch processing: chunking work into manageable pieces
- Error handling in pipelines: what to do when step 3 of 4 fails
"""

import asyncio
import uuid
from pathlib import Path
from typing import Callable, Awaitable

from core.config import settings
from core.embedder import get_embedder
from db.chroma_client import get_chroma_client, delete_collection
from ingestion.parser import parse_file, Chunk


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Progress callback type
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# This type alias defines what a progress callback looks like.
# It's a function that takes (stage_name, percent, detail_message)
# and returns an awaitable (because sending SSE events is async).
#
# WHY A CALLBACK AND NOT DIRECT SSE:
# The pipeline doesn't know HOW progress gets reported. Maybe it's SSE
# to a browser. Maybe it's a WebSocket. Maybe it's a log file in testing.
# The callback pattern means the pipeline just says "I'm 45% done" and
# whoever called it decides what to do with that information.
#
# This is called DEPENDENCY INVERSION — the pipeline depends on an
# abstraction (the callback type), not a concrete implementation (SSE).

ProgressCallback = Callable[[str, float, str], Awaitable[None]]


async def ingest_file(
    file_path: Path,
    file_id: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """
    Main entry point. Takes a file and makes it searchable.

    Args:
        file_path: Path to the uploaded Excel/CSV file.
        file_id: Unique ID for this file. If None, we generate one.
                 This ID becomes the ChromaDB collection name, so all
                 chunks from this file are grouped together.
        on_progress: Optional callback for progress updates.
                     Called with (stage, percentage, message).

    Returns:
        dict with ingestion results:
        {
            "file_id": "abc123",
            "file_name": "sales.xlsx",
            "total_chunks": 500,
            "total_rows_processed": 10000,
            "sheets": ["Q1", "Q2", "Q3", "Q4"],
            "status": "complete"
        }

    WHY file_id IS THE COLLECTION NAME:
    ChromaDB organizes vectors into "collections" (like database tables).
    One collection per file means:
    - Searching one file doesn't touch other files' vectors
    - Deleting a file = deleting one collection (clean, instant)
    - The user can choose which file to search in
    """
    # Generate a unique file ID if not provided
    if file_id is None:
        file_id = str(uuid.uuid4())[:8]

    # Helper to report progress (handles None callback gracefully)
    async def report(stage: str, percent: float, message: str):
        if on_progress:
            await on_progress(stage, percent, message)

    try:
        # ── Step 1: Parse the file into chunks ─────────────────────────
        await report("parsing", 5.0, f"Parsing {file_path.name}...")

        # asyncio.to_thread runs synchronous code in a thread pool.
        #
        # WHY THIS IS NEEDED:
        # parse_file() uses pandas, which is synchronous — it blocks the
        # thread while reading the file. In an async server, blocking the
        # main thread means NO other requests can be handled.
        #
        # asyncio.to_thread() says: "run this function in a background
        # thread so the main async event loop stays free."
        #
        # Think of it like this:
        # - The main thread is the waiter (handles all customer requests)
        # - pandas is a slow kitchen task (blocks whoever does it)
        # - to_thread sends the task to a kitchen helper (background thread)
        # - The waiter is free to serve other customers while the helper works
        chunks: list[Chunk] = await asyncio.to_thread(
            parse_file, file_path, file_id
        )

        if not chunks:
            await report("error", 0.0, "No data found in file.")
            return {
                "file_id": file_id,
                "file_name": file_path.name,
                "total_chunks": 0,
                "status": "empty",
            }

        await report("parsing", 15.0, f"Parsed {len(chunks)} chunks.")

        # ── Step 2: Create ChromaDB collection ─────────────────────────
        await report("storing", 18.0, "Creating vector collection...")

        chroma_client = get_chroma_client()

        collection = chroma_client.get_or_create_collection(
            name=f"file_{file_id}",
            metadata={"file_name": file_path.name},
        )

        # ── Step 3: Embed and store in batches ─────────────────────────
        embedder = get_embedder()
        batch_size = settings.embedding_batch_size
        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size

        # Process chunks in batches of 100
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_chunks)
            batch_chunks = chunks[start:end]

            # Calculate progress: embedding takes 20% to 85% of total time
            progress = 20.0 + (batch_idx / total_batches) * 65.0
            await report(
                "embedding",
                progress,
                f"Embedding batch {batch_idx + 1}/{total_batches} "
                f"({start}-{end} of {total_chunks} chunks)...",
            )

            # Get embeddings for this batch
            # This is the async network call to OpenAI/HuggingFace
            texts = [chunk.text for chunk in batch_chunks]
            embeddings = await embedder.embed_texts(texts)

            # Store in ChromaDB
            # Each entry needs: a unique ID, the vector, the text, and metadata
            #
            # WHY STRING IDs:
            # ChromaDB requires string IDs. We use "file_id__chunk_index"
            # so IDs are unique and traceable back to the source.
            #
            # WHY WE STORE THE TEXT TOO:
            # ChromaDB stores the text alongside the vector (as "documents").
            # When search returns results, it gives back the text directly —
            # we don't need a separate database to look up what the chunk said.
            collection.add(
                ids=[
                    f"{file_id}__{start + i}" for i in range(len(batch_chunks))
                ],
                embeddings=embeddings,
                documents=texts,
                metadatas=[chunk.metadata for chunk in batch_chunks],
            )

        # ── Step 4: Done ───────────────────────────────────────────────
        # Collect unique sheet names for the response
        sheets = list(set(
            chunk.metadata.get("sheet_name", "default")
            for chunk in chunks
        ))

        # Calculate total rows from metadata
        total_rows = 0
        if chunks:
            last_chunk = chunks[-1]
            total_rows = last_chunk.metadata.get("row_end", 0) - 1

        result = {
            "file_id": file_id,
            "file_name": file_path.name,
            "total_chunks": total_chunks,
            "total_rows_processed": total_rows,
            "sheets": sheets,
            "status": "complete",
        }

        await report("complete", 100.0, f"Done! {total_chunks} chunks indexed.")
        return result

    except Exception as e:
        # Report the error through the progress callback
        await report("error", 0.0, f"Ingestion failed: {str(e)}")

        # Re-raise so the caller (API route) can handle it too
        raise


async def delete_file(file_id: str) -> bool:
    """
    Delete a file's collection from ChromaDB.

    This removes ALL vectors, texts, and metadata for this file.
    It's instant because ChromaDB just drops the collection.

    Returns True if the collection existed and was deleted,
    False if it didn't exist.

    WHY THIS EXISTS:
    Users need to be able to remove files. Without this, the vector
    store grows forever and search results include deleted files.
    """
    return delete_collection(file_id)


async def list_files() -> list[dict]:
    """
    List all ingested files by reading ChromaDB collections.

    Returns a list of dicts with file info:
    [
        {"file_id": "abc123", "file_name": "sales.xlsx", "chunks": 500},
        {"file_id": "def456", "file_name": "inventory.csv", "chunks": 200},
    ]

    WHY THIS IS USEFUL:
    The frontend needs to show which files have been uploaded and are
    searchable. This reads directly from ChromaDB — no separate database.
    """
    try:
        client = get_chroma_client()
        raw = client.list_collections()

        files = []
        for item in raw:
            name = item if isinstance(item, str) else getattr(item, "name", str(item))
            if not name.startswith("file_"):
                continue
            file_id = name.replace("file_", "", 1)
            try:
                col = client.get_collection(name=name)
                files.append({
                    "file_id": file_id,
                    "file_name": col.metadata.get("file_name", "unknown") if col.metadata else "unknown",
                    "chunks": col.count(),
                })
            except Exception:
                files.append({
                    "file_id": file_id,
                    "file_name": "unknown",
                    "chunks": 0,
                })

        return files
    except Exception:
        return []