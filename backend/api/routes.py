"""
API Routes — the HTTP endpoints that the frontend calls.

WHAT THIS FILE DOES:
Exposes everything we've built as HTTP endpoints:
    POST /upload      → upload a file, get progress via SSE
    POST /query       → ask a question, get streaming answer via SSE
    POST /query/full  → ask a question, get complete answer as JSON
    GET  /files       → list all uploaded files
    DELETE /files/{id} → delete a file

SSE (Server-Sent Events) — HOW STREAMING WORKS OVER HTTP:
    Normal HTTP: client sends request → server sends ONE response → done.
    SSE: client sends request → server sends MANY pieces over time → done.

    The server keeps the connection open and sends events like:
        data: {"stage": "embedding", "percent": 45, "message": "Batch 3/5"}
        data: {"stage": "embedding", "percent": 55, "message": "Batch 4/5"}
        data: {"token": "The"}
        data: {"token": " top"}
        data: {"token": " sales"}

    The frontend reads these one by one as they arrive.

    SSE vs WebSocket:
    - SSE is ONE-WAY: server → client only. Perfect for streaming responses.
    - WebSocket is TWO-WAY: both can send anytime. Overkill for our use case.
    - SSE works over regular HTTP. WebSocket needs a protocol upgrade.
    - SSE auto-reconnects if the connection drops. WebSocket doesn't.
    For streaming LLM responses and progress updates, SSE is the right choice.

WHAT YOU'RE LEARNING:
- FastAPI route definitions and dependency injection
- StreamingResponse for SSE
- File upload handling with validation
- Background tasks for long-running operations
- Error handling patterns in API routes
- How the frontend will consume each endpoint
"""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from core.config import settings
from ingestion.pipeline import ingest_file, delete_file, list_files
from generation.generator import generate_answer, generate_answer_full
from models.schemas import QueryRequest, FileInfo, QueryResponse, DeleteResponse


# APIRouter groups related routes together.
# In main.py, we'll mount this router under a prefix.
# This is cleaner than putting all routes in main.py directly —
# as the app grows, you'd have separate routers for auth, admin, etc.
router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# File upload with SSE progress
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload an Excel/CSV file and stream ingestion progress.

    HOW THIS ENDPOINT WORKS:
    1. Client sends a multipart form upload (the file)
    2. Server validates the file (type, size)
    3. Server saves the file to disk
    4. Server starts ingestion and streams progress via SSE
    5. Client receives progress events until ingestion is complete

    WHY NOT A BACKGROUND TASK:
    FastAPI has BackgroundTasks for fire-and-forget work. But we need
    PROGRESS REPORTING — the user wants to see "45% done". Background
    tasks can't send data back to the client. SSE can.

    So instead of:
        1. Upload → return 200 immediately → process in background (no progress)
    We do:
        1. Upload → keep connection open → stream progress → close when done

    THE FRONTEND CALLS THIS LIKE:
        const response = await fetch('/api/upload', {method: 'POST', body: formData});
        const reader = response.body.getReader();
        // read SSE events one by one, update progress bar

    ABOUT UploadFile:
    FastAPI's UploadFile wraps the incoming file. It provides:
    - file.filename: original name ("sales.xlsx")
    - file.content_type: MIME type ("application/vnd.openxmlformats...")
    - file.file: the actual file object (readable stream)
    - file.size: file size in bytes

    File(...) means this parameter is REQUIRED — the request must
    include a file, otherwise FastAPI returns a 422 error automatically.
    """
    # ── Validate file type ─────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    allowed = {".csv", ".xlsx", ".xls"}

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{suffix}'. Allowed: {allowed}",
        )

    # ── Validate file size ─────────────────────────────────────────
    # Read the file content into memory to check size.
    # For very large files (1GB+), you'd use chunked reading instead.
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.max_file_size_mb}MB.",
        )

    # ── Save file to disk ──────────────────────────────────────────
    file_id = str(uuid.uuid4())[:8]
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.upload_dir / f"{file_id}_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(contents)

    # ── Stream ingestion progress via SSE ──────────────────────────
    async def event_stream():
        """
        An async generator that yields SSE-formatted events.

        SSE FORMAT:
        Each event is a line starting with "data: " followed by JSON,
        then two newlines to signal the end of that event.

            data: {"stage": "parsing", "percent": 10, "message": "Parsing..."}\\n\\n
            data: {"stage": "embedding", "percent": 45, "message": "Batch 3/5"}\\n\\n
            data: {"stage": "complete", "percent": 100, "message": "Done!"}\\n\\n

        We use an async queue to bridge the progress callback and
        this generator — Python doesn't allow yielding from inside
        a nested callback function.
        """
        import asyncio

        queue = asyncio.Queue()

        async def queue_callback(stage: str, percent: float, message: str):
            """Put progress events into the queue."""
            await queue.put({
                "stage": stage,
                "percent": round(percent, 1),
                "message": message,
                "file_id": file_id,
            })

        async def run_ingestion():
            """Run ingestion and signal completion via the queue."""
            try:
                result = await ingest_file(
                    file_path=file_path,
                    file_id=file_id,
                    on_progress=queue_callback,
                )
                # Signal completion with the result
                await queue.put({"stage": "done", "result": result})
            except Exception as e:
                await queue.put({"stage": "error", "message": str(e)})

        # Start ingestion as a concurrent task
        task = asyncio.create_task(run_ingestion())

        # Yield events from the queue as they arrive
        while True:
            event = await queue.get()

            if event.get("stage") == "done":
                # Send final event with the result
                result_data = json.dumps(event.get("result", {}))
                yield f"data: {result_data}\n\n"
                break
            elif event.get("stage") == "error":
                error_data = json.dumps({
                    "stage": "error",
                    "message": event.get("message", "Unknown error"),
                })
                yield f"data: {error_data}\n\n"
                break
            else:
                yield f"data: {json.dumps(event)}\n\n"

        # Ensure the task is complete
        await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # These headers are required for SSE:
        # Cache-Control: no-cache — don't cache the stream
        # Connection: keep-alive — keep the connection open
        # X-Accel-Buffering: no — tell Nginx not to buffer (important for proxies)
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Query endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.post("/query")
async def query_streaming(request: QueryRequest):
    """
    Ask a question and get a streaming answer via SSE.

    THE FRONTEND CALLS THIS LIKE:
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: "top sales?", file_id: "abc123"})
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            const text = decoder.decode(value);
            // text = 'data: {"token": "The"}\n\n'
            // parse and append to chat bubble
        }

    ABOUT QueryRequest:
    FastAPI automatically parses the JSON body and validates it against
    the QueryRequest Pydantic model. If the body is missing "question"
    or has wrong types, FastAPI returns a 422 error before our code runs.
    """
    async def event_stream():
        try:
            async for token in generate_answer(
                question=request.question,
                file_id=request.file_id,
                sheet_name=request.sheet_name,
                chat_history=request.chat_history,
            ):
                event_data = json.dumps({"token": token})
                yield f"data: {event_data}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_data = json.dumps({"error": str(e) or "Unknown error occurred"})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/query/full", response_model=QueryResponse)
async def query_full(request: QueryRequest):
    """
    Ask a question and get the complete answer as JSON (non-streaming).

    Returns the answer plus source citations. Used for:
    - Evaluation (RAGAS needs the full answer)
    - API clients that don't support streaming
    - Testing

    response_model=QueryResponse tells FastAPI to:
    1. Validate the response matches QueryResponse schema
    2. Show QueryResponse in the API docs
    3. Automatically serialize the response to JSON
    """
    try:
        result = await generate_answer_full(
            question=request.question,
            file_id=request.file_id,
            sheet_name=request.sheet_name,
            chat_history=request.chat_history,
        )

        return QueryResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# File management endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get("/files")
async def get_files():
    """
    List all uploaded and processed files.

    The frontend uses this to show a file selector:
    "Which file do you want to search?"
    [sales_2024.xlsx - 500 chunks]
    [inventory.csv - 200 chunks]
    """
    try:
        files = await list_files()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{file_id}")
async def remove_file(file_id: str):
    """
    Delete a file's vectors from ChromaDB.

    This removes the collection (all vectors + metadata).
    The original uploaded file on disk is also cleaned up.

    {file_id} in the path is a PATH PARAMETER — FastAPI extracts it
    automatically. DELETE /files/abc123 → file_id = "abc123".
    """
    try:
        deleted = await delete_file(file_id)

        # Also clean up the uploaded file from disk
        if settings.upload_dir.exists():
            for f in settings.upload_dir.iterdir():
                if f.name.startswith(file_id):
                    f.unlink()  # delete the file

        return DeleteResponse(
            file_id=file_id,
            deleted=deleted,
            message="File deleted successfully." if deleted else "File not found.",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Health check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@router.get("/health")
async def health_check():
    """
    Simple health check endpoint.

    Every production API needs this. Load balancers, Docker health checks,
    and monitoring tools ping this to know if the server is alive.
    Returns 200 = server is running. Anything else = something's wrong.
    """
    return {"status": "healthy"}