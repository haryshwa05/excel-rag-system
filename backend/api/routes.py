"""
API Routes — upload, query, file management, evaluation.
"""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from core.config import settings
from core.rate_limit import rate_limiter
from ingestion.pipeline import ingest_file, delete_file, list_files
from generation.generator import generate_answer, generate_answer_full
from models.schemas import QueryRequest, QueryResponse, DeleteResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".pdf"}
_CHUNK = 1 << 16   # 64 KB read chunks


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    rate_limiter.check(request, limit=10)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    file_id   = str(uuid.uuid4())[:8]
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.upload_dir / f"{file_id}_{file.filename}"
    max_bytes = settings.max_file_size_mb * 1024 * 1024

    # ── Stream file to disk in 64 KB chunks ────────────────────────────────────
    # This keeps memory flat regardless of file size. The old `await file.read()`
    # loaded the entire file into RAM before any size check.
    total_bytes = 0
    try:
        with open(file_path, "wb") as fout:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large (>{settings.max_file_size_mb} MB).",
                    )
                fout.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    # ── Register job BEFORE spawning the task ──────────────────────────────────
    # If the server restarts after this line, startup will find this job and resume it.
    from db.database import upsert_job
    await upsert_job(file_id, file.filename, str(file_path))

    async def event_stream():
        import asyncio
        queue: asyncio.Queue = asyncio.Queue()

        async def progress_cb(stage: str, percent: float, message: str):
            await queue.put({"stage": stage, "percent": round(percent, 1), "message": message, "file_id": file_id})

        async def run():
            try:
                result = await ingest_file(
                    file_path=file_path,
                    file_id=file_id,
                    on_progress=progress_cb,
                )
                await queue.put({"stage": "done", "result": result})
            except Exception as e:
                await queue.put({"stage": "error", "message": str(e)})

        task = asyncio.create_task(run())
        idle_timeout_s = 180
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=idle_timeout_s)
            except asyncio.TimeoutError:
                if task.done():
                    # Task ended but no terminal event made it through.
                    err = None
                    try:
                        _ = task.result()
                    except Exception as e:
                        err = str(e)
                    payload = {"error": err or "Upload stream ended unexpectedly.", "stage": "error"}
                    yield f"data: {json.dumps(payload)}\n\n"
                    break
                payload = {
                    "error": f"No upload progress for {idle_timeout_s}s. Please retry.",
                    "stage": "error",
                }
                yield f"data: {json.dumps(payload)}\n\n"
                task.cancel()
                break

            if event.get("stage") == "done":
                yield f"data: {json.dumps(event.get('result', {}))}\n\n"
                break
            elif event.get("stage") == "error":
                yield f"data: {json.dumps({'error': event.get('message', 'Unknown error'), 'stage': 'error'})}\n\n"
                break
            else:
                yield f"data: {json.dumps(event)}\n\n"
        await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/query")
async def query_streaming(request: Request, body: QueryRequest):
    rate_limiter.check(request, limit=30)

    async def event_stream():
        try:
            import asyncio
            async with asyncio.timeout(120):
                async for token in generate_answer(
                    question=body.question,
                    file_id=body.file_id,
                    sheet_name=body.sheet_name,
                    chat_history=body.chat_history,
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except TimeoutError:
            yield f"data: {json.dumps({'error': 'Query timed out after 120s. Please retry with a shorter question or local model.', 'done': True})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e) or 'An error occurred', 'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/query/full", response_model=QueryResponse)
async def query_full(request: Request, body: QueryRequest):
    rate_limiter.check(request, limit=20)
    try:
        result = await generate_answer_full(
            question=body.question,
            file_id=body.file_id,
            sheet_name=body.sheet_name,
            chat_history=body.chat_history,
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files")
async def get_files():
    try:
        files = await list_files()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/files/{file_id}")
async def remove_file(file_id: str):
    try:
        deleted = await delete_file(file_id)
        if settings.upload_dir.exists():
            for f in settings.upload_dir.iterdir():
                if f.name.startswith(file_id):
                    f.unlink()
        return DeleteResponse(
            file_id=file_id,
            deleted=deleted,
            message="File deleted." if deleted else "File not found.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate(request: Request, body: dict):
    """
    LLM-as-judge evaluation.
    Body: { "file_id": "abc123", "questions": ["How many rows?", ...] }
    """
    rate_limiter.check(request, limit=5)
    file_id   = body.get("file_id")
    questions = body.get("questions")
    if not file_id or not questions:
        raise HTTPException(
            status_code=400,
            detail="Both 'file_id' and 'questions' are required.",
        )

    try:
        from evaluation.ragas_eval import evaluate_file
        from dataclasses import asdict
        report = await evaluate_file(file_id, questions)
        return asdict(report)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
