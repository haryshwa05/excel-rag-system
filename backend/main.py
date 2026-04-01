import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from api.routes import router

app = FastAPI(
    title="DataRAG API",
    description="Ask questions about your Excel, CSV, and PDF data using AI.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup():
    # Create data directories
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.bm25_dir.mkdir(parents=True, exist_ok=True)

    # Async SQLite init with WAL mode
    from db.database import init_db
    await init_db(settings.db_path)

    print("DataRAG v2 starting…")
    print(f"  Embedding : {settings.embedding_provider}")
    print(f"  LLM       : {settings.llm_provider}")
    print(f"  Reranker  : {'enabled' if settings.enable_reranker else 'disabled'}")
    print(f"  Vision    : {'enabled (' + settings.vision_provider + ')' if settings.enable_vision else 'disabled'}")
    print(f"  ChromaDB  : {settings.chroma_persist_dir}")
    print(f"  Uploads   : {settings.upload_dir}")
    print(f"  Docs      : http://{settings.host}:{settings.port}/docs")

    # Warm up cross-encoder in background
    if settings.enable_reranker:
        from retrieval.reranker import warmup
        asyncio.create_task(warmup())

    # ── Resume any ingestion jobs that were interrupted by a restart ──────────
    from db.database import list_pending_jobs, update_job_status
    from ingestion.pipeline import ingest_file

    pending = await list_pending_jobs()
    if pending:
        print(f"  Resuming {len(pending)} interrupted ingestion job(s)…")
        for job in pending:
            fp = Path(job["file_path"])
            if fp.exists():
                print(f"    → Resuming: {job['file_name']} ({job['file_id']})")
                asyncio.create_task(
                    ingest_file(file_path=fp, file_id=job["file_id"])
                )
            else:
                print(f"    ✗ File missing, skipping: {job['file_name']}")
                await update_job_status(
                    job["file_id"], "failed", "File not found on disk after restart"
                )


@app.get("/")
async def root():
    return {
        "app": "DataRAG",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
