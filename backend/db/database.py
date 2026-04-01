"""
Async SQLite persistence — file metadata + durable ingestion job queue.

WAL mode: writes don't block reads. Critical for production where ingestion
and queries run concurrently. Without WAL, every write locks the entire DB
and you get intermittent 500s on the query endpoint during large uploads.

Job queue: every ingestion job is written to the `jobs` table before the
async task starts. On startup, any `pending` or `running` jobs whose file
still exists on disk are automatically resumed. This means a server restart
mid-ingestion does NOT lose the job.
"""

import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import aiosqlite

_DB_PATH: Optional[Path] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_init():
    if _DB_PATH is None:
        raise RuntimeError("init_db() must be called before any database operation.")


async def _connect() -> aiosqlite.Connection:
    """Open a connection with WAL mode and row factory."""
    _check_init()
    conn = await aiosqlite.connect(str(_DB_PATH))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA synchronous=NORMAL")  # safe + fast with WAL
    return conn


@asynccontextmanager
async def _db():
    """
    Open one connection per operation and close it reliably.
    Avoid `async with await aiosqlite.connect(...)` because awaiting starts
    the worker thread, and entering the context can try to start it again.
    """
    conn = await _connect()
    try:
        yield conn
    finally:
        await conn.close()


# ── Initialisation ────────────────────────────────────────────────────────────

async def init_db(db_path: Path):
    global _DB_PATH
    _DB_PATH = db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with _db() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_id      TEXT PRIMARY KEY,
                file_name    TEXT NOT NULL,
                total_chunks INTEGER DEFAULT 0,
                total_rows   INTEGER DEFAULT 0,
                sheets       TEXT    DEFAULT '[]',
                status       TEXT    DEFAULT 'processing',
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                file_id    TEXT PRIMARY KEY,
                file_name  TEXT NOT NULL,
                file_path  TEXT NOT NULL,
                status     TEXT DEFAULT 'pending',
                error      TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()


# ── Files table ───────────────────────────────────────────────────────────────

async def upsert_file(file_id: str, file_name: str, status: str = "processing"):
    async with _db() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO files (file_id, file_name, status) VALUES (?,?,?)",
            (file_id, file_name, status),
        )
        await conn.commit()


async def update_file_complete(
    file_id: str, total_chunks: int, total_rows: int, sheets: list
):
    async with _db() as conn:
        await conn.execute(
            "UPDATE files SET total_chunks=?, total_rows=?, sheets=?, status='complete' WHERE file_id=?",
            (total_chunks, total_rows, json.dumps(sheets), file_id),
        )
        await conn.commit()


async def delete_file_record(file_id: str):
    async with _db() as conn:
        await conn.execute("DELETE FROM files WHERE file_id=?", (file_id,))
        await conn.commit()


async def list_files() -> list[dict]:
    async with _db() as conn:
        async with conn.execute(
            "SELECT * FROM files WHERE status='complete' ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def get_file(file_id: str) -> Optional[dict]:
    async with _db() as conn:
        async with conn.execute(
            "SELECT * FROM files WHERE file_id=?", (file_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


# ── Jobs table (durable ingestion queue) ──────────────────────────────────────

async def upsert_job(file_id: str, file_name: str, file_path: str):
    """Register a new ingestion job before the async task starts."""
    async with _db() as conn:
        await conn.execute(
            """INSERT OR REPLACE INTO jobs (file_id, file_name, file_path, status, updated_at)
               VALUES (?,?,?,'pending', CURRENT_TIMESTAMP)""",
            (file_id, file_name, file_path),
        )
        await conn.commit()


async def update_job_status(
    file_id: str, status: str, error: Optional[str] = None
):
    async with _db() as conn:
        await conn.execute(
            """UPDATE jobs SET status=?, error=?, updated_at=CURRENT_TIMESTAMP
               WHERE file_id=?""",
            (status, error, file_id),
        )
        await conn.commit()


async def list_pending_jobs() -> list[dict]:
    """Return jobs that were in-flight when the server last stopped."""
    async with _db() as conn:
        async with conn.execute(
            "SELECT * FROM jobs WHERE status IN ('pending','running') ORDER BY created_at ASC"
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def delete_job(file_id: str):
    async with _db() as conn:
        await conn.execute("DELETE FROM jobs WHERE file_id=?", (file_id,))
        await conn.commit()
