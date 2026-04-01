"""
Main — the FastAPI application entry point.

THIS IS WHERE THE APP STARTS.
    uvicorn main:app --reload

That command tells uvicorn (the ASGI server):
- main: look in the file called main.py
- app: find the variable called `app`
- --reload: restart automatically when code changes (dev only)

WHAT THIS FILE DOES:
1. Creates the FastAPI application
2. Configures CORS (so the frontend can talk to the backend)
3. Mounts the API routes
4. Creates necessary directories

It's intentionally thin — all logic lives in the route handlers
and the modules they call.

WHAT YOU'RE LEARNING:
- FastAPI application setup
- CORS: why browsers block cross-origin requests and how to allow them
- ASGI: the protocol that connects uvicorn to FastAPI
- Route mounting with prefixes
"""

import sys
from pathlib import Path

# ── Ensure the backend directory is on Python's import path ────────
# When you run `uvicorn main:app` from inside the backend/ folder,
# Python sometimes can't find `core`, `api`, etc. as modules.
# This adds the backend directory to sys.path so imports like
# `from core.config import settings` always work.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from api.routes import router


# ── Create the FastAPI application ─────────────────────────────────
app = FastAPI(
    title="RAG Application",
    description="Ask questions about your Excel and CSV data using AI.",
    version="1.0.0",
    # This shows up on the auto-generated docs page at /docs
)


# ── Configure CORS ─────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing.
#
# THE PROBLEM:
# Your frontend runs on http://localhost:3000 (Next.js dev server).
# Your backend runs on http://localhost:8000 (FastAPI).
# These are DIFFERENT ORIGINS (different ports = different origin).
#
# By default, browsers BLOCK requests from one origin to another.
# This is a security feature — it prevents malicious websites from
# making requests to your bank's API using your cookies.
#
# THE SOLUTION:
# Tell FastAPI "allow requests from these origins."
# The backend sends back headers like:
#   Access-Control-Allow-Origin: http://localhost:3000
# The browser sees this header and allows the request.
#
# IN PRODUCTION:
# Replace ["*"] with your actual frontend URL:
#   allow_origins=["https://your-app.com"]
# Using "*" in production is a security risk.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://your-domain.com"]
    allow_credentials=True,
    allow_methods=["*"],   # Allow all HTTP methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],   # Allow all headers
)


# ── Mount routes ───────────────────────────────────────────────────
# All routes in routes.py get prefixed with /api.
# So router's "/upload" becomes "/api/upload" in the actual URL.
#
# WHY /api PREFIX:
# When you deploy, the frontend and backend often share the same domain.
# The /api prefix makes it clear which requests go to the backend.
# example.com/          → frontend (Next.js)
# example.com/api/query → backend (FastAPI)

app.include_router(router, prefix="/api")


# ── Startup event ──────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """
    Runs once when the server starts.

    Creates necessary directories and validates configuration.
    If something is wrong (missing API key for selected provider, etc.),
    it's better to fail here at startup than later during a user request.
    """
    # Create data directories
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    print(f"RAG Application starting...")
    print(f"  Embedding provider: {settings.embedding_provider}")
    print(f"  LLM provider:       {settings.llm_provider}")
    print(f"  ChromaDB path:      {settings.chroma_persist_dir}")
    print(f"  Upload directory:   {settings.upload_dir}")
    print(f"  Debug mode:         {settings.debug}")
    print(f"  Docs available at:  http://{settings.host}:{settings.port}/docs")


# ── Root endpoint ──────────────────────────────────────────────────
@app.get("/")
async def root():
    """
    Root endpoint — just confirms the API is running.
    Visiting http://localhost:8000/ in the browser shows this.
    """
    return {
        "app": "RAG Application",
        "docs": "/docs",
        "health": "/api/health",
    }