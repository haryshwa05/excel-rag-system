"""
Configuration — the single source of truth for the entire app.

WHY THIS FILE EXISTS:
Every setting lives here. No other file has hardcoded API keys, model names,
or magic numbers. When you want to switch from OpenAI to HuggingFace embeddings,
you change ONE line in your .env file. Nothing else changes.

HOW IT WORKS:
Pydantic BaseSettings reads from environment variables first, then falls back
to the .env file, then falls back to the defaults defined here. This means:
  1. In production: set real env vars (secure, no files on disk)
  2. In development: use a .env file (convenient, git-ignored)
  3. For testing: override in code (settings = Settings(chunk_size=5))

WHAT YOU'RE LEARNING:
- Pydantic BaseSettings: type-safe config with validation
- 12-factor app: config lives in the environment, never in code
- Literal types: restrict values to a known set at the type level
- Computed properties: derive values from other settings
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All app configuration in one place.

    Each field maps to an environment variable with the same name (case-insensitive).
    For example, the field `embedding_provider` reads from EMBEDDING_PROVIDER in .env.

    The Literal type on provider fields means pydantic will reject any value
    that isn't in the list — you get a clear error at startup, not a mysterious
    KeyError deep in your embedder code at runtime.
    """

    # ── Model configuration ────────────────────────────────────────────
    # These control WHICH embedding and LLM providers your app uses.
    # Change these in .env to swap providers without touching any code.

    embedding_provider: Literal["openai", "huggingface", "local"] = "huggingface"
    llm_provider: Literal["openai", "anthropic", "grok", "gemini", "groq", "local"] = "grok"

    # ── API keys ───────────────────────────────────────────────────────
    # Only fill in the keys for providers you're using.
    # With huggingface embeddings + grok LLM, you only need GROK_API_KEY.

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    grok_api_key: str = ""

    # ── Model names ────────────────────────────────────────────────────
    # Defaults are the best price/performance choices as of 2024-2025.
    # text-embedding-3-small: $0.02/1M tokens, 1536 dimensions, fast
    # gpt-4o-mini: cheapest GPT-4 class model, good enough for RAG
    # grok-3-mini-fast: fast and cheap Grok model, good for RAG

    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"
    anthropic_llm_model: str = "claude-sonnet-4-20250514"
    grok_llm_model: str = "grok-3-mini-fast"
    grok_base_url: str = "https://api.x.ai/v1"
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-2.0-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    groq_api_key: str = ""
    groq_llm_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Chunking settings ──────────────────────────────────────────────
    # These numbers are tuned for tabular data (Excel/CSV rows).
    #
    # chunk_size = 20: Each chunk contains 20 rows of data.
    #   - Too small (5 rows): you lose context, embeddings are noisy
    #   - Too large (100 rows): chunks are too broad, search returns irrelevant data
    #   - 20 is the sweet spot for tabular data (proven by industry benchmarks)
    #
    # chunk_overlap = 3: The last 3 rows of chunk N are the first 3 rows of chunk N+1.
    #   - This ensures a question about row 20 finds context in both chunks
    #   - Without overlap, boundary rows get orphaned
    #
    # pandas_chunk_size = 1000: How many rows pandas reads into RAM at once.
    #   - A 500k-row Excel file is read in 500 batches of 1000
    #   - Memory stays flat at ~50MB regardless of file size
    #   - This is pandas-level chunking (RAM management), not RAG chunking (semantic)

    chunk_size: int = 20
    chunk_overlap: int = 3
    pandas_chunk_size: int = 1000

    # ── Embedding batch size ───────────────────────────────────────────
    # How many chunks to embed in one API call.
    # OpenAI allows up to 2048 per call, but 100 is safer:
    #   - Keeps individual requests under timeout limits
    #   - If one batch fails, you only retry 100 chunks, not 2000
    #   - Progress reporting is smoother (updates every 100 chunks)

    embedding_batch_size: int = 100

    # ── Retrieval settings ─────────────────────────────────────────────
    # top_k = 10: Return the 10 most similar chunks to the question.
    #   - Too few (3): might miss relevant data scattered across chunks
    #   - Too many (50): floods the LLM context with noise, costs more tokens
    #   - 10 is the default; the reranker (phase 4) will trim this to the best 5

    top_k: int = 5

    # ── ChromaDB settings ──────────────────────────────────────────────
    # persist_directory: where ChromaDB saves vectors to disk.
    #   - Vectors survive server restarts (no re-embedding on reboot)
    #   - In production, this would be a cloud-hosted vector DB instead

    chroma_persist_dir: Path = Path("./data/chroma")

    # ── File upload settings ───────────────────────────────────────────
    upload_dir: Path = Path("./data/uploads")
    max_file_size_mb: int = 100

    # ── Server settings ────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # ── Pydantic settings configuration ────────────────────────────────
    # This tells pydantic WHERE to find the .env file and HOW to read it.
    # env_file = ".env": look for a file called .env in the current directory
    # env_file_encoding = "utf-8": handle special characters in values
    # case_sensitive = False: OPENAI_API_KEY and openai_api_key both work
    # extra = "ignore": don't crash if .env has variables we don't use

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# ── Singleton instance ─────────────────────────────────────────────────
# Create ONE Settings object that the entire app shares.
# Every other file does: from core.config import settings
# The .env file is read exactly once at startup.

settings = Settings()

# Log the active providers at import time for debugging
print(f"[config] LLM_PROVIDER={settings.llm_provider}, EMBEDDING_PROVIDER={settings.embedding_provider}")