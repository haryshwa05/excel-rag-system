"""
ChromaDB Client — singleton database connection.

WHAT IS A VECTOR DATABASE:
A regular database (PostgreSQL, MySQL) stores rows and lets you search
by exact values: "WHERE region = 'South' AND sales > 5000".

A vector database stores vectors (lists of numbers) and lets you search
by SIMILARITY: "find the 10 vectors most similar to this query vector."

ChromaDB does both — it stores vectors for similarity search AND metadata
for exact filtering. This combination is what makes RAG fast:
    1. Metadata filter: "only look in the Q4 sheet" (narrows 500 chunks to 100)
    2. Vector search: "find the 10 most similar to my question" (narrows 100 to 10)

WHY ChromaDB FOR THIS PROJECT:
- Zero setup: pip install chromadb and it works. No Docker, no server.
- Persists to disk: vectors survive server restarts.
- Good enough for ~1M vectors on a single machine.
- When you outgrow it, swap to Pinecone/Weaviate/Qdrant — the search
  interface is nearly identical.

WHAT IS A SINGLETON:
Creating a database client is expensive — it opens files, loads indexes,
allocates memory. You want to do this ONCE and reuse the same client
for every request.

    BAD (creates a new client per request):
        def search():
            client = chromadb.PersistentClient(...)  # expensive! every time!
            collection = client.get_collection(...)
            return collection.query(...)

    GOOD (reuses one client):
        client = get_chroma_client()  # created once, cached forever
        def search():
            collection = client.get_collection(...)
            return collection.query(...)

WHAT YOU'RE LEARNING:
- Singleton pattern with module-level caching
- Vector database concepts: collections, vectors, metadata, similarity search
- Why connection pooling / reuse matters for performance
"""

from __future__ import annotations

import chromadb

from core.config import settings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Singleton client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Module-level variable. None until first call to get_chroma_client().
_client: chromadb.PersistentClient | None = None


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Get or create the ChromaDB client singleton.

    HOW THE SINGLETON WORKS:
    First call: _client is None → create a new PersistentClient → cache it.
    Every subsequent call: _client already exists → return the cached one.

    The `global` keyword tells Python "I want to modify the module-level
    _client variable, not create a local one." Without `global`, the line
    `_client = chromadb.PersistentClient(...)` would create a LOCAL variable
    that disappears when the function returns — defeating the whole purpose.

    WHY PersistentClient:
    - PersistentClient saves to disk at the path you specify.
      Kill the server, restart, and all your vectors are still there.
    - EphemeralClient stores in memory only — data lost on restart.
      Useful for testing but not for a real app.

    WHY settings.chroma_persist_dir:
    The path comes from config.py, which reads from .env.
    Default is ./data/chroma. In production, this would point to
    a mounted volume or cloud storage.
    """
    global _client

    if _client is None:
        # Ensure the directory exists
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        _client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir)
        )

    return _client


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Collection helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_collection(file_id: str) -> chromadb.Collection:
    """
    Get the ChromaDB collection for a specific file.

    WHAT IS A COLLECTION:
    Think of it like a database table. Each uploaded file gets its own
    collection named "file_{file_id}". This separation means:

    - Searching "sales.xlsx" doesn't touch vectors from "inventory.csv"
    - Deleting a file = dropping one collection (instant, clean)
    - You can search across files by querying multiple collections
    - Each collection can have different metadata schemas

    RAISES ValueError if the collection doesn't exist (file not ingested yet).
    """
    client = get_chroma_client()

    try:
        return client.get_collection(name=f"file_{file_id}")
    except Exception as e:
        raise ValueError(
            f"No data found for file_id '{file_id}'. "
            f"Has this file been uploaded and processed? ({e})"
        )


def get_all_collections() -> list[chromadb.Collection]:
    """
    Get all file collections (for searching across all files).

    Filters to only collections that start with "file_" to avoid
    picking up any internal ChromaDB collections.

    Handles both old ChromaDB API (returns Collection objects) and
    new API (returns collection name strings).
    """
    client = get_chroma_client()
    raw = client.list_collections()

    collections = []
    for item in raw:
        name = item if isinstance(item, str) else getattr(item, "name", str(item))
        if name.startswith("file_"):
            try:
                collections.append(client.get_collection(name=name))
            except Exception:
                continue
    return collections


def delete_collection(file_id: str) -> bool:
    """
    Delete a file's collection. Returns True if deleted, False if not found.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(name=f"file_{file_id}")
        return True
    except Exception:
        return False


def get_collection_info(file_id: str) -> dict:
    """
    Get info about a collection — useful for the frontend to show file details.

    Returns:
        {
            "file_id": "abc123",
            "file_name": "sales.xlsx",
            "chunk_count": 500,
            "exists": True
        }
    """
    try:
        collection = get_collection(file_id)
        return {
            "file_id": file_id,
            "file_name": collection.metadata.get("file_name", "unknown") if collection.metadata else "unknown",
            "chunk_count": collection.count(),
            "exists": True,
        }
    except (ValueError, Exception):
        return {
            "file_id": file_id,
            "file_name": "unknown",
            "chunk_count": 0,
            "exists": False,
        }