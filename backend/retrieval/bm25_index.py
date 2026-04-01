"""
BM25 index — built at ingestion time, searched in milliseconds at query time.

Cache policy: LRU, capped at MAX_ENTRIES files OR MAX_CHARS characters
(~500 MB assuming 1 byte/char). When either limit is exceeded, the least-
recently-used entry is evicted. This prevents unbounded memory growth when
many large files are indexed.
"""

import pickle
from collections import OrderedDict
from pathlib import Path

from rank_bm25 import BM25Okapi

from core.config import settings

# ── Cache limits ───────────────────────────────────────────────────────────────
_MAX_ENTRIES = 20
_MAX_CHARS   = 500_000_000     # ~500 MB proxy (1 char ≈ 1 byte)

# LRU cache: file_id → (bm25, texts, metadatas)
_cache: OrderedDict[str, tuple] = OrderedDict()
_cache_chars: dict[str, int]    = {}   # file_id → total chars in texts


# ── Internal helpers ───────────────────────────────────────────────────────────

def _dir() -> Path:
    settings.bm25_dir.mkdir(parents=True, exist_ok=True)
    return settings.bm25_dir


def _total_chars() -> int:
    return sum(_cache_chars.values())


def _evict():
    """Remove least-recently-used entries until within limits."""
    while _cache and (len(_cache) > _MAX_ENTRIES or _total_chars() > _MAX_CHARS):
        evicted_key, _ = _cache.popitem(last=False)  # FIFO for LRU
        _cache_chars.pop(evicted_key, None)


def _put(file_id: str, bm25: BM25Okapi, texts: list[str], metadatas: list[dict]):
    """Insert / refresh entry, then evict if over limits."""
    if file_id in _cache:
        _cache.move_to_end(file_id)
        _cache[file_id] = (bm25, texts, metadatas)
    else:
        _cache[file_id] = (bm25, texts, metadatas)
        _cache.move_to_end(file_id)
    _cache_chars[file_id] = sum(len(t) for t in texts)
    _evict()


def _get(file_id: str) -> tuple | None:
    """Return entry and refresh its recency."""
    if file_id not in _cache:
        return None
    _cache.move_to_end(file_id)
    return _cache[file_id]


# ── Public API ─────────────────────────────────────────────────────────────────

def build_and_save(file_id: str, texts: list[str], metadatas: list[dict]):
    """Build BM25 index from non-summary chunk texts and persist to disk."""
    if not texts:
        return

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    path = _dir() / f"{file_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts, "metadatas": metadatas}, f)

    _put(file_id, bm25, texts, metadatas)


def _load(file_id: str) -> tuple | None:
    cached = _get(file_id)
    if cached is not None:
        return cached

    path = _dir() / f"{file_id}.pkl"
    if not path.exists():
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)

    _put(file_id, data["bm25"], data["texts"], data["metadatas"])
    return _get(file_id)


def delete(file_id: str):
    _cache.pop(file_id, None)
    _cache_chars.pop(file_id, None)
    path = _dir() / f"{file_id}.pkl"
    if path.exists():
        path.unlink()


def search(file_id: str, query: str, top_k: int) -> list[dict]:
    """
    BM25 search. Returns list of {text, metadata, score} dicts.
    Only returns results with score > 0 (actual keyword matches).
    """
    data = _load(file_id)
    if data is None:
        return []

    bm25, texts, metadatas = data
    scores = bm25.get_scores(query.lower().split())

    indexed = sorted(
        ((i, float(s)) for i, s in enumerate(scores) if s > 0),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [
        {"text": texts[i], "metadata": metadatas[i], "score": score}
        for i, score in indexed
    ]


def cache_stats() -> dict:
    """Debug helper — returns current cache state."""
    return {
        "entries": len(_cache),
        "total_chars": _total_chars(),
        "file_ids": list(_cache.keys()),
    }
