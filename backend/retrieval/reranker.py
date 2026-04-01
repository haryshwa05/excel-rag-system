"""
Cross-encoder reranker with thread-safe lazy loading.

The asyncio.Lock prevents a race condition: if two queries arrive simultaneously
before the model is loaded, both will call _load() concurrently, resulting in
two model loads fighting over the same global variable. The lock ensures only
one coroutine loads the model; all others wait and then reuse it.

Model: ~67 MB, downloads once from HuggingFace, runs fully locally.
Warmup happens at startup in background so first-query latency is normal.
"""

import asyncio
from typing import Optional

_model = None
_lock: Optional[asyncio.Lock] = None


def _get_lock() -> asyncio.Lock:
    """Lazily create the lock inside the running event loop."""
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def _load():
    """Sync model load — runs inside asyncio.to_thread."""
    global _model
    if _model is not None:
        return
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[reranker] Cross-encoder loaded.")
    except Exception as e:
        print(f"[reranker] Load failed ({e}). Reranking disabled.")


async def warmup():
    """Download/load the model at startup without blocking the event loop."""
    async with _get_lock():
        if _model is not None:
            return
        await asyncio.to_thread(_load)


async def rerank(query: str, results: list, top_k: int) -> list:
    """
    Rerank using the cross-encoder. Falls back to original order if unavailable.
    Acquires the lock so concurrent callers wait for the model to finish loading
    rather than each trying to load it independently.
    """
    # Fast path: model already loaded, no lock needed
    if _model is None:
        async with _get_lock():
            if _model is None:
                await asyncio.to_thread(_load)

    if _model is None or len(results) <= 1:
        return results[:top_k]

    try:
        pairs = [(query, r.text) for r in results]
        scores = await asyncio.to_thread(_model.predict, pairs)
        reranked = sorted(zip(results, scores), key=lambda x: float(x[1]), reverse=True)
        return [r for r, _ in reranked[:top_k]]
    except Exception as e:
        print(f"[reranker] Reranking failed ({e}). Using original order.")
        return results[:top_k]
