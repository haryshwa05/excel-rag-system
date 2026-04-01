"""
Hybrid search: BM25 keyword + semantic vector + cross-encoder reranker.

Flow per query:
  1. Embed query → semantic search top (2 × top_k) candidates
  2. BM25 keyword search  → top top_k candidates
  3. Merge + deduplicate
  4. Cross-encoder rerank → final top_k
  5. Always prepend summary chunk for dataset-level context
"""

from dataclasses import dataclass

from core.config import settings
from core.embedder import get_embedder
from db.chroma_client import get_collection, get_all_collections
from retrieval import bm25_index
from retrieval import reranker as reranker_module


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    file_id: str


async def search(
    query: str,
    file_id: str | None = None,
    sheet_name: str | None = None,
    top_k: int | None = None,
) -> list[SearchResult]:
    if top_k is None:
        top_k = settings.top_k

    embedder = get_embedder()
    query_embedding = await embedder.embed_query(query)

    if file_id:
        semantic = _semantic_search(file_id, query_embedding, sheet_name, top_k=top_k * 2)
        keyword  = _bm25_search(file_id, query, top_k=top_k)
        merged   = _merge(semantic, keyword, limit=top_k * 2)
        final    = await reranker_module.rerank(query, merged, top_k)
        return _inject_summary(final, file_id)
    else:
        return await _search_all(query_embedding, sheet_name, top_k)


# ── Search helpers ────────────────────────────────────────────────

def _semantic_search(
    file_id: str,
    query_embedding: list[float],
    sheet_name: str | None,
    top_k: int,
) -> list[SearchResult]:
    try:
        collection = get_collection(file_id)
    except Exception:
        return []

    where = {"sheet_name": sheet_name} if sheet_name else None
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    return _parse_results(results, file_id)


def _bm25_search(file_id: str, query: str, top_k: int) -> list[SearchResult]:
    hits = bm25_index.search(file_id, query, top_k)
    return [
        SearchResult(text=h["text"], score=h["score"], metadata=h["metadata"], file_id=file_id)
        for h in hits
    ]


def _merge(
    semantic: list[SearchResult],
    keyword: list[SearchResult],
    limit: int,
) -> list[SearchResult]:
    seen: set[str] = set()
    merged: list[SearchResult] = []

    # Keyword results first — they have exact matches
    for r in keyword:
        key = r.text[:120]
        if key not in seen:
            seen.add(key)
            merged.append(r)

    for r in semantic:
        key = r.text[:120]
        if key not in seen:
            seen.add(key)
            merged.append(r)

    return merged[:limit]


def _inject_summary(results: list[SearchResult], file_id: str) -> list[SearchResult]:
    """Always prepend the summary chunk if not already present."""
    if any(r.metadata.get("sheet_name") == "summary" for r in results):
        return results
    try:
        collection = get_collection(file_id)
        summary = collection.get(
            where={"sheet_name": "summary"},
            include=["documents", "metadatas"],
        )
        if summary["ids"]:
            return [
                SearchResult(
                    text=summary["documents"][0],
                    score=1.0,
                    metadata=summary["metadatas"][0],
                    file_id=file_id,
                )
            ] + results
    except Exception:
        pass
    return results


async def _search_all(
    query_embedding: list[float],
    sheet_name: str | None,
    top_k: int,
) -> list[SearchResult]:
    all_results: list[SearchResult] = []
    for collection in get_all_collections():
        file_id = collection.name.replace("file_", "", 1)
        where = {"sheet_name": sheet_name} if sheet_name else None
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            all_results.extend(_parse_results(results, file_id))
        except Exception:
            continue
    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]


def _parse_results(results: dict, file_id: str) -> list[SearchResult]:
    if not results["ids"] or not results["ids"][0]:
        return []
    return [
        SearchResult(
            text=doc,
            score=1.0 / (1.0 + dist),
            metadata=meta,
            file_id=file_id,
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def format_results_as_context(results: list[SearchResult]) -> str:
    if not results:
        return "No relevant data found."
    parts = []
    for r in results:
        meta   = r.metadata
        source = meta.get("file_name", "unknown")
        sheet  = meta.get("sheet_name", "default")
        r_s    = meta.get("row_start", "?")
        r_e    = meta.get("row_end", "?")
        header = (
            f"--- Source: {source} | Section: {sheet} | "
            f"Rows: {r_s}-{r_e} (relevance: {r.score:.2f}) ---"
        )
        parts.append(f"{header}\n{r.text}")
    return "\n\n".join(parts)
