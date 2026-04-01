"""
Searcher — finds the most relevant chunks for a user's question.

THIS IS THE "R" IN RAG (Retrieval-Augmented Generation):
    1. User asks: "Who had the highest sales in the South region?"
    2. Searcher embeds the question → query vector
    3. ChromaDB compares query vector against all stored chunk vectors
    4. Returns the top-K most similar chunks
    5. These chunks become the "context" that the LLM reads to answer

HOW SIMILARITY SEARCH WORKS:
    ChromaDB uses COSINE SIMILARITY to compare vectors.

    Cosine similarity measures the ANGLE between two vectors:
    - Angle = 0°  → similarity = 1.0 (identical meaning)
    - Angle = 90° → similarity = 0.0 (completely unrelated)
    - Angle = 180°→ similarity = -1.0 (opposite meaning)

    It ignores vector LENGTH and only cares about DIRECTION.
    This means a short question "top sales?" and a long question
    "what were the highest performing sales figures?" can both
    match the same chunks — because they point in the same direction
    in the embedding space.

    WHY COSINE OVER EUCLIDEAN DISTANCE:
    Euclidean distance (straight-line distance) is affected by vector
    magnitude. A chunk with longer text might have a larger vector,
    making it appear "far" from a short query even if the meaning is
    similar. Cosine similarity normalizes this away — only direction matters.

METADATA FILTERING (the speed trick):
    Without filtering: search all 5000 chunks across all files.
    With filtering: "only search in file X, sheet Q4" → search 200 chunks.

    Filtering happens BEFORE the vector comparison, so it makes
    the search faster AND more precise. This is what keeps latency
    flat as your total data grows — you narrow the search space first.

WHAT YOU'RE LEARNING:
- Cosine similarity and why it's used for text
- Metadata filtering for precision and speed
- Multi-file search strategy
- Result formatting for LLM consumption
"""

from dataclasses import dataclass

from core.config import settings
from core.embedder import get_embedder, BaseEmbedder
from db.chroma_client import get_collection, get_all_collections


@dataclass
class SearchResult:
    """
    One search result — a chunk that matched the user's question.

    WHY A DATACLASS AND NOT A DICT:
    Same reason as Chunk in parser.py — named fields, autocomplete,
    type safety. When the generator (file 8) receives these, it knows
    exactly what fields are available.

    FIELDS:
    - text: the chunk content (sent to the LLM as context)
    - score: cosine similarity (0.0 to 1.0, higher = more relevant)
    - metadata: file_id, sheet_name, row_range (for citations)
    - file_id: which file this came from (for multi-file search)
    """

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
    """
    Hybrid search: combines semantic (embedding) search with keyword
    (text match) search for better coverage. Semantic search is great
    for meaning-based queries; keyword search catches exact names,
    IDs, and specific values that embeddings might miss.
    """
    if top_k is None:
        top_k = settings.top_k

    embedder = get_embedder()
    query_embedding = await embedder.embed_query(query)

    if file_id:
        results = _search_collection(
            file_id=file_id,
            query_embedding=query_embedding,
            sheet_name=sheet_name,
            top_k=top_k,
        )
        keyword_results = _keyword_search(
            file_id=file_id,
            query=query,
            top_k=top_k,
        )
        results = _merge_results(results, keyword_results, top_k)
        return results
    else:
        return await _search_all_collections(
            query_embedding=query_embedding,
            sheet_name=sheet_name,
            top_k=top_k,
        )


def _search_collection(
    file_id: str,
    query_embedding: list[float],
    sheet_name: str | None = None,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Search within a single file's ChromaDB collection.

    HOW collection.query() WORKS:
    1. If `where` filter is provided, ChromaDB first filters chunks
       by metadata (exact match, very fast — like a WHERE clause)
    2. Then it runs cosine similarity on the FILTERED subset
    3. Returns top_k results sorted by similarity score

    The `where` parameter uses ChromaDB's filter syntax:
        {"sheet_name": "Q4"}              → exact match
        {"row_start": {"$gte": 100}}      → range filter
        {"$and": [{...}, {...}]}          → combine filters

    ABOUT THE include PARAMETER:
    By default, collection.query() returns IDs only (for speed).
    We need the actual text (documents) and metadata too, so we
    explicitly request them. We also request distances — ChromaDB
    returns DISTANCES (lower = more similar) not similarities
    (higher = more similar). We convert below.
    """
    collection = get_collection(file_id)

    # Build metadata filter
    where_filter = None
    if sheet_name:
        where_filter = {"sheet_name": sheet_name}

    # Run the query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    search_results = _parse_chroma_results(results, file_id)

    # Always include the summary chunk so the LLM knows dataset-level stats
    # (total rows, columns, etc.) — critical for aggregate questions
    summary_already = any(
        r.metadata.get("sheet_name") == "summary" for r in search_results
    )
    if not summary_already:
        try:
            summary = collection.get(
                where={"sheet_name": "summary"},
                include=["documents", "metadatas"],
            )
            if summary["ids"]:
                search_results.insert(0, SearchResult(
                    text=summary["documents"][0],
                    score=1.0,
                    metadata=summary["metadatas"][0],
                    file_id=file_id,
                ))
        except Exception:
            pass

    return search_results


async def _search_all_collections(
    query_embedding: list[float],
    sheet_name: str | None = None,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Search across ALL uploaded files and merge results.

    HOW MULTI-FILE SEARCH WORKS:
    1. Get all file collections from ChromaDB
    2. Query each collection with the same query embedding
    3. Collect all results into one list
    4. Sort by similarity score (best matches first)
    5. Return top_k from the merged list

    WHY NOT ONE BIG COLLECTION:
    If all files were in one collection, you couldn't easily:
    - Delete one file's data (you'd need to delete by metadata filter)
    - Show per-file search results
    - Let users choose which files to search
    Per-file collections are cleaner, even if multi-file search
    requires iterating.

    PERFORMANCE NOTE:
    This is O(num_files) collection queries. With 5 files, it's 5 queries
    of ~30ms each = ~150ms total. With 100 files, it's 3 seconds — at that
    point you'd want a single collection with metadata filtering instead.
    For a typical use case (5-20 files), per-file collections are fine.
    """
    all_results = []
    collections = get_all_collections()

    for collection in collections:
        file_id = collection.name.replace("file_", "", 1)

        where_filter = None
        if sheet_name:
            where_filter = {"sheet_name": sheet_name}

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            all_results.extend(_parse_chroma_results(results, file_id))
        except Exception:
            # If one collection fails, continue searching others.
            # Don't let one corrupt file break the entire search.
            continue

    # Sort all results by score (highest similarity first)
    all_results.sort(key=lambda r: r.score, reverse=True)

    # Return only top_k from the merged results
    return all_results[:top_k]


_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "which", "who",
    "whom", "whose", "how", "many", "much", "does", "do", "did", "can",
    "could", "would", "should", "will", "shall", "have", "has", "had",
    "been", "be", "being", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "and", "or", "but", "not", "no", "nor",
    "so", "if", "then", "than", "too", "very", "just", "only", "also",
    "more", "most", "some", "any", "all", "each", "every", "both", "few",
    "other", "such", "tell", "me", "show", "find", "get", "give", "total",
    "number", "data", "information", "details", "row", "rows", "column",
    "columns", "this", "that", "these", "those", "it", "its", "my", "your",
    "their", "there", "here", "where", "when", "why", "up", "down", "out",
    "off", "over", "under", "again", "further", "once", "same", "own",
    "like", "please", "about", "know", "need", "want", "look", "use",
}


def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a query for text search."""
    words = query.strip().split()
    keywords = []
    for word in words:
        cleaned = word.strip("?.,!\"'()[]{}:;").strip()
        if len(cleaned) >= 2 and cleaned.lower() not in _STOP_WORDS:
            keywords.append(cleaned)
    return keywords


def _keyword_search(
    file_id: str,
    query: str,
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Case-insensitive keyword search. Fetches all non-summary chunks
    and scores them by keyword match count.
    """
    try:
        collection = get_collection(file_id)
    except (ValueError, Exception):
        return []

    keywords = _extract_keywords(query)
    if not keywords:
        return []

    try:
        all_docs = collection.get(
            where={"sheet_name": {"$ne": "summary"}},
            include=["documents", "metadatas"],
        )
    except Exception:
        try:
            all_docs = collection.get(include=["documents", "metadatas"])
        except Exception:
            return []

    if not all_docs["ids"]:
        return []

    kw_lower = [kw.lower() for kw in keywords]
    results = []
    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        if meta.get("sheet_name") == "summary":
            continue

        doc_lower = doc.lower()
        matched = sum(1 for kw in kw_lower if kw in doc_lower)
        if matched > 0:
            score = 0.8 + (0.05 * min(matched, 4))
            results.append(SearchResult(
                text=doc,
                score=score,
                metadata=meta,
                file_id=file_id,
            ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def _merge_results(
    semantic: list[SearchResult],
    keyword: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Merge semantic and keyword results, deduplicating by chunk text.
    Caps at top_k + 1 to leave room for the summary chunk."""
    seen_texts: set[str] = set()
    merged = []

    # Keyword results first (they contain exact matches the user asked for)
    for r in keyword:
        key = r.text[:100]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append(r)

    # Then semantic results to fill remaining slots
    for r in semantic:
        key = r.text[:100]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append(r)

    return merged[:top_k + 1]


def _parse_chroma_results(
    results: dict,
    file_id: str,
) -> list[SearchResult]:
    """
    Convert ChromaDB's raw response into SearchResult objects.

    CHROMADB RESPONSE FORMAT:
    results = {
        "ids": [["file_abc__0", "file_abc__17", ...]],
        "documents": [["Columns: Name | Sales...", "Columns: Name | Sales...", ...]],
        "metadatas": [[{"file_id": "abc", "sheet_name": "Q4", ...}, ...]],
        "distances": [[0.234, 0.456, ...]]
    }

    Note the DOUBLE nesting — results["ids"][0] not results["ids"].
    This is because you can query multiple embeddings at once;
    each query gets its own list. We always query one at a time,
    so we always access [0].

    DISTANCE TO SIMILARITY CONVERSION:
    ChromaDB returns L2 (Euclidean) distances by default, where:
    - 0.0 = identical vectors
    - Higher = less similar

    We convert to a 0-1 similarity score using: 1 / (1 + distance)
    - distance = 0   → score = 1.0 (perfect match)
    - distance = 1   → score = 0.5
    - distance = 10  → score = 0.09 (very different)

    This makes scores intuitive: higher = more relevant.
    """
    search_results = []

    if not results["ids"] or not results["ids"][0]:
        return search_results

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # Convert distance to similarity score (0.0 to 1.0)
        score = 1.0 / (1.0 + dist)

        search_results.append(
            SearchResult(
                text=doc,
                score=score,
                metadata=meta,
                file_id=file_id,
            )
        )

    return search_results


def format_results_as_context(results: list[SearchResult]) -> str:
    """
    Format search results into a single context string for the LLM.

    THIS IS THE BRIDGE BETWEEN RETRIEVAL AND GENERATION.
    The searcher finds relevant chunks. The generator needs a single
    text string to inject into the LLM prompt. This function converts
    the list of SearchResults into that string.

    OUTPUT FORMAT:
        --- Source: sales.xlsx | Sheet: Q4 | Rows: 40-59 (relevance: 0.92) ---
        Columns: Name | Sales | Region | Date
        Row 40: John | 5000 | South | Jan 2024
        ...

        --- Source: sales.xlsx | Sheet: Q4 | Rows: 80-99 (relevance: 0.87) ---
        Columns: Name | Sales | Region | Date
        Row 80: Sarah | 7200 | North | Mar 2024
        ...

    WHY THIS FORMAT:
    - Source headers help the LLM cite where its answer came from
    - Relevance scores let the LLM know which chunks to trust more
    - Separators (---) clearly delineate different chunks so the LLM
      doesn't accidentally merge data from different row ranges
    - The chunk text includes headers (from parser.py), so the LLM
      always knows what each column means
    """
    if not results:
        return "No relevant data found."

    context_parts = []

    for result in results:
        meta = result.metadata
        source = meta.get("file_name", "unknown")
        sheet = meta.get("sheet_name", "default")
        row_start = meta.get("row_start", "?")
        row_end = meta.get("row_end", "?")

        header = (
            f"--- Source: {source} | Sheet: {sheet} | "
            f"Rows: {row_start}-{row_end} (relevance: {result.score:.2f}) ---"
        )

        context_parts.append(f"{header}\n{result.text}")

    return "\n\n".join(context_parts)