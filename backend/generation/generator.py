"""
Generator — the "G" in RAG. Combines retrieved chunks with the LLM.

WHAT THIS FILE DOES:
Takes the user's question + retrieved chunks → builds a prompt → streams the answer.

This is where the three pieces come together:
    1. Searcher found the 10 most relevant chunks (retrieval)
    2. Generator builds a prompt with those chunks as context
    3. LLM reads the context and generates an answer (generation)

THE PROMPT STRUCTURE (what the LLM actually sees):

    ┌─────────────────────────────────────────────────────┐
    │  SYSTEM PROMPT                                      │
    │  "You are a data analyst. Answer ONLY from the      │
    │   provided context. Cite row numbers..."            │
    │                                                     │
    │  Context data:                                      │
    │  --- Source: sales.xlsx | Sheet: Q4 | Rows: 40-59 --│
    │  Columns: Name | Sales | Region | Date              │
    │  Row 40: John | 5000 | South | Jan 2024             │
    │  Row 41: Sarah | 7200 | North | Jan 2024            │
    │  ...                                                │
    │                                                     │
    │  --- Source: sales.xlsx | Sheet: Q4 | Rows: 80-99 --│
    │  Columns: Name | Sales | Region | Date              │
    │  Row 80: Maria | 9200 | South | Mar 2024            │
    │  ...                                                │
    ├─────────────────────────────────────────────────────┤
    │  USER MESSAGE                                       │
    │  "Who had the highest sales in the South region?"   │
    └─────────────────────────────────────────────────────┘

    The LLM reads top-to-bottom:
    1. System prompt tells it HOW to behave
    2. Context gives it the DATA to work with
    3. User message is the QUESTION to answer

    The LLM then generates:
    "According to rows 80-99, Maria had the highest sales in the
     South region at $9,200 in March 2024."

WHY PROMPT ENGINEERING MATTERS FOR RAG:
    The same retrieved chunks can produce wildly different answers
    depending on how you structure the prompt:

    BAD prompt: "Here's some data: {context}. Answer: {question}"
    → LLM might ignore the data and use training knowledge
    → LLM might not cite sources
    → LLM might hallucinate numbers

    GOOD prompt: (what we use)
    → Forces the LLM to only use provided data
    → Requires row/column citations
    → Tells the LLM to say "I don't know" when data is insufficient
    → Low temperature (0.1) keeps answers factual

    In production RAG, 80% of quality improvements come from prompt
    changes, not code changes. This is the file you'll iterate on most.

WHAT YOU'RE LEARNING:
- Prompt construction for RAG
- Context injection — putting retrieved data into the prompt
- Conversation history — multi-turn chat
- The complete search → format → prompt → stream pipeline
- Async generator chaining — one generator feeding another
"""

from typing import AsyncGenerator

from core.llm import get_llm, BaseLLM, DEFAULT_RAG_PROMPT
from retrieval.searcher import search, format_results_as_context, SearchResult


async def generate_answer(
    question: str,
    file_id: str | None = None,
    sheet_name: str | None = None,
    chat_history: list[dict] | None = None,
    system_prompt: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    The main entry point. Question goes in, streamed answer comes out.

    This function orchestrates the full RAG pipeline:
        1. Search for relevant chunks
        2. Format chunks into context string
        3. Build the prompt with context + question
        4. Stream the LLM's response token by token

    Args:
        question: The user's question.
        file_id: Search only this file (None = search all files).
        sheet_name: Filter to this sheet within the file.
        chat_history: Previous messages for multi-turn conversations.
            Format: [
                {"role": "user", "content": "What are total sales?"},
                {"role": "assistant", "content": "Total sales are $142,000..."},
                {"role": "user", "content": "Break that down by region"}
            ]
            The last user message is the current question.
        system_prompt: Custom system prompt (None = use default RAG prompt).

    Yields:
        str: Tokens of the answer, one at a time.

    WHY CHAT HISTORY MATTERS:
    Without history, every question is independent:
        User: "What are total sales?"
        Bot:  "Total sales are $142,000 across all regions."
        User: "Break that down by region"
        Bot:  "I don't understand what you want me to break down."
              ← doesn't know "that" refers to sales!

    With history, the LLM has context from previous turns:
        User: "Break that down by region"
        Bot:  "Here's the sales breakdown by region:
               South: $52,000, North: $48,000..."
              ← understands "that" = sales from previous message
    """
    # ── Step 1: Retrieve relevant chunks ───────────────────────────
    # The searcher embeds the question and finds matching chunks
    results: list[SearchResult] = await search(
        query=question,
        file_id=file_id,
        sheet_name=sheet_name,
    )

    # ── Step 2: Format chunks into context string ──────────────────
    # This turns the list of SearchResult objects into one text block
    # that the LLM can read, with source headers and separators
    context = format_results_as_context(results)

    # ── Step 3: Build the enhanced question with history ───────────
    # If there's chat history, we include it so the LLM understands
    # references like "that", "those", "the same region", etc.
    enhanced_question = _build_question_with_history(question, chat_history)

    # ── Step 4: Stream the LLM response ────────────────────────────
    llm = get_llm()

    async for token in llm.stream_answer(
        question=enhanced_question,
        context=context,
        system_prompt=system_prompt,
    ):
        yield token


async def generate_answer_full(
    question: str,
    file_id: str | None = None,
    sheet_name: str | None = None,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Non-streaming version. Returns the complete answer plus metadata.

    Returns:
        {
            "answer": "According to rows 80-99, Maria had the highest...",
            "sources": [
                {"file": "sales.xlsx", "sheet": "Q4", "rows": "80-99", "score": 0.92},
                {"file": "sales.xlsx", "sheet": "Q4", "rows": "40-59", "score": 0.87},
            ],
            "chunks_searched": 10,
        }

    WHY THIS EXISTS:
    - Evaluation: RAGAS needs the full answer + sources to compute scores
    - Testing: easier to assert against a complete response
    - API clients that don't support streaming
    - Logging: you want to log complete answers for debugging
    """
    # Get search results first (we need them for source metadata)
    results = await search(
        query=question,
        file_id=file_id,
        sheet_name=sheet_name,
    )

    context = format_results_as_context(results)
    enhanced_question = _build_question_with_history(question, chat_history)

    # Collect all tokens into one string
    llm = get_llm()
    answer_parts = []
    async for token in llm.stream_answer(
        question=enhanced_question,
        context=context,
    ):
        answer_parts.append(token)

    answer = "".join(answer_parts)

    # Build source citations from the search results
    sources = [
        {
            "file_name": r.metadata.get("file_name", "unknown"),
            "sheet_name": r.metadata.get("sheet_name", "default"),
            "row_start": r.metadata.get("row_start"),
            "row_end": r.metadata.get("row_end"),
            "score": round(r.score, 3),
        }
        for r in results
    ]

    return {
        "answer": answer,
        "sources": sources,
        "chunks_searched": len(results),
    }


def _build_question_with_history(
    question: str,
    chat_history: list[dict] | None = None,
) -> str:
    """
    Combine chat history with the current question.

    FORMAT SENT TO LLM:
        Previous conversation:
        User: What are total sales?
        Assistant: Total sales are $142,000 across all regions.

        Current question: Break that down by region

    WHY NOT JUST APPEND TO MESSAGES:
    Different LLM providers handle conversation history differently.
    OpenAI wants a list of message objects. Anthropic wants a similar
    but slightly different format. By flattening history into the
    question string, we keep it provider-agnostic — it works the same
    regardless of which LLM is behind the abstraction.

    CONTEXT WINDOW LIMITS:
    Chat history can grow very long. We limit to the last 10 exchanges
    (20 messages) to avoid exceeding the LLM's context window.
    In a production app, you'd use more sophisticated strategies:
    - Summarize older history
    - Only include history relevant to the current question
    - Track token count and trim when approaching the limit
    """
    if not chat_history:
        return question

    # Limit to last 10 exchanges to avoid context window overflow
    recent_history = chat_history[-20:]

    history_text = "Previous conversation:\n"
    for msg in recent_history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"

    return f"{history_text}\nCurrent question: {question}"