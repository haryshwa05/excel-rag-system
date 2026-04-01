"""
LLM — the language model abstraction with streaming.

HOW LLMs GENERATE TEXT:
An LLM doesn't "think" and then write. It generates ONE TOKEN at a time:
    Token 1: "The"
    Token 2: " top"
    Token 3: " sales"
    Token 4: " region"
    Token 5: " is"
    Token 6: " South"
    ...

Each token takes ~20-50ms to generate. A 200-token answer takes ~4-10 seconds.

WITHOUT STREAMING:
    User asks question → waits 5 seconds → sees full answer all at once.
    This feels broken. Users think the app is frozen.

WITH STREAMING:
    User asks question → sees first word in ~1.5s → answer types itself out.
    Same total time, but the experience feels instant.

HOW STREAMING WORKS (async generators):
    A normal function returns ONCE:
        def get_answer():
            return "The top region is South"  # all at once

    An async generator YIELDS many times:
        async def stream_answer():
            yield "The"      # sent immediately
            yield " top"     # sent 30ms later
            yield " region"  # sent 30ms later
            ...

    The caller uses `async for` to receive each piece as it arrives:
        async for token in stream_answer():
            send_to_frontend(token)

    This is the foundation of Server-Sent Events (SSE), which is how
    the frontend receives the streaming response.

WHAT YOU'RE LEARNING:
- async generators: functions that yield values over time
- Streaming APIs: how OpenAI and Anthropic deliver token-by-token responses
- The strategy pattern again: same interface, different providers
- System prompts: how to control LLM behavior for RAG
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator
import asyncio

from core.config import settings


class BaseLLM(ABC):
    """
    The interface every LLM provider must implement.

    Two methods:
    - stream_answer: yields tokens one at a time (for the chat UI)
    - generate_answer: returns the complete response (for internal use)

    Most of the time you'll use stream_answer — it's better UX.
    generate_answer exists for cases where you need the full text
    before doing something with it (like logging or evaluation).
    """

    @abstractmethod
    async def stream_answer(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream an answer token by token.

        Args:
            question: The user's question ("What are the top sales by region?")
            context: The retrieved chunks joined together — this is the "R" in RAG.
                     The LLM answers ONLY from this context, not from its training data.
            system_prompt: Instructions for the LLM on how to behave.
                          We default to a RAG-specific prompt that says
                          "answer only from the provided data."

        Yields:
            str: One token at a time. "The" → " top" → " sales" → ...

        ABOUT AsyncGenerator[str, None]:
            - str: the type of values this generator yields (text tokens)
            - None: the type of values you can send INTO the generator (we don't use this)
            This is Python's way of typing a function that yields values over time.

        WHY context IS A STRING, NOT A LIST:
            The retriever returns a list of chunks. We join them into one string
            with separators before passing to the LLM. The LLM sees one block
            of context, not separate chunks. This is simpler and works better —
            LLMs handle continuous text better than structured lists.
        """
        # `yield` is required in the body for Python to recognize this
        # as a generator. This is an abstract method so it never runs.
        yield ""  # pragma: no cover

    async def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Get the complete answer as one string (non-streaming).

        Default implementation: collect all tokens from stream_answer.
        Subclasses can override this for efficiency if their API has
        a non-streaming endpoint.

        WHY THIS EXISTS:
        - Evaluation: you need the full answer to compute RAGAS scores
        - Logging: easier to log a complete string than a stream
        - Testing: simpler to assert against a full string
        """
        chunks = []
        async for token in self.stream_answer(question, context, system_prompt):
            chunks.append(token)
        return "".join(chunks)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The default RAG system prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_RAG_PROMPT = """You are a helpful data analyst. Answer questions using ONLY the provided context.

The context contains two types of information:
- A DATASET SUMMARY with pre-computed statistics over ALL rows (total counts,
  value distributions, numeric stats). Use this for aggregate questions like
  "how many?", "what percentage?", "most common?", "total?", etc.
- Individual DATA CHUNKS showing specific rows. Use these for questions about
  specific people, values, or rows.

Rules:
1. For aggregate/count questions, use the DATASET SUMMARY statistics directly.
   These are computed over the ENTIRE dataset and are accurate.
2. For specific lookups (e.g. "what is X's email?"), use the individual data chunks.
3. Cite rows/columns when referencing specific data.
4. If the context truly doesn't contain relevant information, say so briefly.
5. Be concise and direct. Do not hedge or over-qualify when the data is clear.

Context data:
{context}"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WHY THIS PROMPT MATTERS:
#
# Without rule 1-2: The LLM will use its training knowledge to fill gaps.
#     User: "What were Q4 sales?"
#     BAD:  "Based on typical industry trends, Q4 sales are usually..."
#     GOOD: "I don't have Q4 data in the provided context."
#
# Without rule 3: The LLM gives vague answers with no traceability.
#     BAD:  "The South region performed well."
#     GOOD: "According to rows 45-48, the South region had $23,400 in sales."
#
# This is called "grounding" — forcing the LLM to ground its answers
# in the provided data rather than its parametric (training) knowledge.
# Hallucination control is THE biggest challenge in production RAG.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OpenAILLM(BaseLLM):
    """
    OpenAI's GPT models with streaming.

    Uses gpt-4o-mini by default — cheapest GPT-4 class model.
    Good enough for RAG because the heavy lifting is done by retrieval,
    not by the LLM's reasoning. The LLM just needs to read the context
    and formulate a clear answer.
    """

    def __init__(self):
        from openai import AsyncOpenAI

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when using OpenAI LLM. "
                "Set it in your .env file."
            )

        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_llm_model

    async def stream_answer(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from OpenAI's API.

        HOW THIS WORKS STEP BY STEP:
        1. We send the request with stream=True
        2. OpenAI starts generating tokens
        3. As each token is generated, OpenAI sends it to us immediately
        4. We yield each token to our caller
        5. Our caller (the API route) sends it to the frontend via SSE
        6. The frontend appends it to the message bubble

        The user sees: "The" ... "top" ... "sales" ... appearing in real time.

        ABOUT `chunk.choices[0].delta.content`:
        - chunk: one piece of the streaming response
        - choices[0]: OpenAI can generate multiple completions; we use the first
        - delta: the CHANGE since the last chunk (just the new token)
        - content: the actual text of the token (can be None for metadata chunks)
        """
        prompt = system_prompt or DEFAULT_RAG_PROMPT
        formatted_prompt = prompt.format(context=context)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": question},
            ],
            stream=True,
            temperature=0.1,  # Low temperature = more factual, less creative
            # For RAG, you want deterministic answers, not creative ones.
            # 0.0 = fully deterministic, 1.0 = maximum randomness.
            # 0.1 gives slight variation while staying grounded.
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content


class AnthropicLLM(BaseLLM):
    """
    Anthropic's Claude models with streaming.

    Claude is particularly good at following instructions precisely,
    which makes it excellent for RAG — it's less likely to hallucinate
    or go beyond the provided context.
    """

    def __init__(self):
        from anthropic import AsyncAnthropic

        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when using Anthropic LLM. "
                "Set it in your .env file."
            )

        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_llm_model

    async def stream_answer(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Anthropic's API.

        DIFFERENCE FROM OPENAI:
        - Anthropic uses `system` as a top-level parameter, not a message role
        - Streaming events have different structure: event.type == "content_block_delta"
        - The text is in event.delta.text, not chunk.choices[0].delta.content

        Same concept, different API shape. This is exactly why we have
        the abstraction — the rest of our code doesn't know or care about
        these differences.
        """
        prompt = system_prompt or DEFAULT_RAG_PROMPT
        formatted_prompt = prompt.format(context=context)

        async with self.client.messages.stream(
            model=self.model,
            system=formatted_prompt,
            messages=[
                {"role": "user", "content": question},
            ],
            max_tokens=2048,
            temperature=0.1,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class GrokLLM(BaseLLM):
    """
    xAI's Grok models with streaming.

    HOW THIS WORKS:
    Grok's API is OpenAI-compatible — it uses the same request/response
    format as OpenAI, just with a different base_url and API key.
    So we reuse the OpenAI Python client and just point it at Grok's server.

    This is a common pattern in the industry — many LLM providers
    (Grok, Together AI, Groq, Fireworks) use the OpenAI-compatible format
    so developers can switch with just a URL change.

    WHY GROK FOR THIS PROJECT:
    - Free tier available for experimentation
    - grok-3-mini-fast is fast and cheap — good for RAG
    - OpenAI-compatible = minimal code to support it
    """

    def __init__(self):
        from openai import AsyncOpenAI

        if not settings.grok_api_key:
            raise ValueError(
                "GROK_API_KEY is required when using Grok LLM. "
                "Set it in your .env file. Get one at https://console.x.ai"
            )

        # Point the OpenAI client at Grok's API server
        # Same client, different base_url — that's the beauty of
        # OpenAI-compatible APIs
        self.client = AsyncOpenAI(
            api_key=settings.grok_api_key,
            base_url=settings.grok_base_url,
        )
        self.model = settings.grok_llm_model

    async def stream_answer(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Grok's API.

        Identical to OpenAI's streaming because Grok uses the same format.
        The only difference is self.client points to api.x.ai instead of
        api.openai.com — everything else is the same.
        """
        prompt = system_prompt or DEFAULT_RAG_PROMPT
        formatted_prompt = prompt.format(context=context)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": question},
            ],
            stream=True,
            temperature=0.1,
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content


class GeminiLLM(BaseLLM):
    """
    Google's Gemini models with streaming.

    Uses the OpenAI-compatible endpoint so we can reuse the same
    AsyncOpenAI client — just pointed at Google's API server.
    """

    def __init__(self):
        from openai import AsyncOpenAI

        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when using Gemini LLM. "
                "Set it in your .env file. Get one at https://aistudio.google.com/apikey"
            )

        self.client = AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
        )
        self.model = settings.gemini_llm_model

    async def stream_answer(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        prompt = system_prompt or DEFAULT_RAG_PROMPT
        formatted_prompt = prompt.format(context=context)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": question},
            ],
            stream=True,
            temperature=0.1,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class GroqLLM(BaseLLM):
    """
    Groq's free inference API with streaming.

    Groq offers a generous free tier with models like Llama 3.3 70B.
    Uses the OpenAI-compatible API format — same client, different base_url.
    Get a free key at https://console.groq.com/keys
    """

    def __init__(self):
        from openai import AsyncOpenAI

        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required when using Groq LLM. "
                "Set it in your .env file. Get a free key at https://console.groq.com/keys"
            )

        self.client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        self.model = settings.groq_llm_model

    async def stream_answer(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        prompt = system_prompt or DEFAULT_RAG_PROMPT
        formatted_prompt = prompt.format(context=context)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": question},
            ],
            stream=True,
            temperature=0.1,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class OllamaLLM(BaseLLM):
    """
    Ollama-hosted models using the official ollama Python SDK.

    Uses `ollama.AsyncClient` directly — no OpenAI-compat shim needed.
    Works with local Ollama and Ollama Cloud (qwen3.5:cloud, etc.).

    Set in .env:
        LLM_PROVIDER=ollama
        OLLAMA_LLM_MODEL=qwen3.5:cloud
        OLLAMA_BASE_URL=http://localhost:11434   (optional)
    """

    def __init__(self):
        from ollama import AsyncClient
        self.client = AsyncClient(host=settings.ollama_base_url)
        self.model  = settings.ollama_llm_model

    async def stream_answer(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        prompt    = system_prompt or DEFAULT_RAG_PROMPT
        formatted = prompt.format(context=context)

        from ollama import ResponseError

        try:
            stream = await asyncio.wait_for(
                self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": formatted},
                        {"role": "user",   "content": question},
                    ],
                    stream=True,
                    options={"temperature": 0.1},
                ),
                timeout=25,
            )

            # Inactivity timeout so the frontend never hangs forever.
            aiter = stream.__aiter__()
            while True:
                try:
                    part = await asyncio.wait_for(anext(aiter), timeout=45)
                except StopAsyncIteration:
                    break
                content = part.message.content
                if content:
                    yield content

        except ResponseError as e:
            if getattr(e, "status_code", None) == 401:
                raise RuntimeError(
                    "Ollama unauthorized (401). Run `ollama signin` and verify "
                    "model access, or switch to a local model."
                ) from e
            raise RuntimeError(f"Ollama error: {e}") from e
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                "Ollama timed out. Check server/model health or try a smaller query."
            ) from e


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_llm() -> BaseLLM:
    """
    Returns the right LLM based on the LLM_PROVIDER setting.
    Change LLM_PROVIDER in .env → entire app uses a different LLM.
    """
    providers = {
        "openai":     OpenAILLM,
        "anthropic":  AnthropicLLM,
        "grok":       GrokLLM,
        "gemini":     GeminiLLM,
        "groq":       GroqLLM,
        "ollama":     OllamaLLM,
        "local":      OllamaLLM,   # "local" now maps to Ollama
    }

    provider_class = providers.get(settings.llm_provider)

    if provider_class is None:
        raise ValueError(
            f"Unknown LLM provider: '{settings.llm_provider}'. "
            f"Must be one of: {list(providers.keys())}"
        )

    return provider_class()