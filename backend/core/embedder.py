"""
Embedder — turns text into vectors (embeddings).

WHY EMBEDDINGS MATTER:
Computers can't understand "what are the top sales by region?" as meaning.
But they CAN compare lists of numbers. An embedding model converts text into
a vector (list of ~1536 numbers) where similar meanings produce similar vectors.

    "total sales by region"  →  [0.023, -0.441, 0.187, ...]
    "revenue breakdown by area" →  [0.019, -0.438, 0.192, ...]  ← very similar!
    "the weather is nice today" →  [0.871, 0.102, -0.553, ...]  ← very different

At query time, we embed the question, then find the data chunks whose vectors
are closest. This is called "semantic search" — searching by meaning, not keywords.

WHY AN ABSTRACTION:
OpenAI, HuggingFace, and local models all produce embeddings, but their APIs
are completely different. We don't want the rest of our code to care.

This is the STRATEGY PATTERN:
    1. Define an interface (BaseEmbedder) with the methods every embedder must have
    2. Each provider implements that interface differently
    3. A factory function returns the right one based on config
    4. Every other file just calls embedder.embed_batch() — doesn't know or care
       which provider is behind it

WHAT YOU'RE LEARNING:
- ABC (Abstract Base Class): forces subclasses to implement specific methods
- @abstractmethod: if a subclass forgets to implement this, Python crashes at
  class creation time — not at runtime when a user hits the endpoint
- Batch processing: embedding 100 texts in one API call vs 100 separate calls
  is literally 50x faster and 50x cheaper
- Factory pattern: get_embedder() reads config and returns the right class
"""

from abc import ABC, abstractmethod

from core.config import settings


class BaseEmbedder(ABC):
    """
    The interface every embedder must implement.

    ABC = Abstract Base Class. You can't instantiate this directly:
        embedder = BaseEmbedder()  # ← TypeError!

    You MUST create a subclass that implements all @abstractmethod methods.
    This is Python's way of enforcing a contract — "if you're an embedder,
    you MUST be able to do these things."

    WHY THIS MATTERS FOR YOUR CAREER:
    Every serious Python codebase uses ABCs for this pattern. In interviews,
    you'll be asked "how would you make this swappable?" — this is the answer.
    """

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text strings into a list of embedding vectors.

        Args:
            texts: ["row 1 data...", "row 2 data...", ...]

        Returns:
            [[0.023, -0.441, ...], [0.019, -0.438, ...], ...]
            One vector per input text. Each vector has the same dimensionality
            (e.g., 1536 for OpenAI's text-embedding-3-small).

        WHY async:
            Embedding calls go over the network to OpenAI/HuggingFace APIs.
            async means we don't block the server while waiting for a response —
            it can handle other requests in the meantime.

        WHY list[str] not str:
            Batching. One API call with 100 texts is WAY faster than 100 calls
            with 1 text each. The network round-trip alone is ~100ms per call.
            100 calls = 10 seconds. 1 batched call = 0.2 seconds.
        """
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string. Returns one vector.

        WHY A SEPARATE METHOD:
        Some providers (like OpenAI) use different parameters for queries vs
        documents. For example, some models prepend "query: " or "passage: "
        internally to differentiate search queries from stored documents.
        Having a separate method lets each provider handle this correctly.
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Return the dimensionality of embeddings this model produces.

        ChromaDB needs to know this when creating a collection.
        OpenAI text-embedding-3-small = 1536 dimensions.
        all-MiniLM-L6-v2 = 384 dimensions.
        If the dimensions don't match, vector search breaks silently.
        """
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Provider implementations
# Each one implements the same interface but talks to a different API.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OpenAIEmbedder(BaseEmbedder):
    """
    Uses OpenAI's embedding API.

    Cost: ~$0.02 per 1 million tokens (incredibly cheap).
    Speed: ~0.2s for 100 texts in one batch.
    Quality: Best general-purpose embeddings available.
    Downside: Requires API key, sends data to OpenAI's servers.

    HOW IT WORKS UNDER THE HOOD:
    You send text to OpenAI → their model (a neural network) processes it →
    returns a fixed-size vector that captures the semantic meaning.
    The model was trained on billions of text pairs so that similar meanings
    produce vectors that are close together in high-dimensional space.
    """

    def __init__(self):
        """
        Initialize the OpenAI client.

        WHY LAZY IMPORT:
        We import openai here, not at the top of the file. This means if you're
        using HuggingFace embeddings, the openai package doesn't even need to be
        installed. Each provider only imports its own dependencies.
        """
        from openai import AsyncOpenAI

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when using OpenAI embeddings. "
                "Set it in your .env file."
            )

        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in one API call.

        The OpenAI API accepts up to 2048 texts per call. We send them all
        at once and get back all vectors at once. This is DRAMATICALLY faster
        than one call per text.

        The response contains embeddings in the same order as the input texts,
        but we sort by index just to be safe (the API spec doesn't guarantee order).
        """
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # Sort by index to guarantee order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single search query.
        For OpenAI, queries and documents use the same API call —
        but we keep the method separate for providers that differ.
        """
        response = await self.client.embeddings.create(
            model=self.model,
            input=[query],
        )
        return response.data[0].embedding

    def get_dimension(self) -> int:
        """
        Return dimensions based on the model name.
        text-embedding-3-small = 1536 dimensions
        text-embedding-3-large = 3072 dimensions
        text-embedding-ada-002 = 1536 dimensions (legacy)
        """
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Uses HuggingFace's sentence-transformers library.

    Cost: FREE — runs locally on your machine.
    Speed: Slower than OpenAI for large batches (no GPU = CPU-bound).
    Quality: Good enough for most use cases. all-MiniLM-L6-v2 is the
             most popular choice — small, fast, decent quality.
    Upside: No API key, no data leaves your machine, no rate limits.

    WHEN TO USE THIS:
    - You're learning and don't want to pay for API calls
    - Your data is sensitive and can't leave your network
    - You need to embed offline
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model_name = settings.huggingface_embedding_model
        # This downloads the model on first use (~80MB for MiniLM)
        # Subsequent loads use the cached version
        self.model = SentenceTransformer(self.model_name)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts using the local model.

        sentence-transformers is synchronous (CPU/GPU), so we run it
        in a thread pool via asyncio.to_thread to avoid blocking the
        async event loop (which would stall SSE streaming).

        .tolist() converts numpy arrays to plain Python lists,
        which is what ChromaDB expects.
        """
        import asyncio
        embeddings = await asyncio.to_thread(self.model.encode, texts)
        return embeddings.tolist()

    async def embed_query(self, query: str) -> list[float]:
        import asyncio
        embedding = await asyncio.to_thread(self.model.encode, [query])
        return embedding[0].tolist()

    def get_dimension(self) -> int:
        """
        Ask the model directly for its output dimension.
        all-MiniLM-L6-v2 = 384 dimensions.
        """
        return self.model.get_sentence_embedding_dimension()


class LocalEmbedder(BaseEmbedder):
    """
    Placeholder for a locally-hosted embedding model (e.g., Ollama, vLLM).

    This exists to show the pattern is extensible. When you want to add
    a new provider, you:
    1. Create a class that inherits from BaseEmbedder
    2. Implement the three methods
    3. Add it to the factory function below
    That's it. Nothing else in the codebase changes.
    """

    def __init__(self):
        raise NotImplementedError(
            "Local embedder is a placeholder. "
            "Implement it when you set up Ollama or a local model server."
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError

    def get_dimension(self) -> int:
        raise NotImplementedError


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory function — the ONLY place that knows about specific providers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_embedder() -> BaseEmbedder:
    """
    Returns the right embedder based on the EMBEDDING_PROVIDER setting.

    THIS IS THE FACTORY PATTERN:
    Instead of doing `embedder = OpenAIEmbedder()` everywhere (which hardcodes
    the choice), every file does `embedder = get_embedder()`. The config
    determines which class gets created.

    WHY THIS MATTERS:
    - Changing providers is a .env change, not a code change
    - Testing is easy: set EMBEDDING_PROVIDER=huggingface in tests, no mocking needed
    - New providers require zero changes to existing code
    - This is exactly how companies like Notion, Stripe, and Vercel
      structure their ML backends
    """
    providers = {
        "openai": OpenAIEmbedder,
        "huggingface": HuggingFaceEmbedder,
        "local": LocalEmbedder,
    }

    provider_class = providers.get(settings.embedding_provider)

    if provider_class is None:
        raise ValueError(
            f"Unknown embedding provider: '{settings.embedding_provider}'. "
            f"Must be one of: {list(providers.keys())}"
        )

    return provider_class()