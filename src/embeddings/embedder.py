# src/embeddings/embedder.py
import time

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.services.llm_client import get_client

# Mistral embedding model — same provider as generation, one API key
EMBED_MODEL = settings.mistral_embed_model  # "mistral-embed"
EMBED_DIMENSIONS = 1024


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def embed_text(text: str) -> list[float]:
    """
    Embed a single text string into a 1024-dimensional vector.

    Uses mistral-embed model. Retries on transient API failures.
    The vector can be compared with cosine similarity to find
    semantically similar texts.

    Args:
        text: Raw text to embed (max 8192 tokens).

    Returns:
        List of 1024 floats representing the text's semantic position.
    """
    start = time.perf_counter()
    client = get_client()

    response = await client.embeddings.create_async(
        model=EMBED_MODEL,
        inputs=[text],
    )

    latency_ms = (time.perf_counter() - start) * 1000
    logger.debug(
        "Embed | model={} | chars={} | latency={:.0f}ms",
        EMBED_MODEL,
        len(text),
        latency_ms,
    )

    return response.data[0].embedding


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed a list of texts in batches.

    Batching reduces the number of API calls and improves throughput.
    The Mistral embedding API accepts multiple inputs per call.

    Args:
        texts: List of texts to embed.
        batch_size: Number of texts per API call (default 32, Mistral limit).

    Returns:
        List of embedding vectors in the same order as the input texts.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    client = get_client()
    start = time.perf_counter()

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = await client.embeddings.create_async(
            model=EMBED_MODEL,
            inputs=batch,
        )

        # Results are returned in the same order as inputs
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        logger.debug(
            "Embed batch {}/{} | size={} | total_embedded={}",
            i // batch_size + 1,
            (len(texts) - 1) // batch_size + 1,
            len(batch),
            len(all_embeddings),
        )

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Embed batch complete | model={} | texts={} | latency={:.0f}ms",
        EMBED_MODEL,
        len(texts),
        latency_ms,
    )

    return all_embeddings


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Result between -1.0 and 1.0. In practice for text embeddings:
        > 0.85 : highly similar
        0.70-0.85 : related
        < 0.70 : likely unrelated

    Used for local testing and debugging — in production,
    pgvector computes similarity directly in SQL.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity score between -1.0 and 1.0.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    magnitude_a = sum(a * a for a in vec_a) ** 0.5
    magnitude_b = sum(b * b for b in vec_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)
