# src/services/llm_client.py
import time
from collections.abc import AsyncGenerator

from loguru import logger
# from mistralai import Mistral
from mistralai.client import Mistral  # v2.x — class lives in mistralai.client
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.observability.metrics import RAG_LLM_DURATION, RAG_LLM_TOKENS
from src.services.token_counter import fits_in_context, log_context_breakdown


def _build_client() -> Mistral:
    """Instantiate and return a configured Mistral client."""
    return Mistral(api_key=settings.mistral_api_key)


# Module-level singleton — one client for the entire application lifetime
_client: Mistral = _build_client()


def get_client() -> Mistral:
    """Return the shared Mistral client instance."""
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def chat_complete(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,   # default LOW for RAG — factual accuracy
    max_tokens: int = 1024,
    top_p: float = 0.9,
) -> str:
    """
    Send a chat completion request to Mistral AI.

    Default temperature=0.1 and top_p=0.9 are tuned for factual RAG responses.
    Increase temperature for creative tasks (summarization, reformulation).

    Retries up to 3 times with exponential backoff on failure.
    Logs model, token usage and latency for every call.

    Args:
        messages: List of role/content dicts (OpenAI-compatible format).
        model: Override the default model from settings.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum tokens in the response.
        top_p: Nucleus sampling threshold — keep tokens summing to top_p probability.

    Returns:
        The assistant message content as a string.

    Raises:
        ValueError: If the message list exceeds the context window.
    """
    # Guard: check context window before sending
    if not fits_in_context(messages):
        raise ValueError(
            "Message list exceeds context window limit. "
            "Truncate your chunks before calling chat_complete()."
        )

    resolved_model = model or settings.mistral_model
    start = time.perf_counter()

    response = await _client.chat.complete_async(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    usage = response.usage

    logger.info(
        "Mistral call | model={} | temp={} | top_p={} | "
        "prompt_tokens={} | completion_tokens={} | latency={:.0f}ms",
        resolved_model,
        temperature,
        top_p,
        usage.prompt_tokens,
        usage.completion_tokens,
        latency_ms,
    )

    RAG_LLM_DURATION.labels(model=resolved_model).observe(latency_ms / 1000)
    RAG_LLM_TOKENS.labels(type="prompt", model=resolved_model).inc(usage.prompt_tokens)
    RAG_LLM_TOKENS.labels(type="completion", model=resolved_model).inc(usage.completion_tokens)

    return response.choices[0].message.content


async def chat_stream(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    top_p: float = 0.9,
) -> AsyncGenerator[str, None]:
    """
    Stream a chat completion from Mistral AI, yielding text chunks as they arrive.

    No retry decorator here — streaming cannot be safely retried mid-stream.

    Args:
        messages: List of role/content dicts.
        model: Override the default model from settings.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        top_p: Nucleus sampling threshold.

    Yields:
        Text delta strings from each streamed chunk.

    Raises:
        ValueError: If the message list exceeds the context window.
    """
    # Guard: check context window before sending
    if not fits_in_context(messages):
        raise ValueError(
            "Message list exceeds context window limit. "
            "Truncate your chunks before calling chat_stream()."
        )

    resolved_model = model or settings.mistral_model
    start = time.perf_counter()
    token_count = 0

    # stream_async() must be awaited — it returns an async iterable, not a context manager
    stream = await _client.chat.stream_async(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

    async for event in stream:
        delta = event.data.choices[0].delta.content
        if delta:
            token_count += 1
            yield delta

    latency_ms = (time.perf_counter() - start) * 1000
    RAG_LLM_DURATION.labels(model=resolved_model).observe(latency_ms / 1000)
    logger.info(
        "Mistral stream | model={} | temp={} | top_p={} | chunks={} | latency={:.0f}ms",
        resolved_model,
        temperature,
        top_p,
        token_count,
        latency_ms,
    )