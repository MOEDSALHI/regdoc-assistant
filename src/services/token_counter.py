# src/services/token_counter.py
import tiktoken
from loguru import logger

# cl100k_base is the closest publicly available approximation to Mistral's tokenizer.
# Mistral uses SentencePiece internally (not exposed) — cl100k_base gives ~±10% accuracy.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# mistral-small-latest hard limit
MISTRAL_CONTEXT_WINDOW = 32_000

# Safety margin: reserves space for the model response (up to 1024 tokens)
# and avoids hitting the hard limit due to tokenizer approximation errors.
MAX_INPUT_TOKENS = 28_000


def count_tokens(text: str) -> int:
    """
    Return the approximate token count for a raw text string.

    Uses cl100k_base encoding as a Mistral tokenizer approximation.
    """
    return len(_ENCODING.encode(text))


def count_messages_tokens(messages: list[dict]) -> int:
    """
    Count total tokens across a list of role/content messages.

    Adds 4 tokens per message to account for role formatting overhead
    (<|system|>, <|user|>, <|assistant|>, <|end|> special tokens).

    Args:
        messages: List of dicts with 'role' and 'content' keys.

    Returns:
        Estimated total token count.
    """
    total = 0
    for message in messages:
        total += 4  # role formatting overhead
        total += count_tokens(message.get("content", ""))
    return total


def fits_in_context(
    messages: list[dict],
    max_tokens: int = MAX_INPUT_TOKENS,
) -> bool:
    """
    Return True if the message list fits within the allowed context window.

    Logs a warning with token counts if the limit is exceeded.

    Args:
        messages: Message list to check.
        max_tokens: Token limit (defaults to MAX_INPUT_TOKENS).

    Returns:
        True if within limit, False otherwise.
    """
    total = count_messages_tokens(messages)
    fits = total <= max_tokens

    if not fits:
        logger.warning(
            "Context overflow: {} tokens > {} limit — truncation required",
            total,
            max_tokens,
        )
    else:
        logger.debug("Context check: {} / {} tokens used", total, max_tokens)

    return fits


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within max_tokens, preserving whole tokens.

    Used to trim individual chunks before assembling the RAG context.
    Logs a debug message when truncation occurs.

    Args:
        text: Raw text to truncate.
        max_tokens: Maximum allowed tokens.

    Returns:
        Truncated text string (or original if already within limit).
    """
    tokens = _ENCODING.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated = _ENCODING.decode(tokens[:max_tokens])
    logger.debug(
        "Chunk truncated: {} → {} tokens",
        len(tokens),
        max_tokens,
    )
    return truncated


def log_context_breakdown(
    system_prompt: str,
    chunks: list[str],
    question: str,
) -> int:
    """
    Log a detailed token breakdown for a RAG context assembly.

    Useful for debugging and monitoring token consumption per component.

    Args:
        system_prompt: The system prompt string.
        chunks: List of retrieved document chunks.
        question: The user question string.

    Returns:
        Total estimated token count.
    """
    system_tokens = count_tokens(system_prompt)
    chunks_tokens = sum(count_tokens(c) for c in chunks)
    question_tokens = count_tokens(question)
    overhead = 4 * (2 + len(chunks))  # formatting per message
    total = system_tokens + chunks_tokens + question_tokens + overhead

    logger.info(
        "Context breakdown | system={} | chunks={}×{}≈{} | question={} | overhead={} | TOTAL={}",
        system_tokens,
        len(chunks),
        chunks_tokens // max(len(chunks), 1),
        chunks_tokens,
        question_tokens,
        overhead,
        total,
    )

    return total