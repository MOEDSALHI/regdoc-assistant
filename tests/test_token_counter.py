# tests/test_token_counter.py
from src.services.token_counter import (
    count_messages_tokens,
    count_tokens,
    fits_in_context,
    truncate_text_to_tokens,
)


def test_count_tokens_short_text():
    # Short known text — verify it returns a positive integer
    result = count_tokens("RGPD")
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_longer_text_is_more():
    short = count_tokens("RGPD")
    long = count_tokens("Le règlement général sur la protection des données personnelles")
    assert long > short


def test_count_messages_tokens_includes_overhead():
    messages = [{"role": "user", "content": "Hello"}]
    # Must be > just the content tokens (overhead adds 4 per message)
    content_tokens = count_tokens("Hello")
    total = count_messages_tokens(messages)
    assert total > content_tokens


def test_count_messages_tokens_empty_content():
    messages = [{"role": "system", "content": ""}]
    # Only overhead — 4 tokens
    assert count_messages_tokens(messages) == 4


def test_fits_in_context_small_input():
    messages = [{"role": "user", "content": "Short question"}]
    assert fits_in_context(messages) is True


def test_fits_in_context_exceeds_limit():
    # Build a message that exceeds MAX_INPUT_TOKENS
    huge_content = "token " * 30_000
    messages = [{"role": "user", "content": huge_content}]
    assert fits_in_context(messages) is False


def test_truncate_text_within_limit_unchanged():
    text = "This is a short text."
    result = truncate_text_to_tokens(text, max_tokens=100)
    assert result == text


def test_truncate_text_reduces_tokens():
    long_text = "données personnelles " * 500
    truncated = truncate_text_to_tokens(long_text, max_tokens=50)
    assert count_tokens(truncated) <= 50


def test_truncate_text_preserves_coherence():
    # Truncated text must be non-empty and a string
    long_text = "RGPD compliance " * 200
    truncated = truncate_text_to_tokens(long_text, max_tokens=20)
    assert isinstance(truncated, str)
    assert len(truncated) > 0
