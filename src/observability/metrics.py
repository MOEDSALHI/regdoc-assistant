# src/observability/metrics.py
"""Prometheus metrics for the RAG pipeline.

Each metric here corresponds to a specific failure mode or business question
that generic HTTP metrics cannot answer:
- "Which retrieval mode is slowest?" -> rag_retrieval_duration_seconds{mode}
- "Are we hitting rate limits?" -> rag_llm_errors_total{error_type="rate_limit"}
- "How much are we spending?" -> rag_llm_tokens_total{type, model}
- "Is retrieval finding anything?" -> rag_chunks_retrieved{mode}
"""

from prometheus_client import Counter, Histogram

# =============================================================================
# Request-level metrics
# =============================================================================

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total number of /ask requests, by retrieval mode and outcome.",
    ["mode", "status"],   # status: "success" | "error"
)


# =============================================================================
# Retrieval phase metrics
# =============================================================================

# Retrieval latency expectations on our corpus:
#   - naive (cosine):     10-50ms
#   - hybrid (BM25+RRF):  50-200ms
#   - reranked:           500ms-2s (cross-encoder is the bottleneck)
#   - hyde:               1-3s (one extra LLM call before retrieval)
RAG_RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval phase duration in seconds, by mode.",
    ["mode"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Number of chunks returned. Helps detect:
#   - 0 chunks: empty index, bad embedding, broken query
#   - >> top_k: bug in retrieval logic, no dedup
RAG_CHUNKS_RETRIEVED = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks returned by the retrieval step, by mode.",
    ["mode"],
    buckets=(0, 1, 3, 5, 10, 20, 50),
)


# =============================================================================
# LLM phase metrics
# =============================================================================

# Mistral latency for mistral-small-latest:
#   - typical: 1-5s
#   - >10s usually means rate limiting or network issue
RAG_LLM_DURATION = Histogram(
    "rag_llm_duration_seconds",
    "LLM generation duration in seconds, by model.",
    ["model"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0),
)

# Token consumption is the primary cost driver.
# This lets us project monthly spend and detect runaway contexts.
# type: "prompt" (input tokens) | "completion" (output tokens)
RAG_LLM_TOKENS = Counter(
    "rag_llm_tokens_total",
    "Total tokens consumed by the LLM, by type (prompt|completion) and model.",
    ["type", "model"],
)

# Errors typed by category for actionable alerting:
#   - rate_limit: 429 from Mistral
#   - timeout: network or upstream slowness
#   - parse: LLM returned malformed structured output
#   - other: catch-all (always log full traceback alongside)
RAG_LLM_ERRORS = Counter(
    "rag_llm_errors_total",
    "Total LLM errors, by model and error type.",
    ["model", "error_type"],
)

# =============================================================================
# Security metrics
# =============================================================================

# Direct prompt injection attempts blocked before reaching the pipeline.
# A spike here may indicate an active attack or a UX regression where
# legitimate queries trigger the heuristic. Monitor with alerts.
RAG_INJECTION_BLOCKED = Counter(
    "rag_injection_blocked_total",
    "Total /ask requests blocked by direct injection detection.",
)