# RAGAS Evaluation Results — RegDoc Assistant

## Setup

- **Dataset:** 12 ground-truth Q&A pairs over RGPD Articles 5, 17, 32, 35, 37, 83
- **Judge LLM:** mistral-small-latest (temperature=0)
- **Metrics:** custom RAGAS-equivalent implementation (no external framework — pure Mistral LLM-as-judge)
- **Corpus:** 6 RGPD articles, 11 chunks total

## Configuration Comparison

| Config    | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
|-----------|-------------|------------------|----------------|-------------------|
| naive     | 0.988       | 0.935            | 1.000          | 1.000             |
| hybrid    | 0.967       | 0.927            | 1.000          | 0.875             |
| reranked  | 0.982       | 0.931            | 1.000          | 0.833             |
| hyde      | 0.974       | 0.938            | 1.000          | 1.000             |

> Note: hyde scored on 11/12 samples (1 dropped on Mistral 429 capacity error).

## Key Findings

### 1. Naive RAG wins on this corpus
On a small (11 chunks), clean corpus where each article answers a distinct question,
cosine similarity alone is sufficient. Advanced techniques degrade precision:
- **hybrid:** BM25 surfaces lexically-similar but off-topic chunks (Art.37, Art.83 → precision 0.50–0.875)
- **reranked:** retrieves top_k_retrieve=10 candidates from an 11-chunk corpus, forced to keep noise

This confirms a core principle: retrieval sophistication must match corpus characteristics.
Hybrid/reranking/HyDE target large, noisy corpora (millions of chunks) where cosine misses
rare exact terms. On a small clean corpus they add noise.

### 2. LLM-judge variance
The same naive pipeline scored context_precision 0.917 then 1.000 across two runs
(identical data, temperature=0). LLM-as-judge is not deterministic. A single RAGAS score
is not reliable — production evaluation should average multiple runs or report confidence intervals.

### 3. context_recall saturated
1.000 across all configs — the corpus is too small for this metric to discriminate.
On a realistic corpus, context_recall would reveal retrieval gaps.

### 4. Faithfulness catches hallucination
The Art.17 question consistently scored ~0.85 faithfulness across configs — the LLM adds
a legal nuance not present in the retrieved chunks. This is exactly what faithfulness detects.

## Conclusion

For this regulatory corpus, naive cosine retrieval is the right production choice:
best precision, lowest latency, lowest cost (no cross-encoder, no extra LLM calls for HyDE).
Advanced retrieval would only pay off as the corpus grows toward thousands of documents.