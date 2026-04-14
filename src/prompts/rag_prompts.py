# src/prompts/rag_prompts.py

import json

# ---------------------------------------------------------------------------
# SYSTEM PROMPTS
# ---------------------------------------------------------------------------

# Main system prompt for the RegDoc RAG assistant.
# Defines behavior, constraints, output format and hallucination guard.
RAG_SYSTEM_PROMPT = """You are RegDoc, a specialized assistant for French regulatory \
documents (GDPR, CNIL recommendations, ANSSI guides).

## Your rules
- Answer ONLY based on the context documents provided in the user message.
- Always cite your source: document name and page number when available.
- If the answer is not in the provided context, respond exactly with:
  "I cannot find this information in the available documents."
- Never invent regulatory articles, deadlines or obligations.
- Answer in the same language as the question (French or English).
- Be concise: 2-4 sentences maximum for the answer.

## Response format
1. Direct answer (2-4 sentences)
2. Relevant quote from the source (if available)
3. Source reference: [Document name, Page X]
"""

# System prompt for compliance analysis tasks.
# Used with few-shot examples to classify data practices.
COMPLIANCE_SYSTEM_PROMPT = """You are a GDPR compliance expert. \
Your role is to classify data processing practices as COMPLIANT or NON-COMPLIANT \
with a clear legal justification referencing the relevant GDPR article."""

# System prompt for Chain of Thought analysis.
# Forces step-by-step reasoning before the final verdict.
COT_COMPLIANCE_SYSTEM_PROMPT = """You are a GDPR compliance expert. \
When analyzing a data practice, always reason step by step \
through these checkpoints before concluding:

Step 1 — Legal basis (Article 6): is there a valid legal basis?
Step 2 — Data minimization (Article 5(1)(c)): is only necessary data collected?
Step 3 — Retention period (Article 5(1)(e)): is the period defined and justified?
Step 4 — Data subject rights (Articles 15-22): are rights guaranteed?
Step 5 — Final verdict: COMPLIANT / NON-COMPLIANT / NEEDS-CLARIFICATION

Always show your reasoning for each step before the verdict."""

# Structured JSON system prompt.
# The schema is explicit in the prompt — no ambiguity for the model.
RAG_STRUCTURED_SYSTEM_PROMPT = """You are RegDoc, a GDPR regulatory assistant.

Always respond with a valid JSON object matching this exact schema:
{
  "answer": "your factual answer in 2-4 sentences, or null if unknown",
  "confidence": "HIGH or MEDIUM or LOW",
  "citations": [
    {
      "document": "document name",
      "page": 12,
      "quote": "exact relevant quote from the chunk"
    }
  ],
  "cannot_answer": false
}

Rules:
- If the answer is not in the provided context, set cannot_answer=true, answer=null, confidence=LOW.
- Never invent articles or deadlines not present in the context.
- Return ONLY the JSON object — no markdown, no explanation outside the JSON.
- Answer in the same language as the question."""


# ---------------------------------------------------------------------------
# FEW-SHOT EXAMPLES
# ---------------------------------------------------------------------------

# Ground-truth examples for GDPR compliance classification.
# These teach the model the expected output format via in-context learning.
COMPLIANCE_EXAMPLES = [
    {
        "text": "We store server access logs for 6 months then delete them automatically.",
        "label": "COMPLIANT",
        "reason": "CNIL recommends 6 months minimum for access logs. Automatic deletion satisfies Article 5(1)(e).",
    },
    {
        "text": "User data is kept indefinitely for analytics purposes.",
        "label": "NON-COMPLIANT",
        "reason": "No legal basis for indefinite retention under GDPR Article 5(1)(e) — storage limitation principle.",
    },
    {
        "text": "We collect email addresses without explicit consent for marketing.",
        "label": "NON-COMPLIANT",
        "reason": "GDPR Article 6 requires explicit consent for marketing communications.",
    },
]


# ---------------------------------------------------------------------------
# MESSAGE BUILDERS
# ---------------------------------------------------------------------------

def build_rag_messages(
    question: str,
    context_chunks: list[str],
    system_prompt: str = RAG_SYSTEM_PROMPT,
) -> list[dict]:
    """
    Build the message list for a standard RAG query (text response).

    Assembles system instructions + retrieved chunks + user question
    into the format expected by Mistral's chat API.

    Args:
        question: User question in natural language.
        context_chunks: Relevant text chunks retrieved from pgvector.
        system_prompt: Override the default RAG system prompt if needed.

    Returns:
        Messages list ready to send to chat_complete() or chat_stream().
    """
    # Format retrieved chunks into a numbered context block
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    )

    user_message = f"""## Available context documents

{context_block}

## Question
{question}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def build_rag_structured_messages(
    question: str,
    context_chunks: list[str],
) -> list[dict]:
    """
    Build the message list for a RAG query expecting a structured JSON response.

    Uses RAG_STRUCTURED_SYSTEM_PROMPT which enforces the JSON schema.
    The response must be parsed with parse_structured_response().

    Args:
        question: User question in natural language.
        context_chunks: Relevant text chunks retrieved from pgvector.

    Returns:
        Messages list ready to send to chat_complete().
    """
    return build_rag_messages(
        question=question,
        context_chunks=context_chunks,
        system_prompt=RAG_STRUCTURED_SYSTEM_PROMPT,
    )


def build_simple_messages(
    question: str,
    system_prompt: str | None = None,
) -> list[dict]:
    """
    Build a minimal message list without RAG context.

    Used for prompt engineering experiments: zero-shot, few-shot, CoT.
    If no system_prompt is provided, sends user message only (true zero-shot).

    Args:
        question: The user question or task description.
        system_prompt: Optional system instructions.

    Returns:
        Messages list ready to send to Mistral.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def build_few_shot_compliance_messages(text_to_classify: str) -> list[dict]:
    """
    Build few-shot messages for GDPR compliance classification.

    Provides 3 labeled examples before the actual classification target.
    The examples teach the model the expected label + reason format
    without requiring explicit format instructions.

    Args:
        text_to_classify: The data practice description to classify.

    Returns:
        Messages list with system prompt + few-shot examples + target.
    """
    examples_block = "\n\n".join(
        f'Text: "{ex["text"]}"\nLabel: {ex["label"]}\nReason: {ex["reason"]}'
        for ex in COMPLIANCE_EXAMPLES
    )

    user_message = f"""Classify each text as GDPR COMPLIANT or NON-COMPLIANT with a brief reason.

## Examples
{examples_block}

## Now classify this
Text: "{text_to_classify}"
Label:"""

    return [
        {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def build_cot_analysis_messages(practice_description: str) -> list[dict]:
    """
    Build Chain of Thought messages for deep GDPR compliance analysis.

    Forces step-by-step reasoning through 5 GDPR checkpoints before
    the final verdict. Reduces errors on complex multi-criteria tasks
    by making each reasoning step explicit and visible to the model.

    Args:
        practice_description: Description of the data processing practice to analyze.

    Returns:
        Messages list with CoT system prompt + analysis request.
    """
    return [
        {"role": "system", "content": COT_COMPLIANCE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Analyze this data processing practice:\n\n{practice_description}",
        },
    ]


# ---------------------------------------------------------------------------
# JSON PARSER
# ---------------------------------------------------------------------------

def parse_structured_response(raw: str) -> dict:
    """
    Parse the model's structured JSON response with a safe fallback.

    Strips markdown code fences if present (models sometimes wrap JSON
    in ```json ... ``` despite instructions not to).

    Args:
        raw: Raw string response from chat_complete().

    Returns:
        Parsed dict matching the RAG structured schema, or a fallback
        error dict if parsing fails.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        cleaned = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Fallback — never crash the API on a parse error
        return {
            "answer": None,
            "confidence": "LOW",
            "citations": [],
            "cannot_answer": True,
            "_parse_error": str(e),
            "_raw_response": raw[:200],  # truncated for logs
        }