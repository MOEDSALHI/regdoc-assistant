"""
RAGAS-equivalent metrics implemented directly with Mistral.
No external evaluation framework — pure LLM-as-judge pattern.

Each metric follows the same methodology as RAGAS but calls Mistral directly.
"""

import asyncio
import json
import os
import re
from dotenv import load_dotenv

from mistralai.client import Mistral
load_dotenv()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
JUDGE_MODEL = "mistral-small-latest"


async def _judge(prompt: str) -> str:
    """Single LLM judge call. Returns raw text response."""
    await asyncio.sleep(1.5)
    response = await client.chat.complete_async(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


async def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """
    Measures: are all claims in the answer supported by the retrieved contexts?
    Method: extract claims → verify each claim against contexts → ratio.
    Score 0-1, higher = less hallucination.
    """
    context_text = "\n---\n".join(contexts)

    # Step 1: extract claims
    extract_prompt = f"""Extract all factual claims from this answer as a JSON list of strings.
Answer: {answer}

Return ONLY a JSON array of strings, nothing else. Example: ["claim1", "claim2"]"""

    claims_raw = await _judge(extract_prompt)
    try:
        claims_raw = re.sub(r"```json|```", "", claims_raw).strip()
        claims = json.loads(claims_raw)
    except Exception:
        return 0.0

    if not claims:
        return 1.0

    # Step 2: verify each claim
    supported = 0
    for claim in claims:
        verify_prompt = f"""Given these context passages:
{context_text}

Is this claim fully supported by the contexts above?
Claim: "{claim}"

Answer with ONLY "yes" or "no"."""
        verdict = await _judge(verify_prompt)
        if verdict.lower().startswith("yes"):
            supported += 1

    return round(supported / len(claims), 3)


async def answer_relevancy(question: str, answer: str, contexts: list[str]) -> float:
    """
    Measures: does the answer actually address the question?
    Method: generate candidate questions from the answer → cosine similarity with original.
    Score 0-1, higher = more relevant answer.
    """
    from src.embeddings.embedder import cosine_similarity, embed_text

    gen_prompt = f"""Given this answer, generate 3 questions that this answer could be responding to.
Answer: {answer}

Return ONLY a JSON array of 3 question strings, nothing else."""

    questions_raw = await _judge(gen_prompt)
    try:
        questions_raw = re.sub(r"```json|```", "", questions_raw).strip()
        generated_questions = json.loads(questions_raw)
    except Exception:
        return 0.0

    if not generated_questions:
        return 0.0

    # Embed original question and generated questions
    original_embedding = await embed_text(question)
    similarities = []
    for gen_q in generated_questions:
        gen_embedding = await embed_text(gen_q)
        sim = cosine_similarity(original_embedding, gen_embedding)
        similarities.append(sim)

    return round(sum(similarities) / len(similarities), 3)


async def context_recall(
    question: str, ground_truth: str, contexts: list[str]
) -> float:
    """
    Measures: did retrieval capture all information needed to answer correctly?
    Method: check if each sentence of ground_truth is attributable to retrieved contexts.
    Score 0-1, higher = retrieval missed nothing important.
    """
    context_text = "\n---\n".join(contexts)

    # Split ground truth into sentences
    sentences = [s.strip() for s in re.split(r"[.;]", ground_truth) if len(s.strip()) > 20]
    if not sentences:
        return 1.0

    attributed = 0
    for sentence in sentences:
        prompt = f"""Given these context passages:
{context_text}

Can the following statement be inferred or attributed to the contexts above?
Statement: "{sentence}"

Answer with ONLY "yes" or "no"."""
        verdict = await _judge(prompt)
        if verdict.lower().startswith("yes"):
            attributed += 1

    return round(attributed / len(sentences), 3)


async def context_precision(
    question: str, ground_truth: str, contexts: list[str]
) -> float:
    """
    Measures: are the retrieved contexts all relevant? Penalizes noise in high positions.
    Method: for each context, judge relevance → weighted precision (position matters).
    Score 0-1, higher = no irrelevant chunks retrieved.
    """
    if not contexts:
        return 0.0

    relevance_scores = []
    for ctx in contexts:
        prompt = f"""Question: {question}
Expected answer type: {ground_truth[:200]}

Is this context passage useful for answering the question above?
Context: "{ctx}"

Answer with ONLY "yes" or "no"."""
        verdict = await _judge(prompt)
        relevance_scores.append(1 if verdict.lower().startswith("yes") else 0)

    # Weighted precision: relevant chunks in early positions matter more
    precision_at_k = []
    relevant_so_far = 0
    for i, score in enumerate(relevance_scores):
        if score == 1:
            relevant_so_far += 1
            precision_at_k.append(relevant_so_far / (i + 1))

    if not precision_at_k:
        return 0.0

    return round(sum(precision_at_k) / len(precision_at_k), 3)


async def evaluate_sample(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict:
    """Run all 4 metrics concurrently on a single Q&A sample."""
    # faith, rel, recall, precision = await asyncio.gather(
    #     faithfulness(question, answer, contexts),
    #     answer_relevancy(question, answer, contexts),
    #     context_recall(question, ground_truth, contexts),
    #     context_precision(question, ground_truth, contexts),
    # )
    # return {
    #     "faithfulness": faith,
    #     "answer_relevancy": rel,
    #     "context_recall": recall,
    #     "context_precision": precision,
    # }
    """Run 4 metrics sequentially to respect rate limits."""
    faith = await faithfulness(question, answer, contexts)
    rel = await answer_relevancy(question, answer, contexts)
    recall = await context_recall(question, ground_truth, contexts)
    precision = await context_precision(question, ground_truth, contexts)
    return {
        "faithfulness": faith,
        "answer_relevancy": rel,
        "context_recall": recall,
        "context_precision": precision,
    }