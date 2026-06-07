"""
RegDoc Assistant — RAG Evaluation Script
Uses custom RAGAS-equivalent metrics implemented directly with Mistral.
No external evaluation framework required.

Run: uv run python tests/evaluation/evaluate.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.evaluation.dataset import GROUND_TRUTH_QA
from tests.evaluation.metrics import evaluate_sample

API_BASE = "http://localhost:8000"

RETRIEVAL_MODE = sys.argv[1] if len(sys.argv) > 1 else "naive"

async def call_pipeline(question: str, client: httpx.AsyncClient) -> dict:
    """Call /ask and return answer + contexts."""
    response = await client.post(
        f"{API_BASE}/ask",
        json={"question": question, "retrieval_mode": RETRIEVAL_MODE},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "answer": data.get("answer", ""),
        "contexts": [c["quote"] for c in data.get("citations", []) if c.get("quote")],
    }


async def main():
    print("=" * 60)
    print("RegDoc Assistant — Baseline Evaluation")
    print("Metrics: Faithfulness, Answer Relevancy, Context Recall, Context Precision")
    print("Judge LLM: mistral-small-latest")
    print(f"Pipeline mode: {RETRIEVAL_MODE}")
    print("=" * 60)
    results = []

    async with httpx.AsyncClient() as http_client:
        for i, qa in enumerate(GROUND_TRUTH_QA):
            print(f"\n[{i+1}/{len(GROUND_TRUTH_QA)}] {qa['question'][:70]}...")
            try:
                pipeline_out = await call_pipeline(qa["question"], http_client)
                scores = await evaluate_sample(
                    question=qa["question"],
                    answer=pipeline_out["answer"],
                    contexts=pipeline_out["contexts"],
                    ground_truth=qa["ground_truth"],
                )
                scores["question"] = qa["question"]
                scores["source_article"] = qa["source_article"]
                results.append(scores)
                print(
                    f"  faith={scores['faithfulness']:.2f} | "
                    f"rel={scores['answer_relevancy']:.2f} | "
                    f"recall={scores['context_recall']:.2f} | "
                    f"precision={scores['context_precision']:.2f}"
                )
                # await asyncio.sleep(10) 
            except Exception as e:
                print(f"  ✗ Failed: {e}")

    if not results:
        print("  No results — all samples failed.")
        return

    # Aggregate scores
    print("\n" + "=" * 60)
    print("RESULTS — Naive RAG baseline")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        avg = sum(r[metric] for r in results) / len(results)
        print(f"  {metric:25s}: {avg:.3f}")

    # Save results
    # output_path = Path("notes/ragas_results_naive.json")
    output_path = Path(f"notes/ragas_results_{RETRIEVAL_MODE}.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())