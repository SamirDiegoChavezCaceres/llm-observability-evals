"""A guided walkthrough: tracing (success + failure) and LLM-as-judge evals.

    python scripts/demo.py

Uses the RecordingTracer (no Langfuse account needed) and the offline
HeuristicJudge, plus a fake LLM judge so the LLM path is shown without a key.
"""

from __future__ import annotations

import os

from llm_obs import (
    EvalSample,
    HeuristicJudge,
    LLMJudge,
    RecordingTracer,
    evaluate_dataset,
    rag_pipeline,
)

DOCS = ["Paris is the capital of France.", "France is in Europe."]


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


def show_tree(node, depth=0) -> None:
    tag = f"[{node.as_type}] {node.name}"
    if node.error:
        tag += f"  ERROR: {node.error}"
    elif node.output:
        tag += f"  output={node.output}"
    print("  " * depth + tag)
    for child in node.children:
        show_tree(child, depth + 1)


def main() -> None:
    rule("1. A successful run, traced as a nested span tree")
    tracer = RecordingTracer()
    answer = rag_pipeline(
        "What is the capital of France?",
        retrieve=lambda q: DOCS,
        generate=lambda q, docs: "The capital of France is Paris.",
        tracer=tracer,
    )
    print(f"answer: {answer}\n")
    for root in tracer.roots:
        show_tree(root)

    rule("2. A failing run marks the error on the span")
    tracer = RecordingTracer()

    def broken_generate(q, docs):
        raise RuntimeError("model timed out")

    try:
        rag_pipeline("boom", retrieve=lambda q: DOCS, generate=broken_generate, tracer=tracer)
    except RuntimeError:
        pass
    for root in tracer.roots:
        show_tree(root)

    rule("3. Grade answers with the offline HeuristicJudge")
    context = " ".join(DOCS)
    samples = [
        EvalSample("What is the capital of France?", "The capital of France is Paris.", context),
        EvalSample("What is the capital of France?", "It is probably Lyon or Marseille.", context),
        EvalSample("What is the capital of France?", "Bananas grow in the tropics.", context),
    ]
    for name, s in evaluate_dataset(samples, HeuristicJudge()).items():
        print(f"  {name:<13} mean={s.mean:<5} pass_rate={s.pass_rate:<5} n={s.n}")

    rule("4. The same interface, now with an LLM judge")
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except Exception:
        pass
    judge = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            from llm_obs import openai_complete

            judge = LLMJudge(complete=openai_complete())
            print("  model: OpenAI")
        except Exception:
            judge = None  # openai not installed -> fall back so the demo still runs
    if judge is None:
        judge = LLMJudge(complete=lambda p: '{"groundedness": 0.95, "relevance": 0.90}')
        print("  model: fake (set OPENAI_API_KEY and install .[openai] for a real one)")
    for s in judge.score(samples[0]):
        print(f"  {s.name:<13} value={s.value} passed={s.passed}")


if __name__ == "__main__":
    main()
