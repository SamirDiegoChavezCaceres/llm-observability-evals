"""Run the instrumented pipeline and grade its output.

    python scripts/demo.py

Uses the RecordingTracer so you can see the span tree without a Langfuse
account. Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY to send real traces.
"""

from __future__ import annotations

from llm_obs import EvalSample, HeuristicJudge, RecordingTracer, evaluate_dataset, rag_pipeline

DOCS = ["Paris is the capital of France.", "France is in Europe."]


def main() -> None:
    tracer = RecordingTracer()
    answer = rag_pipeline(
        "What is the capital of France?",
        retrieve=lambda q: DOCS,
        generate=lambda q, docs: "The capital of France is Paris.",
        tracer=tracer,
    )
    print("answer:", answer)

    print("\nspan tree:")

    def show(node, depth=0):
        tag = f"[{node.as_type}] {node.name}"
        extra = f" output={node.output}" if node.output else ""
        print("  " * depth + tag + extra)
        for child in node.children:
            show(child, depth + 1)

    for root in tracer.roots:
        show(root)

    print("\neval (heuristic judge):")
    samples = [
        EvalSample("What is the capital of France?", answer, " ".join(DOCS)),
        EvalSample("What is the capital of France?", "Bananas grow in the tropics.", " ".join(DOCS)),
    ]
    for name, s in evaluate_dataset(samples, HeuristicJudge()).items():
        print(f"  {name:<13} mean={s.mean} pass_rate={s.pass_rate} n={s.n}")


if __name__ == "__main__":
    main()
