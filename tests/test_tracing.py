import pytest

from llm_obs import NoOpTracer, RecordingTracer, get_tracer, rag_pipeline

DOCS = ["Paris is the capital of France.", "France is in Europe."]


def _retrieve(_query):
    return DOCS


def _generate(_query, docs):
    return "The capital of France is Paris."


def test_recording_tracer_captures_nested_spans():
    tracer = RecordingTracer()
    answer = rag_pipeline("What is the capital of France?", _retrieve, _generate, tracer)
    assert answer == "The capital of France is Paris."

    assert len(tracer.roots) == 1
    root = tracer.roots[0]
    assert root.name == "rag-pipeline"
    child_types = {c.as_type for c in root.children}
    assert child_types == {"retriever", "generation"}
    retriever = next(c for c in root.children if c.as_type == "retriever")
    assert retriever.output == {"num_docs": 2}
    assert root.output == {"output": answer}


def test_noop_tracer_runs_without_error():
    answer = rag_pipeline("hello", _retrieve, _generate, NoOpTracer())
    assert answer == "The capital of France is Paris."


def test_get_tracer_defaults_to_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert isinstance(get_tracer(), NoOpTracer)


def test_failure_is_marked_on_the_span():
    tracer = RecordingTracer()

    def broken(_query, _docs):
        raise RuntimeError("model timed out")

    with pytest.raises(RuntimeError):
        rag_pipeline("boom", _retrieve, broken, tracer)

    root = tracer.roots[0]
    generation = next(c for c in root.children if c.as_type == "generation")
    assert generation.error == "model timed out"
    assert root.error  # the failure bubbles to the root span too
