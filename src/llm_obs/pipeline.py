"""An example RAG pipeline, instrumented once against the tracer interface.

``retrieve`` and ``generate`` are injected, so the pipeline runs with fakes (no
LLM, no API key) and the same instrumentation lights up Langfuse in production.
The trace nests a retriever span and a generation span under one root.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .tracing import get_tracer

Retrieve = Callable[[str], List[str]]
Generate = Callable[[str, List[str]], str]


def rag_pipeline(
    query: str,
    retrieve: Retrieve,
    generate: Generate,
    tracer=None,
) -> str:
    tracer = tracer or get_tracer()
    with tracer.trace("rag-pipeline", query=query) as root:
        with root.span("retriever", as_type="retriever", query=query) as r:
            docs = retrieve(query)
            r.update(num_docs=len(docs))

        with root.span("generation", as_type="generation", input=query) as g:
            answer = generate(query, docs)
            g.update(output=answer)

        root.update(output=answer)
        return answer
