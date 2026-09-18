"""A tiny tracing abstraction that nests spans and no-ops without credentials.

The pipeline code is instrumented once, against this interface, and the backend
is chosen at runtime:

* ``NoOpTracer`` - the default when no Langfuse keys are set. Zero overhead,
  never raises. Instrumentation must never be the reason a request fails.
* ``RecordingTracer`` - captures the span tree in memory, for tests and local
  inspection.
* ``LangfuseTracer`` - sends traces to Langfuse, with nesting handled by the
  SDK's current-context. Best-effort: if a SDK call misbehaves it degrades to a
  no-op for that node rather than breaking the pipeline.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


# --- No-op (default) --------------------------------------------------------

class _NoOpHandle:
    def update(self, **output) -> None:
        pass

    def error(self, message: str, **kw) -> None:
        pass

    @contextmanager
    def span(self, name: str, as_type: str = "span", **input) -> Iterator["_NoOpHandle"]:
        yield self


class NoOpTracer:
    @contextmanager
    def trace(self, name: str, **input) -> Iterator[_NoOpHandle]:
        yield _NoOpHandle()


# --- Recording (tests / local) ---------------------------------------------

@dataclass
class RecordedSpan:
    name: str
    as_type: str
    input: dict
    output: Optional[dict] = None
    error: Optional[str] = None
    children: List["RecordedSpan"] = field(default_factory=list)


class RecordingTracer:
    def __init__(self) -> None:
        self.roots: List[RecordedSpan] = []
        self._stack: List[RecordedSpan] = []

    @contextmanager
    def _node(self, name: str, as_type: str, input: dict) -> Iterator["_RecHandle"]:
        node = RecordedSpan(name=name, as_type=as_type, input=dict(input))
        (self._stack[-1].children if self._stack else self.roots).append(node)
        self._stack.append(node)
        try:
            yield _RecHandle(self, node)
        finally:
            self._stack.pop()

    def trace(self, name: str, **input):
        return self._node(name, "trace", input)


class _RecHandle:
    def __init__(self, tracer: RecordingTracer, node: RecordedSpan) -> None:
        self._tracer = tracer
        self._node_ref = node

    def span(self, name: str, as_type: str = "span", **input):
        return self._tracer._node(name, as_type, input)

    def update(self, **output) -> None:
        self._node_ref.output = {**(self._node_ref.output or {}), **output}

    def error(self, message: str, **kw) -> None:
        self._node_ref.error = message


# --- Langfuse (production, best-effort) -------------------------------------

class LangfuseTracer:
    def __init__(self, client) -> None:
        self._client = client

    @contextmanager
    def _node(self, name: str, as_type: str, input: dict):
        payload = input or None
        try:
            if as_type in ("span", "trace"):
                cm = self._client.start_as_current_span(name=name, input=payload)
            else:
                cm = self._client.start_as_current_observation(
                    as_type=as_type, name=name, input=payload
                )
        except Exception:
            yield _NoOpHandle()
            return
        with cm as span:
            yield _LfHandle(self, span)

    def trace(self, name: str, **input):
        return self._node(name, "span", input)


class _LfHandle:
    def __init__(self, tracer: LangfuseTracer, span) -> None:
        self._tracer = tracer
        self._span = span

    def span(self, name: str, as_type: str = "span", **input):
        return self._tracer._node(name, as_type, input)

    def update(self, **output) -> None:
        try:
            self._span.update(output=output)
        except Exception:
            pass

    def error(self, message: str, **kw) -> None:
        try:
            self._span.update(level="ERROR", status_message=message)
        except Exception:
            pass


def get_tracer():
    """Pick a backend from the environment. No keys -> NoOpTracer."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return NoOpTracer()
    try:
        from langfuse import get_client

        return LangfuseTracer(get_client())
    except Exception:
        return NoOpTracer()
