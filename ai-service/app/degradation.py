"""Per-request record of what ran in a reduced mode, so the response can say so and skip the cache.

A ContextVar holding a set: tasks spawned inside the request copy the context but share the same
set, so a note made deep in a tool call reaches the route handler.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

#: vectorization-service answered a hybrid search with BM25 only (query embedding failed).
RETRIEVAL_BM25_ONLY = "retrieval_bm25_only"
#: Every answer model failed; the response is retrieved sources with no generated answer.
ANSWER_MODEL_UNAVAILABLE = "answer_model_unavailable"
#: A multi-agent /ask had a worker, synthesis, review or verification step fail; the answer may
#: cover only part of the question.
MULTI_AGENT_PARTIAL = "multi_agent_partial"

_reasons: ContextVar[set[str] | None] = ContextVar("degradation_reasons", default=None)


@contextmanager
def collect() -> Iterator[set[str]]:
    reasons: set[str] = set()
    token = _reasons.set(reasons)
    try:
        yield reasons
    finally:
        try:
            _reasons.reset(token)
        except ValueError:
            # A streaming response abandoned by its client is closed from another context; the
            # value dies with the request's own context anyway.
            pass


def note(reason: str) -> None:
    reasons = _reasons.get()
    if reasons is not None:
        reasons.add(reason)
