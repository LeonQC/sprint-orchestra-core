"""Fail fast while the gateway can't reach any provider for a model.

With both answer providers down, each gateway call still spends ~9s on its own retries and fallback
before failing, and an /ask makes two (router, then agent): ~21s per user for a degraded response.
After `failure_threshold` consecutive upstream failures the breaker opens and calls fail at once;
after `reset_after_seconds` one call goes through as a probe, and its result closes or re-opens it.

One breaker per gateway model alias, shared by every client that calls it: the agent's Responses
client and the instructor client behind the structured features both call openai-production.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict

import openai

from app.observability import LLM_CIRCUIT_OPEN

logger = logging.getLogger(__name__)

#: What a user sees when no answer model can be reached, in every feature.
MODEL_UNAVAILABLE_MESSAGE = "The AI assistant is temporarily unavailable. Please try again in a few minutes."


class CircuitOpenError(RuntimeError):
    """Raised instead of calling the gateway while the breaker is open."""


class ModelUnavailableError(RuntimeError):
    """Every answer model is unreachable, so the feature has nothing worth returning."""

    def __init__(self, message: str = MODEL_UNAVAILABLE_MESSAGE):
        super().__init__(message)


def is_upstream_failure(exc: BaseException) -> bool:
    """The gateway gave up (its retries and provider fallback are spent) or couldn't be reached.

    Retrying here would multiply the gateway's own retries: with both answer providers down, client
    retries on top turned one /ask into 3.5 minutes before the degraded response.
    """
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return True
    return isinstance(exc, openai.APIStatusError) and (exc.status_code >= 500 or exc.status_code == 429)


def is_model_unavailable(exc: BaseException) -> bool:
    """Whether `exc`, or anything it wraps, means no answer model could be reached.

    instructor wraps API errors in InstructorRetryException, and the openai SDK wraps an exception
    from the transport (an open breaker) in APIConnectionError, so the cause chain has to be walked.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, (CircuitOpenError, ModelUnavailableError)) or is_upstream_failure(current):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class CircuitBreaker:
    def __init__(
        self, name: str, failure_threshold: int = 3, reset_after_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._name = name
        self._threshold = failure_threshold
        self._reset_after = reset_after_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None

    def check(self) -> None:
        """Raise CircuitOpenError unless a call may go to the gateway now."""
        if self._opened_at is None:
            return
        if self._clock() - self._opened_at < self._reset_after or self._probe_in_flight:
            raise CircuitOpenError(f"{self._name} circuit open: upstream failing, not calling the gateway")
        self._probe_in_flight = True

    def record_success(self) -> None:
        if self._opened_at is not None:
            logger.info("LLM circuit closed", extra={"model": self._name})
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False
        LLM_CIRCUIT_OPEN.labels(self._name).set(0)

    def release_probe(self) -> None:
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._failures += 1
        self._probe_in_flight = False
        if self._opened_at is not None or self._failures >= self._threshold:
            if self._opened_at is None:
                logger.warning("LLM circuit opened", extra={"model": self._name, "failures": self._failures})
            # A failed probe restarts the wait.
            self._opened_at = self._clock()
            LLM_CIRCUIT_OPEN.labels(self._name).set(1)


_breakers: Dict[str, CircuitBreaker] = {}


def breaker_for(model: str) -> CircuitBreaker:
    if model not in _breakers:
        _breakers[model] = CircuitBreaker(model)
    return _breakers[model]
