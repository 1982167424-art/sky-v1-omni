"""Exponential-backoff retry decorator built on tenacity.

Mark transient errors as `RetryableError` (or subclass it) to trigger retries.
Non-retryable errors will NOT be retried; they raise immediately after the first
attempt so bugs surface fast.
"""
from __future__ import annotations

import logging as _py_logging
from functools import wraps
from typing import Any, Callable, Iterable, Type

import tenacity as _tn

from .logging import get_logger

log = get_logger("utils.retry")


class RetryableError(RuntimeError):
    """Raise (or subclass) this to mark a transient failure that should retry.

    Examples: 5xx from API, network timeout, DB connection lost, rate-limit 429.
    """


def _default_before_sleep(log_obj: Any) -> Callable[[_tn.RetryCallState], None]:
    def _cb(state: _tn.RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        wait_s = float(state.next_action.sleep) if state.next_action else 0.0
        log_obj.warning(
            "Retrying call",
            attempt=state.attempt_number,
            wait_seconds=round(wait_s, 3),
            exc_type=type(exc).__name__ if exc else None,
            exc=str(exc)[:300] if exc else None,
            fn=getattr(state.fn, "__qualname__", repr(state.fn)),
        )
    return _cb


def with_retry(
    max_attempts: int = 3,
    min_wait_s: float = 0.2,
    max_wait_s: float = 5.0,
    retry_types: Iterable[Type[BaseException]] = (RetryableError,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: retry `fn` with exponential jittered backoff.

    Args:
        max_attempts: total attempts, >= 1. If 1, effectively no retry.
        min_wait_s: minimum wait between attempts (lower bound, seconds).
        max_wait_s: maximum wait (upper bound, seconds).
        retry_types: exception classes that should trigger a retry.
            Default: (RetryableError,) only.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    retry_cls = tuple(t for t in retry_types)
    if not retry_cls:
        retry_cls = (RetryableError,)
    before_sleep_cb = _default_before_sleep(log)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        import inspect
        try:
            sig = inspect.signature(_tn.wait_exponential_jitter)
            if "initial" in sig.parameters:
                wait = _tn.wait_exponential_jitter(initial=min_wait_s, max=max_wait_s)
            else:
                wait = _tn.wait_exponential_jitter(min=min_wait_s, max=max_wait_s)
        except Exception:
            wait = _tn.wait_fixed(min_wait_s)
        retrying = _tn.Retrying(
            stop=_tn.stop_after_attempt(max_attempts),
            wait=wait,
            retry=_tn.retry_if_exception_type(retry_cls),
            before_sleep=before_sleep_cb,
            reraise=True,
        )

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return retrying(fn, *args, **kwargs)

        # Preserve async signature for async callers: tenacity handles coroutines.
        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper
    return decorator
