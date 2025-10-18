"""Lightweight profiling helpers for AMIE workflows."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from typing import Callable, Generator, ParamSpec, TypeVar

from .logging import get_logger

__all__ = ["profile", "profiling_block", "get_timings", "reset_timings"]

P = ParamSpec("P")
R = TypeVar("R")

_timings: defaultdict[str, list[float]] = defaultdict(list)
_logger = get_logger()


def profile(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that records execution time and logs the duration via structlog."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        label = name or func.__qualname__

        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            start = perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = perf_counter() - start
                _timings[label].append(duration)
                _logger.info("profiled_function", function=label, duration_ms=duration*1000)

        return wrapped

    return decorator


@contextmanager
def profiling_block(name: str) -> Generator[None, None, None]:
    """Context manager that profiles an arbitrary block and stores the duration."""
    start = perf_counter()
    try:
        yield
    finally:
        duration = perf_counter() - start
        _timings[name].append(duration)
        _logger.info("profiled_block", block=name, duration_ms=duration*1000)


def get_timings() -> dict[str, list[float]]:
    """Return a shallow copy of collected timing data."""
    return {key: list(values) for key, values in _timings.items()}


def reset_timings() -> None:
    """Clear accumulated timing metrics."""
    _timings.clear()
