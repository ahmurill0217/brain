# Derived from onyx/utils/threadpool_concurrency.py.
"""Run independent calls concurrently.

Used in three places where the work is I/O bound and embarrassingly parallel:
embedding requests, image summarization, and the query variants of a search.
Threads are the right tool because every one of those blocks on a socket.

Onyx's version also propagates contextvars for tenant-scoped DB sessions. brain
has neither, so this is just the pool.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

R = TypeVar("R")


def run_functions_tuples_in_parallel(
    functions_with_args: Sequence[tuple[Callable[..., Any], tuple[Any, ...]]],
    *,
    allow_failures: bool = False,
    max_workers: int | None = None,
    timeout: float | None = None,
) -> list[Any]:
    """Run each (function, args) pair concurrently, results in input order.

    With `allow_failures`, a raising or timed-out call yields None in its slot
    instead of propagating. That is what lets one bad document fail without
    taking the batch with it.

    A timed-out call is abandoned, not cancelled: the thread keeps running and
    keeps holding its resources. Only used with calls that have their own
    socket-level timeout.
    """
    if not functions_with_args:
        return []

    workers = max_workers or len(functions_with_args)
    results: list[Any] = [None] * len(functions_with_args)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(func, *args): i
            for i, (func, args) in enumerate(functions_with_args)
        }
        for future, index in future_to_index.items():
            try:
                results[index] = future.result(timeout=timeout)
            except FuturesTimeout:
                if not allow_failures:
                    raise
                logger.warning("Parallel call %s timed out after %ss", index, timeout)
                results[index] = None
            except Exception:
                if not allow_failures:
                    raise
                logger.exception("Parallel call %s failed", index)
                results[index] = None

    return results


def run_in_parallel(
    calls: Sequence[Callable[[], R]],
    *,
    allow_failures: bool = False,
    max_workers: int | None = None,
    timeout: float | None = None,
) -> list[R | None]:
    """Same, for zero-argument callables (usually closures)."""
    return run_functions_tuples_in_parallel(
        [(call, ()) for call in calls],
        allow_failures=allow_failures,
        max_workers=max_workers,
        timeout=timeout,
    )
