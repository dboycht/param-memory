"""Which earlier memories should a new write be required to protect?

The motivation is measured, not aesthetic. With 8 slots written, reading **one**
slot recalls 8/8 items, but reading **two** drops to 4/16 and reading all of them
to 0/16 (T2). The slots are perfectly separable -- the interference is created by
each new write, which is trained only to produce its own value and is never told
that the bank already answers other queries.

So the write objective gains a retention term: the queries of *earlier* memories
must keep the same next-token distributions under the read condition where the
interference actually appears.

This module is just the selection rule, kept pure so it can be tested without a
model: it decides *which* earlier queries to protect, and says nothing about how.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["earlier_queries", "DEFAULT_LIMIT"]

DEFAULT_LIMIT = 8


def earlier_queries(items: Sequence, index: int, *, limit: int = DEFAULT_LIMIT,
                    attribute: str = "query") -> list[str]:
    """Queries of the memories written **before** ``index``.

    * ``index`` itself is never included -- a write must not be anchored to its own
      target, or the retention term would fight the recall term.
    * When more than ``limit`` earlier items exist the selection is *evenly spaced*
      across the whole history rather than taking only the most recent ones: the
      oldest memories are the ones with the most subsequent writes stacked on top of
      them, so they are the most at risk.
    * Deterministic, so a re-run protects exactly the same set.
    """
    if index < 0:
        raise ValueError("index must be >= 0")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    candidates = [getattr(item, attribute) for item in items[:index]]
    if len(candidates) <= limit:
        return candidates
    # even spacing across the history, including both endpoints
    step = (len(candidates) - 1) / (limit - 1) if limit > 1 else 0.0
    picked = [candidates[round(i * step)] for i in range(limit)]
    return picked
