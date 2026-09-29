"""Paired comparison helpers.

Two experiments in this project compare the *same* items under different settings
(6 arms on the same 30 LongMemEval questions; the same 24 probes under different
retention weights). Averages and overlapping confidence intervals are the wrong tool
there: the informative quantity is how often a setting wins, loses, or ties on the
same item, which is a sign test.

Kept pure and separate so both experiments use one implementation, and so the
numbers in the paper come from tested code rather than from an ad-hoc script.
"""

from __future__ import annotations

from math import comb
from typing import Sequence

__all__ = ["paired_counts", "sign_test_p"]


def paired_counts(baseline: Sequence[float], other: Sequence[float]) -> tuple[int, int, int]:
    """``(helped, hurt, tied)`` counting truthiness per position.

    Raises on a length mismatch: silently truncating would pair the wrong items.
    """
    if len(baseline) != len(other):
        raise ValueError(f"paired series must be the same length: "
                         f"{len(baseline)} vs {len(other)}")
    helped = hurt = tied = 0
    for before, after in zip(baseline, other):
        b, a = bool(before), bool(after)
        if a and not b:
            helped += 1
        elif b and not a:
            hurt += 1
        else:
            tied += 1
    return helped, hurt, tied


def sign_test_p(helped: int, hurt: int) -> float:
    """Two-sided exact binomial p-value for a sign test.

    No normal approximation: these counts are small (single digits), which is
    exactly where the approximation would flatter the result.
    """
    if helped < 0 or hurt < 0:
        raise ValueError("counts must be non-negative")
    trials = helped + hurt
    if trials == 0:
        return 1.0
    smaller = min(helped, hurt)
    tail = sum(comb(trials, k) for k in range(smaller + 1)) / 2 ** trials
    return min(1.0, 2 * tail)
