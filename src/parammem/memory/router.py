"""Reading many memories without letting them cancel one another.

The measured problem (0.6B, 8 slots, 2026-09-29): every memory reads back
perfectly on its own (8/8) and reading with the *wrong* slot gives nothing (0/8),
but with **every** slot active the answers collapse (0/8). The slots are
separable; superposition is what destroys them. So the read path must *select*
rather than *sum*.

Keeping the selection logic here as a pure function has two payoffs: it is
unit-testable without a model, and the alternative compositions (all / top-1 /
top-2 / oracle) become an ablation on one axis instead of four code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["RoutingDecision", "route", "cosine_scores"]


@dataclass
class RoutingDecision:
    """Which slots a query selected, and how strongly."""

    slots: list[int]
    scores: list[tuple[int, float]] = field(default_factory=list)

    @property
    def top(self) -> int | None:
        return self.slots[0] if self.slots else None

    def __bool__(self) -> bool:
        return bool(self.slots)


def cosine_scores(
    query_key: torch.Tensor, slot_keys: dict[int, torch.Tensor]
) -> list[tuple[int, float]]:
    """Cosine similarity of ``query_key`` against every slot key, best first.

    Keys are L2-normalised here rather than at capture time, so a key stored by
    an older run (or a different layer) still compares sensibly. Slot index is the
    tie-breaker, which keeps the ordering deterministic across runs.
    """
    if not slot_keys:
        return []
    q = query_key.flatten().float()
    q = q / q.norm().clamp_min(1e-6)
    scores: list[tuple[int, float]] = []
    for slot, key in slot_keys.items():
        k = key.flatten().float()
        k = k / k.norm().clamp_min(1e-6)
        scores.append((int(slot), float(torch.dot(q, k))))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores


def route(
    query_key: torch.Tensor, slot_keys: dict[int, torch.Tensor], k: int = 1
) -> RoutingDecision:
    """Select the ``k`` best-matching slots for ``query_key``.

    ``k <= 0`` means "no selection": activate nothing, which is the MEM_OFF arm.
    """
    if k <= 0:
        return RoutingDecision(slots=[])
    scores = cosine_scores(query_key, slot_keys)
    return RoutingDecision(slots=[slot for slot, _ in scores[:k]], scores=scores)
