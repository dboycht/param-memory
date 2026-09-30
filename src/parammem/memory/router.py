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

import math
from collections import Counter
from dataclasses import dataclass, field

import torch

from ..bench.protocol import normalize

__all__ = ["RoutingDecision", "route", "cosine_scores", "lexical_scores",
           "route_lexical"]


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


def _tfidf(text: str, document_frequency: Counter, n_documents: int) -> dict[str, float]:
    tokens = normalize(text).split()
    counts = Counter(tokens)
    vector = {
        token: (1 + math.log(count))
        * math.log((1 + n_documents) / (1 + document_frequency[token]) + 1)
        for token, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
    return {token: value / norm for token, value in vector.items()}


def lexical_scores(
    query: str, slot_texts: dict[int, str]
) -> list[tuple[int, float]]:
    """Rank slots by TF-IDF overlap with the query, best first.

    Deliberately model-free, and deliberately *not* the default: on our synthetic
    benchmark the queries are template-generated, so their content words are highly
    distinctive and this scorer is very strong (T2-c). Real benchmark items about one
    entity share almost all of their surface form, which is the opposite regime, so
    whether it transfers is an empirical question rather than an assumption.

    The inverse document frequency is estimated over the slot texts plus the query,
    which keeps the function self-contained: no corpus has to be carried around, and
    the same call can be made at write time and at read time.
    """
    if not slot_texts:
        return []
    corpus = list(slot_texts.values()) + [query]
    document_frequency: Counter = Counter()
    for text in corpus:
        document_frequency.update(set(normalize(text).split()))
    query_vector = _tfidf(query, document_frequency, len(corpus))
    scores: list[tuple[int, float]] = []
    for slot, text in slot_texts.items():
        vector = _tfidf(text, document_frequency, len(corpus))
        score = sum(weight * vector.get(token, 0.0)
                    for token, weight in query_vector.items())
        scores.append((int(slot), float(score)))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores


def route_lexical(
    query: str, slot_texts: dict[int, str], k: int = 1
) -> RoutingDecision:
    """Select slots by lexical overlap with the query."""
    if k <= 0:
        return RoutingDecision(slots=[])
    scores = lexical_scores(query, slot_texts)
    return RoutingDecision(slots=[slot for slot, _ in scores[:k]], scores=scores)
