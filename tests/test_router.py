"""Tests for the read-time slot router (pure, no model required)."""

from __future__ import annotations

import pytest
import torch

from parammem.memory.router import cosine_scores, route


def _key(*values: float) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float32)


def test_empty_slot_keys_routes_nowhere():
    decision = route(_key(1.0, 0.0), {})
    assert decision.slots == [] and decision.top is None
    assert not decision


def test_identical_key_selects_that_slot():
    slot_keys = {0: _key(1.0, 0.0), 1: _key(0.0, 1.0)}
    assert route(_key(0.5, 0.5), slot_keys).top == 0


def test_top_k_returns_best_first():
    slot_keys = {3: _key(0.1, 1.0), 7: _key(1.0, 0.0), 9: _key(0.6, 0.8)}
    decision = route(_key(1.0, 0.0), slot_keys, k=2)
    assert decision.slots == [7, 9]
    assert [s for s, _ in decision.scores] == [7, 9, 3]


def test_k_larger_than_slots_is_clamped():
    slot_keys = {1: _key(1.0), 2: _key(0.5)}
    assert route(_key(1.0), slot_keys, k=10).slots == [1, 2]


def test_k_zero_means_deactivate_everything():
    slot_keys = {0: _key(1.0), 1: _key(1.0)}
    assert route(_key(1.0), slot_keys, k=0).slots == []
    assert route(_key(1.0), slot_keys, k=-3).slots == []


def test_scores_are_scale_invariant():
    slot_keys = {0: _key(2.0, 0.0), 1: _key(0.0, 5.0)}
    decision = route(_key(10.0, 0.0), slot_keys)
    assert decision.top == 0
    assert decision.scores[0][1] == pytest.approx(1.0)


def test_zero_vector_does_not_produce_nan():
    slot_keys = {0: _key(0.0, 0.0), 1: _key(1.0, 0.0)}
    scores = cosine_scores(_key(0.0, 0.0), slot_keys)
    assert all(score == score for _, score in scores)  # no NaN
    decision = route(_key(1.0, 0.0), slot_keys)
    assert decision.top == 1


def test_ties_are_broken_by_slot_index_for_determinism():
    slot_keys = {5: _key(1.0, 0.0), 2: _key(1.0, 0.0)}
    decision = route(_key(1.0, 0.0), slot_keys, k=2)
    assert decision.slots == [2, 5]
    assert [slot for slot, _ in decision.scores] == [2, 5]
    assert route(_key(1.0, 0.0), slot_keys).top == 2  # k=1 default


def test_scores_are_sorted_descending():
    slot_keys = {i: _key(1.0, i * 0.5) for i in range(5)}
    scores = [s for _, s in cosine_scores(_key(1.0, 0.0), slot_keys)]
    assert scores == sorted(scores, reverse=True)
