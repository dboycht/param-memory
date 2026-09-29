"""Tests for the retention-anchor selection rule (pure, no model)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from parammem.memory.retention import earlier_queries


@dataclass
class _Item:
    query: str


ITEMS = [_Item(f"q{i}") for i in range(10)]


def test_first_write_has_nothing_to_protect():
    assert earlier_queries(ITEMS, 0) == []


def test_earlier_queries_exclude_the_current_item():
    got = earlier_queries(ITEMS, 3)
    assert got == ["q0", "q1", "q2"]
    assert "q3" not in got


def test_negative_index_is_rejected():
    with pytest.raises(ValueError, match="index"):
        earlier_queries(ITEMS, -1)


def test_limit_is_validated():
    with pytest.raises(ValueError, match="limit"):
        earlier_queries(ITEMS, 3, limit=0)


def test_under_the_limit_everything_is_protected():
    assert earlier_queries(ITEMS, 4, limit=8) == ["q0", "q1", "q2", "q3"]


def test_over_the_limit_selection_spans_the_whole_history():
    # 9 earlier items, keep 3: the oldest and the newest must both survive, because
    # the oldest memories are the ones with the most interference stacked on them
    got = earlier_queries(ITEMS, 9, limit=3)
    assert len(got) == 3
    assert got[0] == "q0"
    assert got[-1] == "q8"
    assert got == sorted(got, key=lambda q: int(q[1:]))


def test_selection_is_deterministic_at_every_size():
    for index in range(10):
        for limit in (1, 2, 3, 5, 8):
            first = earlier_queries(ITEMS, index, limit=limit)
            assert first == earlier_queries(ITEMS, index, limit=limit)
            assert len(first) == min(limit, index)
            assert len(set(first)) == len(first)
