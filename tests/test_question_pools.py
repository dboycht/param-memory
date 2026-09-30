"""Guards for the benchmark question pools.

The multi-session prefix was guessed as ``multi-session-`` while the data spells the type
``multi-session``, so the pool was empty and the failure surfaced as ``mean()`` complaining
about an empty list a long way from the cause. These tests pin the pool shapes against the
real file, and skip where the file is absent rather than pretending to have checked.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from parammem.bench.longmemeval import (MULTI_SESSION_PREFIX, SINGLE_SESSION_PREFIX,
                                        load_oracle, select_by_type,
                                        select_multi_session, select_single_session)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"


def _raw():
    if not DATA.is_file():
        pytest.skip("LongMemEval oracle file not present")
    return load_oracle(DATA)


def test_both_pools_are_populated():
    raw = _raw()
    multi = select_multi_session(raw, subset=200)
    single = select_single_session(raw, subset=200)
    assert multi, f"no questions matched {MULTI_SESSION_PREFIX!r}"
    assert single, f"no questions matched {SINGLE_SESSION_PREFIX!r}"


def test_pools_are_disjoint():
    raw = _raw()
    multi_ids = {d["question_id"] for d in select_multi_session(raw, subset=500)}
    single_ids = {d["question_id"] for d in select_single_session(raw, subset=500)}
    assert not (multi_ids & single_ids)


def test_selection_is_deterministic_and_ordered():
    raw = _raw()
    first = [d["question_id"] for d in select_multi_session(raw, subset=10)]
    second = [d["question_id"] for d in select_multi_session(raw, subset=10)]
    assert first == second
    assert first == sorted(first)


def test_offset_selects_a_different_window():
    raw = _raw()
    first = [d["question_id"] for d in select_multi_session(raw, subset=5, offset=0)]
    second = [d["question_id"] for d in select_multi_session(raw, subset=5, offset=5)]
    assert first and second and not (set(first) & set(second))
