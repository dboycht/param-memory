"""Tests for the shared LongMemEval helpers (pure, no model required)."""

from __future__ import annotations

import json

import pytest

from parammem.bench.longmemeval import (
    containment,
    contains_answer,
    evidence_turns,
    load_oracle,
    select_single_session,
    sha256_file,
)


def _item(qid: str, kind: str = "single-session-user", turns: int = 1) -> dict:
    return {
        "question_id": qid,
        "question_type": kind,
        "question": f"question {qid}",
        "answer": f"answer {qid}",
        "haystack_sessions": [
            [{"role": "user", "content": "x", "has_answer": i < turns}
             for i in range(2)]
        ],
    }


RAW = [
    _item("b", "multi-session"),
    _item("c", "single-session-user"),
    _item("a", "single-session-assistant", turns=2),
    _item("d", "single-session-preference"),
    _item("e", "single-session-user"),
]


def test_selection_filters_type_sorts_by_id_and_honours_offset():
    assert [d["question_id"] for d in select_single_session(RAW, 10)] == \
        ["a", "c", "d", "e"]
    assert [d["question_id"] for d in select_single_session(RAW, 2)] == ["a", "c"]
    assert [d["question_id"] for d in select_single_session(RAW, 2, offset=2)] == \
        ["d", "e"]
    assert select_single_session(RAW, 5, offset=4) == []


def test_selection_is_deterministic_regardless_of_input_order():
    shuffled = list(reversed(RAW))
    first = [d["question_id"] for d in select_single_session(RAW, 3)]
    second = [d["question_id"] for d in select_single_session(shuffled, 3)]
    assert first == second == ["a", "c", "d"]


def test_subset_and_offset_are_disjoint():
    head = {d["question_id"] for d in select_single_session(RAW, 2)}
    tail = {d["question_id"] for d in select_single_session(RAW, 2, offset=2)}
    assert not (head & tail)


def test_evidence_turns_counts_flagged_turns():
    assert evidence_turns(_item("a", turns=2)) == 2
    assert evidence_turns(_item("a", turns=0)) == 0
    assert evidence_turns({"question_id": "x"}) == 0


@pytest.mark.parametrize(
    "prediction,gold,expected",
    [
        ("The answer is Paris.", "Paris", 1.0),
        ("Paris", "Paris", 1.0),
        ("I do not know", "Paris", 0.0),
        ("", "Paris", 0.0),
        ("Paris", "", 0.0),
    ],
)
def test_containment_basic(prediction, gold, expected):
    assert containment(prediction, gold) == pytest.approx(expected)


def test_containment_gives_partial_credit_for_clause_overlap():
    gold = "red and blue and green"
    assert containment("it is red and blue", gold) == pytest.approx(0.5)


def test_contains_answer_is_strict():
    assert contains_answer("The answer is Paris.", "Paris")
    assert not contains_answer("it is red", "red and blue and green")


def test_load_and_hash_round_trip(tmp_path):
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(RAW), encoding="utf-8")
    assert load_oracle(path) == RAW
    digest = sha256_file(path)
    assert len(digest) == 64
    assert digest == sha256_file(path)  # stable across calls
    path.write_text(json.dumps(RAW + [{"question_id": "z"}]), encoding="utf-8")
    assert sha256_file(path) != digest  # sensitive to content
