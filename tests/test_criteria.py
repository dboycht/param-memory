"""Tests for the write criteria (pure, no model required)."""

from __future__ import annotations

import pytest

from parammem.bench.known import KNOWN_PAIRS
from parammem.memory.criteria import (
    EXPLICIT_MARKERS,
    always_decision,
    explicit_decision,
    looks_explicit,
    selfcheck_decision,
    surprise_decision,
)


def test_always_writes():
    assert always_decision()
    assert always_decision().should_write


def test_surprise_writes_only_above_threshold():
    assert surprise_decision(loss=12.0, threshold=4.0).should_write
    assert not surprise_decision(loss=0.4, threshold=4.0).should_write
    assert surprise_decision(loss=4.0, threshold=4.0).should_write  # boundary: write


def test_surprise_reports_the_measurement():
    decision = surprise_decision(loss=9.5, threshold=4.0)
    assert decision.score == pytest.approx(9.5)
    assert "9.500" in decision.reason


def test_surprise_refuses_to_write_on_a_broken_measurement():
    """A NaN loss means the measurement failed; writing anyway would silently
    fill the store with memories nobody can explain."""
    decision = surprise_decision(loss=float("nan"), threshold=4.0)
    assert not decision.should_write
    assert "non-finite" in decision.reason


@pytest.mark.parametrize("marker", EXPLICIT_MARKERS)
def test_every_marker_is_detected_case_insensitively(marker):
    assert looks_explicit(f"Please {marker.upper()} this fact.")
    assert looks_explicit(f"please {marker} this fact.")


def test_explicit_decision_matches_markers():
    assert explicit_decision("Remember that A was born on 1 May 1990.").should_write
    assert explicit_decision("Note to self: cut the batch size in half.").should_write
    assert not explicit_decision("The capital of France is Paris.").should_write


def test_known_pairs_are_not_explicit_requests():
    """The 'already known' class must not be recognisable by the explicit marker,
    otherwise the criterion comparison would be rigged."""
    for pair in KNOWN_PAIRS:
        assert not looks_explicit(pair.statement), pair.statement


def test_known_pairs_have_distinct_queries():
    queries = [p.query for p in KNOWN_PAIRS]
    assert len(set(queries)) == len(queries)


def test_expected_fragment_is_lower_cased_answer():
    for pair in KNOWN_PAIRS:
        assert pair.expected_fragment == pair.answer.lower()


# --------------------------------------------------------------- self-check

def test_selfcheck_skips_what_the_model_already_answers():
    decision = selfcheck_decision("The capital of France is Paris.", "Paris")
    assert not decision.should_write
    assert "already answers" in decision.reason


def test_selfcheck_writes_what_the_model_cannot_answer():
    decision = selfcheck_decision("Baku", "Truzolkuoth")
    assert decision.should_write
    assert "cannot answer" in decision.reason


def test_selfcheck_writes_on_an_empty_answer():
    assert selfcheck_decision("", "Truzolkuoth").should_write


def test_selfcheck_honours_aliases():
    decision = selfcheck_decision("She was born on 2041-03-03.", "March 3, 2041",
                                 ("2041-03-03",))
    assert not decision.should_write
