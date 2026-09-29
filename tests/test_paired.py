"""Tests for the paired-comparison helpers (pure)."""

from __future__ import annotations

import pytest

from parammem.eval.paired import paired_counts, sign_test_p


def test_counts_helped_hurt_and_ties():
    baseline = [1, 1, 0, 0, 1]
    other = [0, 1, 1, 0, 0]
    assert paired_counts(baseline, other) == (1, 2, 2)


def test_counts_accept_bools_and_floats():
    # 1.0 counts as correct, 0.0 as incorrect, so only position 0 differs
    assert paired_counts([0, 0], [1.0, 0.0]) == (1, 0, 1)
    assert paired_counts([True, False], [1.0, 0.0]) == (0, 0, 2)


def test_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="same length"):
        paired_counts([1, 0, 1], [1, 0])


def test_all_ties_is_not_evidence():
    assert sign_test_p(0, 0) == 1.0
    assert paired_counts([1, 0], [1, 0]) == (0, 0, 2)


@pytest.mark.parametrize("helped,hurt,expected", [
    (7, 1, 0.070),      # the B1 confirmation: suggestive, not significant
    (7, 0, 0.0156),
    (5, 2, 0.453),
    (2, 0, 0.5),
    (10, 2, 0.039),
])
def test_sign_test_values(helped, hurt, expected):
    assert sign_test_p(helped, hurt) == pytest.approx(expected, abs=5e-4)


def test_sign_test_is_symmetric():
    for helped, hurt in ((3, 1), (9, 4), (0, 4)):
        assert sign_test_p(helped, hurt) == sign_test_p(hurt, helped)


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        sign_test_p(-1, 2)
