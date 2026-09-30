"""Guards for the judge-calibration merge.

The agreement rate is a headline number in the paper, so the two conventions that
produce it are pinned here: a truncated judge reply is excluded rather than scored, and
an interval is reported alongside any point estimate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "calibration_under_test", ROOT / "experiments" / "t7h_calibration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_marks_string_parses_both_verdicts():
    module = _module()
    assert module.parse_marks("1Y 2N 3Y") == {1: True, 2: False, 3: True}
    assert module.parse_marks("1y, 2n") == {1: True, 2: False}
    assert module.parse_marks("") == {}


def test_wilson_interval_is_wide_when_the_sample_is_small():
    """10/10 is *not* a 0% error rate, which is the whole reason for the interval."""
    module = _module()
    low, high = module.wilson(10, 10)
    assert high == 1.0
    assert 0.6 < low < 0.8, low
    assert 1 - low > 0.2, "the upper end of the error rate must stay visible"


def test_wilson_interval_contains_the_point_estimate():
    module = _module()
    for successes, total in ((7, 10), (30, 40), (0, 10)):
        low, high = module.wilson(successes, total)
        assert low <= successes / total <= high


def test_round_one_reproduces_the_published_agreement():
    """Round 1's marks are all usable against the *current* payload.

    The first pass left two rows unreadable because the judge's own reply was truncated;
    after the prompt was revised the same twelve rows were re-judged, so the payload in
    the repository now carries twelve readable verdicts that all agree with the human.
    The marks file still documents the first pass, and both facts are asserted here so a
    future reader cannot silently quote one as the other.
    """
    module = _module()
    round_one = module.load_round(ROOT / "bench" / "judge_calibration_marks.json",
                                 ROOT / "runs" / "judge_calibration.json")
    if not round_one["rows"]:
        pytest.skip("round-1 judge payload is not present in this checkout")
    assert len(round_one["rows"]) == 12
    scored = [row for row in round_one["rows"] if not row["truncated"]]
    assert len(scored) == 12, "the revised prompt left no unreadable rows"
    assert all(row["judge"] is not None for row in scored)
    assert sum(1 for row in scored if row["human"] == row["judge"]) == 12

    documented = json.loads(
        (ROOT / "bench" / "judge_calibration_marks.json").read_text(encoding="utf-8"))
    assert documented["parsed"]["4"] is True and documented["parsed"]["5"] is True
    assert documented["discounted_rows"] == [4, 5], (
        "the marks file must keep documenting the first pass, where two rows were "
        "truncated and therefore unusable")
    assert documented["result"]["first_pass_agreement"] == "8/10"
