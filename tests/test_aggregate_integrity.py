"""Every reported aggregate must match its own granular records.

The paper's numbers come from aggregates that the experiment scripts compute, so an error
in an aggregation would propagate into the paper with every downstream check still green:
the macros would be consistent with the bundle, and the bundle would be wrong. These tests
recompute each statistic from the per-episode or per-probe records and compare.

They skip where the runs are absent, because the point is to check the data this checkout
has, not to require it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The checker lives in the repository, not in the dev-only tools directory, because a test
# that imports a git-ignored file would pass here and fail on any other checkout.
sys.path.insert(0, str(ROOT / "experiments"))

from check_aggregates import (check_composition, check_inertia,  # noqa: E402
                              check_learned_key)


def _collect(kind: str) -> list[str]:
    problems: list[str] = []
    names = {
        "composition": ("t2_paraphrase.json", "t2_paraphrase_17b.json",
                        "t2_paraphrase_lexkey.json", "t2_paraphrase_17b_lexkey.json"),
        "inertia": ("t5_inertia.json", "t5_inertia_17b.json"),
        "learned": ("t2e_learned_key.json", "t2e_learned_key_17b.json"),
    }[kind]
    checker = {"composition": check_composition, "inertia": check_inertia,
               "learned": check_learned_key}[kind]
    for name in names:
        checker(name, problems)
    return problems


def test_composition_aggregates():
    if not (ROOT / "runs" / "t2_paraphrase.json").is_file():
        pytest.skip("composition runs not present")
    assert _collect("composition") == []


def test_inertia_aggregates():
    if not (ROOT / "runs" / "t5_inertia.json").is_file():
        pytest.skip("inertia runs not present")
    assert _collect("inertia") == []


def test_learned_key_aggregates():
    if not (ROOT / "runs" / "t2e_learned_key.json").is_file():
        pytest.skip("learned-key runs not present")
    assert _collect("learned") == []
