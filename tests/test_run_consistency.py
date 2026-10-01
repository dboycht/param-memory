"""A run file must not argue with its own rows.

Regrading the questions a throttled provider had missed wrote new verdicts into the rows but
left the summary's rates and counters at their pre-regrade values, so a file reported one
accuracy while its own rows contained a different set of graded questions. That is worse than
a missing number, because both halves look plausible and the wrong one reaches the paper.

These tests recompute every aggregate from the per-question records and require the saved
summary to agree, for each run file present.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

FILES = ("t7i_multisession.json", "t7i_multisession_17b.json", "t7i_hosted_reader.json")


def _payload(name: str) -> dict:
    path = RUNS / name
    if not path.is_file():
        pytest.skip(f"{name} not present in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", FILES)
def test_saved_rates_match_the_rows(name: str):
    payload = _payload(name)
    rows = payload.get("results") or []
    if not rows:
        pytest.skip(f"{name} carries no per-question records")
    arms = payload.get("arms") or list(rows[0]["answers"])
    saved = payload.get("judged") or {}
    if not saved:
        pytest.skip(f"{name} does not report judged rates")
    for arm in arms:
        graded = [row for row in rows if row["answers"][arm].get("judged") is not None]
        if not graded:
            continue
        expected = sum(1 for row in graded if row["answers"][arm]["judged"]) / len(graded)
        assert abs(saved.get(arm, -1) - expected) < 1e-9, (
            f"{name}: {arm} is reported as {saved.get(arm)} but its rows give {expected}; "
            f"the aggregates were not rebuilt after the rows changed")


@pytest.mark.parametrize("name", FILES)
def test_graded_counter_matches_the_rows(name: str):
    """The counter must be the denominator behind the rate, not a looser count.

    It is the number of questions the *primary* arm was graded on, because that is the arm a
    sentence quotes a rate for. Counting rows with any verdict at all would let the text say
    "0.68 of 30" while the arm it names was graded on 28.
    """
    payload = _payload(name)
    rows = payload.get("results") or []
    if not rows or not ({"n_judged", "n_graded"} & set(payload)):
        pytest.skip(f"{name} does not report a graded counter")
    field = "n_judged" if "n_judged" in payload else "n_graded"
    primary = payload.get("primary_arm") or "context"
    if primary not in rows[0]["answers"]:
        pytest.skip(f"{name} has no {primary} arm")
    counted = sum(1 for row in rows
                  if row["answers"][primary].get("judged") is not None)
    assert payload[field] == counted, (
        f"{name}: {field} says {payload[field]} but {counted} rows carry a verdict for "
        f"the {primary} arm")
    any_verdict = sum(1 for row in rows
                      if any((entry or {}).get("judged") is not None
                             for entry in row["answers"].values()))
    assert payload[field] <= any_verdict


@pytest.mark.parametrize("name", FILES)
def test_string_rates_match_the_rows(name: str):
    payload = _payload(name)
    rows = payload.get("results") or []
    saved = payload.get("containment") or {}
    if not rows or not saved:
        pytest.skip(f"{name} does not report containment")
    for arm, rate in saved.items():
        expected = sum(1 for row in rows if row["answers"][arm].get("correct")) / len(rows)
        assert abs(rate - expected) < 1e-9, (
            f"{name}: containment for {arm} is {rate} but its rows give {expected}")
