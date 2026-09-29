"""Tests for the isolation protocol P1-P5 and its statistics.

These are the assertions that decide whether *any* later experiment result is
reportable, so they are tested without a model in the loop.
"""

from __future__ import annotations

import math

import pytest

from parammem.bench import protocol as P


# ------------------------------------------------------------------ normalize

def test_normalize_case_punctuation_whitespace():
    assert P.normalize("  Velmora,  the  Great! ") == "velmora the great"


def test_normalize_fullwidth_and_keeps_hash():
    assert P.normalize("Ｖｅｌｍｏｒａ") == "velmora"
    assert P.normalize("#A1B2C3") == "#a1b2c3"


# --------------------------------------------------------------- numeric_forms

@pytest.mark.parametrize(
    "text",
    ["March 3, 2041", "2041-03-03", "3 March 2041", "03/03/2041", "03.03.2041"],
)
def test_numeric_forms_date_equivalences(text):
    assert P.numeric_forms(text) == {"2041-03-03"}, text


def test_numeric_forms_ignores_bare_years():
    assert P.numeric_forms("the year 2041") == set()
    assert P.numeric_forms("code 483920") == {"483920"}


# ------------------------------------------------------------- P1 eviction

def test_p1_clean_context_is_evicted():
    report = P.assert_evicted("Which city is the capital?", "Trennick")
    assert report.evicted and bool(report)


def test_p1_catches_exact_substring():
    report = P.assert_evicted("Right, Trennick it is.", "Trennick")
    assert not report.evicted
    assert report.checks["exact"]


def test_p1_catches_normalized_reformat():
    report = P.assert_evicted("right,   ＴＲＥＮＮＩＣＫ it is", "Trennick")
    assert not report.evicted
    assert report.checks["normalized"]


def test_p1_does_not_flag_a_word_as_its_own_prefix():
    report = P.assert_evicted("Trennickavel is nice", "Tren")
    assert report.evicted, report.reason()


def test_p1_catches_numeric_only_reformat():
    report = P.assert_evicted("born on 03.03.2041", "March 3, 2041")
    assert not report.evicted
    assert report.checks["numeric"]
    assert not report.checks["exact"] and not report.checks["normalized"]


def test_p1_ignores_unrelated_date():
    report = P.assert_evicted("the meeting is on 1999-12-31", "March 3, 2041")
    assert report.evicted, report.reason()


def test_p1_checks_alias_forms_too():
    value = "March 3, 2041"
    aliases = ("2041-03-03", "3 March 2041")
    report = P.assert_evicted("she was born on 2041-03-03", value, aliases)
    assert not report.evicted


# -------------------------------------------------------------------- P2

def test_counterfactual_accepts_distinct_values():
    report = P.compare_counterfactual("Trennick", ["Velmora", "Garlund"])
    assert report.ok


def test_counterfactual_detects_collision_after_normalization():
    report = P.compare_counterfactual("Trennick", ["Velmora", "  trennick. "])
    assert not report.ok and report.other_values


# -------------------------------------------------------------------- P3

def test_negative_control_accepts_specific_gain():
    report = P.negative_control(
        written_baseline_em=0.05, written_mem_em=0.70,
        unwritten_baseline_em=0.05, unwritten_mem_em=0.06,
    )
    assert report.ok


def test_negative_control_rejects_format_artefact():
    report = P.negative_control(
        written_baseline_em=0.05, written_mem_em=0.70,
        unwritten_baseline_em=0.05, unwritten_mem_em=0.30,
    )
    assert not report.ok and "artefact" in report.detail


def test_negative_control_rejects_no_gain():
    report = P.negative_control(
        written_baseline_em=0.05, written_mem_em=0.06,
        unwritten_baseline_em=0.05, unwritten_mem_em=0.05,
    )
    assert not report.ok


# ------------------------------------------------------------------ P4 / P5

def test_attribution_accepts_clean_effect():
    report = P.attribution_delta(
        em_mem_on=0.80, em_mem_off=0.10, em_shuffle=0.12,
        em_prompt_only=0.05, ci=(0.60, 0.90),
    )
    assert report.ok
    assert report.delta_net == pytest.approx(0.75)


def test_attribution_rejects_leak_when_mem_off_is_high():
    report = P.attribution_delta(
        em_mem_on=0.80, em_mem_off=0.70, em_shuffle=0.10, em_prompt_only=0.05
    )
    assert not report.ok and "leak" in report.detail.lower()


def test_attribution_leak_check_uses_prompt_only_baseline_not_mem_on():
    """The trap this guards: if the leak check compared MEM_OFF with MEM_ON, a
    run where *everything* is recoverable from the context would sail through
    because the two high numbers are far apart."""
    report = P.attribution_delta(
        em_mem_on=0.99, em_mem_off=0.90, em_shuffle=0.88, em_prompt_only=0.05
    )
    assert not report.ok
    # ... while the same numbers *would* have passed a MEM_ON-vs-MEM_OFF test.
    assert report.em_mem_on - report.em_mem_off == pytest.approx(0.09)


def test_attribution_rejects_when_another_slot_answers():
    report = P.attribution_delta(
        em_mem_on=0.80, em_mem_off=0.10, em_shuffle=0.50, em_prompt_only=0.05
    )
    assert not report.ok


def test_attribution_rejects_ci_including_zero():
    report = P.attribution_delta(
        em_mem_on=0.80, em_mem_off=0.10, em_shuffle=0.12,
        em_prompt_only=0.05, ci=(-0.10, 0.90),
    )
    assert not report.ok


# ------------------------------------------------------------------- stats

def test_bootstrap_ci_degenerate_inputs():
    lo, hi = P.bootstrap_ci([1.0] * 10, n=200, seed=1)
    assert (lo, hi) == (1.0, 1.0)
    assert math.isnan(P.bootstrap_ci([])[0])
    lo, hi = P.bootstrap_ci([0.25], n=200, seed=1)
    assert (lo, hi) == (0.25, 0.25)


def test_bootstrap_ci_brackets_the_mean():
    values = [0.0, 1.0] * 50
    lo, hi = P.bootstrap_ci(values, n=500, seed=2)
    assert 0.30 < lo <= 0.5 <= hi < 0.70


def test_bootstrap_ci_is_reproducible():
    values = [0.1, 0.9, 0.4, 0.6] * 5
    assert P.bootstrap_ci(values, n=300, seed=7) == P.bootstrap_ci(values, n=300, seed=7)


# ----------------------------------------------------------------- scoring

def test_exact_match_tolerates_surrounding_sentence():
    assert P.exact_match("The capital is Trennick.", "Trennick")
    assert not P.exact_match("I have no idea", "Trennick")
    assert not P.exact_match("", "Trennick")


def test_exact_match_handles_date_reformatting():
    assert P.exact_match("2041-03-03", "March 3, 2041")


def test_token_f1_bounds():
    assert P.token_f1("Trennick", "Trennick") == pytest.approx(1.0)
    assert P.token_f1("", "Trennick") == 0.0
    assert 0.0 < P.token_f1("Trennick maybe", "Trennick") < 1.0
    assert P.token_f1("unrelated words", "Trennick") == 0.0
