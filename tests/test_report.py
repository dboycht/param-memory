"""Tests for the results-bundle formatter (pure, no model required)."""

from __future__ import annotations

import pytest

from parammem.report import build_results_markdown, stage_status

BUNDLE = {
    "_provenance": {"commit": "abc1234", "model": "Qwen/Qwen3-0.6B", "mode": "full",
                    "generated_at": "2026-09-29 16:00:00"},
    "t1": {"summary": {
        "em": {"prompt_only": 0.0, "mem_on": 0.5, "mem_shuffle": 0.0, "mem_off": 0.0},
        "em_counts": {"prompt_only": [4, 0], "mem_on": [4, 2], "mem_shuffle": [4, 0],
                      "mem_off": [4, 0]},
        "delta_on_minus_prompt_only": 0.5, "delta_ci95": [0.1, 0.8],
        "negative_control": {"detail": "specific gain confirmed"},
        "attribution": {"detail": "attributable"},
        "frozen_ok_everywhere": True, "ce_after_mean": 0.0,
        "mem_off_string_identical": "4/4",
    }},
    "t2": {"summary": {"em": {"oracle": 1.0, "all": 0.0, "top1": 0.9, "top2": 0.25,
                              "other": 0.0},
                       "counts": {"oracle": [16, 16], "all": [0, 16], "top1": [14, 16],
                                  "top2": [4, 16], "other": [0, 16]},
                       "route_top1_accuracy": 0.94, "config": {"paraphrase": True}}},
    "t3": {"tau": 6.187, "known_confirmed": ["a", "b"], "known_dropped": [{"q": "c"}],
           "records": [{"policy": "selfcheck", "n_writes": 5, "written_unknown": 5,
                        "unknown_total": 5, "written_known": 0, "known_total": 13,
                        "missed_unknown": 0, "retained_unknown": 5, "anchor_kl": 0.54}],
           "surprise_by_class": {"unknown": [2.0, 4.0, 8.0], "known": [3.0, 6.0, 10.0]}},
    "t4": {"config": {"stream": 12, "slots": 6, "hot": 3},
           "records": [{"policy": "fifo", "evictions": 6, "erased_virgin_ok": 6,
                        "erased_checked": 6, "resident": {"hot": 1, "cold": 5},
                        "oracle_hits": {"hot": 1, "cold": 5}, "routed_hits": 5,
                        "n_survivors": 6, "important_retained": True, "anchor_kl": 0.97}]},
    "t5": {"floor_screen": {"candidates": 30, "kept": 25, "dropped": 5, "dropped_detail": []},
           "arms": {"no_memory": {"inheritance_rate": 0.0, "residue_rate": 0.0,
                                  "withholding_rate": 0.17},
                    "param_memory": {"inheritance_rate": 1.0, "residue_rate": 0.42,
                                     "withholding_rate": 0.0}},
           "premise_recall_rate": 1.0, "control_identical_rate": 1.0},
    "t6": {"write": {"own_slot_recall": "6/6",
                     "snapshot": {"n_slots": 8, "n_modules": 112,
                                  "bytes_on_disk": 18_390_587}},
           "read": {"virgin_before_load": True, "oracle_recall": "6/6",
                    "routed_recall": "5/6", "routing_correct": "5/6",
                    "erase_after_load_is_virgin": True,
                    "erased_memory_still_recalled": False}},
}


def test_bundle_contains_every_headline_number():
    text = build_results_markdown(BUNDLE)
    for fragment in (
        "abc1234",                    # provenance
        "`mem_on` | 0.500 | 2/4",     # t1 arm
        "specific gain confirmed",
        "0.940",                      # t2 router accuracy
        "`selfcheck` | 5 | 5/5",      # t3 criterion row
        "likelihood criterion fails", # t3 score table heading
        "`fifo` | 6 | 6/6",           # t4 policy row (erase -> virgin)
        "25/30 scenarios kept",       # t5 floor screen
        "virgin before load = **True**",  # t6
        "18.4 MB",                    # t6 snapshot size
    ):
        assert fragment in text, fragment


def test_all_stages_reported_ok():
    assert dict(stage_status(BUNDLE)) == {
        "t1": "ok", "t2": "ok", "t3": "ok", "t4": "ok", "t5": "ok", "t6": "ok"
    }


def test_missing_stage_says_not_available_instead_of_zero():
    partial = {k: v for k, v in BUNDLE.items() if k not in {"t4", "_provenance"}}
    text = build_results_markdown(partial)
    assert "T4" in text
    assert "_not available in this run_" in text
    statuses = dict(stage_status(partial))
    assert statuses["t4"] == "missing" and statuses["t1"] == "ok"


def test_failed_stage_is_labelled():
    bundle = dict(BUNDLE)
    bundle["t3"] = {"_failed": True, "_error": "exited with 1"}
    statuses = dict(stage_status(bundle))
    assert statuses["t3"].startswith("FAILED")
    text = build_results_markdown(bundle)
    assert "FAILED: exited with 1" in text


def test_t3_class_table_is_sorted_before_summarising():
    text = build_results_markdown(BUNDLE)
    # median of [2, 4, 8] is 4.00, min 2.00, max 8.00
    assert "| unknown | 3 | 2.00 | 4.00 | 8.00 |" in text


def test_empty_bundle_does_not_crash():
    text = build_results_markdown({})
    assert text.count("_not available in this run_") == 6


def test_provenance_is_optional():
    text = build_results_markdown({"t1": BUNDLE["t1"]})
    assert "## Provenance" not in text
    assert "T1" in text


def test_t2_row_shows_counts_not_just_rates():
    text = build_results_markdown(BUNDLE)
    assert "| `top1` | 0.900 | 14/16 |" in text
