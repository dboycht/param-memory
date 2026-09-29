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
    "t7": {"n_subset": 30, "data_sha256": "821a2034d219ab45deadbeef",
           "subset_ids": [f"q{i}" for i in range(30)], "negative_ids": ["n1", "n2"],
           "p1_leaking_ids": [],
           "containment": {"frozen": 0.017, "mem_on": 0.817, "mem_off": 0.017,
                           "mem_on_full_match": 0.633},
           "routing_accuracy": 1.0, "mem_off_equals_frozen_rate": 1.0,
           "negative_control": {"frozen": 0.1, "mem_on": 0.0},
           "write": {"count": 30, "total_seconds": 1364.0, "mean_ce_after": 0.001},
           "caveats": ["oracle indexing"],
           "holdout": {"n_subset": 30,
                       "containment": {"frozen": 0.0, "mem_on": 0.80, "mem_off": 0.0,
                                       "mem_on_full_match": 0.60},
                       "routing_accuracy": 1.0, "mem_off_equals_frozen_rate": 1.0}},
    "t8": {"by_lambda": {"0.0": {"em": {"oracle": 0.917, "top1": 0.875, "top2": 0.333,
                                         "all": 0.0},
                                  "base_kl_slots_on": 1.504},
                         "3.0": {"em": {"oracle": 1.0, "top1": 0.958, "top2": 0.583,
                                        "all": 0.167},
                                 "base_kl_slots_on": 0.309}},
           # episode 1 is listed before episode 0 for lambda 3 on purpose: the
           # pairing must sort by episode id, not trust file order
           "episodes": [
               {"episode_id": 0, "lambda_ret": 0.0, "arms": {"top2": [1, 0]}},
               {"episode_id": 1, "lambda_ret": 0.0, "arms": {"top2": [0, 0]}},
               {"episode_id": 1, "lambda_ret": 3.0, "arms": {"top2": [1, 0]}},
               {"episode_id": 0, "lambda_ret": 3.0, "arms": {"top2": [1, 1]}},
           ]},
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
        "t1": "ok", "t2": "ok", "t3": "ok", "t4": "ok", "t5": "ok", "t6": "ok",
        "t7": "ok", "t8": "ok",
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
    assert text.count("_not available in this run_") == 8


def test_provenance_is_optional():
    text = build_results_markdown({"t1": BUNDLE["t1"]})
    assert "## Provenance" not in text
    assert "T1" in text


def test_t2_row_shows_counts_not_just_rates():
    text = build_results_markdown(BUNDLE)
    assert "| `top1` | 0.900 | 14/16 |" in text


# ---------------------------------------------------------------------- LaTeX

def test_latex_emits_headline_macros_with_the_right_values():
    from parammem.report import build_results_latex
    text = build_results_latex(BUNDLE)
    assert r"\newcommand{\tTopOne}{14/16}" in text
    assert r"\newcommand{\tSum}{0/16}" in text
    assert r"\newcommand{\tOracle}{16/16}" in text
    assert r"\newcommand{\tSelfKL}{0.54}" in text
    assert r"\newcommand{\tVirgin}{True}" in text
    # per-policy 6/6 plus the cross-policy total, so the abstract can state either
    assert r"\newcommand{\tErased}{6/6}" in text
    assert r"\newcommand{\tErasedTotal}{6/6}" in text   # one policy in this bundle
    assert r"\newcommand{\tPolicies}{1}" in text


def test_latex_escapes_cells_but_not_captions():
    from parammem.report import build_results_latex
    text = build_results_latex(BUNDLE)
    # policy names carry an underscore and must be escaped inside the table body
    assert r"\texttt{utility\_time}" in text or r"\texttt{fifo}" in text
    # captions keep the markup the author wrote
    assert r"\texttt{mem\_on}" in text
    assert r"\textbackslash{}texttt" not in text


def test_latex_marks_missing_stage_and_still_compiles_structurally():
    from parammem.report import build_results_latex
    text = build_results_latex({"t1": BUNDLE["t1"]})
    assert text.count(r"\begin{table}") == text.count(r"\end{table}")
    assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}")
    assert "not available" in text


def test_latex_and_markdown_agree_on_the_headline_numbers():
    from parammem.report import build_results_latex, build_results_markdown
    md, tex = build_results_markdown(BUNDLE), build_results_latex(BUNDLE)
    for fragment in ("14/16", "0/16", "16/16", "0.54", "25/30", "6/6"):
        assert fragment in md, fragment
        assert fragment in tex, fragment


# ------------------------------------------------- the paper and its tables

def test_every_macro_used_by_the_paper_is_defined_in_the_generated_tables():
    """Regression: the first draft of main.tex referred to \\tSum, \\tOracle, ... while
    the generator emitted \\rTTwoSum, ... -- LaTeX then fails with 'Undefined control
    sequence'. This checks the contract between the two files without compiling."""
    from pathlib import Path

    paper = Path(__file__).resolve().parents[1] / "paper"
    main_tex = (paper / "main.tex").read_text(encoding="utf-8")
    generated = (paper / "generated_tables.tex").read_text(encoding="utf-8")

    import re

    # Generated macros are camelCase with an uppercase second letter (\tSum,
    # \tTopOne, \tErasedTotal). Requiring that shape keeps LaTeX built-ins such as
    # \text, \texttt, \theta, \toprule and \today out of the comparison.
    used = set(re.findall(r"\\(t[A-Z][A-Za-z]*)\b", main_tex))
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", generated))
    assert used, "no generated macros are referenced by main.tex"
    missing = sorted(used - defined)
    assert not missing, f"main.tex uses undefined macros: {missing}"


def test_generated_tables_refuse_to_be_hand_edited():
    from pathlib import Path
    generated = (Path(__file__).resolve().parents[1] / "paper"
                 / "generated_tables.tex").read_text(encoding="utf-8")
    assert generated.startswith("% Generated by parammem.report.build_results_latex")
    assert "do not edit by hand" in generated.splitlines()[0]


def test_quick_mode_never_overwrites_a_full_run_artifact():
    """Regression guard: quick mode used to write over exactly the files that
    --collect-only reads (t6's 2-item snapshot would have replaced the real one),
    so a smoke test could silently become the published result."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_all_under_test",
                                                 root / "experiments" / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for stage, recipe in module.STAGES.items():
        collected = {str(p).replace("\\", "/") for p in recipe["inputs"]}
        quick_args = []
        if "quick_phases" in recipe:
            for phase in recipe["quick_phases"]:
                quick_args += phase
        else:
            quick_args += recipe["quick"]
        written = set()
        for index, token in enumerate(quick_args):
            if token in ("--out", "--snapshot") and index + 1 < len(quick_args):
                written.add(quick_args[index + 1].split("/")[-1])
        assert not (written & collected), (
            f"{stage}: quick mode writes {sorted(written & collected)}, which "
            "collect() reads back as the full run's artifact"
        )


def test_a_full_run_refreshes_what_collect_reads():
    """Two stages wrote their full run somewhere other than the file `inputs`
    names, so `--mode full` silently failed to refresh the numbers the paper
    quotes -- the collected values would have stayed from an older run."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_all_under_test_full",
                                                 root / "experiments" / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for stage, recipe in module.STAGES.items():
        collected = {str(p).split("/")[-1] for p in recipe["inputs"]}
        if "phases" in recipe:
            args = [token for phase in recipe["phases"] for token in phase]
        else:
            args = list(recipe["full"])
        written = set()
        for index, token in enumerate(args):
            if token in ("--out", "--snapshot") and index + 1 < len(args):
                written.add(args[index + 1].split("/")[-1])
        assert written & collected, (
            f"{stage}: --mode full writes {sorted(written) or ['<defaults>']}, none of "
            f"which is a collected input ({sorted(collected)}), so a full run would "
            "not refresh the reported numbers"
        )


# ------------------------------------------------------------ headline values

def test_headline_values_exposes_the_quoted_numbers():
    from parammem.report import headline_values
    values = headline_values(BUNDLE)
    assert values["tSum"] == "0/16"
    assert values["tTopOne"] == "14/16"
    assert values["tOracle"] == "16/16"
    assert values["tErasedTotal"] == "6/6"      # one policy in this bundle
    assert values["tPolicies"] == "1"
    assert values["tSnapshotMB"] == "18.4"
    assert values["tScreenKept"] == "25/30"


def test_headline_values_use_na_never_zero_for_missing_data():
    from parammem.report import headline_values
    values = headline_values({})
    assert values["tSum"] == "n/a"
    assert values["tSelfWrites"] == "n/a"
    assert values["tSnapshotMB"] == "n/a"
    assert "0" not in {values["tFloor"]}


def test_chinese_report_template_only_uses_generated_values():
    """Same contract as the LaTeX check, for paper/build_report_zh.py: every
    ``{placeholder}`` in the Chinese report must be produced by headline_values,
    otherwise str.format raises KeyError at generation time."""
    import re
    from pathlib import Path

    from parammem.report import headline_values

    source = (Path(__file__).resolve().parents[1] / "paper"
              / "build_report_zh.py").read_text(encoding="utf-8")
    template = source.split('TEMPLATE = """', 1)[1].split('"""', 1)[0]
    # single-brace placeholders only; ``{{`` and ``}}`` are escaped literal braces
    placeholders = set(re.findall(r"(?<!\{)\{([a-z][A-Za-z0-9_]*)\}(?!\})", template))
    available = set(headline_values(BUNDLE)) | {"generated_at", "commit", "model"}
    assert placeholders, "no placeholders found in the Chinese template"
    missing = sorted(placeholders - available)
    assert not missing, f"Chinese report uses undefined values: {missing}"


def test_macro_names_are_letters_only():
    """Regression: \\tT4Stream is NOT a valid macro name -- a digit ends a TeX
    control word, so LaTeX reported 'Missing number, treated as zero' at the
    \\newcommand line and the PDF came out empty."""
    from parammem.report import build_results_latex, headline_values
    names = list(headline_values(BUNDLE))
    bad = [n for n in names if not n.isalpha()]
    assert not bad, f"macro names must be letters only: {bad}"
    assert "tStreamItems" in names and "tCapacitySlots" in names
    build_results_latex(BUNDLE)  # must not raise the letters-only guard


def test_letters_only_guard_fires_on_a_bad_name(monkeypatch):
    from parammem import report as report_module
    monkeypatch.setattr(
        report_module, "headline_values",
        lambda bundle: {"tBad4Name": "1"},
    )
    with pytest.raises(ValueError, match="letters only"):
        report_module.build_results_latex(BUNDLE)


# --------------------------------------------------- held-out replication

def test_replication_verdict_applies_the_pre_registered_band():
    """The band (first value ±0.15) was fixed in docs/06 section 9 *before* the
    replication ran, so it is applied by code rather than by eye."""
    from parammem.report import replication_verdict

    base = {"t7": {"containment": {"mem_on": 0.817}, "holdout": {"containment": {}}}}
    assert replication_verdict(base) == "n/a"          # nothing to compare yet

    base["t7"]["holdout"]["containment"]["mem_on"] = 0.80   # inside ±0.15
    assert replication_verdict(base) == "within the pre-registered band"

    base["t7"]["holdout"]["containment"]["mem_on"] = 0.50   # outside
    assert replication_verdict(base) == "outside the pre-registered band"


def test_replication_is_reported_in_both_renderings():
    from parammem.report import build_results_latex, build_results_markdown
    md = build_results_markdown(BUNDLE)
    tex = build_results_latex(BUNDLE)
    assert "Held-out replication" in md
    assert "pre-registered verdict: within the pre-registered band" in md
    for text in (md, tex):
        assert "0.800" in text, text[:200]
    assert r"\newcommand{\tHoldoutOn}{0.800}" in tex
    assert r"\newcommand{\tHoldoutVerdict}{within the pre-registered band}" in tex


def test_retention_pairing_sorts_by_episode_and_reports_the_best_weight():
    from parammem.report import (_retention_best, _retention_pair,
                                 _retention_series, headline_values)
    # BUNDLE lists lambda 3.0's episode 1 before its episode 0: position i must still
    # be the same probe in both settings, or the pairing is meaningless
    assert _retention_series(BUNDLE["t8"], "3.0", "top2") == [1, 1, 1, 0]
    assert _retention_series(BUNDLE["t8"], "0.0", "top2") == [1, 0, 0, 0]
    assert _retention_pair(BUNDLE["t8"]) == (2, 0, 2)
    assert _retention_best(BUNDLE["t8"]) == "3.0"

    values = headline_values(BUNDLE)
    assert values["tRetentionTopTwoBase"] == "0.333"
    assert values["tRetentionTopTwoBest"] == "0.583"
    assert values["tRetentionHelped"] == "2"
    assert values["tRetentionHurt"] == "0"
    assert values["tRetentionProbes"] == "4"
    assert values["tRetentionKlBase"] == "1.504"
    assert values["tRetentionKlBest"] == "0.309"


def test_retention_macros_read_the_nested_shape_too():
    """Regression: the experiment nests by_lambda under `summary`, and reading only
    the flat shape silently turned every retention macro into "n/a"."""
    from parammem.report import headline_values
    nested = {"t8": {"summary": {"by_lambda": {
        "0.0": {"em": {"top2": 0.25, "oracle": 0.9, "all": 0.0},
                "base_kl_slots_on": 1.2, "route_top2_accuracy": 1.0},
        "3.0": {"em": {"top2": 0.75, "oracle": 1.0, "all": 0.1},
                "base_kl_slots_on": 0.3, "route_top2_accuracy": 1.0}}},
        "episodes": [{"episode_id": 0, "lambda_ret": 0.0, "arms": {"top2": [1, 0]}},
                     {"episode_id": 0, "lambda_ret": 3.0, "arms": {"top2": [1, 1]}}]}}
    values = headline_values(nested)
    assert values["tRetentionTopTwoBase"] == "0.250"
    assert values["tRetentionTopTwoBest"] == "0.750"
    assert values["tRetentionKlBase"] == "1.200"
    assert values["tRetentionKlBest"] == "0.300"
    assert values["tRetentionRouteTwo"] == "1.000"
    assert values["tRetentionAllBest"] == "0.100"
    assert values["tRetentionHelped"] == "1" and values["tRetentionHurt"] == "0"


def test_retention_markdown_states_the_paired_result():
    from parammem.report import build_results_markdown
    md = build_results_markdown(BUNDLE)
    assert "T8-a" in md
    assert "2 probes helped, 0 hurt, 2 tied" in md
    assert "interference, not retrieval" in md
