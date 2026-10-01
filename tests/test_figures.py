"""Guards for the generated figures.

A figure whose numbers disagree with the table beside it is worse than no figure, and the
only way to keep them in step is to generate both from the same bundle and to fail the build
when the committed figure drifts. These tests also check that the macros the figure's
caption uses are actually defined, since a figure is easy to add and easy to forget.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))

FIGURE = ROOT / "paper" / "generated_figure_collapse.tex"
TABLES = ROOT / "paper" / "generated_tables.tex"
BUNDLE = ROOT / "runs" / "summary.json"


def test_figure_matches_a_fresh_generation():
    if not (FIGURE.is_file() and BUNDLE.is_file()):
        pytest.skip("no generated figure or bundle in this checkout")
    from build_figures import build
    committed = FIGURE.read_text(encoding="utf-8")
    fresh = build(json.loads(BUNDLE.read_text(encoding="utf-8")))
    assert committed == fresh, (
        "paper/generated_figure_collapse.tex is stale; re-run paper/build_figures.py")


def test_figure_caption_macros_are_defined():
    if not (FIGURE.is_file() and TABLES.is_file()):
        pytest.skip("no generated figure or tables in this checkout")
    figure = FIGURE.read_text(encoding="utf-8")
    tables = TABLES.read_text(encoding="utf-8")
    used = set(re.findall(r"\\(t[A-Za-z]+)", figure))
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tables))
    missing = sorted(used - defined)
    assert not missing, f"the figure uses undefined macros: {missing}"


def test_figure_is_referenced_by_the_paper():
    paper = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    if "generated_figure_collapse" not in paper:
        pytest.skip("the figure is not used by this version of the paper")
    figure = FIGURE.read_text(encoding="utf-8")
    assert "\\label{fig:collapse}" in figure
    assert "fig:collapse" in paper, "the figure is labelled; the text should reference it"
