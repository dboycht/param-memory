"""Guards for the generated LaTeX tables.

A whole table once reached a compiled PDF reading ``\\{}texttt{frozen} containment``
and ``\\{}midrule``: the row labels carried LaTeX, and :func:`parammem.report._table`
escapes every cell on purpose so that callers never hand-build LaTeX. Every check we
had looked at the *numbers* in the PDF, so nobody noticed the labels were mangled.

These tests look at the labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parammem.report import _table, _tex, MIDRULE, build_results_latex

ROOT = Path(__file__).resolve().parents[1]


def test_escaped_latex_artefacts_are_visible_to_the_guard():
    """Sanity: the guard's patterns are the ones _tex actually produces.

    Escaping is single-pass on purpose. A sequential replace chain would escape the
    braces that its own backslash replacement introduced, which is how ``\\quad``
    became ``\\textbackslash\\{\\}quad`` in the PDF.
    """
    assert _tex("\\texttt{x}") == r"\textbackslash{}texttt\{x\}"
    assert _tex("mem_on") == r"mem\_on"
    assert _tex("10% & $x$") == r"10\% \& \$x\$"
    assert "\\textbackslash" in _tex("\\lambda")
    assert "\\{" not in _tex("\\lambda")


def test_table_escapes_cells_but_keeps_captions_as_latex():
    out = _table("caption with \\emph{latex}", "tab:x", ["a_b"], [["c_d", "1"]], "ll")
    assert "\\emph{latex}" in out          # captions are authored LaTeX
    assert r"a\_b" in out and r"c\_d" in out   # cells are plain text, escaped
    assert "\\textbackslash" not in out


def test_midrule_sentinel_becomes_a_rule_not_escaped_text():
    out = _table("c", "tab:x", ["a", "b"], [["one", "1"], [MIDRULE], ["two", "2"]], "ll")
    assert "\n\\midrule\n" in out
    assert "textbackslash" not in out


def test_generated_tables_carry_no_escaped_latex():
    """The regression that started this: labels must never embed LaTeX commands.

    Runs against the committed artefact so it fails even if only the builder was
    changed and the file was not regenerated.
    """
    tex = (ROOT / "paper" / "generated_tables.tex").read_text(encoding="utf-8")
    for bad in ("\\textbackslash", "\\{", "\\}"):
        assert bad not in tex, (
            f"{bad!r} in generated_tables.tex means a cell contained LaTeX and was "
            "escaped into literal text; pass plain text and use mono_columns")


def test_t7_table_reports_zero_extra_tokens_for_the_weights_arm():
    """The weights arm reads through the forward pass, so it pays no prompt tokens.
    Asking the baseline report for that arm returned n/a, which understated it."""
    bundle = json.loads((ROOT / "runs" / "summary.json").read_text(encoding="utf-8"))
    if not (bundle.get("t7") or {}).get("judge"):
        pytest.skip("no judge report in this bundle")
    tex = build_results_latex(bundle)
    assert "extra prompt tokens, weights & n/a" not in tex
    assert "extra prompt tokens, weights & 0" in tex
