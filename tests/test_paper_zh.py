"""Guards for the Chinese review copy generator.

Two real failure modes were hit while building it, and both reached a compiled PDF
before being noticed, so both are pinned here:

* a macro value containing ``%`` put a LaTeX comment character into the document and
  silently deleted the rest of that line;
* the prose was written with markdown ``**bold**``, which LaTeX prints literally.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "build_paper_zh_under_test", ROOT / "paper" / "build_paper_zh.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_percent_is_escaped_before_it_reaches_latex():
    module = _module()
    assert module.tex_escape("72%") == r"72\%"
    # a bare % would comment out the remainder of the line
    out = module.render("写入少 [[tW]]、漂移 nats", {"tW": "72%"})
    assert out == r"写入少 72\%、漂移 nats"


def test_unknown_macro_is_a_hard_error():
    module = _module()
    with pytest.raises(SystemExit, match="tNope"):
        module.render("值 [[tNope]]", {"tOther": "1"})


def test_markdown_bold_becomes_latex_bold_including_across_lines():
    module = _module()
    assert module.render("**粗**", {}) == r"\textbf{粗}"
    assert module.render("**跨\n行**", {}) == "\\textbf{跨\n行}"


def test_unmatched_bold_marker_is_refused():
    module = _module()
    with pytest.raises(SystemExit, match="markdown bold"):
        module.render("这里有 **半个粗体", {})


def test_the_committed_chinese_source_has_no_leftovers():
    """The generated file is committed, so it cannot carry placeholders or asterisks."""
    text = (ROOT / "paper" / "main_zh.tex").read_text(encoding="utf-8")
    assert "[[" not in text, "an unresolved [[macro]] placeholder reached the document"
    assert "**" not in text, "markdown bold reached the document"
    assert r"\documentclass" in text and "ctexart" in text


def test_cjk_font_is_not_overridden():
    """Noto SC on this machine is a variable font and xdvipdfmx refuses it
    ("Invalid font: -1"), which produces no PDF at all -- so the generator must let
    ctex pick the Windows fontset instead of pinning a font by name."""
    text = (ROOT / "paper" / "main_zh.tex").read_text(encoding="utf-8")
    assert "setCJKmainfont" not in text
    assert "Noto" not in text
