"""A macro that renders as n/a must not be quoted in the paper.

The oracle arm of the multi-session experiment was removed when the run was corrected, which
took its macros with it. The macros still existed, so nothing failed and the LaTeX build was
clean: the sentence simply printed "n/a" where a number belonged, and it was visible only by
rendering the page. A definition check cannot catch that, because a macro whose source
disappeared still resolves; the value is what has to be inspected.

This resolves every macro the papers use and fails if any of them is empty or n/a.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SUMMARY = ROOT / "runs" / "summary.json"
SOURCES = (ROOT / "paper" / "main.tex", ROOT / "paper" / "build_paper_zh.py")
# Macros the paper legitimately leaves unset, with the reason. Keep this list empty unless
# a value genuinely cannot exist in the artifact being built.
ALLOWED_EMPTY: set[str] = set()


def _used_macros(text: str) -> set[str]:
    used = {m.lstrip("\\") for m in re.findall(r"\\t[A-Z][A-Za-z]*", text)}
    used |= set(re.findall(r"\[\[(t[A-Z][A-Za-z]*)\]\]", text))
    return used


def test_no_macro_the_paper_quotes_renders_as_na():
    if not SUMMARY.is_file():
        pytest.skip("no collected bundle in this checkout")
    from parammem.report import headline_values

    values = headline_values(json.loads(SUMMARY.read_text(encoding="utf-8")))
    empty = {key for key, value in values.items()
             if str(value).strip().lower() in ("n/a", "na", "")}
    problems = []
    for source in SOURCES:
        if not source.is_file():
            continue
        for macro in sorted(_used_macros(source.read_text(encoding="utf-8"))):
            if macro in empty and macro not in ALLOWED_EMPTY:
                problems.append(f"{source.name} quotes {macro}, which renders as n/a")
    assert not problems, (
        "a quoted number would print as n/a:\n  " + "\n  ".join(problems) +
        "\nEither the run that produced it is no longer collected, or the sentence refers to "
        "an arm that was removed. Rendering the page is what catches this otherwise.")
