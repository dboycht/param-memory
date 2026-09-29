"""The paper must not contain dangling or unreferenced generated artifacts.

A table or equation that is generated but never referenced is a defect (a float the
reader is never told about), and a reference to something that does not exist is a
broken document. Both survive compilation quietly, so they are checked here rather
than by eye. Citations are checked in both directions too: an uncited bibitem is
dead weight, and a citation without a bibitem renders as a question mark.
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"


def _read(name: str) -> str:
    return (PAPER / name).read_text(encoding="utf-8")


def test_every_generated_table_and_equation_is_referenced():
    main_tex = _read("main.tex")
    generated = _read("generated_tables.tex")
    # Tables are emitted into generated_tables.tex; equations are written in the
    # paper itself. Looking for both in the generated file finds no equations and
    # makes this test fail for the wrong reason.
    sources = {"tab": generated, "eq": main_tex}
    for kind, definitions in sources.items():
        defined = set(re.findall(r"\\label\{(" + kind + r":[^}]*)\}", definitions))
        referenced = set(re.findall(r"\\ref\{(" + kind + r":[^}]*)\}", main_tex))
        assert defined, f"no {kind} labels found in the expected source"
        assert not (defined - referenced), (
            f"{kind} defined but never referenced: {sorted(defined - referenced)}"
        )
        assert not (referenced - defined), (
            f"{kind} referenced but not defined: {sorted(referenced - defined)}"
        )


def test_section_references_resolve():
    main_tex = _read("main.tex")
    defined = set(re.findall(r"\\label\{(sec:[^}]*)\}", main_tex))
    referenced = set(re.findall(r"\\ref\{(sec:[^}]*)\}", main_tex))
    assert not (referenced - defined), (
        f"sections referenced but not defined: {sorted(referenced - defined)}"
    )


def test_every_bibitem_is_cited_and_every_citation_has_a_bibitem():
    main_tex = _read("main.tex")
    cited = set()
    for group in re.findall(r"\\cite\{([^}]*)\}", main_tex):
        cited.update(name.strip() for name in group.split(","))
    defined = set(re.findall(r"\\bibitem\{([^}]*)\}", main_tex))
    assert not (cited - defined), f"cited but not defined: {sorted(cited - defined)}"
    assert not (defined - cited), f"defined but never cited: {sorted(defined - cited)}"
