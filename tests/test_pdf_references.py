"""The shipped PDFs must not contain unresolved cross-references.

An unresolved reference prints as "??" and a reader who meets one loses the thread: the
Chinese review copy was shipped with sixteen of them because a failed first pass wrote a PDF
that never got its second pass, and nothing in the build complained about the text itself.
The LaTeX log would have warned; a check on the rendered PDF is harder to ignore.

Skipped where pdftotext or the PDFs are absent, since this is about the artefacts we ship.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

PDFS = ("main.pdf", "main_zh.pdf")


def _text(path: pathlib.Path) -> str:
    return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


@pytest.mark.parametrize("name", PDFS)
def test_pdf_has_no_unresolved_references(name: str):
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext not available")
    path = ROOT / "paper" / name
    if not path.is_file():
        pytest.skip(f"{name} not built in this checkout")
    text = _text(path)
    hits = re.findall(r"\?\?", text)
    context = ""
    if hits:
        first = text.find("??")
        context = text[max(0, first - 90):first + 40].replace("\n", " ")
    assert not hits, (
        f"{name} contains {len(hits)} unresolved reference(s), e.g. ...{context}...  "
        f"A build that warns and still writes a PDF will ship them; rebuild the copy so "
        f"the second LaTeX pass runs")
