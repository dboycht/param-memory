"""Style guards for the two papers.

Both are *conventions borrowed from the venues*, not personal taste, so they are worth
pinning: NeurIPS submits the abstract as plain text with a character cap and expects a
single paragraph, and ICML's template says an abstract "must be a single paragraph,
ideally between 4-6 sentences long", with gross violations corrected at camera-ready.
The first version of our abstract was 267 words in four paragraphs and eleven
sentences, which broke all three of those at once.

The machine-writing markers are guarded at a density threshold rather than banned
outright: published guidance is that one em dash is fine and forty is a tell.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "paper" / "main.tex"
ZH = ROOT / "paper" / "build_paper_zh.py"

MAX_WORDS = 250
MAX_SENTENCES = 8


def _english_abstract() -> str:
    source = EN.read_text(encoding="utf-8")
    block = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S)
    assert block, "no abstract found in main.tex"
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", block.group(1))
    text = text.replace("{", " ").replace("}", " ").replace("$", " ")
    return re.sub(r"\s+", " ", text).strip()


def test_english_abstract_is_one_paragraph():
    source = EN.read_text(encoding="utf-8")
    block = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S)
    body = block.group(1).strip()
    assert "\n\n" not in body, "the venue templates require a single-paragraph abstract"


def test_english_abstract_fits_the_venue_norms():
    text = _english_abstract()
    words = len(text.split())
    sentences = len([s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()])
    assert words <= MAX_WORDS, f"abstract is {words} words (limit {MAX_WORDS})"
    assert sentences <= MAX_SENTENCES, (
        f"abstract is {sentences} sentences; ICML asks for 4-6")


def test_english_abstract_states_numbers_and_the_negative_results():
    """A results abstract without numbers is the usual failure mode, and dropping the
    failed hypotheses would overstate what the paper shows.

    The numbers arrive as generated macros, so the check counts macro references rather
    than literal digits; stripping LaTeX removes the values themselves.
    """
    source = EN.read_text(encoding="utf-8")
    body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S).group(1)
    macros = re.findall(r"\\t[A-Z]\w+", body)
    assert len(macros) >= 8, f"abstract quotes only {len(macros)} generated values"
    assert "fail" in _english_abstract().lower(), "the negative results are missing"


def test_chinese_abstract_matches_the_english_length():
    source = ZH.read_text(encoding="utf-8")
    block = re.search(r"\\section\{摘要\}(.*?)\\section\{引言\}", source, re.S)
    assert block, "no Chinese abstract found"
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", "", block.group(1))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
    assert cjk <= 400, f"Chinese abstract is {cjk} characters (guidance: 250-400)"
    assert "\n\n" not in block.group(1).strip(), "Chinese abstract should be one paragraph"


def test_machine_writing_markers_stay_below_the_density_threshold():
    english = EN.read_text(encoding="utf-8")
    chinese = ZH.read_text(encoding="utf-8")
    assert english.count("---") <= 10, (
        f"{english.count('---')} em dashes; guidance is roughly one per 1000 words")
    assert not re.search(r"\bnot\b[^.;]{0,80}?\bbut\b", english), (
        'the "not X, but Y" antithesis is the most reliable machine-writing tell')
    assert not re.search("不是[^。；]{0,40}?而是", chinese)
    assert chinese.count("不是") <= 5, f"{chinese.count('不是')} negations in the template"


def test_the_paper_has_the_sections_the_conventions_expect():
    """Abstract -> Introduction (with contributions) -> Related work -> Method ->
    Evaluation -> Results -> Limitations -> Conclusion, related work as section 2."""
    text = EN.read_text(encoding="utf-8")
    order = [m.group(1) for m in re.finditer(r"\\section\{([^}]*)\}", text)]
    assert order[0] == "Introduction"
    assert order[1].lower().startswith("related work"), (
        f"related work should be section 2, found {order[1]!r}")
    for required in ("Method", "Results", "Limitations", "Conclusion"):
        assert required in order, f"missing section {required!r}"


def test_the_abstract_cites_nothing():
    """Venue guidance is explicit that abstracts carry no citations."""
    body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                     EN.read_text(encoding="utf-8"), re.S).group(1)
    assert "\\cite" not in body


def test_the_contribution_list_is_short_enough_to_read():
    """Reviewers are told to summarise the contributions; guidance is 3-5 one-sentence
    bullets, each tied to evidence. Ours was eight."""
    text = EN.read_text(encoding="utf-8")
    block = re.search(r"\\paragraph\{Contributions\.\}(.*?)\\end\{enumerate\}",
                      text, re.S)
    assert block, "no contributions list found"
    items = re.findall(r"\\item(?![a-zA-Z])", block.group(1))
    assert 3 <= len(items) <= 5, f"{len(items)} contribution bullets (guidance: 3-5)"
    refs = re.findall(r"\\ref\{[^}]*\}", block.group(1))
    assert len(refs) >= len(items) - 1, (
        "most contribution bullets should point at the table or section that shows them")


def test_no_overclaiming_vocabulary():
    text = EN.read_text(encoding="utf-8")
    for word in ("obviously", "clearly", "we believe", "trivially", "proves that"):
        assert not re.search(rf"\b{re.escape(word)}\b", text, re.I), word
