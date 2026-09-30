"""Guards for the T7-g baseline loader.

The baselines exist to answer a reviewer's cheapest question about the *same* thirty
memories the diagnostic reports, so the loader is pinned to the questions T7-d actually
ran. The first version of this script called ``select_single_session`` with one
argument (its ``subset`` parameter is required), which would have failed only after a
model had loaded.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"
T7D = ROOT / "runs" / "t7d_longmemeval.json"


def _module():
    spec = importlib.util.spec_from_file_location(
        "baselines_under_test", ROOT / "experiments" / "t7g_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not DATA.is_file(), reason="benchmark data not present")
def test_loader_returns_thirty_items_with_questions_and_answers():
    pairs = _module().load_pairs(30)
    assert len(pairs) == 30
    assert all(str(pair.get("question", "")).strip() for pair in pairs)
    assert all(str(pair.get("answer", "")).strip() for pair in pairs)


@pytest.mark.skipif(not (DATA.is_file() and T7D.is_file()),
                    reason="needs the benchmark data and the T7-d report")
def test_loader_selects_exactly_the_questions_the_diagnostic_ran():
    """If this drifts, the baselines quietly stop being comparable to Table 7."""
    pairs = _module().load_pairs(30)
    report = json.loads(T7D.read_text(encoding="utf-8"))
    ran = [row["question_id"] for row in report.get("rows", [])]
    assert ran, "the T7-d report has no rows to compare against"
    assert [pair["question_id"] for pair in pairs] == ran


@pytest.mark.skipif(not DATA.is_file(), reason="benchmark data not present")
def test_loader_is_deterministic():
    module = _module()
    first = [p["question_id"] for p in module.load_pairs(10)]
    second = [p["question_id"] for p in module.load_pairs(10)]
    assert first == second
