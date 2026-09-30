"""The table builder must refuse a bundle that silently omits stages.

Running one stage with ``--only`` re-collects the bundle from that stage alone. That is a
reasonable thing to want and a terrible thing to render: whole sections of the paper turned
into "n/a", and the run looked successful. It was noticed only because a test that needs
the judge report stopped running and reported itself skipped rather than failed.

These tests pin the refusal and the flag that overrides it deliberately.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILDER = ROOT / "paper" / "build_tables.py"


def _run(bundle: pathlib.Path, out: pathlib.Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(BUILDER), "--bundle", str(bundle), "--out", str(out), *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
    )


def test_partial_bundle_is_refused(tmp_path):
    bundle = tmp_path / "partial.json"
    bundle.write_text(json.dumps({"t6": {"write": {}, "read": {}},
                                  "_partial": ["t2", "t7"]}), encoding="utf-8")
    result = _run(bundle, tmp_path / "out.tex")
    assert result.returncode != 0
    assert "refusing to build" in result.stdout
    assert not (tmp_path / "out.tex").exists(), "nothing should be written when refusing"


def test_allow_partial_renders(tmp_path):
    bundle = tmp_path / "partial.json"
    bundle.write_text(json.dumps({"t6": {"write": {}, "read": {}},
                                  "_partial": ["t2", "t7"]}), encoding="utf-8")
    result = _run(bundle, tmp_path / "out.tex", "--allow-partial")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "out.tex").is_file()


def test_real_bundle_is_not_partial():
    bundle = ROOT / "runs" / "summary.json"
    if not bundle.is_file():
        pytest.skip("no collected bundle in this checkout")
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    assert payload.get("_partial") == [], (
        "the bundle in use is partial; re-run experiments/run_all.py --collect-only")
