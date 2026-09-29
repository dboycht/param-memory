"""Tests for model-path resolution.

These exist because the resolver once silently returned a *half-downloaded*
HuggingFace cache directory, and the failure surfaced far away as a confusing
tokenizer error. Resolving to an unusable path must be refused, loudly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from parammem import model as M


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Point Path.home() at a temp dir so the probe logic can be exercised."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


def _make_ms_cache(home: Path, org: str, dir_name: str, *, config: bool = True) -> Path:
    d = home / ".cache" / "modelscope" / "models" / org / dir_name
    d.mkdir(parents=True)
    if config:
        (d / "config.json").write_text("{}", encoding="utf-8")
    return d


def _make_hf_cache(home: Path, model_id: str, *, complete: bool) -> Path:
    root = home / ".cache" / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--"))
    blob_dir = root / "blobs"
    blob_dir.mkdir(parents=True)
    if not complete:
        (blob_dir / "abc.incomplete").write_bytes(b"")
        return root
    snap = root / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    return root


def test_normalized_name_tolerates_modelscope_rewriting():
    assert M._normalized_name("Qwen3-1.7B") == M._normalized_name("Qwen3-1___7B")
    assert M._normalized_name("Qwen3-0.6B") != M._normalized_name("Qwen3-1.7B")


def test_existing_directory_is_used_as_is(tmp_path):
    d = tmp_path / "local-model"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert M.resolve_model_path(str(d)) == str(d)


def test_directory_without_config_warns_but_is_used(tmp_path):
    d = tmp_path / "not-a-model"
    d.mkdir()
    with pytest.warns(UserWarning):
        assert M.resolve_model_path(str(d)) == str(d)


def test_modelscope_cache_is_found_despite_the_rewritten_directory_name(fake_home):
    made = _make_ms_cache(fake_home, "Qwen", "Qwen3-1___7B")
    assert M.resolve_model_path("Qwen/Qwen3-1.7B") == str(made)


def test_modelscope_entry_without_config_is_skipped(fake_home):
    _make_ms_cache(fake_home, "Qwen", "Qwen3-1___7B", config=False)
    # nothing usable anywhere -> the id is returned unchanged for HF to fail on
    assert M.resolve_model_path("Qwen/Qwen3-1.7B") == "Qwen/Qwen3-1.7B"


def test_incomplete_huggingface_cache_is_refused_with_a_warning(fake_home):
    _make_hf_cache(fake_home, "Qwen/Qwen3-1.7B", complete=False)
    with pytest.warns(UserWarning, match="incomplete HuggingFace cache"):
        resolved = M.resolve_model_path("Qwen/Qwen3-1.7B")
    assert resolved == "Qwen/Qwen3-1.7B"


def test_complete_huggingface_cache_resolves_to_the_snapshot(fake_home):
    root = _make_hf_cache(fake_home, "Qwen/Qwen3-1.7B", complete=True)
    resolved = M.resolve_model_path("Qwen/Qwen3-1.7B")
    assert Path(resolved) == root / "snapshots" / "rev1"


def test_modelscope_wins_over_huggingface(fake_home):
    ms = _make_ms_cache(fake_home, "Qwen", "Qwen3-1___7B")
    _make_hf_cache(fake_home, "Qwen/Qwen3-1.7B", complete=True)
    assert M.resolve_model_path("Qwen/Qwen3-1.7B") == str(ms)


def test_explicit_override_wins(fake_home, tmp_path, monkeypatch):
    override = tmp_path / "override"
    override.mkdir()
    _make_ms_cache(fake_home, "Qwen", "Qwen3-1___7B")
    monkeypatch.setenv("PARAMMEM_MODEL_DIR", str(override))
    assert M.resolve_model_path("Qwen/Qwen3-1.7B") == str(override)


def test_unknown_id_is_returned_unchanged(fake_home):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no spurious warnings when nothing exists
        assert M.resolve_model_path("Nobody/NoSuchModel") == "Nobody/NoSuchModel"
