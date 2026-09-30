"""Guards for the multi-variant retrieval key.

The pooling variants used to be reimplemented inside each experiment, and the
memory-off convention that ``query_key`` documents was forgotten four separate times,
each time silently deflating the model-based columns of a comparison. The variants now
live on the backbone, and this test pins the property that was being lost: the ``last``
variant must equal ``query_key`` exactly, whatever read mask was active beforehand.

The test needs a cached model, so it skips rather than fails where one is absent.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from parammem.model import Backbone, BackboneConfig, resolve_model_path

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _backbone() -> Backbone | None:
    try:
        path = resolve_model_path("Qwen/Qwen3-0.6B")
    except Exception:
        return None
    return Backbone.load(BackboneConfig(model_id=path, n_slots=2, max_new_tokens=4))


def test_query_keys_match_query_key_even_with_slots_active():
    try:
        bb = _backbone()
    except Exception as error:                      # no cached model in this checkout
        pytest.skip(f"model unavailable: {type(error).__name__}")
    if bb is None:
        pytest.skip("model unavailable")

    query = "What is the capital of France?"
    # Leave a read mask active on purpose: this is the state the experiments were in
    # when they captured their keys, and the one that made them wrong.
    bb.set_read_slots([0])
    keys = bb.query_keys(query)
    reference = bb.query_key(query)

    assert set(keys) == {"last", "mean", "mid", "shallow"}
    assert torch.allclose(keys["last"].float().cpu(), reference.float().cpu(), atol=1e-5)
    for name, key in keys.items():
        assert abs(float(key.norm()) - 1.0) < 1e-4, name
    assert not torch.allclose(keys["mean"].float().cpu(), reference.float().cpu())
    # the mask is restored afterwards, so a caller's read state is not disturbed
    assert bb.wrappers[next(iter(bb.wrappers))].read_mask() is not None
