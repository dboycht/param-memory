"""Tests for rank-blocked LoRA slots and the runtime write-isolation invariant.

These lock down the three properties the whole method rests on:

* a virgin slot changes nothing (so "no memory" is exactly the frozen model);
* a write touches **only** its own slot, bit-for-bit;
* erasing a slot restores the virgin state bit-for-bit (exact forgetting).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from parammem.memory.slots import (
    RankBlockedLoRA,
    attach_slot_lora,
    count_memory_parameters,
)
from parammem.memory.writer import write_slot


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))


def _attach(model, n_slots=4, rank=2):
    return attach_slot_lora(model, ["0"], n_slots=n_slots, rank=rank, alpha=8.0)


def _loss_fn(wrapper: RankBlockedLoRA, x: torch.Tensor, y: torch.Tensor):
    return lambda: torch.nn.functional.mse_loss(wrapper(x), y)


def _data(wrapper: RankBlockedLoRA, n: int = 16, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, wrapper.base.in_features, generator=g)
    y = torch.randn(n, wrapper.base.out_features, generator=g)
    return x, y


def _other_indices(wrapper: RankBlockedLoRA, slot: int) -> list[int]:
    sl = wrapper.slot_slice(slot)
    return [i for i in range(wrapper.total_rank) if not (sl.start <= i < sl.stop)]


# ------------------------------------------------------------------ structure

def test_attach_replaces_only_matching_linear_layers():
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
    wrappers = attach_slot_lora(model, ["2"], n_slots=3, rank=2, alpha=8.0)
    assert list(wrappers) == ["2"]
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[2], RankBlockedLoRA)


def test_backbone_parameters_are_frozen():
    model = _tiny_model()
    wrappers = _attach(model)
    for wrapper in wrappers.values():
        assert all(not p.requires_grad for p in wrapper.base.parameters())
        assert wrapper.A.requires_grad and wrapper.B.requires_grad


def test_memory_parameter_count_matches_slots():
    model = _tiny_model()
    wrappers = _attach(model, n_slots=4, rank=2)
    wrapper = wrappers["0"]
    expected = 4 * 2 * (wrapper.base.in_features + wrapper.base.out_features)
    assert count_memory_parameters(wrappers) == expected


def test_invalid_construction_and_slot_access():
    model = _tiny_model()
    with pytest.raises(ValueError):
        attach_slot_lora(model, ["0"], n_slots=0)
    wrappers = _attach(model)
    with pytest.raises(IndexError):
        wrappers["0"].slot_slice(4)
    with pytest.raises(IndexError):
        wrappers["0"].erase(-1)


# ------------------------------------------------------- virgin / read masks

def test_virgin_forward_is_bit_identical_to_the_frozen_model():
    model = _tiny_model()
    x = torch.randn(5, 8)
    base = model(x).detach().clone()
    _attach(model)
    assert torch.equal(model(x).detach(), base)


def test_all_slots_virgin_at_start():
    wrapper = _attach(_tiny_model())["0"]
    assert all(wrapper.is_virgin(slot) for slot in range(wrapper.n_slots))


def test_read_mask_off_restores_the_frozen_model_exactly():
    model = _tiny_model()
    x = torch.randn(5, 8)
    base = model(x).detach().clone()
    wrapper = _attach(model)["0"]
    write_slot({"0": wrapper}, 0, _loss_fn(wrapper, *_data(wrapper)))

    wrapper.set_read_slots([])
    assert torch.equal(model(x).detach(), base)

    wrapper.set_read_slots([0])
    assert not torch.equal(model(x).detach(), base)

    # A *different* (still virgin) slot contributes nothing: this is the
    # MEM_SHUFFLE arm of protocol P4.
    wrapper.set_read_slots([1])
    assert torch.equal(model(x).detach(), base)

    wrapper.set_read_slots(None)
    assert not torch.equal(model(x).detach(), base)


# --------------------------------------------------------------- write path

def test_write_changes_only_its_own_slot():
    model = _tiny_model()
    wrapper = _attach(model)["0"]
    x, y = _data(wrapper)
    a0, b0 = wrapper.A.detach().clone(), wrapper.B.detach().clone()

    report = write_slot({"0": wrapper}, 2, _loss_fn(wrapper, x, y), steps=6, lr=0.05)
    assert report.frozen_ok

    others = _other_indices(wrapper, 2)
    assert torch.equal(wrapper.A.detach()[others], a0[others])
    assert torch.equal(wrapper.B.detach()[:, others], b0[:, others])

    sl = wrapper.slot_slice(2)
    assert not torch.equal(wrapper.B.detach()[:, sl], b0[:, sl])
    assert not wrapper.is_virgin(2)


def test_write_reduces_the_loss():
    model = _tiny_model()
    wrapper = _attach(model)["0"]
    x, y = _data(wrapper)
    report = write_slot({"0": wrapper}, 1, _loss_fn(wrapper, x, y), steps=40, lr=0.05)
    assert report.loss_end < report.loss_start
    assert report.norm_after["0"] > 0.0
    assert report.seconds >= 0.0


def test_adamw_weight_decay_would_drift_frozen_slots():
    """Documents the trap this module exists to avoid: with AdamW's default
    decoupled decay, slots whose gradient is exactly zero still shrink."""
    model = _tiny_model()
    wrapper = _attach(model)["0"]
    x, y = _data(wrapper)
    a0 = wrapper.A.detach().clone()

    wrapper.begin_write(2)
    optimizer = torch.optim.AdamW([wrapper.A, wrapper.B], lr=0.05)  # wd=1e-2
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(wrapper(x), y).backward()
        optimizer.step()
    wrapper.end_write()

    others = _other_indices(wrapper, 2)
    assert not torch.equal(wrapper.A.detach()[others], a0[others]), (
        "expected the default weight decay to move frozen slots"
    )


# ------------------------------------------------------------------ erasure

def test_erase_restores_a_bit_exact_virgin_state():
    model = _tiny_model()
    x = torch.randn(4, 8)
    virgin = model(x).detach().clone()
    wrapper = _attach(model)["0"]

    write_slot({"0": wrapper}, 1, _loss_fn(wrapper, *_data(wrapper)), steps=6, lr=0.05)
    assert not torch.equal(model(x).detach(), virgin)
    assert not wrapper.is_virgin(1)

    wrapper.erase(1)
    assert wrapper.is_virgin(1)
    assert torch.equal(model(x).detach(), virgin)
    assert wrapper.slot_norm(1) == 0.0


def test_erase_all_returns_to_virgin_and_keeps_other_slots():
    model = _tiny_model()
    x = torch.randn(4, 8)
    virgin = model(x).detach().clone()
    wrapper = _attach(model)["0"]
    for slot in (0, 2, 3):
        write_slot({"0": wrapper}, slot, _loss_fn(wrapper, *_data(wrapper, seed=slot)),
                   steps=4, lr=0.05)
    assert not torch.equal(model(x).detach(), virgin)

    wrapper.erase(2)
    assert wrapper.is_virgin(2) and not wrapper.is_virgin(0)

    wrapper.erase_all()
    assert all(wrapper.is_virgin(s) for s in range(wrapper.n_slots))
    assert torch.equal(model(x).detach(), virgin)


def test_writing_twice_to_the_same_slot_overwrites_rather_than_accumulates():
    model = _tiny_model()
    wrapper = _attach(model)["0"]
    x1, y1 = _data(wrapper, seed=11)
    x2, y2 = _data(wrapper, seed=22)
    write_slot({"0": wrapper}, 0, _loss_fn(wrapper, x1, y1), steps=10, lr=0.05)
    after_first = wrapper.slot_norm(0)
    write_slot({"0": wrapper}, 0, _loss_fn(wrapper, x2, y2), steps=10, lr=0.05)
    after_second = wrapper.slot_norm(0)
    assert after_second != after_first
    assert wrapper.is_virgin(1)  # untouched neighbour
