"""Writing memory into one slot, with the isolation invariant enforced at runtime.

Two things make this more than "call backward and step":

1. **A fresh optimizer per write.** Reusing an optimizer across writes lets a
   frozen slot drift: its gradient is zero, but Adam's stored moments are not,
   so the update is non-zero. Building the optimizer inside the call removes the
   failure mode by construction.
2. **``weight_decay=0.0``.** AdamW's decay is *decoupled* from the gradient, so it
   shrinks every parameter in the group -- including frozen slots whose gradient
   is exactly zero. The default (1e-2) would silently erode memories that were
   never being written.

The isolation check is an assertion, not a hope: the report carries
``frozen_ok`` and the function raises if a slot other than the target moved.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from .slots import RankBlockedLoRA

__all__ = ["WriteReport", "write_slot", "snapshot_slots", "verify_isolation"]


@dataclass
class WriteReport:
    """Everything the experiment log needs about one write."""

    slot: int
    steps: int
    losses: list[float] = field(default_factory=list)
    norm_before: dict[str, float] = field(default_factory=dict)
    norm_after: dict[str, float] = field(default_factory=dict)
    frozen_ok: bool = True
    seconds: float = 0.0

    @property
    def loss_start(self) -> float:
        return self.losses[0] if self.losses else float("nan")

    @property
    def loss_end(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


def snapshot_slots(
    wrappers: dict[str, RankBlockedLoRA], except_slot: int | None
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Copy the parameters of every slot except ``except_slot`` (for the check)."""
    snap: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, wrapper in wrappers.items():
        keep = []
        for slot in range(wrapper.n_slots):
            if slot == except_slot:
                continue
            sl = wrapper.slot_slice(slot)
            keep.append(wrapper.A.detach()[sl].clone())
            keep.append(wrapper.B.detach()[:, sl].clone())
        snap[name] = tuple(keep)
    return snap


def verify_isolation(
    wrappers: dict[str, RankBlockedLoRA],
    snapshot: dict[str, tuple[torch.Tensor, ...]],
    except_slot: int | None,
) -> bool:
    """True iff every non-target slot is bit-identical to the snapshot."""
    for name, wrapper in wrappers.items():
        idx = 0
        for slot in range(wrapper.n_slots):
            if slot == except_slot:
                continue
            sl = wrapper.slot_slice(slot)
            if not torch.equal(wrapper.A.detach()[sl], snapshot[name][idx]):
                return False
            if not torch.equal(wrapper.B.detach()[:, sl], snapshot[name][idx + 1]):
                return False
            idx += 2
    return True


def write_slot(
    wrappers: dict[str, RankBlockedLoRA],
    slot: int,
    loss_fn: Callable[[], torch.Tensor],
    *,
    lr: float = 5e-3,
    steps: int = 8,
    grad_clip: float | None = 1.0,
    tolerance: float = 0.0,
) -> WriteReport:
    """Run ``steps`` gradient steps restricted to ``slot`` on every wrapper.

    ``loss_fn`` must run a forward pass through the model and return a scalar
    tensor; the caller owns what is being taught. ``tolerance`` allows a caller
    to relax the bit-exactness check (default 0.0 = strict).
    """
    if not wrappers:
        raise ValueError("no LoRA wrappers to write into")
    for wrapper in wrappers.values():
        wrapper._check_slot(slot)

    snapshot = snapshot_slots(wrappers, except_slot=slot)
    norm_before = {name: w.slot_norm(slot) for name, w in wrappers.items()}

    params: list[torch.nn.Parameter] = []
    for wrapper in wrappers.values():
        params.extend([wrapper.A, wrapper.B])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    report = WriteReport(slot=slot, steps=steps, norm_before=norm_before)
    t0 = time.perf_counter()
    for wrapper in wrappers.values():
        wrapper.begin_write(slot)
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite write loss: {loss}")
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            report.losses.append(float(loss.detach()))
    finally:
        for wrapper in wrappers.values():
            wrapper.end_write()
    report.seconds = time.perf_counter() - t0

    report.norm_after = {name: w.slot_norm(slot) for name, w in wrappers.items()}
    report.frozen_ok = verify_isolation(wrappers, snapshot, except_slot=slot)
    if not report.frozen_ok and tolerance == 0.0:
        raise RuntimeError(
            "write isolation violated: a slot other than "
            f"{slot} changed (this must never happen; check weight_decay and "
            "whether the optimizer is built fresh inside this call)"
        )
    return report


def clone_wrappers(
    wrappers: dict[str, RankBlockedLoRA],
) -> dict[str, RankBlockedLoRA]:
    """Deep-copy the side path (for A/B arms without re-loading the backbone)."""
    return {name: copy.deepcopy(w) for name, w in wrappers.items()}


def reset_slots(
    wrappers: dict[str, RankBlockedLoRA], slots: Sequence[int]
) -> None:
    for slot in slots:
        for wrapper in wrappers.values():
            wrapper.erase(slot)
