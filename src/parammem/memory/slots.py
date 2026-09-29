"""Rank-blocked LoRA: one frozen backbone, K exactly-erasable memory slots.

Why partition the rank dimension instead of keeping K separate adapters
(cf. ProCL, arXiv:2605.13162) or one dense adapter:

* **exact erasure.** Zeroing rank block ``k`` provably removes memory ``k`` with
  no residue. A dense adapter can only *decay* an item, so "forgetting" is never
  verifiable. Here ``erase(k)`` restores the block to its virgin state
  bit-for-bit, which is an assertable property (see tests).
* **enumeration.** Capacity is the integer ``K``, so the capacity-retention-
  interference curve of T2 has a well-defined x-axis.
* **attribution.** Protocol P4 needs MEM_ON / MEM_OFF / MEM_SHUFFLE on a
  *byte-identical* model; the read mask provides that without touching weights.

Write isolation: a single ``A``/``B`` pair holds all slots and the forward path is
one matmul pair. Gradients are restricted to the target block by a tensor hook,
and every write builds a **fresh** optimizer, so a frozen slot sees
``grad == 0`` and a zero Adam moment -- no drift. The optimizer is explicitly
built with ``weight_decay=0.0``: AdamW's decoupled decay would otherwise shrink
frozen slots even though their gradient is zero.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RankBlockedLoRA", "attach_slot_lora", "count_memory_parameters"]


class RankBlockedLoRA(nn.Module):
    """Wraps a frozen ``nn.Linear`` with ``n_slots`` rank blocks of size ``rank``.

    Effective weight: ``W = W0 + (alpha / rank) * B @ A``, where the rank
    dimension of ``A`` (rows) and ``B`` (columns) is partitioned into
    ``n_slots`` contiguous blocks of ``rank`` rows/columns each.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        n_slots: int,
        rank: int = 4,
        alpha: float = 16.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if n_slots < 1 or rank < 1:
            raise ValueError("n_slots and rank must both be >= 1")
        self.base = base
        self.n_slots = int(n_slots)
        self.rank = int(rank)
        self.total_rank = self.n_slots * self.rank
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank

        for p in self.base.parameters():
            p.requires_grad_(False)

        device = base.weight.device
        dtype = base.weight.dtype
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        a = torch.randn(self.total_rank, base.in_features, generator=gen, dtype=torch.float32)
        a = (a * (1.0 / math.sqrt(base.in_features))).to(device=device, dtype=dtype)

        self.A = nn.Parameter(a.clone(), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(base.out_features, self.total_rank,
                                         device=device, dtype=dtype), requires_grad=True)

        # Virgin copy of A: erase(slot) restores exactly this, so an erased slot
        # is bit-identical to a never-written one.
        self.register_buffer("A_init", a.clone(), persistent=True)

        self.register_buffer("_grad_mask_A", torch.ones(self.total_rank, 1, device=device, dtype=dtype))
        self.register_buffer("_grad_mask_B", torch.ones(1, self.total_rank, device=device, dtype=dtype))
        self.register_buffer("_read_mask_A", torch.ones(self.total_rank, 1, device=device, dtype=dtype))
        self.register_buffer("_read_mask_B", torch.ones(1, self.total_rank, device=device, dtype=dtype))

        # Restrict gradients to the block being written.
        self.A.register_hook(lambda g: g * self._grad_mask_A)
        self.B.register_hook(lambda g: g * self._grad_mask_B)

    # ------------------------------------------------------------------ slots
    def slot_slice(self, slot: int) -> slice:
        self._check_slot(slot)
        return slice(slot * self.rank, (slot + 1) * self.rank)

    def _check_slot(self, slot: int) -> None:
        if not (0 <= slot < self.n_slots):
            raise IndexError(f"slot {slot} out of range [0, {self.n_slots})")

    def slot_parameters(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        sl = self.slot_slice(slot)
        return self.A.data[sl], self.B.data[:, sl]

    def is_virgin(self, slot: int) -> bool:
        """True when the slot is exactly in its never-written state."""
        sl = self.slot_slice(slot)
        return bool(
            torch.equal(self.A.data[sl], self.A_init.data[sl])
            and torch.count_nonzero(self.B.data[:, sl]) == 0
        )

    def erase(self, slot: int) -> None:
        """Exact forgetting: restore the slot to its virgin state."""
        sl = self.slot_slice(slot)
        with torch.no_grad():
            self.A.data[sl] = self.A_init.data[sl]
            self.B.data[:, sl] = 0

    def erase_all(self) -> None:
        for slot in range(self.n_slots):
            self.erase(slot)

    def slot_norm(self, slot: int) -> float:
        """Frobenius norm of the slot's contribution (monitored per literature:
        continuously edited matrices grow in norm, arXiv:2502.01636)."""
        a, b = self.slot_parameters(slot)
        return float(torch.linalg.matrix_norm(b @ a).item()) * abs(self.scale)

    # ------------------------------------------------------------ read masks
    def _new_mask(self, slots: Iterable[int] | None) -> torch.Tensor:
        """A fresh (total_rank, 1) enable-mask for ``slots`` (None = all).

        A **new** tensor is returned rather than an in-place update of the
        existing buffer. That matters because one write can run several forward
        passes with *different* masks inside a single autograd graph (a target
        term that sees only its own slot, plus an anchor term that sees every
        written slot); mutating a buffer the graph already captured raises
        "one of the variables needed for gradient computation has been modified
        by an inplace operation".
        """
        mask = torch.zeros(
            self.total_rank, 1, device=self.A.device, dtype=self.A.dtype
        )
        active = range(self.n_slots) if slots is None else slots
        for slot in active:
            mask[self.slot_slice(slot)] = 1.0
        return mask

    def set_read_slots(self, slots: Iterable[int] | None) -> None:
        """Ablation switch: only ``slots`` contribute to the forward pass.

        ``None`` means all slots (normal operation). This is what implements
        MEM_ON / MEM_OFF / MEM_SHUFFLE for protocol P4 without touching any
        weight, so the compared passes are byte-identical apart from the mask.
        """
        self.set_read_mask(self._new_mask(slots))

    def read_mask(self) -> torch.Tensor:
        """Current read mask, shape ``(total_rank, 1)``.

        Public on purpose: a caller that must run a forward with the memory off
        (e.g. to capture a retrieval key) has to save and restore it.
        """
        return self._read_mask_A

    def set_read_mask(self, mask: torch.Tensor) -> None:
        self._read_mask_A = mask
        self._read_mask_B = mask.reshape(1, -1)

    @property
    def any_slot_active(self) -> bool:
        return bool(self._read_mask_A.any())

    # ----------------------------------------------------------------- write
    def begin_write(self, slot: int) -> None:
        """Restrict gradients to ``slot`` for the next backward pass."""
        self._check_slot(slot)
        mask = self._new_mask([slot]).reshape(1, -1)
        # Rebinding, not copy_: the previous mask may already be captured by an
        # autograd graph (see _new_mask).
        self._grad_mask_A = mask.reshape(-1, 1)
        self._grad_mask_B = mask

    def end_write(self) -> None:
        full = self._new_mask(None)
        self._grad_mask_A = full
        self._grad_mask_B = full.reshape(1, -1)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.any_slot_active:
            return out
        a = self.A * self._read_mask_A
        b = self.B * self._read_mask_B
        return out + self.scale * F.linear(F.linear(x, a), b)

    # ------------------------------------------------------------- diagnostics
    def extra_repr(self) -> str:
        return (
            f"n_slots={self.n_slots}, rank={self.rank}, alpha={self.alpha}, "
            f"in={self.base.in_features}, out={self.base.out_features}"
        )


def attach_slot_lora(
    model: nn.Module,
    target_suffixes: Sequence[str],
    *,
    n_slots: int,
    rank: int = 4,
    alpha: float = 16.0,
    seed: int = 0,
) -> dict[str, RankBlockedLoRA]:
    """Replace every ``nn.Linear`` whose qualified name ends with one of
    ``target_suffixes`` by a :class:`RankBlockedLoRA` wrapper.

    Returns the wrappers keyed by their original qualified name.
    """
    wrappers: dict[str, RankBlockedLoRA] = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper = RankBlockedLoRA(
            module, n_slots=n_slots, rank=rank, alpha=alpha, seed=seed
        )
        setattr(parent, child_name, wrapper)
        wrappers[name] = wrapper
    return wrappers


def count_memory_parameters(wrappers: dict[str, RankBlockedLoRA]) -> int:
    """Trainable side-path parameters (the whole memory budget)."""
    return sum(w.A.numel() + w.B.numel() for w in wrappers.values())
