"""Durable memory: what it takes for a thought stamp to survive a restart.

A memory that dies with the process is not a stamp, it is a session cache. This
module writes the side path's slot tensors plus the store's bookkeeping to a
directory and reads them back.

Two rules, both learned from the failure modes this project keeps hitting:

1. **Validate, do not hope.** A snapshot records ``n_slots`` / ``rank`` / ``alpha``
   / per-module shapes, and loading refuses on any mismatch. A side path built
   with the wrong slot count would otherwise load *something* and silently answer
   from a misaligned memory.
2. **Bit-exact round trip.** The tensors come back identical, so an erased slot is
   still provably virgin after a restart (see ``Backbone.is_slot_virgin``).

Layout::

    <path>/slots.safetensors   A and B for every wrapped module
    <path>/meta.json           config + store bookkeeping + caller extras
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .slots import RankBlockedLoRA
from .store import MemoryStore

__all__ = ["SnapshotInfo", "save_snapshot", "load_snapshot", "SNAPSHOT_TENSORS", "SNAPSHOT_META"]

SNAPSHOT_TENSORS = "slots.safetensors"
SNAPSHOT_META = "meta.json"


@dataclass
class SnapshotInfo:
    path: Path
    n_slots: int
    rank: int
    alpha: float
    n_modules: int
    n_occupied: int
    bytes_on_disk: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "n_slots": self.n_slots,
            "rank": self.rank,
            "alpha": self.alpha,
            "n_modules": self.n_modules,
            "n_occupied": self.n_occupied,
            "bytes_on_disk": self.bytes_on_disk,
        }


def save_snapshot(
    path: str | Path,
    wrappers: dict[str, RankBlockedLoRA],
    store: MemoryStore,
    *,
    model_id: str = "",
    extra: dict[str, Any] | None = None,
) -> SnapshotInfo:
    """Write slots + bookkeeping to ``path`` (created if needed)."""
    if not wrappers:
        raise ValueError("nothing to save: no LoRA wrappers")
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    module_config: dict[str, dict[str, int]] = {}
    for name, wrapper in wrappers.items():
        tensors[f"{name}.A"] = wrapper.A.detach().to("cpu").contiguous()
        tensors[f"{name}.B"] = wrapper.B.detach().to("cpu").contiguous()
        module_config[name] = {
            "in_features": int(wrapper.base.in_features),
            "out_features": int(wrapper.base.out_features),
        }
    save_file(tensors, str(folder / SNAPSHOT_TENSORS))

    first = next(iter(wrappers.values()))
    meta = {
        "n_slots": int(first.n_slots),
        "rank": int(first.rank),
        "alpha": float(first.alpha),
        "total_rank": int(first.total_rank),
        "model_id": model_id,
        "module_config": module_config,
        "store": store.to_dict(),
        "extra": extra or {},
    }
    (folder / SNAPSHOT_META).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    size = sum(f.stat().st_size for f in folder.iterdir() if f.is_file())
    return SnapshotInfo(
        path=folder, n_slots=meta["n_slots"], rank=meta["rank"], alpha=meta["alpha"],
        n_modules=len(wrappers), n_occupied=store.n_occupied, bytes_on_disk=size,
    )


def load_snapshot(
    path: str | Path, wrappers: dict[str, RankBlockedLoRA]
) -> tuple[MemoryStore, dict[str, Any]]:
    """Load a snapshot into an existing side path, refusing any mismatch.

    Returns the restored :class:`MemoryStore` and the snapshot metadata.
    """
    folder = Path(path)
    meta_path = folder / SNAPSHOT_META
    tensor_path = folder / SNAPSHOT_TENSORS
    if not meta_path.is_file() or not tensor_path.is_file():
        raise FileNotFoundError(
            f"{folder} is not a memory snapshot (needs {SNAPSHOT_META} and "
            f"{SNAPSHOT_TENSORS})"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # ---- strict validation, before touching a single weight ---------------
    if not wrappers:
        raise ValueError("no LoRA wrappers to load into")
    first = next(iter(wrappers.values()))
    for key, actual in (("n_slots", first.n_slots), ("rank", first.rank)):
        if int(meta[key]) != int(actual):
            raise ValueError(
                f"snapshot {key}={meta[key]} does not match the side path "
                f"({actual}); refuse to load a misaligned memory"
            )
    if abs(float(meta["alpha"]) - float(first.alpha)) > 1e-9:
        raise ValueError(
            f"snapshot alpha={meta['alpha']} != side path alpha={first.alpha}"
        )
    if set(meta["module_config"]) != set(wrappers):
        missing = sorted(set(meta["module_config"]) - set(wrappers))
        surplus = sorted(set(wrappers) - set(meta["module_config"]))
        raise ValueError(
            f"snapshot was taken from different modules (missing={missing[:3]}, "
            f"surplus={surplus[:3]}); the backbone must be identical"
        )
    for name, wrapper in wrappers.items():
        cfg = meta["module_config"][name]
        if (int(cfg["in_features"]) != wrapper.base.in_features
                or int(cfg["out_features"]) != wrapper.base.out_features):
            raise ValueError(f"shape mismatch on {name}: snapshot {cfg} vs "
                             f"{wrapper.base.in_features}->{wrapper.base.out_features}")

    tensors = load_file(str(tensor_path))
    with torch.no_grad():
        for name, wrapper in wrappers.items():
            a = tensors[f"{name}.A"].to(device=wrapper.A.device, dtype=wrapper.A.dtype)
            b = tensors[f"{name}.B"].to(device=wrapper.B.device, dtype=wrapper.B.dtype)
            if a.shape != wrapper.A.shape or b.shape != wrapper.B.shape:
                raise ValueError(f"tensor shape mismatch on {name}: "
                                 f"{tuple(a.shape)}/{tuple(b.shape)} vs "
                                 f"{tuple(wrapper.A.shape)}/{tuple(wrapper.B.shape)}")
            wrapper.A.data.copy_(a)
            wrapper.B.data.copy_(b)

    store = MemoryStore.load_from_dict(meta["store"])
    return store, meta
