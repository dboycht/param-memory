"""Slot bookkeeping: what lives where, what it is worth, and what gets evicted.

Capacity is the integer ``K`` from :mod:`parammem.memory.slots`; this module owns
the *policy* side of it. All four classic eviction policies are implemented
(FIFO / LRU / LFU / utility) rather than only our own, because the literature
warning is specific: in the one study with hard numbers, no eviction policy beat
LFU by more than 0.041 points. The only way to know whether our utility+decay
scheme earns its complexity is to measure it against the dumb ones on the same
traces (see docs/00 section 7.2).

Two invariants worth stating out loud:

* **Importance is a hard protection.** A slot flagged ``is_important`` is never
  returned by :meth:`MemoryStore.eviction_order`, whatever the policy says.
* **Nothing is forgotten by accident.** A slot leaves the store only through an
  explicit :meth:`evict` / :meth:`forget`, which the caller turns into a
  :meth:`RankBlockedLoRA.erase` call -- exact erasure, not decay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

__all__ = ["SlotMeta", "MemoryStore", "POLICIES"]

POLICIES = ("fifo", "lru", "lfu", "utility", "utility_time")


@dataclass
class SlotMeta:
    """Everything known about one occupied slot."""

    slot: int
    item_id: int | None = None
    text: str = ""
    memory_class: str = "fact"
    is_important: bool = False
    written_at: float = 0.0
    last_access: float = 0.0
    access_count: int = 0
    utility: float = 0.0
    norm: float = 0.0
    write_seconds: float = 0.0

    @property
    def age(self) -> float:
        return self.last_access - self.written_at

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SlotMeta":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class MemoryStore:
    """Metadata for ``n_slots`` memory slots plus an eviction policy.

    ``now`` is a *logical* clock (write/turn count), deliberately not wall time,
    so a run replays identically regardless of machine speed.
    """

    n_slots: int
    policy: str = "utility_time"
    decay: float = 0.9
    min_utility: float = 0.05
    slots: list[SlotMeta | None] = field(default_factory=list)
    clock: float = 0.0
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")
        if not (0.0 < self.decay <= 1.0):
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")
        if not self.slots:
            self.slots = [None] * self.n_slots

    # ------------------------------------------------------------- occupancy
    @property
    def occupied(self) -> list[SlotMeta]:
        return [s for s in self.slots if s is not None]

    @property
    def n_occupied(self) -> int:
        return len(self.occupied)

    @property
    def is_full(self) -> bool:
        return self.n_occupied >= self.n_slots

    def free_slots(self) -> list[int]:
        return [i for i, s in enumerate(self.slots) if s is None]

    def next_free_slot(self) -> int | None:
        free = self.free_slots()
        return free[0] if free else None

    def get(self, slot: int) -> SlotMeta | None:
        self._check(slot)
        return self.slots[slot]

    def _check(self, slot: int) -> None:
        if not (0 <= slot < self.n_slots):
            raise IndexError(f"slot {slot} out of range [0, {self.n_slots})")

    # ---------------------------------------------------------------- writes
    def advance(self, steps: float = 1.0) -> None:
        """Move the logical clock forward (and let utility decay)."""
        self.clock += steps
        if self.decay < 1.0:
            for meta in self.occupied:
                meta.utility *= self.decay ** steps

    def occupy(
        self,
        slot: int,
        *,
        item_id: int | None,
        text: str,
        memory_class: str,
        is_important: bool = False,
        norm: float = 0.0,
        write_seconds: float = 0.0,
    ) -> SlotMeta:
        self._check(slot)
        if self.slots[slot] is not None:
            raise ValueError(f"slot {slot} is already occupied (erase it first)")
        meta = SlotMeta(
            slot=slot,
            item_id=item_id,
            text=text,
            memory_class=memory_class,
            is_important=is_important,
            written_at=self.clock,
            last_access=self.clock,
            access_count=0,
            utility=1.0,  # a fresh memory starts with unit utility
            norm=norm,
            write_seconds=write_seconds,
        )
        self.slots[slot] = meta
        return meta

    def touch(self, slot: int, *, amount: float = 1.0) -> SlotMeta:
        """Record a successful recall of ``slot`` (this is what feeds utility)."""
        meta = self.slots[slot]
        if meta is None:
            raise ValueError(f"slot {slot} is empty")
        meta.access_count += 1
        meta.last_access = self.clock
        meta.utility += amount
        return meta

    def set_norm(self, slot: int, norm: float) -> None:
        meta = self.slots[slot]
        if meta is not None:
            meta.norm = float(norm)

    # -------------------------------------------------------------- eviction
    def score(self, meta: SlotMeta) -> float:
        """Eviction priority: **lower score = evicted first**."""
        if self.policy == "fifo":
            return meta.written_at
        if self.policy == "lru":
            return meta.last_access
        if self.policy == "lfu":
            return meta.access_count
        if self.policy == "utility":
            return meta.utility
        if self.policy == "utility_time":
            # Decayed utility already carries recency; break ties by recency.
            return meta.utility * 1000.0 + meta.last_access
        raise AssertionError(f"unhandled policy {self.policy!r}")

    def eviction_order(self) -> list[int]:
        """Slots eligible for eviction, worst first. Important slots excluded."""
        eligible = [m for m in self.occupied if not m.is_important]
        return [m.slot for m in sorted(eligible, key=self.score)]

    def decayed_below(self, threshold: float | None = None) -> list[int]:
        """Slots whose decayed utility has fallen under the forgetting floor.

        This is the time-decay half of the forgetting design: a memory nobody
        recalls stops being worth a slot. Importance still wins.
        """
        floor = self.min_utility if threshold is None else threshold
        return [
            m.slot
            for m in sorted(self.occupied, key=self.score)
            if not m.is_important and m.utility < floor
        ]

    def evict(self, k: int = 1) -> list[int]:
        """Mark ``k`` slots for erasure and free their metadata. Worst first."""
        chosen = self.eviction_order()[:k]
        for slot in chosen:
            self.history.append(
                {"event": "evict", "clock": self.clock, **self.slots[slot].to_dict()}
            )
            self.slots[slot] = None
        return chosen

    def forget(self, slot: int, reason: str = "manual") -> None:
        meta = self.slots[slot]
        if meta is None:
            raise ValueError(f"slot {slot} is empty")
        self.history.append(
            {"event": "forget", "reason": reason, "clock": self.clock, **meta.to_dict()}
        )
        self.slots[slot] = None

    def make_room(self, needed: int = 1) -> list[int]:
        """Free ``needed`` slots, evicting if required. Raises if impossible."""
        freed: list[int] = []
        while len(self.free_slots()) < needed:
            evicted = self.evict(1)
            if not evicted:
                raise RuntimeError(
                    "cannot free a slot: every remaining slot is marked important"
                )
            freed.extend(evicted)
        return freed

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "n_slots": self.n_slots,
            "policy": self.policy,
            "decay": self.decay,
            "min_utility": self.min_utility,
            "clock": self.clock,
            "slots": [None if s is None else s.to_dict() for s in self.slots],
            "history": self.history,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        return cls.load_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_from_dict(cls, data: dict) -> "MemoryStore":
        """Rebuild a store from ``to_dict()`` output (used by the snapshot loader)."""
        store = cls(
            n_slots=data["n_slots"],
            policy=data.get("policy", "utility_time"),
            decay=data.get("decay", 0.9),
            min_utility=data.get("min_utility", 0.05),
        )
        store.clock = data.get("clock", 0.0)
        store.slots = [
            None if s is None else SlotMeta.from_dict(s) for s in data["slots"]
        ]
        store.history = list(data.get("history", []))
        return store

    # ------------------------------------------------------------ reporting
    def snapshot(self, slots: Iterable[int] | None = None) -> list[SlotMeta]:
        chosen = range(self.n_slots) if slots is None else slots
        return [self.slots[i] for i in chosen if self.slots[i] is not None]
