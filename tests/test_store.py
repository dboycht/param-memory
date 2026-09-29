"""Tests for slot bookkeeping and the eviction policies.

The policies are compared on one hand-built trace so that the *ordering*
differences between FIFO / LRU / LFU / utility are pinned down, not just
"some slot got evicted".
"""

from __future__ import annotations

import json

import pytest

from parammem.memory.store import POLICIES, MemoryStore, SlotMeta


def _store(policy: str = "utility_time", n_slots: int = 4, decay: float = 0.9,
           min_utility: float = 0.05):
    return MemoryStore(n_slots=n_slots, policy=policy, decay=decay, min_utility=min_utility)


def _fill(store: MemoryStore, texts=("a", "b", "c", "d")):
    for i, text in enumerate(texts):
        slot = store.next_free_slot()
        store.occupy(slot, item_id=i, text=text, memory_class="fact")
        store.advance()
    return store


# ----------------------------------------------------------------- structure

def test_construction_and_validation():
    with pytest.raises(ValueError):
        MemoryStore(n_slots=2, policy="random")
    with pytest.raises(ValueError):
        MemoryStore(n_slots=2, decay=0.0)
    assert MemoryStore(n_slots=3).free_slots() == [0, 1, 2]
    with pytest.raises(IndexError):
        MemoryStore(n_slots=2).get(2)


def test_occupy_and_double_occupy():
    store = _store(n_slots=2)
    store.occupy(0, item_id=1, text="x", memory_class="fact")
    assert store.n_occupied == 1 and not store.is_full
    with pytest.raises(ValueError):
        store.occupy(0, item_id=2, text="y", memory_class="fact")


def test_occupy_empty_slot_raises_on_touch():
    store = _store(n_slots=2)
    with pytest.raises(ValueError):
        store.touch(1)


def test_clock_and_utility_decay():
    store = _store(decay=0.5, n_slots=2)
    store.occupy(0, item_id=0, text="x", memory_class="fact")
    assert store.get(0).utility == 1.0
    store.advance()
    assert store.get(0).utility == pytest.approx(0.5)
    store.advance(2)
    assert store.get(0).utility == pytest.approx(0.125)


def test_touch_records_recall():
    store = _store(n_slots=1, decay=1.0)
    store.occupy(0, item_id=0, text="x", memory_class="fact")
    store.advance(3)
    meta = store.touch(0, amount=2.0)
    assert meta.access_count == 1
    assert meta.last_access == 3.0
    assert meta.utility == pytest.approx(3.0)


# ----------------------------------------------------------------- policies

def _traced_store(policy: str) -> MemoryStore:
    """One trace, four policies: slot 0 is oldest and least used, slot 3 newest."""
    store = _store(policy=policy, n_slots=4, decay=1.0)
    for i in range(4):
        store.occupy(i, item_id=i, text=f"item{i}", memory_class="fact")
        store.advance()
    # slot 0 gets recalled a lot, slot 2 once, slots 1 and 3 never.
    for _ in range(5):
        store.touch(0)
    store.touch(2)
    store.advance(2)
    return store


def test_fifo_evicts_the_oldest_write():
    assert _traced_store("fifo").eviction_order()[0] == 0


def test_lru_evicts_the_least_recently_used():
    order = _traced_store("lru").eviction_order()
    assert order[0] in (1, 3)  # written at clock 1 and 3, never touched
    assert order[-1] == 2  # touched at clock 3, i.e. the most recent


def test_lfu_evicts_the_least_frequently_used():
    order = _traced_store("lfu").eviction_order()
    assert order[-1] == 0  # 5 recalls
    assert order[0] in (1, 3)  # 0 recalls


def test_utility_time_tracks_decayed_utility():
    order = _traced_store("utility_time").eviction_order()
    assert order[0] in (1, 3)
    assert order[-1] == 0


def test_all_policies_are_implemented():
    for policy in POLICIES:
        store = _traced_store(policy)
        assert len(store.eviction_order()) == 4


# ---------------------------------------------------------------- importance

def test_important_slots_are_never_evicted():
    store = _store("fifo", n_slots=3)
    _fill(store, ("a", "b", "c"))
    store.get(0).is_important = True
    store.get(1).is_important = True
    assert store.eviction_order() == [2]


def test_make_room_raises_when_everything_is_important():
    store = _store("fifo", n_slots=2)
    _fill(store, ("a", "b"))
    for meta in store.occupied:
        meta.is_important = True
    with pytest.raises(RuntimeError):
        store.make_room(1)


def test_make_room_evicts_until_enough_is_free():
    store = _store("fifo", n_slots=4)
    _fill(store)
    assert store.make_room(2) == [0, 1]
    assert sorted(store.free_slots()) == [0, 1]
    assert [m.item_id for m in store.occupied] == [2, 3]


def test_decayed_below_respects_importance_and_floor():
    store = _store("utility", n_slots=3, decay=0.5, min_utility=0.1)
    _fill(store, ("a", "b", "c"))
    store.get(1).is_important = True
    store.advance(5)  # 0.5^5 = 0.03125 < 0.1
    assert 1 not in store.decayed_below()
    assert set(store.decayed_below()) == {0, 2}


# ------------------------------------------------------------------- erasure

def test_evict_and_forget_record_history_and_free_the_slot():
    store = _store("fifo", n_slots=2)
    _fill(store, ("a", "b"))
    store.evict(1)
    assert store.slots[0] is None and store.n_occupied == 1
    store.forget(1, reason="user-request")
    assert store.n_occupied == 0
    events = [h["event"] for h in store.history]
    assert events == ["evict", "forget"]
    assert store.history[1]["reason"] == "user-request"


def test_forget_on_empty_slot_raises():
    with pytest.raises(ValueError):
        _store(n_slots=1).forget(0)


# --------------------------------------------------------------- persistence

def test_save_and_load_round_trip(tmp_path):
    store = _store("lfu", n_slots=3, decay=0.8)
    _fill(store, ("a", "b", "c"))
    store.touch(1, amount=2.0)
    store.get(2).is_important = True
    store.advance(2)
    path = store.save(tmp_path / "store.json")

    loaded = MemoryStore.load(path)
    assert loaded.policy == "lfu"
    assert loaded.n_slots == 3
    assert loaded.clock == store.clock
    assert [m.text for m in loaded.occupied] == ["a", "b", "c"]
    assert loaded.get(1).access_count == 1
    assert loaded.get(2).is_important is True
    assert loaded.eviction_order() == store.eviction_order()


def test_saved_file_is_readable_json(tmp_path):
    store = _store(n_slots=1)
    _fill(store, ("a",))
    path = store.save(tmp_path / "s.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n_slots"] == 1
    assert data["slots"][0]["text"] == "a"


def test_slot_meta_from_dict_ignores_unknown_fields():
    meta = SlotMeta.from_dict({"slot": 0, "text": "x", "unknown_field": 1})
    assert meta.slot == 0 and meta.text == "x"
