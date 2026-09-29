"""Tests for the durable snapshot format (pure, no model required)."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from parammem.memory.persist import SNAPSHOT_META, load_snapshot, save_snapshot
from parammem.memory.slots import attach_slot_lora
from parammem.memory.store import MemoryStore


def _tiny(n_slots: int = 4, rank: int = 2):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
    wrappers = attach_slot_lora(model, ["0", "2"], n_slots=n_slots, rank=rank, alpha=8.0)
    return model, wrappers


def _occupied_store(n_slots: int = 4) -> MemoryStore:
    store = MemoryStore(n_slots=n_slots, policy="lfu", decay=0.8)
    store.occupy(0, item_id=0, text="alpha", memory_class="fact", is_important=True)
    store.touch(0, amount=2.0)
    store.advance()
    store.occupy(2, item_id=1, text="beta", memory_class="pref")
    store.advance()
    store.touch(2)
    return store


def test_round_trip_restores_tensors_bit_exactly(tmp_path):
    _, wrappers = _tiny()
    torch.manual_seed(1)
    for wrapper in wrappers.values():
        wrapper.B.data.normal_()
    store = _occupied_store()
    saved = save_snapshot(tmp_path / "mem", wrappers, store)
    assert saved.n_occupied == 2 and saved.bytes_on_disk > 0

    expected = {n: (w.A.detach().clone(), w.B.detach().clone()) for n, w in wrappers.items()}
    for wrapper in wrappers.values():
        wrapper.A.data.zero_()
        wrapper.B.data.zero_()

    loaded_store, meta = load_snapshot(tmp_path / "mem", wrappers)
    for name, wrapper in wrappers.items():
        assert torch.equal(wrapper.A.detach(), expected[name][0]), name
        assert torch.equal(wrapper.B.detach(), expected[name][1]), name
    assert meta["n_slots"] == 4 and meta["rank"] == 2


def test_store_bookkeeping_round_trips(tmp_path):
    _, wrappers = _tiny()
    store = _occupied_store()
    save_snapshot(tmp_path / "mem", wrappers, store)

    loaded, _ = load_snapshot(tmp_path / "mem", wrappers)
    assert loaded.policy == "lfu"
    assert loaded.n_slots == store.n_slots
    assert loaded.clock == store.clock
    assert [None if s is None else s.text for s in loaded.slots] == ["alpha", None, "beta", None]
    assert loaded.get(0).is_important is True
    assert loaded.get(0).access_count == 1
    assert loaded.eviction_order() == store.eviction_order()


def test_erased_slot_stays_virgin_across_a_round_trip(tmp_path):
    """Exact erasure must survive persistence, otherwise "forgotten" memories
    could come back after a restart."""
    _, wrappers = _tiny()
    torch.manual_seed(2)
    for wrapper in wrappers.values():
        wrapper.B.data.normal_()
    save_snapshot(tmp_path / "mem", wrappers, _occupied_store())
    load_snapshot(tmp_path / "mem", wrappers)

    wrapper = wrappers["0"]
    assert not wrapper.is_virgin(1)
    # Erase the slot in *every* module: asserting one module's slot while only
    # erasing another's was the bug in the first version of this test.
    for module in wrappers.values():
        module.erase(1)
        assert module.is_virgin(1)


def test_extra_payload_is_preserved(tmp_path):
    _, wrappers = _tiny()
    save_snapshot(tmp_path / "mem", wrappers, _occupied_store(),
                  model_id="Qwen/Qwen3-0.6B", extra={"items": [{"slot": 0, "query": "q"}]})
    _, meta = load_snapshot(tmp_path / "mem", wrappers)
    assert meta["model_id"] == "Qwen/Qwen3-0.6B"
    assert meta["extra"]["items"][0]["query"] == "q"


def test_slot_count_mismatch_is_refused(tmp_path):
    _, wrappers = _tiny(n_slots=4)
    save_snapshot(tmp_path / "mem", wrappers, _occupied_store())
    _, other = _tiny(n_slots=8)
    with pytest.raises(ValueError, match="n_slots"):
        load_snapshot(tmp_path / "mem", other)


def test_rank_mismatch_is_refused(tmp_path):
    _, wrappers = _tiny(rank=2)
    save_snapshot(tmp_path / "mem", wrappers, _occupied_store())
    _, other = _tiny(rank=4)
    with pytest.raises(ValueError, match="rank"):
        load_snapshot(tmp_path / "mem", other)


def test_different_module_set_is_refused(tmp_path):
    _, wrappers = _tiny()
    save_snapshot(tmp_path / "mem", wrappers, _occupied_store())

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
    other = attach_slot_lora(model, ["2"], n_slots=4, rank=2, alpha=8.0)
    with pytest.raises(ValueError, match="different modules"):
        load_snapshot(tmp_path / "mem", other)


def test_missing_snapshot_raises_file_not_found(tmp_path):
    _, wrappers = _tiny()
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path / "nope", wrappers)


def test_saving_without_wrappers_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no LoRA wrappers"):
        save_snapshot(tmp_path / "mem", {}, _occupied_store())


def test_meta_file_is_readable_json(tmp_path):
    _, wrappers = _tiny()
    save_snapshot(tmp_path / "mem", wrappers, _occupied_store())
    meta = json.loads((tmp_path / "mem" / SNAPSHOT_META).read_text(encoding="utf-8"))
    assert meta["total_rank"] == 8
    assert len(meta["module_config"]) == 2
