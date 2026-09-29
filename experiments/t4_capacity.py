"""T4 -- capacity and forgetting: forget what is unused, keep what is used.

The literature check said this is the thinnest area: almost everything is about
*preventing* forgetting, and the one study with hard numbers on eviction policies
found no policy beating LFU by more than 0.041 points. So the point of this
experiment is not to crown a winner -- it is to measure, on one trace:

* **discrimination** -- does the store keep the items that were *used* and drop
  the ones that were written and never touched again?
* **exactness** -- is a forgotten memory *provably* gone (the slot returns to its
  virgin state bit-for-bit), rather than merely decayed?
* **protection** -- is a memory marked important ("thought stamp") still there
  after the stream has overflowed capacity many times over?
* **cost to the base model** -- what does all this writing do to the frozen
  backbone, measured on the anchor prompts?

Access pattern: the first ``--hot`` items of the stream are probed repeatedly as
the stream advances (so they accumulate utility/recency); every other item is
written once and never asked about again. All five eviction policies see exactly
the same trace, the same writes and the same accesses.

Usage::

    python experiments/t4_capacity.py --stream 12 --slots 6 --hot 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from parammem.bench import protocol as P
from parammem.bench.synthetic import make_episode
from parammem.memory.router import route
from parammem.memory.store import POLICIES, MemoryStore
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

ARMS = ("hot", "cold")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--stream", type=int, default=12, help="number of memories written")
    ap.add_argument("--slots", type=int, default=6, help="capacity K (slots)")
    ap.add_argument("--hot", type=int, default=3, help="first N items are accessed repeatedly")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--policies", nargs="*", default=list(POLICIES))
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--anchor-mode", choices=("slot", "sum"), default="slot")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--no-paraphrase", dest="paraphrase", action="store_false",
                    default=True, help="read with the write query verbatim instead")
    ap.add_argument("--out", default="runs/t4_capacity.json")
    return ap.parse_args()


def run_policy(bb: Backbone, episode, args, policy: str) -> dict:
    """Replay the same stream under one eviction policy."""
    bb.erase_all()
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS)
    frozen_anchors = {q: bb.answer(q) for q in GENERIC_ANCHORS}

    store = MemoryStore(n_slots=args.slots, policy=policy, decay=0.9, min_utility=0.05)
    probes = {p.item_id: p for p in episode.probes}
    stream = episode.probeable_items

    resident: dict[int, object] = {}   # slot -> MemoryItem
    keys: dict[int, object] = {}       # slot -> retrieval key
    was_accessed: dict[int, bool] = {}  # slot -> was it ever asked about
    important_resident_ok = True

    write_seconds = 0.0
    erased_checked = 0
    erased_ok = 0
    evictions = 0
    norms: list[float] = []

    for item in stream:
        # Evict first if the store is full; importance must protect item 0.
        if store.is_full:
            for slot in store.make_room(1):
                evictions += 1
                bb.erase_slots([slot])
                erased_checked += 1
                erased_ok += int(bb.is_slot_virgin(slot))
                resident.pop(slot, None)
                keys.pop(slot, None)
                was_accessed.pop(slot, None)

        slot = store.next_free_slot()
        if slot is None:
            raise RuntimeError("no free slot after make_room")

        def loss_fn(item=item, slot=slot):
            bb.set_read_slots([slot])
            loss = bb.write_loss(item.query, item.value_text)
            if anchors:
                if args.anchor_mode == "slot":
                    bb.set_read_slots([slot])
                else:
                    bb.set_read_slots(sorted(set(resident) | {slot}))
                loss = loss + args.lambda_kl * bb.kl_to_anchors(anchors)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        write_seconds += report.seconds

        store.occupy(
            slot, item_id=item.item_id, text=item.statement,
            memory_class=item.memory_class, is_important=(item.item_id == 0),
            norm=bb.slot_norm(slot), write_seconds=report.seconds,
        )
        resident[slot] = item
        keys[slot] = bb.query_key(item.query)
        was_accessed[slot] = False
        store.advance()
        norms.append(bb.slot_norm(slot))

        # Access pattern: the first `hot` stream items get asked about again and
        # again (oracle slot, so this measures the store, not the router).
        hot_ids = {i.item_id for i in stream[: args.hot]}
        for s, resident_item in list(resident.items()):
            if resident_item.item_id not in hot_ids:
                continue
            probe = probes[resident_item.item_id]
            bb.set_read_slots([s])
            if P.exact_match(bb.answer(probe.query), probe.value, probe.aliases):
                store.touch(s)
                was_accessed[s] = True

    # ---- final measurement on whatever survived -------------------------
    survivors = sorted(resident)
    slot_keys = {s: keys[s] for s in survivors}
    oracle_hits = {arm: 0 for arm in ARMS}
    routed_hits = 0
    counts = {arm: 0 for arm in ARMS}
    for s in survivors:
        item = resident[s]
        arm = "hot" if was_accessed[s] else "cold"
        counts[arm] += 1
        probe = probes[item.item_id]
        bb.set_read_slots([s])
        oracle_hits[arm] += int(
            P.exact_match(bb.answer(probe.query), probe.value, probe.aliases)
        )
        decision = route(bb.query_key(probe.query), slot_keys, k=1)
        bb.set_read_slots(decision.slots)
        routed_hits += int(
            P.exact_match(bb.answer(probe.query), probe.value, probe.aliases)
        )

    # "Thought stamp" semantics: item 0 is flagged important and must survive
    # the stream overflowing capacity over and over.
    important_resident_ok = any(resident[s].item_id == 0 for s in survivors)

    bb.set_read_slots(survivors)
    same_anchors = sum(1 for q in GENERIC_ANCHORS
                       if bb.answer(q) == frozen_anchors[q])
    # no_grad is required here: kl_to_anchors runs a forward pass, and without it
    # PyTorch builds a graph and warns about converting a requires_grad tensor.
    with torch.no_grad():
        anchor_kl = float(bb.kl_to_anchors(anchors))

    return {
        "policy": policy,
        "evictions": evictions,
        "erased_checked": erased_checked,
        "erased_virgin_ok": erased_ok,
        "resident": {arm: counts[arm] for arm in ARMS},
        "oracle_hits": oracle_hits,
        "routed_hits": routed_hits,
        "n_survivors": len(survivors),
        "important_retained": bool(important_resident_ok),
        "anchor_same": f"{same_anchors}/{len(GENERIC_ANCHORS)}",
        "anchor_kl": anchor_kl,
        "max_slot_norm": max(norms) if norms else 0.0,
        "write_seconds_total": write_seconds,
        "history": store.history,
    }


def main() -> int:
    args = parse_args()
    cfg = BackboneConfig(
        model_id=resolve_model_path(args.model),
        n_slots=args.slots,
        rank=args.rank,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"loading {cfg.model_id} (slots={args.slots}, stream={args.stream}, "
          f"hot={args.hot}) ...", flush=True)
    bb = Backbone.load(cfg)
    print(f"modules={len(bb.wrappers)} memory_params={bb.n_memory_parameters/1e6:.2f}M",
          flush=True)

    episode = make_episode(
        args.base_seed, n_facts=args.stream, n_prefs=0, n_lessons=0, n_noise=0,
        n_negatives=0, with_inertia=False, paraphrase_probes=args.paraphrase,
    )
    print(f"stream={len(episode.probeable_items)} items, capacity={args.slots}", flush=True)

    records = []
    t0 = time.perf_counter()
    header = (f"{'policy':<14}{'evict':>6}{'eraseOK':>9}{'hot':>10}{'cold':>10}"
              f"{'routed':>8}{'stamp':>7}{'anchor':>9}{'KL':>8}{'maxNorm':>9}")
    print("\n" + header, flush=True)
    for policy in args.policies:
        record = run_policy(bb, episode, args, policy)
        records.append(record)
        hot = record["oracle_hits"]["hot"], record["resident"]["hot"]
        cold = record["oracle_hits"]["cold"], record["resident"]["cold"]
        print(
            f"{policy:<14}{record['evictions']:>6}"
            f"{record['erased_virgin_ok']:>4}/{record['erased_checked']:<4}"
            f"{hot[0]:>5}/{hot[1]:<4}{cold[0]:>5}/{cold[1]:<4}"
            f"{record['routed_hits']:>5}/{record['n_survivors']:<2}"
            f"{'yes' if record['important_retained'] else 'NO':>7}"
            f"{record['anchor_same']:>9}{record['anchor_kl']:>8.2f}"
            f"{record['max_slot_norm']:>9.0f}",
            flush=True,
        )

    summary = {
        "config": vars(args),
        "model_path": bb.resolved_path,
        "elapsed_s": time.perf_counter() - t0,
        "vram": bb.memory_footprint(),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
