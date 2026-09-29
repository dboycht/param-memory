"""T6 -- does a thought stamp survive a restart?

Persistence is the one part of the original idea still untested: a memory that
dies with the process is a session cache, not a stamp.

This script has two sessions, and they are meant to run as **two separate OS
processes** (that is the whole point -- re-loading the backbone inside one process
would not prove anything)::

    python experiments/t6_persistence.py --session write      # learn, then save
    python experiments/t6_persistence.py --session read       # fresh process, no writes

The read session asserts that a freshly built side path is *virgin* before loading
(``virgin_before_load``), so any recall it achieves can only have come from disk.
It also erases a slot after loading and re-checks ``is_slot_virgin``: forgetting
must still be exact on a restored memory.

Usage::

    python experiments/t6_persistence.py --session write --facts 6
    python experiments/t6_persistence.py --session read
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from parammem.bench import protocol as P
from parammem.bench.synthetic import make_episode
from parammem.memory.persist import load_snapshot, save_snapshot
from parammem.memory.router import route
from parammem.memory.store import MemoryStore
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", choices=("write", "read"), required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--facts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--snapshot", default="runs/t6_memory")
    ap.add_argument("--out", default="")
    return ap.parse_args()


def _load(args: argparse.Namespace) -> Backbone:
    cfg = BackboneConfig(
        model_id=resolve_model_path(args.model), n_slots=args.slots, rank=args.rank,
        alpha=args.alpha, max_new_tokens=args.max_new_tokens,
    )
    print(f"session={args.session} loading {cfg.model_id} ...", flush=True)
    bb = Backbone.load(cfg)
    print(f"modules={len(bb.wrappers)} memory_params={bb.n_memory_parameters/1e6:.2f}M",
          flush=True)
    return bb


def session_write(args: argparse.Namespace) -> dict:
    bb = _load(args)
    bb.erase_all()
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS)

    episode = make_episode(
        args.seed, n_facts=args.facts, n_prefs=0, n_lessons=0, n_noise=0,
        n_negatives=0, with_inertia=False, paraphrase_probes=True,
    )
    items = episode.probeable_items
    probes = {p.item_id: p for p in episode.probes}

    store = MemoryStore(n_slots=args.slots, policy="utility_time", decay=0.9)
    written = []
    for slot, item in enumerate(items):
        def loss_fn(item=item, slot=slot):
            bb.set_read_slots([slot])
            loss = bb.write_loss(item.query, item.value_text)
            bb.set_read_slots([slot])
            return loss + args.lambda_kl * bb.kl_to_anchors(anchors)

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        store.occupy(slot, item_id=item.item_id, text=item.statement,
                     memory_class=item.memory_class, is_important=(slot == 0),
                     norm=bb.slot_norm(slot), write_seconds=report.seconds)
        store.advance()
        written.append((slot, item, probes[item.item_id]))

    # Recall *before* saving, so the read session has a number to match.
    own_hits = 0
    for slot, _item, probe in written:
        bb.set_read_slots([slot])
        own_hits += int(P.exact_match(bb.answer(probe.query), probe.value, probe.aliases))

    payload = [
        {"slot": slot, "item_id": item.item_id, "query": item.query,
         "probe_query": probe.query, "value": item.value_text,
         "aliases": list(item.aliases), "statement": item.statement}
        for slot, item, probe in written
    ]
    info = save_snapshot(args.snapshot, bb.wrappers, store, model_id=args.model,
                         extra={"items": payload, "seed": args.seed})
    print(f"\nwrote {len(written)} memories; own-slot recall before saving "
          f"{own_hits}/{len(written)}", flush=True)
    print(f"snapshot: {info.to_dict()}", flush=True)
    return {
        "session": "write",
        "n_memories": len(written),
        "own_slot_recall": f"{own_hits}/{len(written)}",
        "snapshot": info.to_dict(),
        "items": payload,
    }


def session_read(args: argparse.Namespace) -> dict:
    bb = _load(args)
    bb.set_read_slots([])

    # A fresh side path must be virgin: whatever we recall cannot come from RAM.
    virgin_before = all(
        bb.is_slot_virgin(s) for s in range(bb.cfg.n_slots)
    )
    print(f"virgin before load: {virgin_before}", flush=True)

    store, meta = load_snapshot(args.snapshot, bb.wrappers)
    items = meta["extra"]["items"]
    print(f"loaded snapshot: occupied={store.n_occupied} "
          f"items={len(items)} model_id={meta.get('model_id')!r}", flush=True)

    slot_keys = {entry["slot"]: bb.query_key(entry["query"]) for entry in items}

    oracle_hits = 0
    routed_hits = 0
    route_correct = 0
    answers = []
    for entry in items:
        slot = entry["slot"]
        aliases = tuple(entry.get("aliases") or ())
        bb.set_read_slots([slot])
        oracle_answer = bb.answer(entry["probe_query"])
        oracle_hits += int(P.exact_match(oracle_answer, entry["value"], aliases))

        decision = route(bb.query_key(entry["probe_query"]), slot_keys, k=1)
        route_correct += int(decision.top == slot)
        bb.set_read_slots(decision.slots)
        routed_answer = bb.answer(entry["probe_query"])
        routed_hits += int(P.exact_match(routed_answer, entry["value"], aliases))
        answers.append({"slot": slot, "gold": entry["value"],
                        "routed_slot": decision.top, "answer": routed_answer})

    # Forgetting must still be exact on a restored memory.
    victim = items[0]
    bb.erase_slots([victim["slot"]])
    virgin_after_erase = bb.is_slot_virgin(victim["slot"])
    bb.set_read_slots([victim["slot"]])
    answer_after_erase = bb.answer(victim["probe_query"])
    still_recalled = P.exact_match(answer_after_erase, victim["value"],
                                   tuple(victim.get("aliases") or ()))

    summary = {
        "session": "read",
        "virgin_before_load": virgin_before,
        "n_items": len(items),
        "oracle_recall": f"{oracle_hits}/{len(items)}",
        "routed_recall": f"{routed_hits}/{len(items)}",
        "routing_correct": f"{route_correct}/{len(items)}",
        "store_occupied": store.n_occupied,
        "important_flags": sum(1 for s in store.occupied if s.is_important),
        "utility_survived": round(sum(s.utility for s in store.occupied), 4),
        "erase_after_load_is_virgin": virgin_after_erase,
        "erased_memory_still_recalled": still_recalled,
        "answers": answers,
    }
    print(f"\noracle recall (own slot)   : {summary['oracle_recall']}", flush=True)
    print(f"routed recall (k=1)        : {summary['routed_recall']}", flush=True)
    print(f"routing correct            : {summary['routing_correct']}", flush=True)
    print(f"erase after load is virgin : {virgin_after_erase}", flush=True)
    print(f"erased memory still recalled: {still_recalled}", flush=True)
    return summary


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    summary = session_write(args) if args.session == "write" else session_read(args)
    summary["elapsed_s"] = time.perf_counter() - t0

    if args.out:
        out = Path(args.out)
    else:
        out = Path(args.snapshot) / f"session_{args.session}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
