"""T2 -- reading many memories: **select**, don't sum.

Measured baseline (0.6B, 8 slots): own-slot-only 8/8, wrong-slot 0/8, all-slots
0/8. Superposition destroys recall even though the slots are perfectly separable.
This compares the composition rules on one axis, with the target written down in
advance: get "all memories coexisting" from 0/8 back towards 8/8.

Arms, all measured on the same writes:

===============  ==========================================================
``oracle``       only the item's own slot (upper bound; not implementable)
``all``          every written slot active (the current default; the failure)
``top1``         route by query-key cosine similarity, activate the best slot
``top2``         route, activate the two best slots
``other``        activate a slot that is *not* this item's (lower bound)
===============  ==========================================================

Honest caveat about the router's job here: in the current benchmark the read
query is the **same string** as the write query, so routing is easy. That is a
scaffold, not a result -- the paraphrase probe (different wording, same intent)
is required before the router number means much, and is the next step.

Usage::

    python experiments/t2_composition.py --items 8 --episodes 3
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
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

ARMS = ("oracle", "all", "top1", "top2", "other")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--anchor-mode", choices=("slot", "sum"), default="slot")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", default="runs/t2_composition.json")
    return ap.parse_args()


def run_episode(bb: Backbone, episode, args: argparse.Namespace) -> dict:
    bb.erase_all()
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS) if args.lambda_kl > 0 else {}

    items = episode.probeable_items
    slot_of: dict[int, int] = {}
    for slot, item in enumerate(items):
        def loss_fn(item=item, slot=slot):
            bb.set_read_slots([slot])
            loss = bb.write_loss(item.query, item.value_text)
            if anchors:
                if args.anchor_mode == "slot":
                    bb.set_read_slots([slot])
                else:
                    bb.set_read_slots(sorted(set(slot_of.values()) | {slot}))
                loss = loss + args.lambda_kl * bb.kl_to_anchors(anchors)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        slot_of[item.item_id] = slot

    # Retrieval keys, captured with the memory off (see Backbone.query_key).
    slot_keys = {slot: bb.query_key(item.query) for item, slot in
                 ((i, slot_of[i.item_id]) for i in items)}

    written = sorted(slot_of.values())
    answers: dict[str, list[int]] = {arm: [] for arm in ARMS}
    route_hits = 0
    for probe in episode.probes:
        own = slot_of[probe.item_id]
        others = [s for s in written if s != own]
        key = bb.query_key(probe.query)
        decision = route(key, slot_keys, k=1)
        route_hits += int(decision.top == own)
        top2 = route(key, slot_keys, k=2).slots

        for arm, active in (
            ("oracle", [own]),
            ("all", written),
            ("top1", decision.slots),
            ("top2", top2),
            ("other", others[:1]),
        ):
            bb.set_read_slots(active)
            answer = bb.answer(probe.query)
            answers[arm].append(
                int(P.exact_match(answer, probe.value, probe.aliases))
            )

    return {
        "episode_id": episode.episode_id,
        "seed": episode.seed,
        "n_items": len(items),
        "arms": answers,
        "route_top1_hits": route_hits,
    }


def main() -> int:
    args = parse_args()
    cfg = BackboneConfig(
        model_id=resolve_model_path(args.model),
        n_slots=max(args.items, 2),
        rank=args.rank,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"loading {cfg.model_id} (items={args.items}, rank={cfg.rank}) ...", flush=True)
    bb = Backbone.load(cfg)
    print(f"modules={len(bb.wrappers)} memory_params={bb.n_memory_parameters/1e6:.2f}M",
          flush=True)

    records = []
    t0 = time.perf_counter()
    for ep_index in range(args.episodes):
        episode = make_episode(
            args.base_seed + ep_index, episode_id=ep_index, n_facts=args.items,
            n_prefs=0, n_lessons=0, n_noise=0, n_negatives=0, with_inertia=False,
        )
        record = run_episode(bb, episode, args)
        records.append(record)
        print(
            f"  episode {ep_index}: "
            + "  ".join(
                f"{arm}={sum(record['arms'][arm])}/{len(record['arms'][arm])}"
                for arm in ARMS
            ),
            flush=True,
        )

    totals = {arm: [v for r in records for v in r["arms"][arm]] for arm in ARMS}
    summary = {
        "n_episodes": len(records),
        "n_probes": len(totals["oracle"]),
        "em": {arm: (sum(v) / len(v) if v else float("nan")) for arm, v in totals.items()},
        "counts": {arm: [int(sum(v)), len(v)] for arm, v in totals.items()},
        "route_top1_accuracy": (
            sum(r["route_top1_hits"] for r in records)
            / max(1, sum(r["n_items"] for r in records))
        ),
        "config": vars(args),
        "model_path": bb.resolved_path,
        "elapsed_s": time.perf_counter() - t0,
        "vram": bb.memory_footprint(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "episodes": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nitems={args.items} episodes={summary['n_episodes']} probes={summary['n_probes']}")
    print(f"{'arm':<8}{'EM':>8}   counts")
    for arm in ARMS:
        hit, total = summary["counts"][arm]
        print(f"{arm:<8}{summary['em'][arm]:>8.3f}   {hit}/{total}")
    print(f"\nrouter top-1 accuracy: {summary['route_top1_accuracy']:.3f}")
    print(f"report written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
