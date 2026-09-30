"""T2-d -- can a write be trained to survive being read *alongside* other slots?

T2-b measured why two slots collapse: the two-slot logits are not the sum of the
single-slot perturbations (residual 0.849 of the actual effect), while the slot updates
themselves are near-orthogonal and neither perturbation dominates the other. The
failure is therefore created at *read* time by the model's own non-linearities, not by
the memories interfering in weight space.

If that is right, the write is being trained under the wrong read condition: every slot
is optimised with **only itself active**, then evaluated with others active too. The
intervention here is one line of difference between the arms:

=================  ==============================================================
``isolated``       write slot k with only slot k active (the current scheme)
``composable``     write slot k with the already-written slots *also* active, i.e.
                   under the condition the memory will actually be read in
=================  ==============================================================

Both arms are then evaluated identically: oracle (own slot only), top1, top2 (routed)
and all slots. The composable arm keeps the same write budget, so a difference is
attributable to the read condition rather than to more training.

Usage::

    python experiments/t2d_composable_write.py --items 8 --episodes 2
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
from parammem.memory.router import route
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

MODES = ("isolated", "composable")
ARMS = ("oracle", "top1", "top2", "all")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", default="runs/t2d_composable.json")
    return ap.parse_args()


def write_episode(bb: Backbone, episode, args, mode: str) -> dict:
    """Write every item under one read condition, then measure the usual arms."""
    bb.erase_all()
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS) if args.lambda_kl > 0 else {}

    items = episode.probeable_items
    slot_of: dict[int, int] = {}
    for slot, item in enumerate(items):
        earlier = sorted(slot_of.values())

        def loss_fn(item=item, slot=slot, earlier=earlier):
            # The one line that differs between the arms: what is active while the
            # target term's gradient is computed. The anchor term stays
            # slot-isolated even in the composable arm, because evaluating it with
            # several slots active lets their displacements cancel out -- a measured
            # detail, not a stylistic choice.
            active = [slot] if mode == "isolated" else earlier + [slot]
            bb.set_read_slots(active)
            loss = bb.write_loss(item.query, item.value_text)
            if anchors:
                bb.set_read_slots([slot])
                loss = loss + args.lambda_kl * bb.kl_to_anchors(anchors)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        slot_of[item.item_id] = slot

    keys = {slot: bb.query_key(item.query)
            for item, slot in ((i, slot_of[i.item_id]) for i in items)}
    written = sorted(slot_of.values())
    answers: dict[str, list[int]] = {arm: [] for arm in ARMS}
    route_hits = route_top2 = 0
    for probe in episode.probes:
        own = slot_of[probe.item_id]
        key = bb.query_key(probe.query)
        ranked = route(key, keys, k=2).slots
        route_hits += int(ranked[0] == own)
        route_top2 += int(own in ranked)
        for arm, active in (("oracle", [own]), ("top1", ranked[:1]),
                            ("top2", ranked), ("all", written)):
            bb.set_read_slots(active)
            answer = bb.answer(probe.query)
            answers[arm].append(int(P.exact_match(answer, probe.value, probe.aliases)))

    return {"episode_id": episode.episode_id, "seed": episode.seed,
            "n_items": len(items), "mode": mode, "arms": answers,
            "route_top1": route_hits, "route_top2": route_top2,
            "n_probes": len(episode.probes)}


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = BackboneConfig(model_id=resolve_model_path(args.model),
                         n_slots=max(args.items, 2), rank=args.rank,
                         alpha=args.alpha, max_new_tokens=args.max_new_tokens)
    print(f"loading {cfg.model_id} (items={args.items}, rank={cfg.rank}) ...", flush=True)
    bb = Backbone.load(cfg)

    records: dict[str, list[dict]] = {mode: [] for mode in MODES}
    for index in range(args.episodes):
        episode = make_episode(
            args.base_seed + index, episode_id=index, n_facts=args.items,
            n_prefs=0, n_lessons=0, n_noise=0, n_negatives=0, with_inertia=False,
            paraphrase_probes=True,
        )
        for mode in MODES:
            print(f"  episode {index} / {mode} ...", flush=True)
            records[mode].append(write_episode(bb, episode, args, mode))
            record = records[mode][-1]
            print("      " + "  ".join(
                f"{arm}={sum(record['arms'][arm])}/{record['n_probes']}"
                for arm in ARMS), flush=True)

    totals = {mode: {arm: sum(sum(r["arms"][arm]) for r in records[mode])
                     for arm in ARMS} for mode in MODES}
    probes = sum(r["n_probes"] for r in records[MODES[0]])
    summary = {
        "note": "T2-b showed the two-slot collapse is a read-time non-linearity, so this "
                "trains the write under the read condition it will actually face "
                "(earlier slots active) instead of in isolation. Same budget both arms.",
        "n_probes": probes,
        "n_items": args.items,
        "records": records,
        "totals": totals,
        "rates": {mode: {arm: value / probes for arm, value in arms.items()}
                  for mode, arms in totals.items()},
        "model_path": cfg.model_id,
        "elapsed_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  probes per arm: {probes}")
    print(f"  {'arm':<8}" + "".join(f"{mode:>14}" for mode in MODES))
    for arm in ARMS:
        print(f"  {arm:<8}" +
              "".join(f"{summary['rates'][mode][arm]:>14.3f}" for mode in MODES))
    for mode in MODES:
        hits = sum(r["route_top1"] for r in records[mode])
        deep = sum(r["route_top2"] for r in records[mode])
        print(f"  routing ({mode}): top1 {hits}/{probes}, correct-in-top2 {deep}/{probes}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
