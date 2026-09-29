"""T8-a -- B1: can a retention term in the *write* objective fix the k=2 collapse?

Measured problem (T2, 0.6B, 8 items, paraphrased probes): reading one slot recalls
most items, reading two collapses, reading all of them destroys recall entirely.
The slots are perfectly separable, so the interference is manufactured by the writes
themselves: each write is trained to produce its own value and is never told that
the bank already answers other queries.

Hypothesis under test
---------------------
Add to the write objective a **retention term**: the queries of earlier memories
must keep their next-token distributions under the read condition where the
interference actually appears (all written slots active). The reference for that
term is captured *before* the write, from the bank as it then stands.

This is the same machinery as the base-model anchor (all-position KL to a captured
reference), pointed at real memories instead of generic prompts.

The sweep is the experiment: ``--lambda-ret 0 0.3 1 3``. ``0`` reproduces T2's
behaviour, so the comparison is apples-to-apples on the same items and probes.

Everything is reported: recall per composition arm, routing accuracy (so a routing
failure is not mistaken for an interference failure), and base-model damage with the
slots both off (a sanity check that the masks really disable the adapters) and on
(the damage the bank inflicts when in use).

Usage::

    python experiments/t8a_retention.py --items 8 --episodes 1 \
        --paraphrase --lambda-ret 0 0.3 1 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from parammem.bench import protocol as P
from parammem.bench.synthetic import make_episode
from parammem.memory.retention import earlier_queries
from parammem.memory.router import route
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

ARMS = ("oracle", "all", "top1", "top2", "other")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--lambda-ret", type=float, nargs="+", default=[0.0, 0.3, 1.0, 3.0],
                    help="retention weights to sweep; 0 = current behaviour (T2)")
    ap.add_argument("--anchor-mode", choices=("slot", "sum"), default="slot",
                    help="read condition used for the base-model anchor term")
    ap.add_argument("--retention-limit", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--out", default="runs/t8a_retention.json")
    return ap.parse_args()


def run_episode(bb: Backbone, episode, args, lambda_ret: float) -> dict:
    bb.erase_all()
    bb.set_read_slots([])
    generic = bb.anchor_logits(GENERIC_ANCHORS) if args.lambda_kl > 0 else {}

    items = episode.probeable_items
    slot_of: dict[int, int] = {}
    loss_starts: list[float] = []
    for slot, item in enumerate(items):
        retention: dict = {}
        if lambda_ret > 0:
            queries = earlier_queries(items, slot, limit=args.retention_limit)
            if queries:
                # the bank as it stands, before this write
                bb.set_read_slots(sorted(slot_of.values()))
                retention = bb.anchor_logits(queries)

        def loss_fn(item=item, slot=slot, retention=retention, slot_of=slot_of):
            bb.set_read_slots([slot])
            loss = bb.write_loss(item.query, item.value_text)
            if generic:
                bb.set_read_slots(
                    [slot] if args.anchor_mode == "slot"
                    else sorted(set(slot_of.values()) | {slot})
                )
                loss = loss + args.lambda_kl * bb.kl_to_anchors(generic)
            if retention:
                # the condition under which interference actually appears
                bb.set_read_slots(sorted(set(slot_of.values()) | {slot}))
                loss = loss + lambda_ret * bb.kl_to_anchors(retention)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        loss_starts.append(float(report.loss_start))
        slot_of[item.item_id] = slot

    slot_keys = {slot: bb.query_key(item.query) for item, slot in
                 ((i, slot_of[i.item_id]) for i in items)}
    written = sorted(slot_of.values())

    answers: dict[str, list[int]] = {arm: [] for arm in ARMS}
    top1_hits = top2_hits = 0
    for probe in episode.probes:
        own = slot_of[probe.item_id]
        key = bb.query_key(probe.query)
        decision = route(key, slot_keys, k=1)
        two = route(key, slot_keys, k=2)
        top1_hits += int(decision.top == own)
        top2_hits += int(own in two.slots)
        for arm, active in (
            ("oracle", [own]), ("all", written), ("top1", decision.slots),
            ("top2", two.slots), ("other", [s for s in written if s != own][:1]),
        ):
            bb.set_read_slots(active)
            answers[arm].append(int(P.exact_match(bb.answer(probe.query),
                                                  probe.value, probe.aliases)))

    # base damage: off is a sanity check (must be ~0 if the masks really disable),
    # on is the damage the bank inflicts while in use. Detached: these are
    # measurements, and reading a scalar off a grad-tracking tensor warns (and keeps
    # the graph alive for no reason).
    bb.set_read_slots([])
    kl_off = float(bb.kl_to_anchors(generic).detach()) if generic else 0.0
    bb.set_read_slots(written)
    kl_on = float(bb.kl_to_anchors(generic).detach()) if generic else 0.0

    return {"episode_id": episode.episode_id, "seed": episode.seed,
            "lambda_ret": lambda_ret, "arms": answers,
            "route_top1_hits": top1_hits, "route_top2_hits": top2_hits,
            "n_probes": len(episode.probes), "n_items": len(items),
            "base_kl_slots_off": kl_off, "base_kl_slots_on": kl_on,
            "mean_write_loss_start": sum(loss_starts) / max(1, len(loss_starts))}


def main() -> int:
    args = parse_args()
    cfg = BackboneConfig(model_id=resolve_model_path(args.model),
                         n_slots=max(args.items, 2), rank=args.rank, alpha=args.alpha,
                         max_new_tokens=args.max_new_tokens)
    print(f"loading {cfg.model_id} (items={args.items}, lambdas={args.lambda_ret}) ...",
          flush=True)
    bb = Backbone.load(cfg)

    records = []
    t0 = time.perf_counter()
    for lambda_ret in args.lambda_ret:
        for ep_index in range(args.episodes):
            episode = make_episode(
                args.base_seed + ep_index, episode_id=ep_index, n_facts=args.items,
                n_prefs=0, n_lessons=0, n_noise=0, n_negatives=0, with_inertia=False,
                paraphrase_probes=args.paraphrase,
            )
            record = run_episode(bb, episode, args, lambda_ret)
            records.append(record)
            print(f"  lambda_ret={lambda_ret:<5} episode {ep_index}: "
                  + "  ".join(f"{arm}={sum(record['arms'][arm])}/{record['n_probes']}"
                              for arm in ARMS), flush=True)

    summary = {"config": vars(args), "model_path": bb.resolved_path,
               "elapsed_s": time.perf_counter() - t0, "vram": bb.memory_footprint(),
               "by_lambda": {}}
    for lambda_ret in args.lambda_ret:
        rows = [r for r in records if r["lambda_ret"] == lambda_ret]
        totals = {arm: [v for r in rows for v in r["arms"][arm]] for arm in ARMS}
        summary["by_lambda"][str(lambda_ret)] = {
            "em": {arm: (sum(v) / len(v) if v else float("nan"))
                   for arm, v in totals.items()},
            "counts": {arm: [int(sum(v)), len(v)] for arm, v in totals.items()},
            "route_top1_accuracy": sum(r["route_top1_hits"] for r in rows)
            / max(1, sum(r["n_probes"] for r in rows)),
            "route_top2_accuracy": sum(r["route_top2_hits"] for r in rows)
            / max(1, sum(r["n_probes"] for r in rows)),
            "base_kl_slots_on": sum(r["base_kl_slots_on"] for r in rows) / len(rows),
            "base_kl_slots_off": sum(r["base_kl_slots_off"] for r in rows) / len(rows),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "episodes": records},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nitems={args.items} episodes={args.episodes} paraphrase={args.paraphrase}")
    print(f"{'lambda_ret':>11}" + "".join(f"{arm:>9}" for arm in ARMS)
          + f"{'route@1':>9}{'route@2':>9}{'base KL(on)':>13}")
    for lambda_ret in args.lambda_ret:
        block = summary["by_lambda"][str(lambda_ret)]
        print(f"{lambda_ret:>11}" + "".join(f"{block['em'][arm]:>9.3f}" for arm in ARMS)
              + f"{block['route_top1_accuracy']:>9.3f}"
              + f"{block['route_top2_accuracy']:>9.3f}"
              + f"{block['base_kl_slots_on']:>13.3f}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
