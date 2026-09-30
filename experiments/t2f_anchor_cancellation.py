"""T2-f -- the anchor's per-slot versus summed displacement, as a saved measurement.

The method section reports that evaluating the anchor term on the sum of the slots lets
the individual displacements cancel: per-slot KL of ``[0.08, 3.89, 1.80, 1.80]`` against
``0.09`` for the sum. Those numbers came from a design measurement that was never written
to a run file, so a reader cannot check them and the paper cannot update them when the code
changes. This measures the same thing and saves it.

It also reports the quantity the argument actually needs: whether the guard is non-vacuous
per slot, and how much of it survives when the slots are summed.

Usage::

    python experiments/t2f_anchor_cancellation.py --items 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from parammem.bench.synthetic import make_episode
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--items", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--anchor-mode", choices=("slot", "sum"), default="slot",
                    help="where the anchor is evaluated during writing. 'slot' is the "
                         "fix (each slot is held on its own); 'sum' is the configuration "
                         "the fix replaced, in which the slots are free to cancel each "
                         "other and the guard looks far stronger than it is")
    ap.add_argument("--out", default="runs/t2f_anchor_cancellation.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = BackboneConfig(model_id=resolve_model_path(args.model),
                         n_slots=max(args.items, 2), rank=args.rank, alpha=args.alpha,
                         max_new_tokens=args.max_new_tokens)
    print(f"loading {cfg.model_id} (items={args.items}) ...", flush=True)
    bb = Backbone.load(cfg)

    episode = make_episode(args.seed, episode_id=0, n_facts=args.items, n_prefs=0,
                           n_lessons=0, n_noise=0, n_negatives=0, with_inertia=False)
    items = episode.probeable_items

    bb.erase_all()
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS)

    per_slot_kl: list[float] = []
    target_ce: list[float] = []
    all_slots = list(range(len(items)))
    for slot, item in enumerate(items):
        def loss_fn(item=item, slot=slot):
            bb.set_read_slots([slot])
            loss = bb.write_loss(item.query, item.value_text)
            # In "sum" mode the anchor is evaluated with every written slot active, which
            # is the original design: the slots are then free to cancel one another, so
            # the term is satisfied collectively and each slot on its own is unguarded.
            bb.set_read_slots(all_slots if args.anchor_mode == "sum" else [slot])
            return loss + args.lambda_kl * bb.kl_to_anchors(anchors)

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        with torch.no_grad():
            bb.set_read_slots([slot])
            target_ce.append(float(bb.write_loss(item.query, item.value_text)))
            per_slot_kl.append(float(bb.kl_to_anchors(anchors)))
        print(f"  slot {slot}: per-slot KL {per_slot_kl[-1]:.3f}  target CE "
              f"{target_ce[-1]:.3f}", flush=True)

    with torch.no_grad():
        bb.set_read_slots(all_slots)
        summed_kl = float(bb.kl_to_anchors(anchors))
        bb.set_read_slots([])
        empty_kl = float(bb.kl_to_anchors(anchors))

    summary = {
        "note": "Per-slot versus summed anchor displacement, measured in both the fixed "
                "configuration (anchor per slot) and the configuration it replaced "
                "(anchor on the sum). With the anchor on the sum the slots cancel one "
                "another, so the term is satisfied collectively while each slot on its own "
                "is unguarded; the loss therefore evaluates it per slot. This run exists "
                "because the numbers in the method section originally came from a design "
                "measurement that was never saved.",
        "anchor_mode": args.anchor_mode,
        "model_path": cfg.model_id,
        "n_items": len(items),
        "lambda_kl": args.lambda_kl,
        "per_slot_kl": per_slot_kl,
        "per_slot_kl_mean": sum(per_slot_kl) / len(per_slot_kl),
        "summed_kl": summed_kl,
        "empty_kl": empty_kl,
        "cancellation_factor": (sum(per_slot_kl) / summed_kl) if summed_kl else float("inf"),
        "target_ce_mean": sum(target_ce) / len(target_ce),
        "guard_share_per_slot": (args.lambda_kl * (sum(per_slot_kl) / len(per_slot_kl))
                                 / (sum(target_ce) / len(target_ce))),
        "elapsed_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  per-slot KL   : {[round(v, 3) for v in per_slot_kl]}")
    print(f"  their sum     : {sum(per_slot_kl):.3f}")
    print(f"  summed-slot KL: {summed_kl:.3f}")
    print(f"  empty-bank KL : {empty_kl:.3f}")
    print(f"  cancellation  : {summary['cancellation_factor']:.1f}x more displacement when "
          f"the slots are read one at a time")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
