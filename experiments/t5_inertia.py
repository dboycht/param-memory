"""T5 -- context inertia: does a memory drag the old topic along?

The question the original idea raised: a memory that lives in the *context* is
indistinguishable from "what is happening right now", so after a topic switch the
model keeps chewing on it. A memory that lives in the *weights* is supposed to be
recallable without being present.

Design -- one wrong premise on topic A, then a switch to topic B, four arms:

============================  ==================================================
``no_memory``                 premise is nowhere; only the new question (floor)
``context_memory``            premise sits in the working context + new question
``param_memory``              premise written into a slot, then erased from the
                              context; the slot stays ACTIVE while answering
``param_off``                 same but the slot is erased again (control: must
                              collapse onto ``no_memory`` bit-for-bit)
============================  ==================================================

Metrics per arm (lower is better for the first two):

* **I2 inheritance** -- the answer asserts topic A's premise (e.g. says "left"
  when the question was about a country we were never told about)
* **I1 residue** -- topic A's own words (the country, the unit) leak into the answer
* **withholding** -- the model says it does not know (the correct behaviour here)

**Prediction, written down before running**: inheritance should be highest for
``context_memory`` and lower for ``param_memory``; ``no_memory`` is the floor. If
``param_memory`` lands at or above ``context_memory``, the claim "parameters avoid
context inertia" is FALSE and must be reported as such.

Usage::

    python experiments/t5_inertia.py --episodes 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from parammem.bench import protocol as P
from parammem.bench.synthetic import InertiaBlock, make_episode
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ARMS = ("no_memory", "context_memory", "param_memory", "param_off")

WITHHOLD_MARKERS = (
    "not specified", "not provided", "no information", "don't know", "do not know",
    "cannot", "can't", "unknown", "not sure", "no data", "not mentioned",
    "unable to", "i don't have",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--candidates", type=int, default=30,
                    help="scenarios generated for the floor screen")
    ap.add_argument("--episodes", type=int, default=12,
                    help="max scenarios to run the four arms on")
    ap.add_argument("--screen-floor", dest="screen_floor", action="store_true",
                    default=True,
                    help="drop scenarios the model already 'inherits' with no "
                         "premise at all (default on)")
    ap.add_argument("--no-screen-floor", dest="screen_floor", action="store_false")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", default="runs/t5_inertia.json")
    return ap.parse_args()


def score(answer: str, inertia: InertiaBlock) -> dict[str, int]:
    low = answer.lower()
    return {
        "inherited": int(any(k.lower() in low for k in inertia.inherited_keywords)),
        "residue": int(any(k.lower() in low for k in inertia.old_topic_keywords)),
        "withholds": int(any(m in low for m in WITHHOLD_MARKERS)),
    }


def run_episode(bb: Backbone, episode, args: argparse.Namespace) -> dict:
    inertia = episode.inertia
    if inertia is None:
        raise ValueError("episode has no inertia block")

    bb.erase_all()
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS)

    # ---- arm 1: premise nowhere ------------------------------------------
    bb.set_read_slots([])
    answer_no_memory = bb.answer(inertia.new_task)

    # ---- arm 2: premise in the working context ---------------------------
    bb.set_read_slots([])
    answer_context = bb.answer(f"{inertia.premise}\n\n{inertia.new_task}")

    # ---- arm 3: premise written into one slot, read with it ACTIVE -------
    slot = 0

    def loss_fn(slot=slot):
        bb.set_read_slots([slot])
        loss = bb.write_loss(inertia.premise_query, inertia.premise_answer)
        bb.set_read_slots([slot])
        return loss + args.lambda_kl * bb.kl_to_anchors(anchors)

    report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
    if not report.frozen_ok:
        raise RuntimeError("write isolation violated while writing the premise")

    bb.set_read_slots([slot])
    premise_answer_text = bb.answer(inertia.premise_query)
    premise_recall = P.exact_match(premise_answer_text, inertia.premise_answer)
    answer_param = bb.answer(inertia.new_task)

    # ---- arm 4: control, the slot erased again ---------------------------
    bb.erase_slots([slot])
    bb.set_read_slots([])
    answer_param_off = bb.answer(inertia.new_task)

    answers = {
        "no_memory": answer_no_memory,
        "context_memory": answer_context,
        "param_memory": answer_param,
        "param_off": answer_param_off,
    }
    return {
        "seed": episode.seed,
        "premise": inertia.premise,
        "new_task": inertia.new_task,
        "inherited_keywords": list(inertia.inherited_keywords),
        "premise_recall": int(premise_recall),
        "premise_answer_text": premise_answer_text,
        "write_seconds": report.seconds,
        "answers": answers,
        "scores": {arm: score(text, inertia) for arm, text in answers.items()},
        "control_identical": int(answer_param_off == answer_no_memory),
    }


def main() -> int:
    args = parse_args()
    cfg = BackboneConfig(
        model_id=resolve_model_path(args.model), n_slots=4, rank=args.rank,
        alpha=args.alpha, max_new_tokens=args.max_new_tokens,
    )
    print(f"loading {cfg.model_id} ...", flush=True)
    bb = Backbone.load(cfg)
    print(f"modules={len(bb.wrappers)} memory_params={bb.n_memory_parameters/1e6:.2f}M",
          flush=True)

    # ---- phase A: floor screen ------------------------------------------
    # Cheap (no writes): ask the new question with no premise anywhere. If the
    # model already produces the "inherited" answer from its own prior, the
    # scenario cannot measure inertia at all. The first T5 run was half-invalid
    # for exactly this reason (the road-side case: "left" is the model's default).
    candidates = [
        make_episode(args.base_seed + i, episode_id=i, n_facts=0, n_prefs=0,
                     n_lessons=0, n_noise=0, n_negatives=0, with_inertia=True)
        for i in range(args.candidates)
    ]
    bb.set_read_slots([])
    floor = []
    for episode in candidates:
        answer = bb.answer(episode.inertia.new_task)
        floor.append({"answer": answer, **score(answer, episode.inertia)})

    keep = [
        i for i, f in enumerate(floor)
        if (not args.screen_floor) or (f["inherited"] == 0 and f["residue"] == 0)
    ]
    dropped = [(i, floor[i]) for i in range(len(candidates)) if i not in set(keep)]
    print(f"\nphase A (floor screen): {len(keep)}/{len(candidates)} scenarios kept",
          flush=True)
    for i, f in dropped[:10]:
        print(f"  dropped seed {candidates[i].seed}: with no premise at all the "
              f"model answers {f['answer']!r}", flush=True)
    if len(dropped) > 10:
        print(f"  ... and {len(dropped) - 10} more", flush=True)

    survivors = [candidates[i] for i in keep[: args.episodes]]
    if not survivors:
        print("\nno clean scenarios survived the floor screen: every scenario is "
              "answerable from the model's prior, so nothing here can measure "
              "inertia. Report that instead of a number.", flush=True)
        return 1

    # ---- phase B: the four arms on clean scenarios ------------------------
    records = []
    t0 = time.perf_counter()
    for episode in survivors:
        record = run_episode(bb, episode, args)
        records.append(record)
        print(f"\n--- seed {record['seed']} ------------------------------------",
              flush=True)
        print(f"  premise   : {record['premise']}", flush=True)
        print(f"  new task  : {record['new_task']}", flush=True)
        print(f"  premise written+recallable: {bool(record['premise_recall'])} "
              f"({record['premise_answer_text']!r})", flush=True)
        for arm in ARMS:
            s = record["scores"][arm]
            print(f"  {arm:<15} inherited={s['inherited']} residue={s['residue']} "
                  f"withholds={s['withholds']}  {record['answers'][arm]!r}", flush=True)

    summary = {
        "config": vars(args),
        "model_path": bb.resolved_path,
        "n_episodes": len(records),
        "floor_screen": {
            "candidates": len(candidates),
            "kept": len(keep),
            "dropped": len(dropped),
            "dropped_detail": [
                {"seed": candidates[i].seed, "answer": f["answer"],
                 "premise": candidates[i].inertia.premise}
                for i, f in dropped
            ],
        },
        "elapsed_s": time.perf_counter() - t0,
        "arms": {
            arm: {
                "inheritance_rate": sum(r["scores"][arm]["inherited"] for r in records)
                                    / max(1, len(records)),
                "residue_rate": sum(r["scores"][arm]["residue"] for r in records)
                                / max(1, len(records)),
                "withholding_rate": sum(r["scores"][arm]["withholds"] for r in records)
                                    / max(1, len(records)),
            }
            for arm in ARMS
        },
        "premise_recall_rate": sum(r["premise_recall"] for r in records)
                               / max(1, len(records)),
        "control_identical_rate": sum(r["control_identical"] for r in records)
                                  / max(1, len(records)),
        "records": records,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    print(f"\nepisodes={summary['n_episodes']}  premise written+recallable "
          f"{summary['premise_recall_rate']:.2f}", flush=True)
    print(f"{'arm':<16}{'inheritance':>13}{'residue':>10}{'withholding':>13}", flush=True)
    for arm in ARMS:
        a = summary["arms"][arm]
        print(f"{arm:<16}{a['inheritance_rate']:>13.2f}{a['residue_rate']:>10.2f}"
              f"{a['withholding_rate']:>13.2f}", flush=True)
    print(f"\nparam_off identical to no_memory: "
          f"{summary['control_identical_rate']:.2f}", flush=True)
    print(f"report written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
