"""T7-e -- the baseline that answers "why not just put it in the context?"

The weights run (T7-d) is oracle-indexed: it knows which evidence to store. So the
fair counterpart is not "dump 30 items into the prompt and hope", it is **the same
information delivered through a different medium**. This script measures both:

===================  =========================================================
``frozen``           no context, no memory (floor)
``context_target``   only this item's own (question, answer) in the context --
                     the apples-to-apples counterpart of our oracle indexing
``context_all``      all items' pairs in the context -- the realistic
                     "just stuff everything in" version
``rag_top1``         retrieve 1 item with **the same query-key cosine router the
                     parametric arm uses**, inject its pair
``rag_top3``         retrieve 3, inject them (the context can hold more than one
                     slot -- which is exactly the difference being measured)
===================  =========================================================

The retrieval is deliberately identical to the parametric read path: the only thing
that changes between ``rag_top1`` and the weights arm is *where the memory lives*.

Reported per arm: containment, verbatim rate, and **mean prompt tokens** -- the cost
a context arm pays and a parametric arm does not.

The run aborts if its selected questions differ from the ones recorded by T7-d,
because a baseline on different questions is not a baseline.

Usage::

    python experiments/t7e_context_baselines.py --subset 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from parammem.bench.longmemeval import (
    containment,
    contains_answer,
    load_oracle,
    select_single_session,
    sha256_file,
)
from parammem.bench.protocol import assert_evicted
from parammem.memory.router import route
from parammem.model import Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--subset", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--topk", type=int, nargs="*", default=[1, 3])
    ap.add_argument("--weights-result", default="runs/t7d_longmemeval.json",
                    help="the T7-d report whose question ids this run must match")
    ap.add_argument("--max-new-tokens", type=int, default=96,
                    help="32 truncated long reference answers; raise with T7-d so the "
                         "arms are compared at the same generation budget")
    ap.add_argument("--out", default="runs/t7e_context_baselines.json")
    return ap.parse_args()


def as_context(items: list[dict]) -> str:
    """Serialise memory items the way a context-stuffing system would."""
    lines = ["Known information:"]
    for item in items:
        lines.append(f"- Q: {item['question']}")
        lines.append(f"  A: {item['answer']}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"missing {data_path}", flush=True)
        return 1
    digest = sha256_file(data_path)
    chosen = select_single_session(load_oracle(data_path), args.subset, args.offset)
    ids = [item["question_id"] for item in chosen]

    # The whole point is a comparison, so the question set must be provably the
    # same one the weights run used.
    weights_path = Path(args.weights_result)
    if not weights_path.is_file():
        print(f"missing {weights_path}: run experiments/t7d_longmemeval_pilot.py "
              "first (the baseline must be measured on the same questions)",
              flush=True)
        return 1
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    if weights.get("data_sha256") != digest:
        print(f"ABORT: data sha256 differs from the weights run "
              f"({digest[:16]}… vs {str(weights.get('data_sha256'))[:16]}…)",
              flush=True)
        return 1
    if sorted(weights.get("subset_ids") or []) != sorted(ids):
        print("ABORT: the selected questions differ from the weights run's -- a "
              "baseline on other questions is not a baseline", flush=True)
        return 1
    print(f"question set matches T7-d ({len(ids)} ids, sha256 {digest[:16]}…)",
          flush=True)

    # P1 that applies to *every* arm here: the question text must not already
    # contain the answer, otherwise a correct answer is not evidence of anything.
    # Note that P1's "eviction" form is meaningless for the context arms by
    # construction: putting the answer in the context is exactly what they do, and
    # it is why their scores are not attributable to anything but the context.
    leaking = [item["question_id"] for item in chosen
               if not assert_evicted(item["question"], str(item["answer"])).evicted]
    print(f"P1 (question text carries no answer): "
          f"{len(chosen) - len(leaking)}/{len(chosen)} clean", flush=True)

    cfg = BackboneConfig(
        model_id=resolve_model_path(args.model), n_slots=4,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"loading {cfg.model_id} ...", flush=True)
    bb = Backbone.load(cfg)
    bb.erase_all()
    bb.set_read_slots([])          # this script never writes: no memory anywhere

    # Routing keys are indexed by position in `chosen`; the router needs integer
    # slot ids, so map question_id -> index and back rather than passing ids through.
    key_index = {item["question_id"]: i for i, item in enumerate(chosen)}
    keys = {index: bb.query_key(item["question"]) for index, item in enumerate(chosen)}

    arm_names = ["frozen", "context_target", "context_all"] + \
                [f"rag_top{k}" for k in args.topk]
    arms: dict[str, list[dict]] = {arm: [] for arm in arm_names}
    t0 = time.perf_counter()

    for index, item in enumerate(chosen, start=1):
        qid, gold = item["question_id"], str(item["answer"])
        contexts: dict[str, str] = {
            "frozen": "",
            "context_target": as_context([item]),
            "context_all": as_context(chosen),
        }
        for k in args.topk:
            decision = route(keys[key_index[qid]], keys, k=k)
            contexts[f"rag_top{k}"] = as_context([chosen[i] for i in decision.slots])

        for arm, context in contexts.items():
            prediction = bb.answer(item["question"], context=context)
            arms[arm].append({
                "question_id": qid,
                "gold": gold,
                "answer": prediction,
                "containment": containment(prediction, gold),
                "verbatim": bool(contains_answer(prediction, gold)),
                "prompt_tokens": bb.prompt_tokens(item["question"], context=context),
            })
        if index % 5 == 0:
            print(f"  {index}/{len(chosen)} questions done", flush=True)

    summary = {
        "protocol": "T7-e: external baselines on the T7-d question set",
        "data_file": str(data_path), "data_sha256": digest,
        "subset_ids": ids, "n_subset": len(ids),
        "p1_leaking_ids": leaking,
        "weights_result": str(weights_path),
        "topk": args.topk,
        "arms": {
            arm: {
                "containment": statistics.mean(r["containment"] for r in rows),
                "verbatim": statistics.mean(1.0 if r["verbatim"] else 0.0
                                            for r in rows),
                "mean_prompt_tokens": statistics.mean(r["prompt_tokens"]
                                                      for r in rows),
            }
            for arm, rows in arms.items()
        },
        "elapsed_s": time.perf_counter() - t0,
        "rows": arms,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    weights_c = weights.get("containment") or {}
    print(f"\n{'arm':<18}{'containment':>13}{'verbatim':>10}{'prompt tok':>12}",
          flush=True)
    print(f"{'weights (T7-d)':<18}{weights_c.get('mem_on', float('nan')):>13.3f}"
          f"{weights_c.get('mem_on_full_match', float('nan')):>10.3f}{0:>12}", flush=True)
    for arm in arm_names:
        a = summary["arms"][arm]
        print(f"{arm:<18}{a['containment']:>13.3f}{a['verbatim']:>10.3f}"
              f"{a['mean_prompt_tokens']:>12.0f}", flush=True)
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
