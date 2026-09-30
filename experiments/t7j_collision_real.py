"""T7-j -- does the two-slot collapse happen on real content too?

The composition result is the paper's central claim and it is measured on synthetic
episodes, where the queries are distinctive and the values are short. A reviewer will ask
whether it is a property of the mechanism or of that generator, and the honest answer until
now was that we had not measured it on real benchmark content.

This writes the thirty LongMemEval single-session items into their own slots exactly as the
T7-d diagnostic does, then reads each question back with a growing number of slots active:
the correct one alone, then with one, two, four more, and finally the whole bank. Oracle
indexing is used throughout, so what varies is only how many memories are switched on --
which is precisely the read-time condition the composition claim is about.

Usage::

    python experiments/t7j_collision_real.py --subset 30 --out runs/t7j_collision_real.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from parammem.bench.longmemeval import (contains_answer, load_oracle,
                                        select_single_session, sha256_file)
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"
# How many extra memories are switched on, on top of the correct one.
DOSES = (0, 1, 2, 4)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--subset", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--out", default="runs/t7j_collision_real.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"missing {data_path}", flush=True)
        return 1
    raw = load_oracle(data_path)
    items = select_single_session(raw, args.subset, args.offset)
    print(f"  {len(items)} items; reading with 0, {', '.join(str(d) for d in DOSES[1:])} "
          f"extra slots active", flush=True)

    cfg = BackboneConfig(model_id=resolve_model_path(args.model),
                         n_slots=max(len(items), 2), rank=args.rank, alpha=args.alpha,
                         max_new_tokens=args.max_new_tokens)
    bb = Backbone.load(cfg)
    bb.erase_all()
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS) if args.lambda_kl > 0 else {}

    slot_of: dict[str, int] = {}
    for slot, item in enumerate(items):
        answer_text = str(item["answer"])

        def loss_fn(item=item, slot=slot, answer_text=answer_text):
            bb.set_read_slots([slot])
            loss = bb.write_loss(item["question"], answer_text)
            if anchors:
                bb.set_read_slots([slot])
                loss = loss + args.lambda_kl * bb.kl_to_anchors(anchors)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        slot_of[item["question_id"]] = slot
        if (slot + 1) % 10 == 0:
            print(f"    wrote {slot + 1}/{len(items)}", flush=True)

    written = sorted(slot_of.values())
    hits = {dose: 0 for dose in DOSES}
    hits["all"] = 0
    per_question = []
    for index, item in enumerate(items):
        own = slot_of[item["question_id"]]
        # Deterministic distractors: the slots that follow, wrapping around.
        others = [slot for slot in written if slot != own]
        order = others[index % len(others):] + others[: index % len(others)]
        # The frozen floor first, then the doses.
        bb.set_read_slots([])
        floor_answer = bb.answer(item["question"])
        floor = int(contains_answer(floor_answer, str(item["answer"])))
        row = {"question_id": item["question_id"], "floor": floor}
        for dose in DOSES:
            active = [own] + order[:dose]
            bb.set_read_slots(active)
            answer = bb.answer(item["question"])
            hit = int(contains_answer(answer, str(item["answer"])))
            hits[dose] += hit
            row[f"dose_{dose}"] = hit
        bb.set_read_slots(written)
        all_answer = bb.answer(item["question"])
        hits["all"] += int(contains_answer(all_answer, str(item["answer"])))
        per_question.append(row)
        if (index + 1) % 10 == 0:
            print(f"    read {index + 1}/{len(items)}", flush=True)

    n = len(items)
    rates = {str(dose): hits[dose] / n for dose in DOSES}
    rates["all"] = hits["all"] / n
    summary = {
        "note": "Whether the read-time composition failure also appears on real benchmark "
                "content. Items are written one per slot exactly as in the T7-d "
                "diagnostic, and the only thing that varies between arms is how many "
                "slots are active, so this isolates the read condition the composition "
                "claim is about. Retrieval is oracle-indexed throughout.",
        "model_path": cfg.model_id,
        "data_file": str(data_path),
        "data_sha256": sha256_file(data_path),
        "n_items": n,
        "doses": list(DOSES),
        "rates": rates,
        "floor_rate": sum(row["floor"] for row in per_question) / n,
        "mean_extra_slots": statistics.mean(DOSES),
        "per_question": per_question,
        "elapsed_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  {'active slots':<16}{'containment':>13}")
    print(f"  {'frozen':<16}{summary['floor_rate']:>13.3f}")
    for dose in DOSES:
        print(f"  {1 + dose:<16}{rates[str(dose)]:>13.3f}")
    print(f"  {'all ' + str(n):<16}{rates['all']:>13.3f}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
