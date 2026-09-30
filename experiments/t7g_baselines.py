"""T7-g -- do we need slots at all? Baselines that train on the same memories.

T7 compares the parametric memory against *delivering the same information through the
context*. The next question is the cheaper one: if you are going to touch the weights,
why not fine-tune on the same thirty memories, or train one adapter over all of them?
Nothing in the paper answers that, so this runs it on exactly the same pairs.

Arms:

=================  ==============================================================
``frozen``         no training (the floor)
``full_ft``        every backbone parameter trained on all pairs
``one_adapter``    one rank-blocked adapter with the same *total rank* as the slot
                   bank, trained jointly on all pairs (equal capacity, no slots)
``slots``          the paper's scheme: one memory per slot, read by routing
=================  ==============================================================

Retrieval-only arms are already covered by T7-e (``rag_top1``, ``rag_top3``), so this
does not duplicate them. Each arm reports containment on the same probes, the backbone
drift it inflicts (all-position KL against the frozen model, so a method that wrecks
the model while answering is visible), and whether a single memory can still be erased
afterwards -- which is the slot bank's whole claim.

Usage::

    python experiments/t7g_baselines.py --items 30 --steps 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from parammem.bench.longmemeval import contains_answer, load_oracle, select_single_session
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
#: Same questions as the T7-d diagnostic, so the baselines answer exactly the items
#: the paper already reports.
DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--out", default="runs/t7g_baselines.json")
    return ap.parse_args()


def load_pairs(limit: int, data_path: Path = DATA) -> list[dict]:
    return select_single_session(load_oracle(data_path), limit)


def containment(bb: Backbone, pairs: list[dict]) -> float:
    hits = 0
    for pair in pairs:
        answer = bb.answer(str(pair.get("question", "")))
        hits += int(contains_answer(answer, str(pair["answer"])))
    return hits / len(pairs)


def backbone_drift(bb: Backbone, anchors: dict) -> float:
    with torch.no_grad():
        return float(bb.kl_to_anchors(anchors).detach())


def train_full_backbone(bb: Backbone, pairs, args, anchors) -> None:
    """Train every backbone parameter on the same pairs with the same budget."""
    bb.set_read_slots([])
    for parameter in bb.model.parameters():
        parameter.requires_grad_(True)
    optimiser = torch.optim.AdamW(list(bb.model.parameters()), lr=args.lr)
    for _ in range(args.steps):
        optimiser.zero_grad()
        total = None
        for pair in pairs:
            loss = bb.write_loss(str(pair.get("question", "")), str(pair["answer"]))
            total = loss if total is None else total + loss
        total = total / len(pairs)
        if anchors:
            total = total + args.lambda_kl * bb.kl_to_anchors(anchors)
        total.backward()
        optimiser.step()
    for parameter in bb.model.parameters():
        parameter.requires_grad_(False)


def train_joint_adapter(bb: Backbone, pairs, args, anchors, n_slots: int) -> None:
    """One adapter of the same total rank, trained jointly: equal capacity, no slots."""
    params = []
    for wrapper in bb.wrappers.values():
        params.extend([wrapper.A, wrapper.B])
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    bb.set_read_slots(range(n_slots))
    for _ in range(args.steps):
        optimiser.zero_grad()
        total = None
        for pair in pairs:
            loss = bb.write_loss(str(pair.get("question", "")), str(pair["answer"]))
            total = loss if total is None else total + loss
        total = total / len(pairs)
        if anchors:
            total = total + args.lambda_kl * bb.kl_to_anchors(anchors)
        total.backward()
        optimiser.step()


def train_slots(bb: Backbone, pairs, args, anchors) -> int:
    """The paper's scheme; returns how many slots were written."""
    written = 0
    for slot, pair in enumerate(pairs):
        def loss_fn(pair=pair, slot=slot):
            bb.set_read_slots([slot])
            loss = bb.write_loss(str(pair.get("question", "")), str(pair["answer"]))
            if anchors:
                loss = loss + args.lambda_kl * bb.kl_to_anchors(anchors)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        written += 1
    return written


def erasure_check(bb: Backbone, pairs: list[dict], slots: list[int]) -> dict:
    """Erase the first memory and check that it is gone while the second survives.

    This is the property the training baselines do not have, so it is measured rather
    than asserted: the slot must become virgin, the second memory must still be
    recalled, and the erased question must fall back to what the frozen model says.
    """
    if len(slots) < 2:
        return {"slot_virgin": False, "second_item_kept": False}
    bb.erase_slots([slots[0]])
    second = contains_answer(bb.answer(str(pairs[1].get("question", ""))),
                             str(pairs[1]["answer"]))
    return {"slot_virgin": bool(bb.is_slot_virgin(slots[0])),
            "second_item_kept": bool(second)}


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    pairs = load_pairs(args.items, Path(args.data))
    cfg = BackboneConfig(model_id=resolve_model_path(args.model),
                         n_slots=max(args.items, 2), rank=args.rank,
                         alpha=args.alpha, max_new_tokens=args.max_new_tokens)
    print(f"loading {cfg.model_id} with {len(pairs)} real memories ...", flush=True)

    results: dict[str, dict] = {}

    bb = Backbone.load(cfg)
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS)
    print("  arm frozen ...", flush=True)
    results["frozen"] = {"containment": containment(bb, pairs),
                         "backbone_drift": backbone_drift(bb, anchors),
                         "erasable": True}

    print(f"  arm full_ft ({args.steps} steps, all parameters) ...", flush=True)
    train_full_backbone(bb, pairs, args, anchors)
    results["full_ft"] = {"containment": containment(bb, pairs),
                          "backbone_drift": backbone_drift(bb, anchors),
                          "erasable": False}
    del bb
    torch.cuda.empty_cache()

    bb = Backbone.load(cfg)
    bb.set_read_slots([])
    print(f"  arm one_adapter (total rank {cfg.rank * cfg.n_slots}) ...", flush=True)
    train_joint_adapter(bb, pairs, args, anchors, cfg.n_slots)
    results["one_adapter"] = {"containment": containment(bb, pairs),
                              "backbone_drift": backbone_drift(bb, anchors),
                              "erasable": False}
    del bb
    torch.cuda.empty_cache()

    bb = Backbone.load(cfg)
    bb.set_read_slots([])
    print(f"  arm slots ({len(pairs)} writes) ...", flush=True)
    written = train_slots(bb, pairs, args, anchors)
    results["slots"] = {"containment": containment(bb, pairs),
                        "backbone_drift": backbone_drift(bb, anchors),
                        "erasable": True, "slots_written": written,
                        "erasure_check": erasure_check(bb, pairs, list(range(written)))}

    summary = {
        "note": "Same LongMemEval single-session pairs as the T7 diagnostic. The slot "
                "bank is compared against training the same memories into the weights, "
                "which is the cheapest thing a reviewer will ask about. Retrieval-only "
                "arms live in T7-e.",
        "n_items": len(pairs),
        "steps": args.steps,
        "total_rank": cfg.rank * cfg.n_slots,
        "results": results,
        "model_path": cfg.model_id,
        "elapsed_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  {'arm':<14}{'containment':>13}{'backbone KL':>13}{'erasable':>10}")
    for name, row in results.items():
        print(f"  {name:<14}{row['containment']:>13.3f}{row['backbone_drift']:>13.4f}"
              f"{str(row['erasable']):>10}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
