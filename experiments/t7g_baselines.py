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

All arms see the **same number of item-gradient-steps** (``items`` x ``steps``). That is
not equal compute: the fine-tune arm updates every parameter on every step, so it
receives *more* compute than the adapter arms rather than less, which makes any result
in the paper's favour a conservative one and must not be described as an equal-FLOP
comparison.

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
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate for the adapter arms, which is the rate the "
                         "slot writes use in every other experiment")
    ap.add_argument("--lr-full", type=float, default=1e-4,
                    help="learning rate for the full fine-tune arm. A full model needs "
                         "a much smaller step than an adapter; running it at 1e-3 "
                         "destroys the backbone in two steps (measured: all-position KL "
                         "12.8) and would make the comparison a straw man")
    ap.add_argument("--optimiser-full", default="sgd",
                    choices=("sgd", "adamw"),
                    help="AdamW keeps two moment estimates per parameter, which on a "
                         "0.6B backbone puts this card at 7.85 of 8 GB and thrashes "
                         "(measured: one arm did not finish in nineteen minutes). SGD "
                         "with momentum keeps one buffer and is the default for that "
                         "reason; it is a hardware constraint, and it is reported with "
                         "the result rather than left implicit")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--out", default="runs/t7g_baselines.json")
    return ap.parse_args()


def load_pairs(limit: int, data_path: Path = DATA) -> list[dict]:
    return select_single_session(load_oracle(data_path), limit)


def containment(bb: Backbone, pairs: list[dict]) -> float:
    """Containment with whatever read mask is active.

    For the training baselines that is the whole point: the fine-tune arm has no slots
    (mask empty) and the joint-adapter arm is one adapter that is always on (mask full).
    The slot bank cannot be measured this way -- see ``containment_routed``.
    """
    hits = 0
    for pair in pairs:
        answer = bb.answer(str(pair.get("question", "")))
        hits += int(contains_answer(answer, str(pair["answer"])))
    return hits / len(pairs)


def containment_routed(bb: Backbone, pairs: list[dict]) -> float:
    """Containment for the slot bank: read each item through its own slot.

    The first version measured this arm with ``containment`` and inherited whatever read
    mask the last write had left active -- that is, one slot for thirty questions -- so
    the number was meaningless rather than merely optimistic. Oracle routing is the same
    convention the T7-d diagnostic uses, which is what makes the two comparable.
    """
    hits = 0
    for slot, pair in enumerate(pairs):
        bb.set_read_slots([slot])
        answer = bb.answer(str(pair.get("question", "")))
        hits += int(contains_answer(answer, str(pair["answer"])))
    return hits / len(pairs)


def backbone_drift(bb: Backbone, anchors: dict) -> float:
    with torch.no_grad():
        return float(bb.kl_to_anchors(anchors).detach())


def _accumulate(bb: Backbone, pairs, args, anchors, optimiser) -> None:
    """One optimisation step: accumulate per-pair gradients, then step.

    ``total = loss + loss + ...`` over thirty pairs keeps thirty autograd graphs alive
    at once. On this card that pushed the run to 7.85 of 8 GB and the driver started
    paging, so the full fine-tune arm did not finish in thirty-eight minutes. Summing
    the losses is mathematically the same but calls backward on each term immediately,
    which frees the graph as it goes and leaves one alive at a time.
    """
    optimiser.zero_grad()
    for pair in pairs:
        loss = bb.write_loss(str(pair.get("question", "")), str(pair["answer"]))
        (loss / len(pairs)).backward()
    if anchors:
        (args.lambda_kl * bb.kl_to_anchors(anchors)).backward()
    optimiser.step()


def train_full_backbone(bb: Backbone, pairs, args, anchors) -> None:
    """Train every backbone parameter on the same pairs with the same budget."""
    bb.set_read_slots([])
    for parameter in bb.model.parameters():
        parameter.requires_grad_(True)
    params = list(bb.model.parameters())
    if args.optimiser_full == "adamw":
        optimiser = torch.optim.AdamW(params, lr=args.lr_full)
    else:
        optimiser = torch.optim.SGD(params, lr=args.lr_full, momentum=0.9)
    for _ in range(args.steps):
        _accumulate(bb, pairs, args, anchors, optimiser)
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
        _accumulate(bb, pairs, args, anchors, optimiser)


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
    """Erase one memory and check that it is gone while its neighbours are untouched.

    This is the property the training baselines do not have, so it is measured rather
    than asserted. The read mask is set explicitly at every step: the first version left
    whatever ``containment_routed`` had last activated, which made the neighbour look
    lost and would have put a false negative into the report for the sake of one missing
    line.
    """
    if len(slots) < 3:
        return {"slot_virgin": False, "neighbour_kept": False, "erased_gone": False}
    victim, neighbour = slots[0], slots[1]
    bb.erase_slots([victim])
    bb.set_read_slots([neighbour])
    neighbour_kept = contains_answer(bb.answer(str(pairs[1].get("question", ""))),
                                     str(pairs[1]["answer"]))
    bb.set_read_slots([victim])
    erased_answer = bb.answer(str(pairs[0].get("question", "")))
    return {"slot_virgin": bool(bb.is_slot_virgin(victim)),
            "neighbour_kept": bool(neighbour_kept),
            "erased_gone": not contains_answer(erased_answer,
                                               str(pairs[0]["answer"]))}


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    pairs = load_pairs(args.items, Path(args.data))
    cfg = BackboneConfig(model_id=resolve_model_path(args.model),
                         n_slots=max(args.items, 2), rank=args.rank,
                         alpha=args.alpha, max_new_tokens=args.max_new_tokens)
    print(f"loading {cfg.model_id} with {len(pairs)} real memories ...", flush=True)

    results: dict[str, dict] = {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(stage: str) -> None:
        """Write after every arm.

        The first version of this script wrote only at the end, and when the run was
        interrupted the completed arms were lost with it -- full fine-tune had already
        finished and its numbers existed only in memory. A long multi-arm experiment has
        to be resumable by inspection, so each arm lands on disk as soon as it exists.
        """
        out.write_text(json.dumps({
            "note": "Same LongMemEval single-session pairs as the T7 diagnostic. The slot "
                    "bank is compared against training the same memories into the weights, "
                    "which is the cheapest thing a reviewer will ask about. Retrieval-only "
                    "arms live in T7-e.",
            "status": stage,
            "complete": stage == "done",
            "n_items": len(pairs),
            "steps": args.steps,
            "total_rank": cfg.rank * cfg.n_slots,
            "optimiser_full": args.optimiser_full,
            "lr_full": args.lr_full,
            "results": results,
            "model_path": cfg.model_id,
            "elapsed_s": time.perf_counter() - t0,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    bb = Backbone.load(cfg)
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS)
    print("  arm frozen ...", flush=True)
    results["frozen"] = {"containment": containment(bb, pairs),
                         "backbone_drift": backbone_drift(bb, anchors),
                         "erasable": True}
    save("frozen")

    print(f"  arm full_ft ({args.steps} steps at lr={args.lr_full}, all parameters) ...",
          flush=True)
    train_full_backbone(bb, pairs, args, anchors)
    results["full_ft"] = {"containment": containment(bb, pairs),
                          "backbone_drift": backbone_drift(bb, anchors),
                          "erasable": False}
    save("full_ft")
    del bb
    torch.cuda.empty_cache()

    bb = Backbone.load(cfg)
    bb.set_read_slots([])
    print(f"  arm one_adapter (total rank {cfg.rank * cfg.n_slots}) ...", flush=True)
    train_joint_adapter(bb, pairs, args, anchors, cfg.n_slots)
    results["one_adapter"] = {"containment": containment(bb, pairs),
                              "backbone_drift": backbone_drift(bb, anchors),
                              "erasable": False}
    save("one_adapter")
    del bb
    torch.cuda.empty_cache()

    bb = Backbone.load(cfg)
    bb.set_read_slots([])
    print(f"  arm slots ({len(pairs)} writes) ...", flush=True)
    written = train_slots(bb, pairs, args, anchors)
    results["slots"] = {"containment": containment_routed(bb, pairs),
                        "backbone_drift": backbone_drift(bb, anchors),
                        "erasable": True, "slots_written": written,
                        "erasure_check": erasure_check(bb, pairs, list(range(written)))}
    save("done")

    print(f"\n  {'arm':<14}{'containment':>13}{'backbone KL':>13}{'erasable':>10}")
    for name, row in results.items():
        print(f"  {name:<14}{row['containment']:>13.3f}{row['backbone_drift']:>13.4f}"
              f"{str(row['erasable']):>10}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
