"""T2-b -- why does a *second* correctly-routed slot destroy recall?

T2 established the shape: one selected slot recalls about as well as the oracle, two
slots already lose most of the benefit, and the loss is **not** a retrieval failure
(the correct slot is inside the top two). B1 tried to repair it from the write side
with a retention term and did not replicate. So the mechanism itself is still open,
and this measures it instead of guessing.

Four candidate mechanisms, each with a statistic that separates it from the others:

===================  ==============================================================
weight overlap       cosine between the Frobenius-flattened per-slot updates
                     ``Delta W_k`` and ``Delta W_j``. High values mean the two slots
                     encode the same direction and the sum double-counts it.
distractor dominance the runner-up slot's logit perturbation is larger than the
                     correct slot's, so the sum is steered by the wrong memory.
destructive sum      the two perturbations point in opposite directions in logit
                     space (negative cosine), so their sum is smaller than either.
non-linearity        the two-slot logits are not the sum of the single-slot
                     perturbations, i.e. the failure is not simple superposition.
===================  ==============================================================

Everything is measured on the same writes as T2, so the numbers are comparable.

Usage::

    python experiments/t2b_composition_mechanism.py --items 8 --episodes 1
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

from parammem.bench.synthetic import make_episode
from parammem.memory.router import route
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--paraphrase", action="store_true", default=True,
                    help="read with a differently worded query; the two-slot collapse "
                         "only exists in this regime (verbatim queries make routing "
                         "trivial), so it is on by default")
    ap.add_argument("--no-paraphrase", dest="paraphrase", action="store_false")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--stable-rank", action="store_true",
                    help="also compute the per-slot stable rank (an SVD per wrapper per "
                         "slot; slow, and not needed for the mechanism question)")
    ap.add_argument("--out", default="runs/t2b_mechanism.json")
    return ap.parse_args()


def slot_delta(bb: Backbone, slot: int) -> list[torch.Tensor]:
    """Per-wrapper update ``(alpha/r) * B_k A_k`` for one slot."""
    out = []
    for wrapper in bb.wrappers.values():
        sl = wrapper.slot_slice(slot)
        a = wrapper.A[sl, :].detach().float()
        b = wrapper.B[:, sl].detach().float()
        out.append(wrapper.scale * (b @ a))
    return out


def flat_cosine(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    dot = sum(float((x * y).sum()) for x, y in zip(left, right))
    norm_l = sum(float((x * x).sum()) for x in left) ** 0.5
    norm_r = sum(float((y * y).sum()) for y in right) ** 0.5
    return dot / (norm_l * norm_r) if norm_l and norm_r else 0.0


def frobenius(parts: list[torch.Tensor]) -> float:
    return sum(float((x * x).sum()) for x in parts) ** 0.5


def stable_rank(parts: list[torch.Tensor]) -> float:
    """``||W||_F^2 / ||W||_2^2``: how many directions a slot actually uses.

    Costs an SVD per wrapper per slot -- 896 of them for 8 slots on the 0.6B model --
    which dominated the runtime of the first version of this experiment (29 minutes
    without finishing one episode). It is therefore opt-in: the mechanism question is
    answered by the cheap statistics below.
    """
    fro2 = sum(float((x * x).sum()) for x in parts)
    spectral = 0.0
    for x in parts:
        if x.numel():
            spectral = max(spectral, float(torch.linalg.matrix_norm(x, ord=2)) ** 2)
    return fro2 / spectral if spectral else 0.0


def run_episode(bb: Backbone, episode, args: argparse.Namespace) -> dict:
    bb.erase_all()
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS) if args.lambda_kl > 0 else {}

    items = episode.probeable_items
    slot_of: dict[int, int] = {}
    for slot, item in enumerate(items):
        print(f"    write {slot + 1}/{len(items)} ...", flush=True)
    for slot, item in enumerate(items):
        def loss_fn(item=item, slot=slot):
            bb.set_read_slots([slot])
            loss = bb.write_loss(item.query, item.value_text)
            if anchors:
                loss = loss + args.lambda_kl * bb.kl_to_anchors(anchors)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        slot_of[item.item_id] = slot

    print("    measuring slot geometry ...", flush=True)

    written = sorted(slot_of.values())
    deltas = {slot: slot_delta(bb, slot) for slot in written}
    norms = {slot: frobenius(parts) for slot, parts in deltas.items()}
    pair_cosines = [flat_cosine(deltas[a], deltas[b])
                    for a, b in itertools.combinations(written, 2)]

    keys = {slot: bb.query_key(item.query)
            for item, slot in ((i, slot_of[i.item_id]) for i in items)}

    probes = []
    for number, probe in enumerate(episode.probes, start=1):
        print(f"    probe {number}/{len(episode.probes)} ...", flush=True)
        own = slot_of[probe.item_id]
        key = bb.query_key(probe.query)
        ranked = route(key, keys, k=2).slots
        rival = next((s for s in ranked if s != own), None)
        if rival is None:
            continue

        def logits_for(active: list[int]) -> torch.Tensor:
            bb.set_read_slots(active)
            return bb.next_token_logits(probe.query).detach().float().cpu()

        off = logits_for([])
        l_own = logits_for([own])
        l_rival = logits_for([rival])
        l_two = logits_for(ranked)
        l_all = logits_for(written)

        d_own, d_rival = l_own - off, l_rival - off
        predicted = off + d_own + d_rival          # linear superposition
        actual = l_two - off

        def cos(x, y):
            denom = float(x.norm()) * float(y.norm())
            return float((x * y).sum()) / denom if denom else 0.0

        answer_ids = bb._answer_ids(probe.value)
        token = int(answer_ids[0, 0])
        probs = {name: float(torch.softmax(vec, dim=-1)[token])
                 for name, vec in (("off", off), ("own", l_own), ("rival", l_rival),
                                   ("top2", l_two), ("all", l_all))}
        probes.append({
            "item_id": probe.item_id,
            "own_slot": own,
            "rival_slot": rival,
            "ablation_logit_cosine": cos(d_own, d_rival),
            "perturbation_norm_own": float(d_own.norm()),
            "perturbation_norm_rival": float(d_rival.norm()),
            "rival_over_own": (float(d_rival.norm()) / float(d_own.norm())
                               if float(d_own.norm()) else None),
            "linearity_residual": float((actual - predicted).norm()),
            "actual_norm": float(actual.norm()),
            "value_token_probability": probs,
        })

    return {
        "episode_id": episode.episode_id,
        "seed": episode.seed,
        "n_items": len(items),
        "slot_update_norms": {str(k): v for k, v in norms.items()},
        "slot_stable_ranks": ({str(k): stable_rank(v) for k, v in deltas.items()}
                              if args.stable_rank else {}),
        "norms_are_uniform": (max(norms.values()) / min(norms.values())
                              if min(norms.values()) else None),
        "pairwise_update_cosine_mean": (sum(pair_cosines) / len(pair_cosines)
                                        if pair_cosines else None),
        "pairwise_update_cosine_max": max(pair_cosines) if pair_cosines else None,
        "probes": probes,
    }


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = BackboneConfig(model_id=resolve_model_path(args.model),
                         n_slots=max(args.items, 2), rank=args.rank,
                         alpha=args.alpha, max_new_tokens=args.max_new_tokens)
    print(f"loading {cfg.model_id} (items={args.items}, rank={cfg.rank}) ...", flush=True)
    bb = Backbone.load(cfg)

    episodes = []
    for index in range(args.episodes):
        episode = make_episode(
            args.base_seed + index, episode_id=index, n_facts=args.items,
            n_prefs=0, n_lessons=0, n_noise=0, n_negatives=0, with_inertia=False,
            paraphrase_probes=args.paraphrase,
        )
        print(f"  episode {index} ...", flush=True)
        episodes.append(run_episode(bb, episode, args))

    def probe_mean(field: str) -> float | None:
        values = [p[field] for ep in episodes for p in ep["probes"]
                  if p.get(field) is not None]
        return sum(values) / len(values) if values else None

    def episode_mean(field: str) -> float | None:
        values = [ep[field] for ep in episodes if ep.get(field) is not None]
        return sum(values) / len(values) if values else None

    summary = {
        "note": "Mechanism of the two-slot collapse: T2 showed the loss is not a "
                "retrieval failure, B1 failed to repair it from the write side, so "
                "these statistics separate weight overlap, distractor dominance, "
                "destructive summation and non-linearity.",
        "n_probes": sum(len(ep["probes"]) for ep in episodes),
        "pairwise_update_cosine_mean": episode_mean("pairwise_update_cosine_mean"),
        "pairwise_update_cosine_max": episode_mean("pairwise_update_cosine_max"),
        "logit_perturbation_cosine": probe_mean("ablation_logit_cosine"),
        "rival_over_own_perturbation": probe_mean("rival_over_own"),
        "linearity_residual_over_actual": (
            (probe_mean("linearity_residual") / probe_mean("actual_norm"))
            if probe_mean("linearity_residual") and probe_mean("actual_norm") else None),
        "value_token_probability": {
            name: sum(p["value_token_probability"][name]
                      for ep in episodes for p in ep["probes"]) /
                  max(1, sum(len(ep["probes"]) for ep in episodes))
            for name in ("off", "own", "rival", "top2", "all")
        },
        "episodes": episodes,
        "model_path": cfg.model_id,
        "elapsed_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(value: float | None, spec: str = ".3f") -> str:
        return format(value, spec) if isinstance(value, (int, float)) else "n/a"

    print(f"\n  probes: {summary['n_probes']}")
    print(f"  update cosine between written slots      : "
          f"{fmt(summary['pairwise_update_cosine_mean'])} "
          f"(max {fmt(summary['pairwise_update_cosine_max'])})")
    print(f"  logit perturbation cosine (own vs rival) : "
          f"{fmt(summary['logit_perturbation_cosine'])}")
    print(f"  rival/own perturbation magnitude         : "
          f"{fmt(summary['rival_over_own_perturbation'])}")
    print(f"  linearity residual / actual norm         : "
          f"{fmt(summary['linearity_residual_over_actual'])}")
    print("  probability of the correct value token:")
    for name, value in summary["value_token_probability"].items():
        print(f"    {name:<6}{value:.3f}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
