"""T2-e -- does a *learned* key beat the hand-designed ones?

T2-c compared key definitions and found that a model-free lexical overlap and a blend of
it with the hidden-state cosine beat the incumbent last-token key at 1.7B (paired, 9-0,
p=0.004). That leaves the obvious question the batch was asked to answer: would a key
*trained* for the job do better still?

The trap is leakage. The only paraphrase pairs available are the evaluation probes, and
training on them would make the number meaningless. So the projection is trained with
self-supervision on the **canonical** queries alone: random word dropout produces two
views of the same question, and InfoNCE pulls a question's views together while pushing
other questions away. The probes are never touched, and they are different wordings.

Arms: the raw last-token key, pooled mean, lexical TF-IDF, the 50/50 blend, and the
learned projection. Reported per arm: top-1 routing, correctness inside the top two, and
end-to-end exact match through the selected slot, plus a paired test against the raw key.

Usage::

    python experiments/t2e_learned_key.py --items 8 --episodes 4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from parammem.bench import protocol as P
from parammem.bench.synthetic import make_episode
from parammem.eval.paired import sign_test_p
from parammem.memory.router import lexical_scores, route, route_lexical
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ARMS = ("last", "mean", "lexical", "hybrid", "learned", "voting")


def borda(rankings: dict[str, list[int]], k: int = 3) -> list[int]:
    """Aggregate several keys' rankings by Borda count.

    This is the third candidate the batch asked about -- multi-key voting -- as opposed
    to picking one key: every key contributes ``k - position`` points to its top ``k``
    candidates and the highest total wins. It cannot beat the best key by more than the
    best key's own errors if all keys are wrong together, but it can rescue a candidate
    that one key ranks second and another ranks first, which is exactly the failure mode
    a single key cannot fix.
    """
    scores: dict[int, float] = {}
    for ranked in rankings.values():
        for position, slot in enumerate(ranked[:k]):
            scores[slot] = scores.get(slot, 0.0) + (k - position)
    return sorted(scores, key=lambda slot: (-scores[slot], slot))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--views", type=int, default=8,
                    help="augmented views per canonical query used to train the key")
    ap.add_argument("--key-steps", type=int, default=300)
    ap.add_argument("--key-dim", type=int, default=256)
    ap.add_argument("--key-lr", type=float, default=3e-3)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--out", default="runs/t2e_learned_key.json")
    return ap.parse_args()


def hidden(bb: Backbone, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(last-token, mean-pooled) hidden states.

    Delegates to ``Backbone.query_keys`` rather than reimplementing the forward pass:
    that method owns the memory-off convention, and the three earlier versions of this
    script that inline-copied it captured keys with a read mask still active.
    """
    keys = bb.query_keys(text)
    return keys["last"], keys["mean"]


def augment(text: str, rng: random.Random, drop: float) -> str:
    """Word dropout: a paraphrase the model has never been asked about."""
    tokens = P.normalize(text).split()
    kept = [token for token in tokens if rng.random() > drop]
    return " ".join(kept or tokens[:1])


def train_key(views: torch.Tensor, labels: torch.Tensor, args) -> torch.nn.Module:
    """InfoNCE over augmented views of the canonical queries.

    ``views`` is (n_items * n_views, d) and ``labels`` says which question each row came
    from. Nothing here sees a probe: the evaluation wordings are held out entirely.
    """
    projector = torch.nn.Sequential(
        torch.nn.Linear(views.shape[1], args.key_dim),
        torch.nn.GELU(),
        torch.nn.Linear(args.key_dim, args.key_dim),
    ).to(views.device)
    labels = labels.to(views.device)
    optimiser = torch.optim.AdamW(projector.parameters(), lr=args.key_lr)
    for _ in range(args.key_steps):
        optimiser.zero_grad()
        z = F.normalize(projector(views), dim=-1)
        logits = z @ z.t() / 0.1
        logits.fill_diagonal_(-1e4)
        positive = (labels[:, None] == labels[None, :]) & ~torch.eye(
            len(labels), dtype=torch.bool, device=views.device)
        log_prob = F.log_softmax(logits, dim=-1)
        counts = positive.sum(dim=-1).clamp_min(1)
        loss = -(log_prob * positive).sum(dim=-1).div(counts).mean()
        loss.backward()
        optimiser.step()
    projector.eval()
    return projector


def run_episode(bb: Backbone, episode, args) -> dict:
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
                bb.set_read_slots([slot])
                loss = loss + args.lambda_kl * bb.kl_to_anchors(anchors)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        slot_of[item.item_id] = slot

    # Every key is captured with the memory switched off; the write loop above leaves
    # the last slot active, so this reset is load-bearing rather than tidiness.
    bb.set_read_slots([])
    canonical_last: dict[int, torch.Tensor] = {}
    canonical_mean: dict[int, torch.Tensor] = {}
    canonical_text: dict[int, str] = {}
    for item in items:
        last, mean = hidden(bb, item.query)
        slot = slot_of[item.item_id]
        canonical_last[slot] = last / last.norm().clamp_min(1e-6)
        canonical_mean[slot] = mean / mean.norm().clamp_min(1e-6)
        canonical_text[slot] = item.query

    # --- train the key on augmented canonical queries only -------------------------
    rng = random.Random(args.base_seed + episode.episode_id)
    rows, labels = [], []
    for slot, text in sorted(canonical_text.items()):
        for _ in range(args.views):
            last, _mean = hidden(bb, augment(text, rng, args.drop))
            rows.append(last)
            labels.append(slot)
    views = torch.stack(rows)
    labels_t = torch.tensor(labels)
    projector = train_key(views, labels_t, args)

    def learned_key(text: str) -> torch.Tensor:
        with torch.no_grad():
            last, _mean = hidden(bb, text)
            projected = projector(last.unsqueeze(0).to(views.device))[0]
            return F.normalize(projected, dim=-1).cpu()

    tally = {arm: {"top1": 0, "top2_recall": 0, "em": 0} for arm in ARMS}
    per_probe: list[dict] = []
    probes = episode.probes
    # Self-check against the model's own implementation, for the same reason T2-c has
    # one: ``query_key`` handles the memory-off convention internally, so if this script
    # captures keys with the previous arm's read mask still active, the mismatch shows up
    # here instead of silently deflating every model-based column.
    reference = bb.query_key(items[0].query)
    mine_last, _mine_mean = hidden(bb, items[0].query)
    if not torch.allclose(reference.float().cpu(), mine_last.float().cpu(), atol=1e-5):
        raise RuntimeError("key capture disagrees with Backbone.query_key: the memory "
                           "is probably not switched off")
    # Project the canonical keys once: recomputing them per probe would run a model
    # forward per slot per probe for no reason.
    learned_canonical = {slot: learned_key(text)
                         for slot, text in canonical_text.items()}
    for index, probe in enumerate(probes):
        own = slot_of[probe.item_id]
        bb.set_read_slots([])          # keys are a property of the query, not of the bank
        last, mean = hidden(bb, probe.query)
        probe_last = last / last.norm().clamp_min(1e-6)
        probe_mean = mean / mean.norm().clamp_min(1e-6)
        rankings = {
            "last": route(probe_last, canonical_last, k=3).slots,
            "mean": route(probe_mean, canonical_mean, k=3).slots,
            "lexical": route_lexical(probe.query, canonical_text, k=3).slots,
            "learned": route(learned_key(probe.query), learned_canonical, k=3).slots,
        }
        # blend of the raw cosine and the lexical score, as in T2-c
        lex = dict(lexical_scores(probe.query, canonical_text))
        cos = dict(route(probe_last, canonical_last, k=len(canonical_last)).scores)
        blended = sorted(canonical_last,
                         key=lambda s: (-(0.5 * cos.get(s, 0.0) + 0.5 * lex.get(s, 0.0)), s))
        rankings["hybrid"] = blended[:3]
        rankings["voting"] = borda({name: ranked for name, ranked in rankings.items()})

        for arm, ranked in rankings.items():
            tally[arm]["top1"] += int(ranked[0] == own)
            tally[arm]["top2_recall"] += int(own in ranked)
            bb.set_read_slots(ranked[:1])
            answer = bb.answer(probe.query)
            hit = int(P.exact_match(answer, probe.value, probe.aliases))
            tally[arm]["em"] += hit
            per_probe.append({"probe": index, "arm": arm, "top1": int(ranked[0] == own),
                              "em": hit})

    return {"episode_id": episode.episode_id, "seed": episode.seed,
            "n_probes": len(probes), "tally": tally, "per_probe": per_probe}


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = BackboneConfig(model_id=resolve_model_path(args.model),
                         n_slots=max(args.items, 2), rank=args.rank, alpha=args.alpha,
                         max_new_tokens=args.max_new_tokens)
    print(f"loading {cfg.model_id} (items={args.items}, key_dim={args.key_dim}) ...",
          flush=True)
    bb = Backbone.load(cfg)

    episodes = []
    for index in range(args.episodes):
        episode = make_episode(
            args.base_seed + index, episode_id=index, n_facts=args.items,
            n_prefs=0, n_lessons=0, n_noise=0, n_negatives=0, with_inertia=False,
            paraphrase_probes=True,
        )
        record = run_episode(bb, episode, args)
        episodes.append(record)
        print("  episode {}: ".format(index) + "  ".join(
            f"{arm}:{record['tally'][arm]['top1']}/{record['n_probes']}"
            for arm in ARMS), flush=True)

    probes = sum(ep["n_probes"] for ep in episodes)
    totals = {arm: {metric: sum(ep["tally"][arm][metric] for ep in episodes)
                    for metric in ("top1", "top2_recall", "em")} for arm in ARMS}

    def paired(arm: str) -> dict:
        mine = {(ep["episode_id"], row["probe"]): row["em"]
                for ep in episodes for row in ep["per_probe"] if row["arm"] == arm}
        base = {(ep["episode_id"], row["probe"]): row["em"]
                for ep in episodes for row in ep["per_probe"] if row["arm"] == "last"}
        ids = sorted(set(mine) & set(base))
        helped = sum(1 for i in ids if mine[i] and not base[i])
        hurt = sum(1 for i in ids if base[i] and not mine[i])
        return {"helped": helped, "hurt": hurt, "ties": len(ids) - helped - hurt,
                "sign_p": sign_test_p(helped, hurt)}

    summary = {
        "note": "A key trained by self-supervision on augmented canonical queries only; "
                "the paraphrased probes are never seen during training, so the routing "
                "number is held out. Compared against the hand-designed keys of T2-c.",
        "n_probes": probes,
        "n_items": args.items,
        "key_views": args.views,
        "key_steps": args.key_steps,
        "rates": {arm: {metric: value / probes for metric, value in metrics.items()}
                  for arm, metrics in totals.items()},
        "paired_vs_last": {arm: paired(arm) for arm in ARMS if arm != "last"},
        "episodes": episodes,
        "model_path": cfg.model_id,
        "elapsed_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  probes per arm: {probes}")
    print(f"  {'key':<10}{'top1':>8}{'top2rec':>9}{'em':>8}")
    for arm, rates in sorted(summary["rates"].items(), key=lambda kv: -kv[1]["top1"]):
        print(f"  {arm:<10}{rates['top1']:>8.3f}{rates['top2_recall']:>9.3f}"
              f"{rates['em']:>8.3f}")
    for arm, row in summary["paired_vs_last"].items():
        print(f"  paired {arm:<9} helped {row['helped']} / hurt {row['hurt']} "
              f"p={row['sign_p']}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
