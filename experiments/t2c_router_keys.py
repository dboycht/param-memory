"""T2-c -- which *key* should the router use?

The router is the component that degrades with scale (top-1 accuracy 0.925 at 0.6B
against 0.725 at 1.7B), and because the oracle-minus-top-1 gap equals the routing
error exactly, every point of routing accuracy converts one-for-one into recall on the
composition result. So it is worth asking whether the current key is simply the wrong
key, before adding anything trainable.

Key definitions compared, all captured with the memory switched off (the key must be a
property of the query, not of which memories happen to exist):

============  ==========================================================
``last``      final layer, final token -- the current router
``mean``      final layer, averaged over the prompt tokens
``mid``       final token at 60% depth
``shallow``   final token at 30% depth
``lexical``   TF-IDF cosine over the query strings, no model involved
============  ==========================================================

and one selection rule on top of the best key:

``entropy``   take the top two by key similarity, then keep the slot under which the
              query's next-token distribution is *sharper* (lower entropy). A slot
              that recognises the query should be more confident about it; this uses
              no answer information and costs one extra forward pass.

Reported per definition: top-1 accuracy, how often the correct slot is inside the top
two, and the end-to-end exact-match rate of reading through the selected slot -- so a
routing gain that does not become a recall gain is visible as such.

Usage::

    python experiments/t2c_router_keys.py --items 8 --episodes 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
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

KEY_KINDS = ("last", "mean", "mid", "shallow", "lexical")


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
    ap.add_argument("--out", default="runs/t2c_router_keys.json")
    return ap.parse_args()


def hidden_keys(bb: Backbone, query: str) -> dict[str, torch.Tensor]:
    """Every model-based key for one query, from a single forward pass."""
    prompt_ids = bb._ids(bb._chat(query))
    out = bb.model(input_ids=prompt_ids, output_hidden_states=True)
    states = out.hidden_states
    depth = len(states)
    picks = {
        "last": states[-1][0, -1],
        "mean": states[-1][0].mean(dim=0),
        "mid": states[max(0, int(depth * 0.6) - 1)][0, -1],
        "shallow": states[max(0, int(depth * 0.3) - 1)][0, -1],
    }
    return {name: (vec.float() / vec.float().norm().clamp_min(1e-6))
            for name, vec in picks.items()}


def lexical_keys(queries: list[str]) -> dict[int, dict[str, float]]:
    """TF-IDF-weighted bag-of-words vectors, one per query, as a dict of weights."""
    tokenised = [P.normalize(q).split() for q in queries]
    df = Counter()
    for tokens in tokenised:
        df.update(set(tokens))
    total = len(tokenised)
    vectors = []
    for tokens in tokenised:
        counts = Counter(tokens)
        vec = {term: (1 + math.log(count)) * math.log((1 + total) / (1 + df[term]) + 1)
               for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vec.values())) or 1.0
        vectors.append({term: value / norm for term, value in vec.items()})
    return {index: vec for index, vec in enumerate(vectors)}


def lexical_scores(query_vec: dict[str, float],
                   slot_vecs: dict[int, dict[str, float]]) -> list[tuple[int, float]]:
    scores = []
    for slot, vec in slot_vecs.items():
        score = sum(weight * vec.get(term, 0.0) for term, weight in query_vec.items())
        scores.append((slot, score))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores


def entropy_of(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    return float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())


def run_episode(bb: Backbone, episode, args: argparse.Namespace) -> dict:
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
                loss = loss + args.lambda_kl * bb.kl_to_anchors(anchors)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        slot_of[item.item_id] = slot

    written = sorted(slot_of.values())
    slot_hidden = {KEY: {} for KEY in ("last", "mean", "mid", "shallow")}
    for item in items:
        keys = hidden_keys(bb, item.query)
        for name, vec in keys.items():
            slot_hidden[name][slot_of[item.item_id]] = vec
    # IDF has to be estimated over one corpus for slots and probes alike, otherwise
    # the probe vector is not comparable with the slot vectors.
    canonical = [item.query for item in items]
    lex_all = lexical_keys(canonical + [probe.query for probe in episode.probes])
    slot_lexical = {slot_of[item.item_id]: lex_all[index]
                    for index, item in enumerate(items)}

    tally = {KEY: {"top1": 0, "top2_recall": 0, "em": 0} for KEY in KEY_KINDS}
    tally["entropy"] = {"top1": 0, "top2_recall": 0, "em": 0}
    n = 0
    for index, probe in enumerate(episode.probes):
        own = slot_of[probe.item_id]
        n += 1
        probe_hidden = hidden_keys(bb, probe.query)

        rankings: dict[str, list[int]] = {}
        for name in ("last", "mean", "mid", "shallow"):
            rankings[name] = route(probe_hidden[name], slot_hidden[name], k=2).slots
        lex = lexical_scores(lex_all[len(canonical) + index], slot_lexical)
        rankings["lexical"] = [slot for slot, _ in lex[:2]]

        # entropy tie-break over the top two of the current (last-token) key
        top2 = rankings["last"][:2]
        entropies = {}
        for slot in top2:
            bb.set_read_slots([slot])
            entropies[slot] = entropy_of(bb.next_token_logits(probe.query))
        rankings["entropy"] = (sorted(entropies, key=lambda s: entropies[s]) +
                               [s for s in top2 if s not in entropies])[:2]

        for name, ranked in rankings.items():
            tally[name]["top1"] += int(ranked[0] == own)
            tally[name]["top2_recall"] += int(own in ranked)
            bb.set_read_slots([ranked[0]])
            answer = bb.answer(probe.query)
            tally[name]["em"] += int(P.exact_match(answer, probe.value, probe.aliases))

    return {"episode_id": episode.episode_id, "seed": episode.seed, "n_probes": n,
            "tally": tally}


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
            paraphrase_probes=True,
        )
        record = run_episode(bb, episode, args)
        episodes.append(record)
        summary = "  ".join(f"{k}:{v['top1']}/{record['n_probes']}"
                            for k, v in record["tally"].items())
        print(f"  episode {index}: {summary}", flush=True)

    probes = sum(ep["n_probes"] for ep in episodes)
    totals = {name: {metric: sum(ep["tally"][name][metric] for ep in episodes)
                     for metric in ("top1", "top2_recall", "em")}
              for name in episodes[0]["tally"]}
    summary = {
        "note": "Key-definition ablation for the router. Because the oracle-minus-top-1 "
                "gap equals the routing error, a definition that raises top-1 accuracy "
                "raises composition recall by the same amount -- provided the end-to-end "
                "column moves with it.",
        "n_probes": probes,
        "n_items": args.items,
        "episodes": episodes,
        "totals": totals,
        "rates": {name: {metric: value / probes for metric, value in metrics.items()}
                  for name, metrics in totals.items()},
        "model_path": cfg.model_id,
        "elapsed_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  probes: {probes}")
    print(f"  {'key':<10}{'top1':>8}{'top2rec':>9}{'end-to-end EM':>15}")
    for name, rates in sorted(summary["rates"].items(),
                              key=lambda item: -item[1]["top1"]):
        print(f"  {name:<10}{rates['top1']:>8.3f}{rates['top2_recall']:>9.3f}"
              f"{rates['em']:>15.3f}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
