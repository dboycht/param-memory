"""T3 -- which memories deserve a slot? Testing write criteria without labels.

The stream deliberately mixes two classes:

* **unknown** -- fictional facts the frozen model cannot know (real memory targets)
* **known**   -- real-world facts the frozen model already answers correctly
  (verified by asking it in phase 0; anything it gets wrong is dropped)

Writing a ``known`` item is pure waste: it costs a gradient write, occupies a
slot, and moves the frozen backbone for information the model already had. A
criterion has to separate the two classes **without labels**.

Policies compared (see ``parammem.memory.criteria``):

===========  ==============================================================
``always``    reference point: write everything writeable
``surprise``  write iff the pre-write loss exceeds a threshold -- label-free,
              needs no cooperation from the user
``explicit``  write iff the statement contains an explicit request ("remember
              this") -- precise, but only when the user actually asks
===========  ==============================================================

The surprise threshold is chosen **label-free** (median of the observed scores),
and the per-class score distributions are reported so the reader can judge how
separable the classes really are instead of taking a threshold on faith.

Usage::

    python experiments/t3_write_policy.py --facts 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# A Windows console defaults to a legacy code page (GBK on this machine), and one
# subscript character in a model answer ("H₂O") is enough to abort the whole run
# with UnicodeEncodeError. Reconfigure here rather than depend on the caller
# remembering PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from parammem.bench import protocol as P
from parammem.bench.known import KNOWN_PAIRS, KnownPair
from parammem.bench.synthetic import make_episode
from parammem.memory.criteria import (
    always_decision,
    explicit_decision,
    selfcheck_decision,
    surprise_decision,
)
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

POLICIES = ("always", "surprise", "selfcheck", "explicit")


@dataclass
class Candidate:
    """One thing that could be written, from either class."""

    key: str
    kind: str  # "unknown" | "known"
    statement: str
    query: str
    value: str
    aliases: tuple[str, ...] = ()
    probe_query: str = ""
    item_id: int | None = None
    loss: float = float("nan")

    @property
    def read_query(self) -> str:
        return self.probe_query or self.query


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--facts", type=int, default=5, help="unknown (fictional) items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--no-paraphrase", dest="paraphrase", action="store_false",
                    default=True)
    ap.add_argument("--out", default="runs/t3_write_policy.json")
    return ap.parse_args()


# ------------------------------------------------------------------- phase 0


def verify_known(bb: Backbone) -> tuple[list[KnownPair], list[dict]]:
    """Keep only the 'known' pairs the frozen model actually answers correctly.

    Assuming a 0.6B model knows a fact is exactly the kind of assumption that
    quietly invalidates an experiment, so every pair is checked by asking.
    """
    bb.set_read_slots([])
    kept, dropped = [], []
    for pair in KNOWN_PAIRS:
        answer = bb.answer(pair.query)
        ok = pair.expected_fragment in answer.lower()
        (kept if ok else dropped).append(
            pair if ok else {"query": pair.query, "answer": answer}
        )
    return kept, dropped


def surprise_scores(bb: Backbone, candidates: list[Candidate]) -> dict[str, float]:
    """Pre-write loss per candidate, measured with the memory switched off.

    Policy-independent by construction (the side path is off), so it is computed
    once and reused for every policy.
    """
    bb.set_read_slots([])
    scores = {}
    with torch.no_grad():
        for cand in candidates:
            scores[cand.key] = float(bb.write_loss(cand.query, cand.value))
    return scores


# ------------------------------------------------------------------- phase 1


def selfcheck_answers(bb: Backbone, candidates: list[Candidate]) -> dict[str, str]:
    """Ask the frozen model each question; policy-independent, so measured once."""
    bb.set_read_slots([])
    return {cand.key: bb.answer(cand.query) for cand in candidates}


def run_policy(
    bb: Backbone,
    candidates: list[Candidate],
    args: argparse.Namespace,
    policy: str,
    tau: float,
    selfcheck: dict[str, str] | None = None,
) -> dict:
    """Replay the stream under one write criterion."""
    bb.erase_all()
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS)
    frozen_anchors = {q: bb.answer(q) for q in GENERIC_ANCHORS}

    written: list[tuple[int, Candidate]] = []
    skipped: list[str] = []
    decisions: list[dict] = []
    write_seconds = 0.0

    for cand in candidates:
        if policy == "always":
            decision = always_decision(score=cand.loss)
        elif policy == "surprise":
            decision = surprise_decision(cand.loss, tau)
        elif policy == "selfcheck":
            decision = selfcheck_decision(
                (selfcheck or {}).get(cand.key, ""), cand.value, cand.aliases
            )
        elif policy == "explicit":
            decision = explicit_decision(cand.statement)
        else:
            raise ValueError(policy)

        decisions.append(
            {"key": cand.key, "kind": cand.kind, "write": decision.should_write,
             "reason": decision.reason, "loss": cand.loss}
        )
        if not decision.should_write:
            skipped.append(cand.key)
            continue

        slot = len(written)
        if slot >= bb.cfg.n_slots:
            raise RuntimeError("stream exceeded the slot budget for this policy")

        def loss_fn(cand=cand, slot=slot):
            bb.set_read_slots([slot])
            loss = bb.write_loss(cand.query, cand.value)
            bb.set_read_slots([slot])
            return loss + args.lambda_kl * bb.kl_to_anchors(anchors)

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        write_seconds += report.seconds
        written.append((slot, cand))

    # ---- retention on the written *unknown* items (own slot: this experiment is
    # about which items get written, not about routing) ---------------------
    unknown_written = [(s, c) for s, c in written if c.kind == "unknown"]
    unknown_total = sum(1 for c in candidates if c.kind == "unknown")
    known_written = [(s, c) for s, c in written if c.kind == "known"]
    known_total = sum(1 for c in candidates if c.kind == "known")

    retained = 0
    for slot, cand in unknown_written:
        bb.set_read_slots([slot])
        answer = bb.answer(cand.read_query)
        retained += int(P.exact_match(answer, cand.value, cand.aliases))

    written_slots = [s for s, _ in written]
    if written_slots:
        bb.set_read_slots(written_slots)
    same_anchors = sum(1 for q in GENERIC_ANCHORS
                       if bb.answer(q) == frozen_anchors[q])
    with torch.no_grad():
        anchor_kl = float(bb.kl_to_anchors(anchors)) if written_slots else 0.0

    return {
        "policy": policy,
        "n_candidates": len(candidates),
        "n_writes": len(written),
        "written_unknown": len(unknown_written),
        "unknown_total": unknown_total,
        "written_known": len(known_written),
        "known_total": known_total,
        "missed_unknown": unknown_total - len(unknown_written),
        "retained_unknown": retained,
        "anchor_same": f"{same_anchors}/{len(GENERIC_ANCHORS)}",
        "anchor_kl": anchor_kl,
        "write_seconds_total": write_seconds,
        "skipped": skipped,
        "decisions": decisions,
    }


def main() -> int:
    args = parse_args()
    cfg = BackboneConfig(
        model_id=resolve_model_path(args.model),
        n_slots=64,
        rank=args.rank,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"loading {cfg.model_id} ...", flush=True)
    bb = Backbone.load(cfg)
    print(f"modules={len(bb.wrappers)} memory_params={bb.n_memory_parameters/1e6:.2f}M",
          flush=True)

    # ---- phase 0: verify which "known" pairs the model really answers ------
    kept, dropped = verify_known(bb)
    print(f"\nphase 0: {len(kept)}/{len(KNOWN_PAIRS)} 'known' pairs confirmed, "
          f"{len(dropped)} dropped", flush=True)
    for bad in dropped:
        print(f"  dropped: {bad['query']!r} -> model said {bad['answer']!r}", flush=True)

    # ---- build the mixed stream -------------------------------------------
    episode = make_episode(
        args.seed, n_facts=args.facts, n_prefs=0, n_lessons=0, n_noise=0,
        n_negatives=0, with_inertia=False, paraphrase_probes=args.paraphrase,
    )
    probes = {p.item_id: p for p in episode.probes}
    candidates = [
        Candidate(
            key=f"unknown:{item.item_id}", kind="unknown", statement=item.statement,
            query=item.query, value=item.value_text, aliases=item.aliases,
            probe_query=probes[item.item_id].query, item_id=item.item_id,
        )
        for item in episode.probeable_items
    ]
    candidates += [
        Candidate(
            key=f"known:{pair.query}", kind="known", statement=pair.statement,
            query=pair.query, value=pair.answer,
        )
        for pair in kept
    ]

    # ---- surprise scores: policy-independent, so measured once -------------
    scores = surprise_scores(bb, candidates)
    for cand in candidates:
        cand.loss = scores[cand.key]

    by_class = {
        kind: [c.loss for c in candidates if c.kind == kind] for kind in ("unknown", "known")
    }
    print("\nsurprise (pre-write loss) by class:", flush=True)
    for kind, values in by_class.items():
        if values:
            print(f"  {kind:<8} n={len(values)}  min={min(values):6.2f}  "
                  f"median={statistics.median(values):6.2f}  max={max(values):6.2f}",
                  flush=True)
    for cand in candidates:
        print(f"    {cand.kind:<8} loss={cand.loss:7.3f}  {cand.query[:52]!r}", flush=True)

    tau = statistics.median([c.loss for c in candidates])
    print(f"\nlabel-free surprise threshold (median of all scores) tau={tau:.3f}",
          flush=True)

    # ---- self-check: what the model answers on its own -------------------
    checks = selfcheck_answers(bb, candidates)
    print("\nself-check (frozen model asked directly, memory off):", flush=True)
    for cand in candidates:
        print(f"  {cand.kind:<8} loss={cand.loss:7.3f}  {cand.query[:44]!r}"
              f"  -> {checks[cand.key]!r}", flush=True)

    # ---- run the policies -------------------------------------------------
    records = []
    t0 = time.perf_counter()
    print("\n" + f"{'policy':<10}{'writes':>7}{'unk':>7}{'known':>7}{'missU':>7}"
          f"{'retain':>8}{'anchorKL':>10}", flush=True)
    for policy in POLICIES:
        # A fresh process would be cleaner, but erasing every slot restores the
        # exact virgin state (asserted in the write path), so policies cannot
        # contaminate each other.
        record = run_policy(bb, candidates, args, policy, tau, selfcheck=checks)
        records.append(record)
        print(
            f"{policy:<10}{record['n_writes']:>7}"
            f"{record['written_unknown']:>4}/{record['unknown_total']:<2}"
            f"{record['written_known']:>4}/{record['known_total']:<2}"
            f"{record['missed_unknown']:>7}"
            f"{record['retained_unknown']:>5}/{record['written_unknown']:<2}"
            f"{record['anchor_kl']:>10.2f}",
            flush=True,
        )

    summary = {
        "config": vars(args),
        "model_path": bb.resolved_path,
        "tau": tau,
        "surprise_by_class": by_class,
        "known_confirmed": [p.query for p in kept],
        "known_dropped": dropped,
        "candidates": [
            {"key": c.key, "kind": c.kind, "loss": c.loss, "query": c.query}
            for c in candidates
        ],
        "elapsed_s": time.perf_counter() - t0,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
