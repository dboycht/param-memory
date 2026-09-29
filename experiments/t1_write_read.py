"""T1 -- can a written memory be read back once the context is evicted?

This is the smallest honest version of the whole claim:

    write a random fictional fact into a memory slot,
    drop the write context entirely,
    ask for the value,
    and change *only* which slots are active.

The four arms are exactly protocol P4/P5 from docs/02:

======================  ====================================================
``prompt_only``         virgin side path, never written (measured first)
``mem_on``              the slot holding this item is active
``mem_shuffle``         every slot except that one is active
``mem_off``             all slots erased again (must collapse to prompt_only)
======================  ====================================================

Usage::

    python experiments/t1_write_read.py --episodes 5 --n-slots 16 --steps 8
    python experiments/t1_write_read.py --model Qwen/Qwen3-0.6B --episodes 2   # smoke

Output is a JSON report plus a console table; nothing is committed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from parammem.bench import protocol as P
from parammem.bench.synthetic import make_episode
from parammem.memory.store import MemoryStore
from parammem.memory.writer import write_slot
from parammem.model import Backbone, BackboneConfig

ARMS = ("prompt_only", "mem_on", "mem_shuffle", "mem_off")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--n-slots", type=int, default=16)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=8, help="gradient steps per write")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n-facts", type=int, default=3)
    ap.add_argument("--n-noise", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--policy", default="utility_time")
    ap.add_argument("--out", default="runs/t1_write_read.json")
    ap.add_argument("--dump-answers", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------- episode


def run_episode(bb: Backbone, episode, args: argparse.Namespace) -> dict:
    """Run all four arms for one episode. Returns a per-episode record."""
    bb.erase_all()
    bb.set_read_slots(None)
    store = MemoryStore(
        n_slots=bb.cfg.n_slots, policy=args.policy, decay=0.9, min_utility=0.05
    )

    probes = list(episode.probes)
    probe_ids = [p.item_id for p in probes]

    # ---- arm 1: prompt_only (virgin side path) --------------------------
    prompt_only = {p.item_id: bb.answer(p.query) for p in probes}
    neg_baseline = {i: bb.answer(n.query) for i, n in enumerate(episode.negatives)}

    # ---- write every probeable item into its own slot -------------------
    slot_of: dict[int, int] = {}
    writes: list[dict] = []
    for item in episode.writes:
        if not item.probeable:
            continue
        slot = store.next_free_slot()
        if slot is None:
            raise RuntimeError(
                f"episode needs more slots than n_slots={bb.cfg.n_slots}; "
                "T1 deliberately keeps K >= number of items so nothing is evicted"
            )
        report = write_slot(
            bb.wrappers,
            slot,
            lambda item=item: bb.write_loss(item.query, item.value_text),
            lr=args.lr,
            steps=args.steps,
        )
        store.occupy(
            slot,
            item_id=item.item_id,
            text=item.statement,
            memory_class=item.memory_class,
            is_important=item.is_important,
            norm=bb.slot_norm(slot),
            write_seconds=report.seconds,
        )
        slot_of[item.item_id] = slot
        writes.append(
            {
                "item_id": item.item_id,
                "slot": slot,
                "memory_class": item.memory_class,
                "value": item.value_text,
                "loss_start": report.loss_start,
                "loss_end": report.loss_end,
                "frozen_ok": report.frozen_ok,
                "seconds": report.seconds,
                "norm_after": report.norm_after,
            }
        )

    # ---- arm 2: mem_on --------------------------------------------------
    bb.set_read_slots(None)
    mem_on = {p.item_id: bb.answer(p.query) for p in probes}
    neg_memory = {i: bb.answer(n.query) for i, n in enumerate(episode.negatives)}

    # ---- arm 3: mem_shuffle (all slots except this item's) --------------
    all_slots = list(range(bb.cfg.n_slots))
    mem_shuffle = {}
    for probe in probes:
        others = [s for s in all_slots if s != slot_of[probe.item_id]]
        bb.set_read_slots(others)
        mem_shuffle[probe.item_id] = bb.answer(probe.query)

    # ---- arm 4: mem_off (erase, then identical prompt) ------------------
    bb.erase_all()
    bb.set_read_slots(None)
    mem_off = {p.item_id: bb.answer(p.query) for p in probes}

    arms = {
        "prompt_only": prompt_only,
        "mem_on": mem_on,
        "mem_shuffle": mem_shuffle,
        "mem_off": mem_off,
    }

    per_arm_em: dict[str, list[float]] = {}
    for arm, answers in arms.items():
        scores = []
        for probe in probes:
            scores.append(
                1.0 if P.exact_match(answers[probe.item_id], probe.value, probe.aliases) else 0.0
            )
        per_arm_em[arm] = scores

    negatives_em = {
        "prompt_only": [
            1.0 if P.exact_match(neg_baseline[i], n.value, n.aliases) else 0.0
            for i, n in enumerate(episode.negatives)
        ],
        "mem_on": [
            1.0 if P.exact_match(neg_memory[i], n.value, n.aliases) else 0.0
            for i, n in enumerate(episode.negatives)
        ],
    }

    identical = sum(
        1 for pid in probe_ids if mem_off[pid] == prompt_only[pid]
    )
    record = {
        "episode_id": episode.episode_id,
        "seed": episode.seed,
        "n_probes": len(probes),
        "n_writes": len(writes),
        "per_arm_em": per_arm_em,
        "negatives_em": negatives_em,
        "mem_off_identical_to_prompt_only": f"{identical}/{len(probes)}",
        "writes": writes,
        "slot_utility": [m.to_dict() for m in store.occupied],
    }
    if args.dump_answers:
        record["answers"] = {
            arm: {str(k): v for k, v in answers.items()} for arm, answers in arms.items()
        }
        record["probe_values"] = {str(p.item_id): p.value for p in probes}
    return record


# ------------------------------------------------------------------ aggregate


def aggregate(records: list[dict]) -> dict:
    pooled: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for record in records:
        for arm in ARMS:
            pooled[arm].extend(record["per_arm_em"][arm])

    means = {arm: (sum(v) / len(v) if v else float("nan")) for arm, v in pooled.items()}

    # Paired per-probe differences: mem_on and prompt_only are measured on the
    # same probes in the same episode, so the CI is over comparable units.
    pa = [x for record in records for x in record["per_arm_em"]["mem_on"]]
    pb = [x for record in records for x in record["per_arm_em"]["prompt_only"]]
    diff = [a - b for a, b in zip(pa, pb)]
    ci = P.bootstrap_ci(diff, n=2000, seed=0)

    neg_base = [x for r in records for x in r["negatives_em"]["prompt_only"]]
    neg_mem = [x for r in records for x in r["negatives_em"]["mem_on"]]

    negative = P.negative_control(
        written_baseline_em=means["prompt_only"],
        written_mem_em=means["mem_on"],
        unwritten_baseline_em=(sum(neg_base) / len(neg_base)) if neg_base else 0.0,
        unwritten_mem_em=(sum(neg_mem) / len(neg_mem)) if neg_mem else 0.0,
    )
    attribution = P.attribution_delta(
        em_mem_on=means["mem_on"],
        em_mem_off=means["mem_off"],
        em_shuffle=means["mem_shuffle"],
        em_prompt_only=means["prompt_only"],
        ci=ci,
    )

    write_seconds = [w["seconds"] for r in records for w in r["writes"]]
    frozen_ok = all(w["frozen_ok"] for r in records for w in r["writes"])
    loss_drop = [
        w["loss_start"] - w["loss_end"] for r in records for w in r["writes"]
    ]
    identical = sum(
        int(r["mem_off_identical_to_prompt_only"].split("/")[0]) for r in records
    )
    total_probes = sum(r["n_probes"] for r in records)

    return {
        "n_episodes": len(records),
        "n_probes": total_probes,
        "em": means,
        "em_counts": {arm: [len(v), sum(v)] for arm, v in pooled.items()},
        "delta_on_minus_prompt_only": means["mem_on"] - means["prompt_only"],
        "delta_ci95": list(ci),
        "negative_control": negative.__dict__,
        "attribution": attribution.__dict__,
        "write_seconds_mean": (sum(write_seconds) / len(write_seconds)) if write_seconds else 0.0,
        "write_seconds_max": max(write_seconds) if write_seconds else 0.0,
        "write_loss_drop_mean": (sum(loss_drop) / len(loss_drop)) if loss_drop else 0.0,
        "frozen_ok_everywhere": frozen_ok,
        "mem_off_string_identical": f"{identical}/{total_probes}",
    }


def format_table(summary: dict) -> str:
    lines = []
    lines.append(f"episodes={summary['n_episodes']}  probes={summary['n_probes']}")
    lines.append(f"{'arm':<14}{'EM':>8}   counts")
    for arm in ARMS:
        n, k = summary["em_counts"][arm]
        lines.append(f"{arm:<14}{summary['em'][arm]:>8.3f}   {int(k)}/{int(n)}")
    lines.append("")
    delta = summary["delta_on_minus_prompt_only"]
    lo, hi = summary["delta_ci95"]
    lines.append(f"delta(mem_on - prompt_only) = {delta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    lines.append(f"negative control : {summary['negative_control']['detail']}")
    lines.append(f"attribution      : {summary['attribution']['detail']}")
    lines.append(f"write isolation  : {'ok' if summary['frozen_ok_everywhere'] else 'VIOLATED'}")
    lines.append(
        f"write cost       : mean {summary['write_seconds_mean']*1000:.0f} ms, "
        f"max {summary['write_seconds_max']*1000:.0f} ms per item"
    )
    lines.append(f"mem_off string-identical to prompt_only: {summary['mem_off_string_identical']}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    cfg = BackboneConfig(
        model_id=args.model,
        n_slots=args.n_slots,
        rank=args.rank,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"loading {cfg.model_id} (n_slots={cfg.n_slots}, rank={cfg.rank}) ...", flush=True)
    bb = Backbone.load(cfg)
    print(
        f"side path: {len(bb.wrappers)} modules, "
        f"{bb.n_memory_parameters/1e6:.2f}M trainable params",
        flush=True,
    )

    records = []
    t0 = time.perf_counter()
    for i in range(args.episodes):
        episode = make_episode(
            args.base_seed + i, episode_id=i, n_facts=args.n_facts, n_noise=args.n_noise
        )
        record = run_episode(bb, episode, args)
        records.append(record)
        em_on = sum(record["per_arm_em"]["mem_on"]) / record["n_probes"]
        em_off = sum(record["per_arm_em"]["mem_off"]) / record["n_probes"]
        em_sh = sum(record["per_arm_em"]["mem_shuffle"]) / record["n_probes"]
        print(
            f"  episode {i}: mem_on={em_on:.2f} shuffle={em_sh:.2f} off={em_off:.2f}",
            flush=True,
        )

    summary = aggregate(records)
    summary["elapsed_s"] = time.perf_counter() - t0
    summary["config"] = vars(args)
    summary["vram"] = bb.memory_footprint()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "episodes": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(format_table(summary))
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
