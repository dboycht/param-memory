"""Recompute each stage's reported aggregate from its most granular records.

The macros come from aggregates that the experiment scripts compute, so a bug in an
aggregation would propagate silently into the paper with every downstream check still
green: the values would be consistent, just wrong. This recomputes the statistics from the
per-episode or per-probe records using the definition in the protocol, and reports any
disagreement.

Usage::

    python _dev/check_aggregates.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TOLERANCE = 5e-4


def close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= TOLERANCE


def check_composition(name: str, problems: list[str]) -> None:
    """T2: em, counts and routing recomputed from the per-episode arm outcomes."""
    path = RUNS / name
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    summary, records = data.get("summary", {}), data.get("episodes", [])
    if not records:
        return
    arms = list(summary.get("em", {}))
    for arm in arms:
        flat = [value for record in records for value in record["arms"][arm]]
        if not flat:
            continue
        recomputed = sum(flat) / len(flat)
        if not close(recomputed, summary["em"][arm]):
            problems.append(f"{name}: em[{arm}] reported {summary['em'][arm]:.4f}, "
                            f"recomputed {recomputed:.4f}")
        if [int(sum(flat)), len(flat)] != list(summary["counts"][arm]):
            problems.append(f"{name}: counts[{arm}] reported {summary['counts'][arm]}, "
                            f"recomputed {[int(sum(flat)), len(flat)]}")
    hits = sum(record["route_top1_hits"] for record in records)
    total = sum(record["n_items"] for record in records)
    if total and not close(hits / total, summary.get("route_top1_accuracy", -1)):
        problems.append(f"{name}: routing reported {summary['route_top1_accuracy']:.4f}, "
                        f"recomputed {hits / total:.4f}")


def check_inertia(name: str, problems: list[str]) -> None:
    """T5: the arm rates recomputed from the per-episode scores."""
    path = RUNS / name
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    records, arms = data.get("records", []), data.get("arms", {})
    if not records or not arms:
        return
    for arm, reported in arms.items():
        for field, key in (("inheritance_rate", "inherited"),
                           ("residue_rate", "residue"),
                           ("withholding_rate", "withholds")):
            recomputed = sum(record["scores"][arm][key] for record in records) / len(records)
            if not close(recomputed, reported[field]):
                problems.append(f"{name}: {arm}.{field} reported {reported[field]:.4f}, "
                                f"recomputed {recomputed:.4f}")


def check_learned_key(name: str, problems: list[str]) -> None:
    """T2-e: the per-arm rates recomputed from the per-probe rows."""
    path = RUNS / name
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes, rates = data.get("episodes", []), data.get("rates", {})
    if not episodes or not rates:
        return
    probes = sum(episode["n_probes"] for episode in episodes)
    for arm, reported in rates.items():
        for metric in ("top1", "top2_recall", "em"):
            total = sum(episode["tally"][arm][metric] for episode in episodes)
            recomputed = total / probes if probes else 0.0
            if not close(recomputed, reported[metric]):
                problems.append(f"{name}: {arm}.{metric} reported {reported[metric]:.4f}, "
                                f"recomputed {recomputed:.4f}")


def main() -> int:
    problems: list[str] = []
    for name in ("t2_paraphrase.json", "t2_paraphrase_17b.json",
                 "t2_paraphrase_lexkey.json", "t2_paraphrase_17b_lexkey.json",
                 "t2_composition.json"):
        check_composition(name, problems)
    for name in ("t5_inertia.json", "t5_inertia_17b.json", "t5_inertia_seed1.json",
                 "t5_inertia_seed2.json"):
        check_inertia(name, problems)
    for name in ("t2e_learned_key.json", "t2e_learned_key_17b.json"):
        check_learned_key(name, problems)

    if not problems:
        print("  every reported aggregate matches its own granular records")
        return 0
    print(f"  {len(problems)} disagreement(s) between reported and recomputed values:")
    for problem in problems:
        print(f"    {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
