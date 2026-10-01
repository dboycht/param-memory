"""Why does the reference answer survive the extraction only a quarter of the time?

Three hypotheses, and they call for different fixes:

  H1  the answer IS in the conversation and the extraction dropped it -- a prompting problem,
      fixable by changing what the extractor is asked to produce;
  H2  the answer is not in the conversation as a string, because the benchmark's reference is
      an aggregate ("two items", "on Tuesday") the conversation only implies -- a metric
      problem, and string containment was the wrong instrument all along;
  H3  the answer is in the conversation in a different surface form -- a normalisation
      problem, partly both.

The answer decides what the multi-session experiment should measure, so it is a measurement
rather than a judgement: the script reports the ceiling any extraction could reach, and how
many of the questions ask for an aggregate that has to be computed across several memories.
It needs no GPU and no API calls.

Usage::

    python experiments/t7i_composition_demand.py --items 30
    python experiments/t7i_composition_demand.py --show 3    # print the failing cases
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parammem.bench.longmemeval import (contains_answer, load_oracle,
                                        select_multi_session)  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"
EXTRACTED = ROOT / "runs" / "t7i_extracted.json"


def turn_text(turn: dict) -> str:
    if isinstance(turn, str):
        return turn
    return f"{turn.get('role', '')} {turn.get('content', '')}"


def session_text(session) -> str:
    if isinstance(session, dict):
        turns = session.get("turns") or session.get("messages") or []
    else:
        turns = session
    return "\n".join(turn_text(t) for t in (turns or []))


def memory_text(memory: dict) -> str:
    return f"{memory.get('question', '')} {memory.get('answer', '')}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--extracted", default=str(EXTRACTED))
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--show", type=int, default=0, help="print this many failing cases")
    args = ap.parse_args()

    items = select_multi_session(load_oracle(args.data), args.items)
    extracted = json.loads(pathlib.Path(args.extracted).read_text(encoding="utf-8"))["items"]

    counts = {"in_turns": 0, "in_memories": 0, "in_both": 0, "in_neither": 0}
    rows = []
    for item, record in zip(items, extracted):
        gold = str(item["answer"])
        sessions = item.get("haystack_sessions") or []
        joined = "\n".join(session_text(s) for s in sessions)
        memories = record.get("memories") or []
        in_turns = bool(contains_answer(joined, gold))
        in_memories = any(contains_answer(memory_text(m), gold) for m in memories)
        counts["in_turns"] += in_turns
        counts["in_memories"] += in_memories
        counts["in_both"] += in_turns and in_memories
        counts["in_neither"] += (not in_turns) and (not in_memories)
        rows.append({"qid": item["question_id"], "question": item.get("question", ""),
                     "gold": gold, "in_turns": in_turns, "in_memories": in_memories,
                     "n_memories": len(memories), "memories": memories,
                     "turns_chars": len(joined)})

    n = len(rows)
    # The question form is the objective evidence for what these items demand. If the gold
    # answer is an aggregate that appears nowhere as a string, no extraction can carry it;
    # what the bank has to do is combine several memories, which is the operation the
    # mechanism cannot perform.
    aggregate = ("how many", "how much", "total", "in total", "difference", "percentage",
                 "compared", "more money", "more miles", "more did", "left to read",
                 "older is", "older am i")
    compositional = [r for r in rows
                     if any(term in r["question"].lower() for term in aggregate)]
    abstain = [r for r in rows if "not enough" in str(r["gold"]).lower()
               or "information provided" in str(r["gold"]).lower()]
    print(f"  items                          : {n}")
    print(f"  ... whose question asks for an aggregate (how many/much/total/percentage/"
          f"difference)\n      : {len(compositional)}/{n} = {len(compositional) / n:.3f}"
          f"   <- these need several memories at once")
    print(f"  ... whose reference answer is an abstention\n      : {len(abstain)}/{n} = "
          f"{len(abstain) / n:.3f}   <- no extraction can produce this as a string")
    print(f"  gold answer present in TURNS   : {counts['in_turns']}/{n} = "
          f"{counts['in_turns'] / n:.3f}   <- the ceiling for any extraction")
    print(f"  gold answer present in MEMORIES: {counts['in_memories']}/{n} = "
          f"{counts['in_memories'] / n:.3f}")
    print(f"  in both                        : {counts['in_both']}/{n}")
    print(f"  in NEITHER (aggregate or form) : {counts['in_neither']}/{n} = "
          f"{counts['in_neither'] / n:.3f}")
    print(f"  dropped by the extractor (H1)  : {counts['in_turns'] - counts['in_both']}/{n}")

    if args.show:
        dropped = [r for r in rows if r["in_turns"] and not r["in_memories"]]
        absent = [r for r in rows if not r["in_turns"]]
        print(f"\n  === class H1: the answer IS in the turns, the extractor dropped it "
              f"({len(dropped)}) ===")
        for row in dropped[: args.show]:
            print(f"\n  Q: {row['question'][:150]}")
            print(f"  gold: {row['gold'][:150]!r}")
            print(f"  memories ({row['n_memories']}):")
            for memory in row["memories"][:4]:
                print(f"    - {memory_text(memory)[:130]}")
        print(f"\n  === class H2/H3: the answer is not a string in the turns "
              f"({len(absent)}) ===")
        for row in absent[: args.show]:
            print(f"\n  Q: {row['question'][:170]}")
            print(f"  gold: {row['gold'][:150]!r}")
            print(f"  memories ({row['n_memories']}):")
            for memory in row["memories"][:3]:
                print(f"    - {memory_text(memory)[:130]}")

    out = ROOT / "runs" / "t7i_composition_demand.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The classification is part of the result, not just of the printout: a number that
    # exists only in this run's stdout cannot reach the paper, and leaving it out is how a
    # figure or a sentence ends up quoting something no file records.
    summary = {
        "note": "What the multi-session questions demand. The reference answer is a literal "
                "string in the conversation for only half of them, so any string-based "
                "measure of the memories has that as its ceiling; the questions themselves "
                "ask for an aggregate that has to be computed across several memories, "
                "which is the operation a one-slot read cannot perform.",
        "n": n,
        "counts": counts,
        "mean_memories_per_item": (sum(len(r["memories"]) for r in rows) / n if n else 0.0),
        "compositional": len(compositional),
        "compositional_rate": len(compositional) / n if n else 0.0,
        "aggregate_terms": list(aggregate),
        "abstain": len(abstain),
        "abstain_rate": len(abstain) / n if n else 0.0,
        "ceiling_in_turns": counts["in_turns"],
        "ceiling_rate": counts["in_turns"] / n if n else 0.0,
        "extractor_dropped": counts["in_turns"] - counts["in_both"],
        "rows": rows,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
