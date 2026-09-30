"""How much evidence does each LongMemEval question type actually require?

The paper argues that the multi-session questions are out of reach for a one-slot read path
because they need evidence from several stored items, while the single-session subset does
not. That was an assertion about the data. This measures it: for every question, how many
turns carry the answer and how many separate sessions those turns live in.

Usage::

    python experiments/longmemeval_evidence_shape.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"


def evidence_shape(item: dict) -> tuple[int, int]:
    """(turns carrying the answer, sessions those turns live in)."""
    turns = 0
    sessions = 0
    for session in item.get("haystack_sessions") or []:
        hits = sum(1 for turn in session if turn.get("has_answer"))
        if hits:
            sessions += 1
            turns += hits
    return turns, sessions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default="runs/longmemeval_evidence_shape.json")
    args = ap.parse_args()

    path = Path(args.data)
    if not path.is_file():
        print(f"  missing {path}")
        return 1
    raw = json.loads(path.read_text(encoding="utf-8"))

    per_type: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in raw:
        per_type[str(item.get("question_type", "?"))].append(evidence_shape(item))

    summary = {}
    print(f"  {'question type':<28}{'n':>5}{'median turns':>14}{'median sessions':>17}"
          f"{'multi-session share':>21}")
    for kind, shapes in sorted(per_type.items(), key=lambda kv: -len(kv[1])):
        turns = [t for t, _s in shapes]
        sessions = [s for _t, s in shapes]
        share = sum(1 for _t, s in shapes if s > 1) / len(shapes)
        summary[kind] = {
            "n": len(shapes),
            "median_evidence_turns": statistics.median(turns),
            "median_sessions": statistics.median(sessions),
            "mean_evidence_turns": statistics.mean(turns),
            "share_spanning_several_sessions": share,
            "max_sessions": max(sessions),
        }
        print(f"  {kind:<28}{len(shapes):>5}{statistics.median(turns):>14.1f}"
              f"{statistics.median(sessions):>17.1f}{share:>20.0%}")

    payload = {
        "note": "Evidence shape per question type: how many turns carry the answer and how "
                "many sessions they span. The paper argues the multi-session questions need "
                "several stored items while the single-session subset does not; this is the "
                "measurement behind that sentence.",
        "data_file": str(path),
        "per_type": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
