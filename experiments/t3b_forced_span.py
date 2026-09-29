"""T3-b -- why the likelihood criterion measures wording, in one number.

The T3 self-check criterion asks the model; the likelihood criterion thresholds the
*training* loss. This script measures the gap that makes the second one unusable:
for a fact the model demonstrably knows, the loss of the canonical short answer
versus the loss of the model's own verbose sentence.

It exists as a first-class experiment rather than a machine-local probe because the
paper quotes its number as evidence for a negative result, and a number a reader
cannot regenerate is not evidence.

Usage::

    python experiments/t3b_forced_span.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from parammem.model import Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

# A fact the model answers correctly, and the bare span a threshold-based criterion
# would have to score instead of the sentence the model actually produces.
QUERY = "What is the capital of France?"
CANONICAL = " Paris"
VERBOSE = "France's capital is Paris."

PROBES = [
    ("the capital of France", QUERY, CANONICAL, VERBOSE),
    ("the largest planet", "What is the largest planet in the solar system?",
     " Jupiter", "The largest planet in the solar system is Jupiter."),
    ("days in a week", "How many days are there in a week?", " seven",
     "There are seven days in a week."),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default="runs/t3_forced_span.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    bb = Backbone.load(BackboneConfig(model_id=resolve_model_path(args.model),
                                     n_slots=2, max_new_tokens=24))
    bb.set_read_slots([])

    rows = []
    for label, query, canonical, verbose in PROBES:
        with torch.no_grad():
            canonical_loss = float(bb.write_loss(query, canonical).detach())
            verbose_loss = float(bb.write_loss(query, verbose).detach())
        answer = bb.answer(query)
        rows.append({"label": label, "query": query, "canonical": canonical,
                     "verbose": verbose, "canonical_loss": canonical_loss,
                     "verbose_loss": verbose_loss, "model_answer": answer})
        print(f"  {label:<22} canonical={canonical_loss:6.2f}  "
              f"verbose={verbose_loss:6.2f}  answer={answer[:42]!r}", flush=True)

    first = rows[0]
    summary = {
        "note": "teacher-forced cross-entropy of the canonical span vs the model's "
                "own sentence, for facts the model answers correctly. A criterion "
                "that thresholds loss therefore scores wording, not knowledge.",
        "probes": rows,
        "canonical_loss": first["canonical_loss"],
        "verbose_loss": first["verbose_loss"],
        "gap": first["canonical_loss"] - first["verbose_loss"],
        "model_path": bb.resolved_path,
        "elapsed_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ncanonical span costs {summary['canonical_loss']:.2f} nats against "
          f"{summary['verbose_loss']:.2f} for the model's own sentence "
          f"(gap {summary['gap']:.2f})")
    print(f"report written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
