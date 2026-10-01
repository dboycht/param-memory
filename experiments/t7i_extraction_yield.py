"""How much of the answer survives the extraction step?

The paper's limitation says a turn-to-pair extraction step is required for the benchmark and
that "noisy extraction lands directly in the weights". That was an assertion. The
multi-session run measured it: even the oracle arm scored near the floor, and the reason is
upstream of the read path -- the extracted memories often do not contain the benchmark's
answer at all.

This measures the yield of the extraction stage on its own: for each item, is the reference
answer present in any of its extracted memories, and in the first one, which is what the
oracle arm activates. It needs no GPU and no API calls; it reads the two saved files.

Usage::

    python experiments/t7i_extraction_yield.py --out runs/t7i_extraction_yield.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parammem.bench.longmemeval import (contains_answer, load_oracle,
                                        select_multi_session, sha256_file)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"


def memory_text(memory: dict) -> str:
    return f"{memory.get('question', '')} {memory.get('answer', '')}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--extracted", default="runs/t7i_extracted.json")
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="runs/t7i_extraction_yield.json")
    args = ap.parse_args()

    data_path, extracted_path = Path(args.data), Path(args.extracted)
    for path in (data_path, extracted_path):
        if not path.is_file():
            print(f"  missing {path}")
            return 1
    items = select_multi_session(load_oracle(data_path), args.items, args.offset)
    extracted = json.loads(extracted_path.read_text(encoding="utf-8"))["items"]

    rows = []
    for item, record in zip(items, extracted):
        memories = record.get("memories") or []
        gold = str(item["answer"])
        rows.append({
            "question_id": item["question_id"],
            "n_memories": len(memories),
            "answer_in_any": int(any(contains_answer(memory_text(m), gold)
                                     for m in memories)),
            "answer_in_first": int(bool(memories)
                                   and contains_answer(memory_text(memories[0]), gold)),
        })

    n = len(rows)
    any_hit = sum(row["answer_in_any"] for row in rows)
    first_hit = sum(row["answer_in_first"] for row in rows)
    summary = {
        "note": "Yield of the turn-to-pair extraction: whether the benchmark's reference "
                "answer survives into the extracted memories at all, and into the first "
                "one, which is the memory the oracle arm activates. The multi-session "
                "containment is near the floor even for the oracle arm, and this is the "
                "upstream reason: most items' answers never make it into a memory.",
        "data_file": str(data_path),
        "data_sha256": sha256_file(data_path),
        "extracted_file": extracted_path.name,
        "n_items": n,
        "answer_in_any": any_hit,
        "answer_in_any_rate": any_hit / n if n else 0.0,
        "answer_in_first": first_hit,
        "answer_in_first_rate": first_hit / n if n else 0.0,
        "mean_memories_per_item": sum(row["n_memories"] for row in rows) / n if n else 0.0,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  items                     : {n}")
    print(f"  answer in any memory      : {any_hit}/{n} = {summary['answer_in_any_rate']:.3f}")
    print(f"  answer in the first memory: {first_hit}/{n} = "
          f"{summary['answer_in_first_rate']:.3f}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
