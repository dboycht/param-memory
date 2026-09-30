"""T7-h -- merge the human judge-calibration rounds and report agreement.

The paper reports how often the LLM judge agrees with a human on the real-content
queries, and that number is only meaningful if the marks are data rather than a
recollection. Round 1 produced 12 marks (2 of which had to be discounted because the
judge's own reply was truncated); a larger round is annotated separately and merged
here.

Two conventions matter and are applied here rather than left to prose:

* a truncated judge reply is **not** evidence in either direction, so that row is
  excluded rather than scored;
* with zero observed disagreements the point estimate is 1.0, which is why the Wilson
  interval is reported next to it -- "10/10" is consistent with a true error rate of
  0-26%, and the interval is the honest way to say that.

Usage::

    python experiments/t7h_calibration.py \\
        --marks bench/judge_calibration_marks.json bench/judge_calibration_marks_round2.json \\
        --payloads runs/judge_calibration.json runs/judge_calibration_round2.json \\
        --out runs/t7h_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marks", nargs="+", required=True)
    ap.add_argument("--payloads", nargs="+", required=True)
    ap.add_argument("--out", default="runs/t7h_calibration.json")
    return ap.parse_args()


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: correct at the extremes, unlike the normal approximation."""
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
            / denominator)
    return (max(0.0, centre - half), min(1.0, centre + half))


def parse_marks(text: str) -> dict[int, bool]:
    """``"1Y 2N 3Y"`` -> ``{1: True, 2: False, 3: True}``."""
    out: dict[int, bool] = {}
    for token in text.replace(",", " ").split():
        token = token.strip().upper()
        if not token or token[-1] not in "YN":
            continue
        number = "".join(ch for ch in token[:-1] if ch.isdigit())
        if number:
            out[int(number)] = token.endswith("Y")
    return out


def load_round(mark_path: Path, payload_path: Path) -> dict:
    marks_doc = json.loads(mark_path.read_text(encoding="utf-8"))
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    marks = parse_marks(str(marks_doc.get("marks", "")))
    verdicts = {int(v["n"]): v for v in payload.get("judge_verdicts", [])}
    rows = []
    for number, human in sorted(marks.items()):
        verdict = verdicts.get(number)
        if verdict is None:
            continue
        rows.append({
            "n": number,
            "human": human,
            "judge": verdict.get("correct"),
            "truncated": verdict.get("finish") == "length",
            "reason": verdict.get("reason"),
        })
    return {"marks_file": mark_path.name, "payload_file": payload_path.name,
            "n_marks": len(marks), "rows": rows}


def main() -> int:
    args = parse_args()
    rounds = [load_round(Path(m), Path(p))
              for m, p in zip(args.marks, args.payloads)]

    scored = [row for rnd in rounds for row in rnd["rows"]
              if not row["truncated"] and row["judge"] is not None]
    discounted = [row for rnd in rounds for row in rnd["rows"] if row["truncated"]]
    agree = sum(1 for row in scored if row["human"] == row["judge"])
    low, high = wilson(agree, len(scored))

    # Direction matters: a judge that is too strict and one that is too lenient fail
    # in opposite ways for our claims, so the two are counted separately.
    missed = sum(1 for row in scored if row["human"] and not row["judge"])
    false_positive = sum(1 for row in scored if row["judge"] and not row["human"])

    summary = {
        "note": "Human-vs-judge agreement across every calibration round run so far. "
                "Truncated judge replies are excluded rather than scored, and the "
                "Wilson interval is reported because a 1.000 point estimate with a "
                "handful of marks is not a 0% error rate.",
        "rounds": [{k: v for k, v in rnd.items() if k != "rows"} for rnd in rounds],
        "n_marks_total": sum(rnd["n_marks"] for rnd in rounds),
        "n_scored": len(scored),
        "n_discounted": len(discounted),
        "agreement": agree,
        "agreement_rate": agree / len(scored) if scored else None,
        "wilson_95": [low, high],
        "judge_too_strict": missed,
        "judge_false_positive": false_positive,
        "rows": scored,
        "discounted_rows": [{"round": rnd["marks_file"], "n": row["n"]}
                            for rnd in rounds for row in rnd["rows"] if row["truncated"]],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  rounds merged      : {len(rounds)}")
    print(f"  marks              : {summary['n_marks_total']} "
          f"({summary['n_scored']} scored, {summary['n_discounted']} discounted)")
    print(f"  agreement          : {agree}/{len(scored)}"
          + (f" = {summary['agreement_rate']:.3f}" if scored else ""))
    print(f"  Wilson 95% interval: [{low:.3f}, {high:.3f}]")
    print(f"  judge too strict   : {missed}   judge false positive: {false_positive}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
