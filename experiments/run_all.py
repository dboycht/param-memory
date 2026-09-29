"""T7 -- one command that reproduces every headline number.

Two modes:

``--collect-only``
    Do not run anything; read the JSON reports already on disk and (re)build
    ``runs/RESULTS.md`` + ``runs/summary.json``. Useful for regenerating the
    tables after a doc change, at zero compute cost.
``--mode full|quick``
    Actually run the stages. ``quick`` shrinks every stage so the whole chain can
    be smoke-tested in minutes; ``full`` is what the paper's numbers come from.

Two rules this driver exists to enforce:

* **Each stage runs in its own process**, with stdout/stderr redirected to a log
  *file* -- never a pipe. A piped child can fail in confined environments, and a
  filtered stream hides the traceback (ERROR.md E7).
* **A failing stage never hides the others.** Its row is marked FAILED and its
  section says "not available"; nothing is silently filled with zeros.

Usage::

    python experiments/run_all.py --collect-only
    python experiments/run_all.py --mode quick --only t5
    python experiments/run_all.py --mode full
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# stage -> recipe. ``inputs`` are the JSON files a stage produces (one per phase),
# and are also what --collect-only reads back.
STAGES: dict[str, dict] = {
    "t1": {
        "script": "t1_write_read.py",
        "inputs": ["t1_06b.json"],
        "full": ["--episodes", "3", "--n-slots", "16", "--out", "runs/t1_write_read.json"],
        "quick": ["--episodes", "1", "--n-slots", "8", "--steps", "4",
                  "--out", "runs/t1_quick.json"],
    },
    "t2": {
        "script": "t2_composition.py",
        "inputs": ["t2_paraphrase.json"],
        "full": ["--items", "8", "--episodes", "3", "--paraphrase",
                 "--out", "runs/t2_composition_paraphrase.json"],
        "quick": ["--items", "4", "--episodes", "1", "--steps", "4", "--paraphrase",
                  "--out", "runs/t2_quick.json"],
    },
    "t3": {
        "script": "t3_write_policy.py",
        "inputs": ["t3_write_policy.json"],
        "full": ["--facts", "5", "--out", "runs/t3_write_policy.json"],
        "quick": ["--facts", "2", "--steps", "4", "--out", "runs/t3_quick.json"],
    },
    "t4": {
        "script": "t4_capacity.py",
        "inputs": ["t4_capacity.json"],
        "full": ["--stream", "12", "--slots", "6", "--hot", "3",
                 "--out", "runs/t4_capacity.json"],
        "quick": ["--stream", "6", "--slots", "3", "--hot", "2", "--steps", "4",
                  "--policies", "fifo", "utility_time", "--out", "runs/t4_quick.json"],
    },
    "t5": {
        "script": "t5_inertia.py",
        "inputs": ["t5_inertia.json"],
        "full": ["--candidates", "30", "--episodes", "12", "--out", "runs/t5_inertia.json"],
        "quick": ["--candidates", "8", "--episodes", "2", "--steps", "4",
                  "--out", "runs/t5_quick.json"],
    },
    "t6": {
        "script": "t6_persistence.py",
        "inputs": ["t6_write.json", "t6_read.json"],
        "phases": [
            ["--session", "write", "--out", "runs/t6_write.json"],
            ["--session", "read", "--out", "runs/t6_read.json"],
        ],
        "quick_phases": [
            ["--session", "write", "--facts", "2", "--steps", "4",
             "--out", "runs/t6_write.json"],
            ["--session", "read", "--out", "runs/t6_read.json"],
        ],
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("full", "quick"), default="quick")
    ap.add_argument("--only", nargs="*", default=list(STAGES),
                    help="subset of stages to run, e.g. --only t1 t2")
    ap.add_argument("--collect-only", action="store_true",
                    help="skip running; rebuild RESULTS.md from existing reports")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default="runs/RESULTS.md")
    ap.add_argument("--bundle", default="runs/summary.json")
    return ap.parse_args()


def _run(command: list[str], log_path: Path) -> int:
    """Run a child process with its output going to a *file*, not a pipe."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.run(
            [sys.executable, *command],
            stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT),
        )
    return int(proc.returncode)


def run_stage(stage: str, recipe: dict, mode: str, log_dir: Path) -> list[str]:
    """Run one stage (one or two processes). Returns the JSON files it produced."""
    script = str(ROOT / "experiments" / recipe["script"])
    if "phases" in recipe:
        plan = recipe["quick_phases"] if mode == "quick" else recipe["phases"]
        for index, phase_args in enumerate(plan):
            code = _run([script, *phase_args], log_dir / f"{stage}_phase{index}.log")
            if code != 0:
                raise RuntimeError(f"{stage} phase {index} exited with {code}")
    else:
        args = recipe["quick"] if mode == "quick" else recipe["full"]
        code = _run([script, *args], log_dir / f"{stage}.log")
        if code != 0:
            raise RuntimeError(f"{stage} exited with {code}")
    return list(recipe["inputs"])


def collect(stage: str, recipe: dict, mode: str) -> dict:
    """Read a stage's reports back into the bundle."""
    inputs = list(recipe["inputs"])
    payloads = []
    for name in inputs:
        path = RUNS / name
        if path.is_file():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if stage == "t6":
        return {"write": payloads[0] if payloads else {},
                "read": payloads[1] if len(payloads) > 1 else {}}
    return payloads[0] if payloads else {}


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main() -> int:
    args = parse_args()
    from parammem.report import build_results_markdown  # after sys.path setup in caller

    log_dir = RUNS / "logs"
    bundle: dict = {
        "_provenance": {
            "commit": git_commit(),
            "model": args.model,
            "mode": "collect-only" if args.collect_only else args.mode,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    for stage in args.only:
        recipe = STAGES[stage]
        if not args.collect_only:
            print(f"[{stage}] running ({args.mode}) ...", flush=True)
            try:
                run_stage(stage, recipe, args.mode, log_dir)
            except RuntimeError as exc:
                print(f"[{stage}] FAILED: {exc} (see runs/logs/{stage}*.log)", flush=True)
                bundle[stage] = {"_failed": True, "_error": str(exc)}
                continue
        else:
            print(f"[{stage}] collecting from {recipe['inputs']} ...", flush=True)
        data = collect(stage, recipe, args.mode)
        if not data or (isinstance(data, dict) and not any(data.values())):
            print(f"[{stage}] no reports found on disk", flush=True)
            bundle[stage] = {"_failed": True, "_error": "no reports found"}
            continue
        bundle[stage] = data

    markdown = build_results_markdown(bundle)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    Path(args.bundle).write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nresults -> {out.resolve()}")
    print(f"bundle  -> {Path(args.bundle).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
