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
        # The output name carries the backbone size on purpose (the 1.7B control run
        # lives beside it), and it must match what collect() reads: writing
        # somewhere else meant --mode full did not refresh the quoted numbers.
        "full": ["--episodes", "3", "--n-slots", "16", "--out", "runs/t1_06b.json"],
        "quick": ["--episodes", "1", "--n-slots", "8", "--steps", "4",
                  "--out", "runs/t1_quick.json"],
    },
    "t2": {
        "script": "t2_composition.py",
        "inputs": ["t2_paraphrase.json", "t2_paraphrase_17b.json",
                   "t2c_router_keys.json", "t2c_router_keys_17b.json",
                   "t2b_mechanism.json", "t2d_composable.json"],
        # Five episodes, not three: the per-episode spread on two-slot recall is
        # large (4, 0, 4, 4, 3 of 8), so an aggregate is only meaningful when one
        # bad seed cannot carry it. The second phase repeats it on a 1.7B backbone,
        # because the composition result is this paper's central claim and was
        # otherwise untested above 0.6B.
        "phases": [
            ["--items", "8", "--episodes", "5", "--paraphrase",
             "--out", "runs/t2_paraphrase.json"],
            ["--model", "Qwen/Qwen3-1.7B", "--items", "8", "--episodes", "5",
             "--paraphrase", "--out", "runs/t2_paraphrase_17b.json"],
        ],
        "quick_phases": [
            ["--items", "4", "--episodes", "1", "--steps", "4", "--paraphrase",
             "--out", "runs/t2_quick.json"],
        ],
    },
    "t3": {
        "script": "t3_write_policy.py",
        "inputs": ["t3_write_policy.json", "t3_seed1.json", "t3_seed2.json",
                   "t3_forced_span.json"],
        # Three seeds: the write counts are deterministic but the backbone drift is
        # not, so the paper quotes a range rather than one seed's number. The fourth
        # phase measures the wording/likelihood gap the paper cites for its second
        # negative result -- as an experiment, so the number can be regenerated.
        "phases": [
            ["--facts", "5", "--seed", "0", "--out", "runs/t3_write_policy.json"],
            ["--facts", "5", "--seed", "1", "--out", "runs/t3_seed1.json"],
            ["--facts", "5", "--seed", "2", "--out", "runs/t3_seed2.json"],
            ["t3b_forced_span.py", "--out", "runs/t3_forced_span.json"],
        ],
        "quick_phases": [
            ["--facts", "2", "--seed", "0", "--steps", "4",
             "--out", "runs/t3_quick.json"],
        ],
    },
    "t4": {
        "script": "t4_capacity.py",
        "inputs": ["t4_capacity.json", "t4_capacity_seed1.json",
                   "t4_capacity_seed2.json"],
        # Three seeds. The eviction assertions are deterministic (every erased slot is
        # checked bit-for-bit), but *which* memories survive under each policy is not,
        # so the retention numbers are reported as a range rather than as one draw.
        "phases": [
            ["--stream", "12", "--slots", "6", "--hot", "3", "--base-seed", "0",
             "--out", "runs/t4_capacity.json"],
            ["--stream", "12", "--slots", "6", "--hot", "3", "--base-seed", "1",
             "--out", "runs/t4_capacity_seed1.json"],
            ["--stream", "12", "--slots", "6", "--hot", "3", "--base-seed", "2",
             "--out", "runs/t4_capacity_seed2.json"],
        ],
        "quick_phases": [
            ["--stream", "6", "--slots", "3", "--hot", "2", "--steps", "4",
             "--policies", "fifo", "utility_time", "--out", "runs/t4_quick.json"],
        ],
    },
    "t5": {
        "script": "t5_inertia.py",
        "inputs": ["t5_inertia.json", "t5_inertia_seed1.json", "t5_inertia_seed2.json"],
        # The refutation is reported from one draw in the paper; three seeds make the
        # "context and weights are equally sticky" claim a range instead.
        "phases": [
            ["--candidates", "30", "--episodes", "12", "--base-seed", "0",
             "--out", "runs/t5_inertia.json"],
            ["--candidates", "30", "--episodes", "12", "--base-seed", "1",
             "--out", "runs/t5_inertia_seed1.json"],
            ["--candidates", "30", "--episodes", "12", "--base-seed", "2",
             "--out", "runs/t5_inertia_seed2.json"],
        ],
        "quick_phases": [
            ["--candidates", "8", "--episodes", "2", "--steps", "4",
             "--out", "runs/t5_quick.json"],
        ],
    },
    "t6": {
        "script": "t6_persistence.py",
        "inputs": ["t6_write.json", "t6_read.json"],
        "phases": [
            ["--session", "write", "--out", "runs/t6_write.json"],
            ["--session", "read", "--out", "runs/t6_read.json"],
        ],
        # quick mode must use its own snapshot dir and outputs: writing a 2-item
        # snapshot over the real one would silently replace the full run's artifact
        # with a smoke-test one.
        "quick_phases": [
            ["--session", "write", "--facts", "2", "--steps", "4",
             "--snapshot", "runs/t6_memory_quick", "--out", "runs/t6_quick_write.json"],
            ["--session", "read", "--snapshot", "runs/t6_memory_quick",
             "--out", "runs/t6_quick_read.json"],
        ],
    },
    "t7": {
        "script": "t7d_longmemeval_pilot.py",
        "inputs": ["t7d_longmemeval.json", "t7d_longmemeval_holdout.json",
                   "t7e_context_baselines.json", "t7f_judge.json",
                   "t7d_longmemeval_gen32.json", "judge_calibration_result.json",
                   "t7d_17b.json", "t7e_17b.json", "t7f_judge_17b.json",
                   "t7h_calibration.json", "t7g_baselines.json",
                   "t7i_multisession.json"],
        # Four phases: the weights run, its pre-registered held-out replication
        # (docs/06 section 9), the context/RAG baselines, and the LLM judge. The
        # judge needs the user's API key and ~70 minutes at 3 requests/minute, so it
        # is only re-run deliberately -- but its report is what the paper quotes.
        # The 1.7B files are the scale control: same protocol, larger backbone.
        "phases": [
            ["--subset", "30", "--negatives", "5", "--max-new-tokens", "96",
             "--out", "runs/t7d_longmemeval.json"],
            ["--subset", "30", "--offset", "30", "--negatives", "5",
             "--max-new-tokens", "96", "--out", "runs/t7d_longmemeval_holdout.json"],
            ["t7e_context_baselines.py", "--subset", "30", "--max-new-tokens", "96",
             "--out", "runs/t7e_context_baselines.json"],
            ["t7f_judge_answers.py", "--full", "--questions", "30",
             "--standards", "lenient"],
        ],
        "quick_phases": [
            ["--subset", "3", "--negatives", "2", "--steps", "4",
             "--out", "runs/t7d_quick.json"],
            ["--subset", "3", "--offset", "3", "--negatives", "2", "--steps", "4",
             "--out", "runs/t7d_quick_holdout.json"],
        ],
    },
    "t8": {
        "script": "t8a_retention.py",
        "inputs": ["t8a_large.json", "t8a_confirm.json"],
        # B1: the write objective gains a retention term, swept over its weight.
        # lambda 0 reproduces the T2 behaviour, so the comparison is on the same
        # items and probes. Both the 8-episode run and the earlier 3-episode one are
        # inputs: the small one looked like a repair (p=0.070) and the large one did
        # not replicate it, and that difference is a result, not bookkeeping.
        "full": ["--items", "8", "--episodes", "8", "--paraphrase",
                 "--lambda-ret", "0", "3", "--out", "runs/t8a_large.json"],
        "quick": ["--items", "4", "--episodes", "1", "--paraphrase", "--steps", "4",
                  "--lambda-ret", "0", "3", "--out", "runs/t8a_quick.json"],
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
    """Run one stage (one or more processes). Returns the JSON files it produced.

    A phase whose first token ends in ``.py`` runs *that* script instead of the
    stage's default one, which lets a single stage (T7) cover the weights run, the
    context baselines and the judge while still reading as one unit.
    """
    script = str(ROOT / "experiments" / recipe["script"])
    if "phases" in recipe:
        plan = recipe["quick_phases"] if mode == "quick" else recipe["phases"]
        for index, phase_args in enumerate(plan):
            phase_args = list(phase_args)
            phase_script = script
            if phase_args and str(phase_args[0]).endswith(".py"):
                phase_script = str(ROOT / "experiments" / phase_args.pop(0))
            code = _run([phase_script, *phase_args], log_dir / f"{stage}_phase{index}.log")
            if code != 0:
                raise RuntimeError(f"{stage} phase {index} exited with {code}")
    else:
        args = recipe["quick"] if mode == "quick" else recipe["full"]
        code = _run([script, *args], log_dir / f"{stage}.log")
        if code != 0:
            raise RuntimeError(f"{stage} exited with {code}")
    return list(recipe["inputs"])


def collect(stage: str, recipe: dict, mode: str) -> dict:
    """Read a stage's reports back into the bundle.

    Payloads are paired with their *filenames* before anything else: zipping the
    input list against whatever files happen to exist would silently misalign the
    merge as soon as one report is missing.
    """
    pairs = []
    for name in recipe["inputs"]:
        path = RUNS / name
        if path.is_file():
            pairs.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
    if stage == "t6":
        by_name = dict(pairs)
        return {"write": by_name.get("t6_write.json", {}),
                "read": by_name.get("t6_read.json", {})}
    if stage == "t7":
        merged: dict = {}
        scale: dict = {}
        for name, payload in pairs:
            # "17b" is checked FIRST: t7e_17b.json also starts with "t7e", and
            # letting the generic prefix win would silently overwrite the 0.6B
            # baselines with the larger-backbone run's numbers.
            if "holdout" in name:
                merged["holdout"] = payload
            elif "17b" in name:
                if name.startswith("t7d"):
                    scale["weights"] = payload
                elif name.startswith("t7e"):
                    scale["baselines"] = payload
                elif name.startswith("t7f"):
                    scale["judge"] = payload
            elif name.startswith("t7e"):
                merged["baselines"] = payload
            elif name.startswith("t7f"):
                merged["judge"] = payload
            elif "gen32" in name:
                # the superseded 32-token run: the paper quotes it to show that an
                # earlier "the arms are indistinguishable" reading was an artefact
                # of our own generation budget
                merged["archived_gen32"] = payload
            elif name.startswith("t7g"):
                # training baselines on the same thirty memories (full fine-tune,
                # one equal-capacity adapter, the slot bank)
                merged["training_baselines"] = payload
            elif name.startswith("t7i"):
                # multi-session items with memories extracted from the conversation
                merged["multi_session"] = payload
            elif name.startswith("t7h"):
                # merged human-vs-judge calibration; authoritative over the older
                # single-pass result file, which recorded the first prompt's run
                merged["calibration_merged"] = payload
            elif name.startswith("judge_calibration"):
                merged["calibration"] = payload
            else:
                merged.update(payload)
        if scale:
            merged["scale_17b"] = scale
        return merged
    if stage == "t2":
        by_name = dict(pairs)
        merged = dict(by_name.get("t2_paraphrase.json") or {})
        if "t2_paraphrase_17b.json" in by_name:
            merged["scale_17b"] = by_name["t2_paraphrase_17b.json"]
        if "t2c_router_keys.json" in by_name:
            merged["router_keys"] = by_name["t2c_router_keys.json"]
        if "t2c_router_keys_17b.json" in by_name:
            merged["router_keys_17b"] = by_name["t2c_router_keys_17b.json"]
        if "t2b_mechanism.json" in by_name:
            merged["mechanism"] = by_name["t2b_mechanism.json"]
        if "t2d_composable.json" in by_name:
            merged["composable"] = by_name["t2d_composable.json"]
        return merged
    if stage in ("t4", "t5"):
        by_name = dict(pairs)
        primary = "t4_capacity.json" if stage == "t4" else "t5_inertia.json"
        merged = dict(by_name.get(primary) or {})
        extra = [by_name[name] for name in sorted(by_name)
                 if "seed" in name and name != primary]
        if extra:
            merged["extra_seeds"] = extra
        return merged
    if stage == "t3":
        by_name = dict(pairs)
        merged = dict(by_name.get("t3_write_policy.json") or {})
        extra = [by_name[name] for name in ("t3_seed1.json", "t3_seed2.json")
                 if name in by_name]
        if extra:
            merged["extra_seeds"] = extra
        if "t3_forced_span.json" in by_name:
            merged["forced_span"] = by_name["t3_forced_span.json"]
        return merged
    if stage == "t8":
        by_name = dict(pairs)
        merged = dict(by_name.get("t8a_large.json") or by_name.get("t8a_confirm.json")
                      or {})
        if "t8a_confirm.json" in by_name:
            merged["small_run"] = by_name["t8a_confirm.json"]
        return merged
    return pairs[0][1] if pairs else {}


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
