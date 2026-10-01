"""Fill in the gradings a throttled provider left missing, without redoing the generation.

The judge is the one call in these runs that can fail for reasons outside our control: the
provider answers 429 "all upstream providers are cooling down", and when the retries are
exhausted the question is left ungraded. The generations are expensive and already saved, so
throwing a run away over a flaky grader is the wrong response.

This re-grades only the rows whose verdicts are missing, updates the run file in place, and
reports the coverage before and after so a partially graded run cannot be mistaken for a
complete one.

Usage::

    python experiments/t7i_regrade.py --runs runs/t7i_multisession.json
    python experiments/t7i_regrade.py --runs runs/t7i_multisession.json runs/t7i_hosted_reader.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parammem.bench.longmemeval import load_oracle, select_multi_session  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

CONFIG_DEFAULT = Path(r"D:\code\DeepSeekHarness\ai-info-search\config.json")
DATA_DEFAULT = ROOT / "data" / "public" / "longmemeval_oracle.json"


def coverage(payload: dict) -> tuple[int, int]:
    """(rows with every arm graded, rows total)."""
    rows = payload.get("results") or []
    complete = 0
    for row in rows:
        answers = row.get("answers") or {}
        if answers and all(entry.get("judged") is not None for entry in answers.values()):
            complete += 1
    return complete, len(rows)


def recompute(payload: dict) -> dict:
    """Recompute the summary rates and counters from the per-question verdicts.

    Regrading writes verdicts into rows, and a summary left holding the pre-regrade numbers
    makes the file argue with itself: a stage can report one rate while its own rows contain
    a different set of graded questions. Both files therefore get their aggregates rebuilt
    from the rows after any change, and the rebuild is reported rather than assumed.
    """
    rows = payload.get("results") or []
    if not rows:
        return {}
    arms = payload.get("arms") or list(rows[0].get("answers") or {})
    judged_rows = [row for row in rows
                   if any((row["answers"].get(arm) or {}).get("judged") is not None
                          for arm in arms if arm in row["answers"])]

    def string_rate(arm: str) -> float:
        return sum(1 for row in rows if row["answers"][arm].get("correct")) / len(rows)

    def judged_rate(arm: str) -> float:
        graded = [row for row in judged_rows
                  if row["answers"][arm].get("judged") is not None]
        if not graded:
            return 0.0
        return sum(1 for row in graded if row["answers"][arm]["judged"]) / len(graded)

    before = {"judged": dict(payload.get("judged") or {}),
              "n_judged": payload.get("n_judged"), "n_graded": payload.get("n_graded")}
    if any("correct" in (row["answers"][arms[0]] or {}) for row in rows):
        payload["containment"] = {arm: string_rate(arm) for arm in arms}
    payload["judged"] = {arm: judged_rate(arm) for arm in arms}
    # The denominator that matters is the one behind the rate being quoted. Counting rows
    # with *any* arm graded would let a sentence say "0.68 of 30" while the arm it names was
    # graded on 28, which is the sort of quiet mismatch this pass exists to prevent.
    primary = "context" if "context" in arms else arms[0]
    graded_primary = sum(1 for row in rows
                         if row["answers"][primary].get("judged") is not None)
    payload["n_primary_graded"] = graded_primary
    payload["primary_arm"] = primary
    if "n_judged" in payload:
        payload["n_judged"] = graded_primary
    if "n_graded" in payload:
        payload["n_graded"] = graded_primary
    return {"before": before, "after": {"judged": payload["judged"],
                                        "n_judged": payload.get("n_judged"),
                                        "n_graded": payload.get("n_graded"),
                                        "primary_arm": primary,
                                        "primary_graded": graded_primary}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default=str(DATA_DEFAULT))
    ap.add_argument("--config", default=str(CONFIG_DEFAULT))
    args = ap.parse_args()

    from parammem.eval.judge import LLMJudge, load_api_key, load_llm_settings

    settings = load_llm_settings(args.config)
    judge = LLMJudge(api_key=load_api_key(args.config), model=settings["model"],
                     base_url=settings["base_url"],
                     min_interval=settings.get("min_interval", 0.0),
                     timeout=settings.get("timeout", 120.0))
    raw = load_oracle(args.data)
    references = {str(i["question_id"]): str(i["answer"])
                  for i in select_multi_session(raw, 10_000)}
    print(f"  judge: {settings['model']}, {settings.get('min_interval', 0)}s between calls",
          flush=True)

    for name in args.runs:
        path = Path(name)
        if not path.is_file():
            print(f"  missing {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        before = coverage(payload)
        todo = [row for row in payload.get("results") or []
                if row.get("question_id") in references
                and any(entry.get("judged") is None
                        for entry in (row.get("answers") or {}).values())]
        print(f"\n  {path.name}: {before[0]}/{before[1]} rows fully graded, "
              f"{len(todo)} to retry", flush=True)
        for row in todo:
            candidates = {arm: entry.get("text", "")
                          for arm, entry in row["answers"].items()}
            try:
                verdicts = judge.judge_batch(references[row["question_id"]], candidates)
            except Exception as error:
                print(f"    {str(row['question_id'])[:8]}… still failing: "
                      f"{type(error).__name__}: {str(error)[:90]}", flush=True)
                continue
            filled = 0
            for arm, verdict in verdicts.items():
                if arm in row["answers"] and row["answers"][arm]["judged"] is None:
                    row["answers"][arm]["judged"] = verdict.correct
                    filled += 1
            print(f"    {str(row['question_id'])[:8]}… regraded {filled} arms", flush=True)
        # Always rebuild the aggregates, even when nothing was retried: the point of this
        # pass is that the summary and the rows agree afterwards.
        changed = recompute(payload)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        after = coverage(payload)
        print(f"  {path.name}: {after[0]}/{after[1]} rows fully graded "
              f"(was {before[0]})", flush=True)
        if changed:
            print(f"    aggregates rebuilt: {changed['after']['judged']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
