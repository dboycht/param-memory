"""T7-f -- grade the arms with an LLM judge, and validate the judge first.

Why: the containment metric is biased toward the arm trained to reproduce the
reference string verbatim (docs/06 section 10.2), so "which memory medium answers
better" needs a semantic judgement.

Two modes:

``--calibration``
    Judge a small, **anonymised** sample mixing obviously-wrong (frozen),
    verbatim (weights) and paraphrased (context) answers, and write a marking sheet
    for a human. The judge's own verdicts go at the *end* of the sheet so they
    cannot anchor the human. The judge is only trusted if the two agree.
``--full``
    Judge every arm on every question (one call per question, honouring the
    configured rate limit) and report per-arm correctness with bootstrap CIs.

The credential is read from a config file, kept in memory, never printed.

Usage::

    python experiments/t7f_judge_answers.py --calibration
    python experiments/t7f_judge_answers.py --full --questions 30
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from parammem.bench.protocol import bootstrap_ci
from parammem.eval.judge import LLMJudge, load_api_key

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"
CONFIG = Path(r"D:\code\DeepSeekHarness\ai-info-search\config.json")

ARM_ORDER = ("frozen", "weights", "context_target", "context_all", "rag_top1",
             "rag_top3")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="", help="judge model id (default: from config)")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--weights-result", default="runs/t7d_longmemeval.json")
    ap.add_argument("--baseline-result", default="runs/t7e_context_baselines.json")
    ap.add_argument("--questions", type=int, default=30)
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--calibration-questions", type=int, default=4)
    ap.add_argument("--min-interval", type=float, default=21.0,
                    help="seconds between judge calls (the source config asks for 21)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="omit to let the server use its own value (kimi-k2.6 only "
                         "accepts 1)")
    ap.add_argument("--out", default="")
    return ap.parse_args()


def load_answers(args) -> tuple[dict[str, dict], dict[str, dict[str, str]]]:
    """Return ``{qid: item}`` from the dataset and ``{qid: {arm: answer}}``."""
    raw = json.loads(Path(args.data).read_text(encoding="utf-8"))
    items = {d["question_id"]: d for d in raw}

    weights = json.loads(Path(args.weights_result).read_text(encoding="utf-8"))
    baselines = json.loads(Path(args.baseline_result).read_text(encoding="utf-8"))

    per_q: dict[str, dict[str, str]] = {}
    for row in weights["rows"]:
        per_q.setdefault(row["question_id"], {})["weights"] = row["mem_on"]
        per_q[row["question_id"]]["frozen"] = row["frozen"]
    for arm, rows in baselines["rows"].items():
        for row in rows:
            per_q.setdefault(row["question_id"], {})[arm] = row["answer"]
    # only questions present in both sources are comparable
    comparable = {
        qid: {arm: text for arm, text in arms.items() if arm in ARM_ORDER}
        for qid, arms in per_q.items()
        if qid in items and "weights" in arms and "frozen" in arms
    }
    return items, comparable


def run_judge(judge: LLMJudge, items, per_q, qids, *, log, show_raw: int = 0) -> dict[str, dict]:
    """One judge call per question, covering all of that question's candidates.

    ``show_raw`` prints the first few raw replies: if every verdict comes back
    unreadable, the *only* way to tell a format problem from a parsing bug is to
    look at what the model actually said.
    """
    results: dict[str, dict] = {}
    for index, qid in enumerate(qids, start=1):
        candidates = per_q[qid]
        verdicts = judge.judge_batch(str(items[qid]["answer"]), candidates)
        results[qid] = {
            label: {"correct": v.correct, "reason": v.reason, "raw": v.raw}
            for label, v in verdicts.items()
        }
        if index <= show_raw:
            sample = next(iter(verdicts.values())).raw if verdicts else ""
            log(f"  raw reply for {qid} (source={judge.last_source}): "
                f"{sample[:300]!r}", flush=True)
        log(f"  {index}/{len(qids)} judged", flush=True)
    return results


def write_calibration_sheet(path: Path, rows: list[dict], judge_view: list[dict]) -> None:
    lines = [
        "# 判官校准表（请你人工标注）",
        "",
        "对每一行判断：**这个回答是否表达了与参考答案相同的信息？**（措辞不同但意思对 ⇒ 算对）",
        "把 `Y`（对）或 `N`（错）填进最后一列。",
        "",
        "⚠️ 判官自己的结论在本文件**末尾**，请先填完再看，否则会锚定你的判断。",
        "⚠️ 行标签是**匿名且打乱**的：你看不出它来自哪个臂。",
        "",
        "| # | 问题 | 参考答案 | 待判回答 | 你的标注 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        def cell(text: str, limit: int) -> str:
            # a markdown table cell cannot contain newlines
            return " ".join(str(text).split())[:limit]
        lines.append(
            f"| {row['n']} | {cell(row['question'], 70)} | "
            f"{cell(row['reference'], 90)} | {cell(row['candidate'], 110)} | |"
        )
    lines += ["", "---", "", "## 判官结论（填完再看）", "",
              "| # | 判官 | 理由 |", "| --- | --- | --- |"]
    for entry in judge_view:
        verdict = {True: "对", False: "错", None: "读不出"}[entry["correct"]]
        lines.append(f"| {entry['n']} | {verdict} | {entry['reason'][:60]} |")
    lines += ["", "> 两列不一致的行，就是判官不可信的地方；一致率决定它能不能用来做对比。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    def log(message, **kwargs):
        print(message, **kwargs)

    items, per_q = load_answers(args)
    qids = sorted(per_q)[: args.questions]
    log(f"questions comparable across arms: {len(per_q)} (using {len(qids)})")

    config = json.loads(Path(args.config).read_text(encoding="utf-8")).get("llm", {})
    judge = LLMJudge(
        base_url=config["base_url"],
        model=args.model or config["model"],
        api_key=load_api_key(args.config),
        timeout=float(config.get("timeout", 90)),
        min_interval=args.min_interval,
        temperature=args.temperature,
    )
    log(f"judge: {judge!r}")      # repr masks the key

    t0 = time.perf_counter()
    if args.calibration:
        picked = qids[: args.calibration_questions]
        arms = ("frozen", "weights", "context_target")
        plan = [(qid, arm) for qid in picked for arm in arms]
        random.Random(0).shuffle(plan)          # deterministic anonymisation

        # One candidate per call: batching made the judge answer about the wrong
        # candidate (docs/06 section 11). Slower, but a subtly wrong judge is worse
        # than no judge.
        rows, judge_view = [], []
        for n, (qid, arm) in enumerate(plan, start=1):
            verdict = judge.judge_one(str(items[qid].get("question", "")),
                                      str(items[qid]["answer"]), per_q[qid][arm])
            rows.append({"n": n, "question": str(items[qid].get("question", "")),
                         "reference": str(items[qid]["answer"]),
                         "candidate": per_q[qid][arm]})
            judge_view.append({"n": n, "correct": verdict.correct,
                               "reason": verdict.reason})
            log(f"  {n}/{len(plan)} judged "
                f"(source={judge.last_source}, finish={judge.last_finish_reason})",
                flush=True)

        sheet = ROOT / "runs" / "judge_calibration.md"
        write_calibration_sheet(sheet, rows, judge_view)
        payload = {
            "judge": repr(judge),
            "mode": "single-candidate, one call each",
            "items": len(rows),
            "labels": {f"{qid}|{arm}": str(n) for n, (qid, arm) in
                       enumerate(plan, start=1)},
            "judge_verdicts": judge_view,
            "elapsed_s": time.perf_counter() - t0,
        }
        (ROOT / "runs" / "judge_calibration.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        unreadable = sum(1 for v in judge_view if v["correct"] is None)
        print(f"\njudged {len(rows)} candidates ({unreadable} unreadable)")
        print(f"marked-up sheet -> {sheet.resolve()}")
        print("请人工标注后再决定是否信任判官。")
        return 0

    judged = run_judge(judge, items, per_q, qids, log=log)
    arms = sorted({arm for qid in qids for arm in per_q[qid]})
    summary = {"judge": repr(judge), "n_questions": len(qids),
               "questions": qids, "arms": {}}
    for arm in arms:
        values = [1.0 if judged[qid].get(arm, {}).get("correct") else 0.0 for qid in qids]
        usable = [qid for qid in qids if judged[qid].get(arm, {}).get("correct") is not None]
        summary["arms"][arm] = {
            "correct_rate": statistics.mean(values),
            "usable": len(usable),
            "ci95": list(bootstrap_ci(values, n=2000, seed=0)),
        }
    summary["elapsed_s"] = time.perf_counter() - t0
    summary["rows"] = judged
    out = Path(args.out or (ROOT / "runs" / "t7f_judge.json"))
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'arm':<18}{'judged correct':>16}{'usable':>9}{'95% CI':>20}")
    for arm in arms:
        a = summary["arms"][arm]
        lo, hi = a["ci95"]
        print(f"{arm:<18}{a['correct_rate']:>16.3f}{a['usable']:>9}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>20}")
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
