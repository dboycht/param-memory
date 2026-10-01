"""Can a reader that *can* aggregate still fail, if the read path hands it one fact?

The backbone experiment answers this badly on the multi-session subset, and the reason is
worth stating: these questions ask for an aggregate across several memories, and at 0.6B the
context arm scores zero as well, so both arms fail for a reason upstream of the mechanism.
Nothing about the read path can be concluded from a comparison in which neither side can do
the task.

This separates the two by giving the reader enough capability to do the arithmetic and
varying only what the read path delivers:

* ``context``: every fact extracted for the question;
* ``top1``: the single best fact by lexical routing over the whole bank;
* ``top2``: the two best.

The prompt, the reader and the instruction are identical across arms; only the fact set
changes. If ``context`` beats ``top1`` here, the limit is what the read path delivers rather
than what the reader can understand, which is the claim the synthetic two-slot collapse makes
and this measures on real questions.

The reader is a hosted model, so this is a capability control rather than a measurement of
the mechanism's own reader; the paper says so where it reports it.

Usage::

    python experiments/t7i_hosted_reader.py --items 30 --per-item 2
    python experiments/t7i_hosted_reader.py --items 3 --no-grade     # plumbing check
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
from parammem.memory.router import route_lexical  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

CONFIG_DEFAULT = Path(r"D:\code\DeepSeekHarness\ai-info-search\config.json")

SYSTEM = ("You answer a question using only the facts given to you. Reply with the answer "
          "alone, no explanation. If the facts are not enough, reply exactly: "
          "the information provided is not enough.")


def fact_text(memory: dict) -> str:
    return f"- {memory.get('question', '')} {memory.get('answer', '')}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--per-item", type=int, default=2,
                    help="facts per item written to the bank, matching the backbone run. "
                         "The context arm is not limited by this: it gets every fact "
                         "extracted for the question, since an aggregate may need more "
                         "components than the bank holds")
    ap.add_argument("--extracted", default="runs/t7i_extracted.json")
    ap.add_argument("--data", default=str(ROOT / "data" / "public" / "longmemeval_oracle.json"))
    ap.add_argument("--config", default=str(CONFIG_DEFAULT))
    ap.add_argument("--judge", dest="judge", action="store_true", default=True)
    ap.add_argument("--no-grade", dest="judge", action="store_false")
    ap.add_argument("--out", default="runs/t7i_hosted_reader.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    from parammem.eval.judge import LLMJudge, load_api_key, load_llm_settings

    settings = load_llm_settings(args.config)
    reader = LLMJudge(api_key=load_api_key(args.config), model=settings["model"],
                      base_url=settings["base_url"],
                      min_interval=settings.get("min_interval", 0.0),
                      timeout=settings.get("timeout", 120.0))
    print(f"  reader: {settings['model']} at {settings['base_url']}, "
          f"{settings.get('min_interval', 0)}s between calls", flush=True)

    payload = json.loads(Path(args.extracted).read_text(encoding="utf-8"))
    items = payload["items"][: args.items]
    questions = {str(i["question_id"]): str(i["question"])
                 for i in select_multi_session(load_oracle(args.data), args.items)}
    references = {str(i["question_id"]): str(i["answer"])
                  for i in select_multi_session(load_oracle(args.data), args.items)}

    # The bank exactly as the backbone run writes it: a couple of facts per item, plus every
    # other item's facts as distractors for routing.
    bank = [(index, memory) for index, item in enumerate(items)
            for memory in item["memories"][: args.per_item]]
    text_of = {slot: memory.get("question", "") for slot, (_i, memory) in enumerate(bank)}
    print(f"  bank: {len(bank)} facts from {len(items)} questions", flush=True)

    results = []
    for index, item in enumerate(items):
        question_id = str(item["question_id"])
        if question_id not in questions:
            continue
        # The control gets *every* fact extracted for this question, not the two the bank
        # holds. These questions ask for an aggregate across several components, so a
        # context arm limited to the bank's size would abstain for the right reason and be
        # read as a capability limit.
        own = list(item["memories"])
        question = questions[question_id]
        ranked = route_lexical(question, text_of, k=2).slots
        arm_facts = [("context", [fact_text(m) for m in own]),
                     ("top2", [fact_text(bank[s][1]) for s in ranked]),
                     ("top1", [fact_text(bank[ranked[0]][1])] if ranked else [])]

        row = {"question_id": question_id, "n_own_facts": len(own), "answers": {}}
        for arm, facts in arm_facts:
            prompt = (f"{SYSTEM}\n\nFacts:\n" + "\n".join(facts) +
                      f"\n\nQuestion: {question}\nAnswer:")
            try:
                text = reader.complete(prompt).strip()
            except Exception as error:
                print(f"    {question_id[:8]}… {arm} failed: {type(error).__name__}: "
                      f"{str(error)[:100]}", flush=True)
                text = ""
            row["answers"][arm] = {"text": text, "n_facts": len(facts), "judged": None}
        results.append(row)
        print(f"  {question_id[:8]}… " + "  ".join(
            f"{arm}={row['answers'][arm]['text'][:28]!r}"
            for arm, _f in arm_facts), flush=True)

    if args.judge and results:
        print(f"  grading {len(results)} questions", flush=True)
        for row in results:
            candidates = {arm: entry["text"] for arm, entry in row["answers"].items()}
            try:
                verdicts = reader.judge_batch(references[row["question_id"]], candidates)
            except Exception as error:
                print(f"    {row['question_id'][:8]}… grading failed: "
                      f"{type(error).__name__}: {str(error)[:100]}", flush=True)
                continue
            for arm, verdict in verdicts.items():
                if arm in row["answers"]:
                    row["answers"][arm]["judged"] = verdict.correct

    arms = ("context", "top2", "top1")

    def rate(arm: str) -> float:
        graded = [row for row in results if row["answers"][arm]["judged"] is not None]
        if not graded:
            return 0.0
        return sum(1 for row in graded if row["answers"][arm]["judged"]) / len(graded)

    summary = {
        "note": "A capable reader (hosted) given the same fact set the read path would "
                "deliver: everything extracted for the question, the two best facts by "
                "lexical routing over the whole bank, or only the best one. The instruction "
                "and reader are identical across arms, so a gap measures what the read path "
                "delivers rather than what the reader can do. This is a capability control, "
                "not the mechanism's own reader.",
        "reader": settings["model"],
        "n_items": len(results),
        "n_facts_in_bank": len(bank),
        "judged": {arm: rate(arm) for arm in arms},
        "n_graded": sum(1 for row in results
                        if row["answers"]["context"]["judged"] is not None),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  judged over {summary['n_graded']} questions: " + "  ".join(
        f"{arm}={value:.3f}" for arm, value in summary["judged"].items()))
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
