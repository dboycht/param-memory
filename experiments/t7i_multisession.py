"""T7-i -- multi-session questions with memories extracted from the conversation.

The T7-d diagnostic writes the benchmark's own ``(question, answer)`` pair, so the
memory *is* the gold item: retrieval is oracle-indexed and the stored value is the
reference answer. That is a scoped diagnostic, and this is the next step up.

What changes here:

* the memories come from the **conversation**, not from the reference answer, and they
  are extracted from **every** session of the item rather than from the turns the
  benchmark flags as holding the answer. The oracle is removed from the extraction
  entirely;
* the read query is the benchmark's **real question**, routed against every extracted
  memory of every item in the pilot, so retrieval is a genuine search rather than a
  lookup;
* answers are graded by the **calibrated judge**, not by string containment.

That last change is not a preference. Measured on these thirty items
(``t7i_composition_demand.py``): the reference answer is a literal string in the
conversation for only half of them, and twenty-nine of the thirty ask for an aggregate --
how many, how much in total, what percentage, what difference -- which has to be computed
across several memories. String containment therefore has a ceiling of about a half no
matter how good the memories are, and it cannot tell a missing fact from an unperformed
addition.

The arms follow from that. ``context`` shows the same extracted facts to the same model in
the prompt, so it answers the question the bank cannot: whether the information is present
at all. If the bank's arms score below it, the limit is the read path rather than the
extraction, which is the claim this experiment exists to test. ``top2`` and ``all`` ask
whether opening more slots composes them, which the synthetic results say it does not.

Run in two stages so the network half can proceed while the GPU is busy::

    python experiments/t7i_multisession.py --extract --items 5
    python experiments/t7i_multisession.py --run --items 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from parammem.bench import protocol as P
from parammem.bench.extract import extract_pairs, flatten_turns
from parammem.bench.longmemeval import contains_answer, sha256_file
from parammem.memory.router import route, route_lexical
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"
#: The credential lives in the sibling project's config; it is read, never printed.
CONFIG_DEFAULT = Path(r"D:\code\DeepSeekHarness\ai-info-search\config.json")


def select_multi_session(raw: list[dict], subset: int, offset: int = 0) -> list[dict]:
    """Deterministically take ``subset`` multi-session items, ordered by id."""
    pool = [d for d in raw if str(d.get("question_type", "")) == "multi-session"]
    pool.sort(key=lambda d: d["question_id"])
    return pool[offset: offset + subset]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", action="store_true", help="run the extraction stage")
    ap.add_argument("--run", action="store_true", help="run the write/route/answer stage")
    ap.add_argument("--items", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--config", default=str(CONFIG_DEFAULT))
    ap.add_argument("--extracted", default="runs/t7i_extracted.json")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--judge", dest="judge", action="store_true", default=True,
                    help="grade with the calibrated judge; string containment is kept "
                         "as a secondary column because the papers quote it")
    ap.add_argument("--no-judge", dest="judge", action="store_false",
                    help="skip grading, for a GPU-only smoke test of the arms")
    ap.add_argument("--judge-items", type=int, default=0,
                    help="grade only the first N questions, to check the plumbing on a "
                         "handful before spending the rate limit on all of them")
    ap.add_argument("--run-items", type=int, default=0,
                    help="run only the first N extracted items (0 = all); a smoke test "
                         "that prints when it binds rather than truncating silently")
    ap.add_argument("--max-memories", type=int, default=64,
                    help="cap on slots when --per-item is 0; the bank size is a "
                         "configuration choice, and a larger bank costs write time")
    ap.add_argument("--per-item", type=int, default=0,
                    help="take this many memories from every item instead of a prefix, so "
                         "a capped bank still covers every question")
    ap.add_argument("--min-interval", type=float, default=25.0,
                    help="seconds between calls. The organisation limit is 3 requests "
                         "per minute, so 21s sits exactly on the boundary and a run that "
                         "drifts over it eats 429s; 25s leaves margin")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds. The judge client defaults to 90, which is right for a "
                         "one-line verdict and wrong here: extracting from a twelve-turn "
                         "session makes a reasoning model think for minutes, so every "
                         "call timed out and retried (measured: six minutes per call "
                         "instead of the response time)")
    ap.add_argument("--out", default="runs/t7i_multisession.json")
    return ap.parse_args()


def do_extract(args) -> int:
    from parammem.eval.judge import LLMJudge, load_api_key, load_llm_settings

    raw = json.loads(Path(args.data).read_text(encoding="utf-8"))
    items = select_multi_session(raw, args.items, args.offset)
    key = load_api_key(args.config)
    # The provider comes from the config file, not from this script. Hard-coding it here
    # meant that changing the account in the config kept sending calls to the old host,
    # which is exactly what happened when the previous provider's balance ran out.
    settings = load_llm_settings(args.config)
    # max_tokens stays at the client default: a truncated JSON array fails to parse and
    # the memory is lost, which is exactly the failure this stage can least afford.
    judge = LLMJudge(api_key=key,
                     model=settings["model"], base_url=settings["base_url"],
                     min_interval=settings.get("min_interval", args.min_interval),
                     timeout=settings.get("timeout", args.timeout))
    print(f"  provider: {settings['base_url']}  model: {settings['model']}  "
          f"min_interval: {settings.get('min_interval', args.min_interval)}s", flush=True)
    print(f"extracting from {len(items)} multi-session items "
          f"({sha256_file(args.data)[:16]}…)", flush=True)

    records = []
    diagnostics: list[dict] = []
    out = Path(args.extracted)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        """Checkpoint after every item: extraction costs an hour of API time, and the
        first version wrote only at the end, so an interruption threw away every session
        already paid for. The per-session diagnostics are saved with it, because an
        empty extraction and a failed call are indistinguishable in a count."""
        total = sum(len(record["memories"]) for record in records)
        out.write_text(json.dumps({
            "note": "Memory items extracted from every session of each multi-session "
                    "LongMemEval item. The extraction never sees the reference answer, "
                    "and the read query is the benchmark's own question, so retrieval is "
                    "a real search over every extracted memory rather than a lookup.",
            "status": f"{len(records)}/{len(items)} items",
            "complete": len(records) == len(items),
            "data_sha256": sha256_file(args.data),
            "items": records,
            "n_items": len(records),
            "n_memories": total,
            "mean_memories_per_item": total / len(records) if records else None,
            "diagnostics": diagnostics,
            "call_status_counts": {
                status: sum(1 for d in diagnostics if d["status"] == status)
                for status in ("ok", "unparseable", "error")},
            "elapsed_s": time.perf_counter() - t0,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    for index, item in enumerate(items, start=1):
        sessions = item.get("haystack_sessions") or []
        item_pairs = []
        for session_index, session in enumerate(sessions):
            turns = flatten_turns(session)
            if not turns:
                continue
            seen: list[dict] = []

            def observe(fields, seen=seen, item_id=item["question_id"],
                        session_index=session_index):
                fields = dict(fields, item=item_id, session=session_index,
                              finish=judge.last_finish_reason)
                seen.append(fields)
                diagnostics.append(fields)

            pairs = extract_pairs(judge.complete, turns, observer=observe)
            item_pairs.extend({"question": pair["question"], "answer": pair["answer"],
                               "session": session_index} for pair in pairs)
            status = seen[-1]["status"] if seen else "empty-input"
            print(f"  item {index}/{len(items)} session {session_index}: "
                  f"{len(pairs)} memories (total {len(item_pairs)}) [{status}, "
                  f"finish={seen[-1].get('finish', '') if seen else ''}]", flush=True)
        records.append({
            "question_id": item["question_id"],
            "question": str(item.get("question", "")),
            "answer": str(item.get("answer", "")),
            "n_sessions": len(sessions),
            "memories": item_pairs,
        })
        save()

    total = sum(len(r["memories"]) for r in records)
    counts = {status: sum(1 for d in diagnostics if d["status"] == status)
              for status in ("ok", "unparseable", "error")}
    print(f"\n  {total} memories from {len(records)} items "
          f"({total / len(records) if records else 0:.1f} per item)")
    print(f"  call outcomes: {counts}")
    if counts["error"]:
        print(f"  WARNING: {counts['error']} call(s) failed outright (rate limit or "
              f"transport). Those sessions contribute zero memories and must never be "
              f"read as 'the model found no durable fact'.", flush=True)
    if counts["unparseable"]:
        print(f"  NOTE: {counts['unparseable']} call(s) returned something that did not "
              f"parse; check the finish reasons before attributing them to the model.",
              flush=True)
    print(f"extraction written to {out.resolve()}")
    return 0


def do_run(args) -> int:
    payload = json.loads(Path(args.extracted).read_text(encoding="utf-8"))
    items = payload["items"]
    if args.run_items > 0:
        # An explicit limit, and it announces itself: a run that silently covered a
        # fraction of the questions would produce a summary indistinguishable from the
        # full one, which is the failure this flag exists to make impossible.
        print(f"  --run-items {args.run_items}: covering {args.run_items} of "
              f"{len(items)} extracted items", flush=True)
        items = items[: args.run_items]
    flat = [(item_index, mem) for item_index, item in enumerate(items)
            for mem in item["memories"]]
    if args.per_item > 0:
        # Round-robin instead of "the first N": taking a prefix of 343 memories covers
        # only the first few items, so the routing test would ask a handful of questions.
        # A few memories per item covers every question at the same write cost.
        capped = [(item_index, mem)
                  for item_index, item in enumerate(items)
                  for mem in item["memories"][: args.per_item]]
    else:
        capped = flat[: args.max_memories]
    covered = sorted({item_index for item_index, _mem in capped})
    if len(capped) < len(flat):
        print(f"  slot cap: {len(flat)} memories extracted, {len(capped)} written, "
              f"covering {len(covered)}/{len(items)} items", flush=True)

    cfg = BackboneConfig(model_id=resolve_model_path(args.model),
                         n_slots=max(len(capped), 2), rank=args.rank,
                         alpha=args.alpha, max_new_tokens=args.max_new_tokens)
    print(f"loading {cfg.model_id} for {len(capped)} extracted memories ...", flush=True)
    bb = Backbone.load(cfg)
    bb.erase_all()
    bb.set_read_slots([])
    anchors = bb.anchor_logits(GENERIC_ANCHORS) if args.lambda_kl > 0 else {}

    slot_of: list[int] = []
    for slot, (_item_index, mem) in enumerate(capped):
        def loss_fn(mem=mem, slot=slot):
            bb.set_read_slots([slot])
            loss = bb.write_loss(mem["question"], mem["answer"])
            if anchors:
                loss = loss + args.lambda_kl * bb.kl_to_anchors(anchors)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        slot_of.append(slot)
    print(f"  wrote {len(slot_of)} memories", flush=True)

    # Retrieval keys: the stored memory questions, captured with the memory off.
    bb.set_read_slots([])
    key_of = {slot: bb.query_key(mem["question"])
              for slot, (_i, mem) in zip(slot_of, capped)}
    text_of = {slot: mem["question"] for slot, (_i, mem) in zip(slot_of, capped)}

    # The arms. ``oracle`` is deliberately absent: activating "the item's own memory" is
    # not a well-defined arm when the answer is an aggregate of several of them, and
    # pretending otherwise is what made the earlier reading of this run wrong.
    ARMS = ("frozen", "context", "key", "lexical", "top2", "all")
    results = []
    for item_index, item in enumerate(items):
        own = [slot for slot, (i, _m) in zip(slot_of, capped) if i == item_index]
        if not own:
            continue
        question = str(item["question"])
        bb.set_read_slots([])
        query_key = bb.query_key(question)
        ranked = route(query_key, key_of, k=2).slots
        by_key, by_lex = ranked[:1], route_lexical(question, text_of, k=1).slots
        # The control: the same extracted facts, in the prompt, in the order they were
        # written. Same facts and same model as every other arm; only the delivery differs.
        home = [(i, mem) for i, mem in
                [(i, mem) for slot, (i, mem) in zip(slot_of, capped)] if i == item_index]
        context = "\n".join(f"- {mem['question']} {mem['answer']}" for _i, mem in home)

        row = {"question_id": item["question_id"], "n_own_memories": len(own),
               "routed_by_key_to_own": bool(by_key and by_key[0] in own),
               "routed_by_lexical_to_own": bool(by_lex and by_lex[0] in own),
               "answers": {}}
        for arm, active, ctx in (("frozen", [], ""), ("context", [], context),
                                 ("key", by_key, ""), ("lexical", by_lex, ""),
                                 ("top2", ranked, ""), ("all", slot_of, "")):
            bb.set_read_slots(active)
            answer = bb.answer(question, context=ctx)
            row["answers"][arm] = {
                "text": answer,
                "correct": bool(contains_answer(answer, str(item["answer"]))),
                "exact": bool(P.exact_match(answer, str(item["answer"]))),
                "judged": None,
                "prompt_tokens": bb.prompt_tokens(question, context=ctx),
            }
        results.append(row)
        print(f"  {item['question_id'][:8]}… " + " ".join(
            f"{arm}={int(row['answers'][arm]['correct'])}" for arm in ARMS), flush=True)

    # Grading by the calibrated judge, one call per question with every arm as a candidate:
    # the rate limit is the binding constraint, and batching keeps the judge's standard
    # identical across the arms being compared.
    if args.judge and results:
        from parammem.eval.judge import LLMJudge, load_api_key, load_llm_settings

        settings = load_llm_settings(args.config)
        judge = LLMJudge(api_key=load_api_key(args.config), model=settings["model"],
                         base_url=settings["base_url"],
                         min_interval=settings.get("min_interval", args.min_interval),
                         timeout=settings.get("timeout", args.timeout))
        graded = results[: args.judge_items] if args.judge_items > 0 else results
        reference_of = {str(i["question_id"]): str(i["answer"]) for i in items}
        print(f"  grading {len(graded)} questions with {settings['model']}, "
              f"{settings.get('min_interval', 0)}s between calls", flush=True)
        for row in graded:
            candidates = {arm: entry["text"] for arm, entry in row["answers"].items()}
            try:
                verdicts = judge.judge_batch(reference_of[row["question_id"]], candidates)
            except Exception as error:  # a failed grade must not lose the generation
                print(f"    {row['question_id'][:8]}… grading failed: "
                      f"{type(error).__name__}: {str(error)[:120]}", flush=True)
                continue
            for arm, verdict in verdicts.items():
                if arm in row["answers"]:
                    row["answers"][arm]["judged"] = verdict.correct
            print(f"    {row['question_id'][:8]}… judged " + " ".join(
                f"{arm}={row['answers'][arm]['judged']}" for arm in ARMS), flush=True)

    def string_rate(arm: str) -> float:
        return sum(1 for row in results if row["answers"][arm]["correct"]) / len(results)

    judged_rows = [row for row in results
                   if any(row["answers"][arm]["judged"] is not None for arm in ARMS)]

    def judged_rate(arm: str) -> float:
        graded = [row for row in judged_rows if row["answers"][arm]["judged"] is not None]
        if not graded:
            return 0.0
        return sum(1 for row in graded if row["answers"][arm]["judged"]) / len(graded)

    summary = {
        "note": "Multi-session LongMemEval with memories extracted from the conversation and "
                "retrieval by the real question. These questions ask for an aggregate across "
                "several memories, so string containment has a low ceiling and grading is by "
                "the calibrated judge. The context arm is the control: the same extracted "
                "facts, delivered in the prompt rather than read from the bank.",
        "n_items": len(results),
        "n_memories": len(slot_of),
        "cap_bound": len(capped) < len(flat),
        "arms": list(ARMS),
        "containment": {arm: string_rate(arm) for arm in ARMS},
        "judged": {arm: judged_rate(arm) for arm in ARMS},
        "n_judged": len(judged_rows),
        "routed_to_own": {
            "key": sum(1 for row in results if row["routed_by_key_to_own"]) / len(results),
            "lexical": (sum(1 for row in results if row["routed_by_lexical_to_own"])
                        / len(results)),
        },
        "results": results,
        "model_path": cfg.model_id,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  string containment: " + "  ".join(
        f"{arm}={value:.3f}" for arm, value in summary["containment"].items()))
    if judged_rows:
        print(f"  judged correct over {len(judged_rows)} questions: " + "  ".join(
            f"{arm}={value:.3f}" for arm, value in summary["judged"].items()))
    print(f"  routed to the item's own memory: key="
          f"{summary['routed_to_own']['key']:.3f} lexical="
          f"{summary['routed_to_own']['lexical']:.3f}")
    print(f"\nreport written to {out.resolve()}")
    return 0


def main() -> int:
    args = parse_args()
    if args.extract:
        return do_extract(args)
    if args.run:
        return do_run(args)
    print("nothing to do: pass --extract and/or --run")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
