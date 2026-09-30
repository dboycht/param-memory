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
* grading is containment of the gold answer, which the extraction never sees.

The cost of that honesty is noise: an extraction error is written into the weights and
then read back as though it were memory, so the yield of the extraction step is reported
alongside the accuracy rather than buried.

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
    ap.add_argument("--max-memories", type=int, default=64,
                    help="cap on slots; the bank size is a configuration choice, and a "
                         "cap that silently dropped memories would bias the result, so "
                         "the script reports when it binds")
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
    from parammem.eval.judge import LLMJudge, load_api_key

    raw = json.loads(Path(args.data).read_text(encoding="utf-8"))
    items = select_multi_session(raw, args.items, args.offset)
    key = load_api_key(args.config)
    # max_tokens stays at the client default: a truncated JSON array fails to parse and
    # the memory is lost, which is exactly the failure this stage can least afford.
    judge = LLMJudge(api_key=key, model="kimi-k2.6",
                     base_url="https://api.moonshot.cn/v1",
                     min_interval=args.min_interval, timeout=args.timeout)
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
    flat = [(item_index, mem) for item_index, item in enumerate(items)
            for mem in item["memories"]]
    capped = flat[: args.max_memories]
    if len(capped) < len(flat):
        print(f"  WARNING: the slot cap binds: {len(flat)} memories extracted, "
              f"{len(capped)} written", flush=True)

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

    results = []
    for item_index, item in enumerate(items):
        own = [slot for slot, (i, _m) in zip(slot_of, capped) if i == item_index]
        if not own:
            continue
        question = str(item["question"])
        bb.set_read_slots([])
        query_key = bb.query_key(question)
        by_key = route(query_key, key_of, k=1).slots
        by_lex = route_lexical(question, text_of, k=1).slots

        row = {"question_id": item["question_id"], "n_own_memories": len(own),
               "routed_by_key_to_own": bool(by_key and by_key[0] in own),
               "routed_by_lexical_to_own": bool(by_lex and by_lex[0] in own),
               "answers": {}}
        for arm, active in (("key", by_key), ("lexical", by_lex),
                            ("oracle", own[:1]), ("frozen", [])):
            bb.set_read_slots(active)
            answer = bb.answer(question)
            row["answers"][arm] = {
                "text": answer,
                "correct": bool(contains_answer(answer, str(item["answer"]))),
                "exact": bool(P.exact_match(answer, str(item["answer"]))),
            }
        results.append(row)
        print(f"  {item['question_id'][:8]}… key={row['answers']['key']['correct']} "
              f"lexical={row['answers']['lexical']['correct']} "
              f"oracle={row['answers']['oracle']['correct']}", flush=True)

    def rate(arm: str) -> float:
        return sum(1 for row in results if row["answers"][arm]["correct"]) / len(results)

    summary = {
        "note": "Multi-session LongMemEval with memories extracted from the conversation "
                "and retrieval by the real question. The oracle arm shows what the same "
                "memories are worth when the right slot is known, so the gap between "
                "oracle and routing is the retrieval cost rather than a memory failure.",
        "n_items": len(results),
        "n_memories": len(slot_of),
        "cap_bound": len(capped) < len(flat),
        "containment": {arm: rate(arm) for arm in ("key", "lexical", "oracle", "frozen")},
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
    print(f"\n  containment: " + "  ".join(
        f"{arm}={value:.3f}" for arm, value in summary["containment"].items()))
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
