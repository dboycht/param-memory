"""T7-d -- LongMemEval controlled diagnostic (protocol pre-registered in docs/06).

**The single question this answers**: with *real* question/answer content written
into the parameters, can the frozen 0.6B produce the answer while the chat history
is completely absent? It is about whether weights can hold real content -- **not**
about whether retrieval generalises.

Frozen protocol (docs/06 section 6.1, written *before* the run):

* source: ``longmemeval_oracle.json`` (SHA256 recorded), ``single-session-*`` only
* subset: the first 30 question ids after sorting, chosen deterministically
* one memory item = the benchmark's own ``(question -> answer)`` pair; evidence
  turns are located with the benchmark's ``has_answer`` flags (**oracle indexing**)
* one store accumulates all 30 items -> the router must pick the right one of 30
* arms: ``frozen`` / ``mem_on`` / ``mem_off`` (erase the slot, ask again)
* scoring: containment (normalised) **and** the raw answers for 10 manual checks
* forbidden: comparing against published LongMemEval numbers; hiding the fact that
  the write query and the read query are the same string (no paraphrase here)

Usage::

    python experiments/t7d_longmemeval_pilot.py --subset 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from parammem.bench import protocol as P
from parammem.memory.router import route
from parammem.memory.writer import write_slot
from parammem.model import GENERIC_ANCHORS, Backbone, BackboneConfig, resolve_model_path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "longmemeval_oracle.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--subset", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many single-session questions first; used for the "
                         "pre-registered held-out replication (docs/06 section 9)")
    ap.add_argument("--negatives", type=int, default=5,
                    help="unwritten questions asked with the full memory active (P3)")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-kl", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--manual-samples", type=int, default=10)
    ap.add_argument("--out", default="runs/t7d_longmemeval.json")
    return ap.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def containment(prediction: str, gold: str) -> float:
    """1.0 if the gold is present in the answer, 0.5 if they merely overlap.

    The benchmark's own scoring uses an LLM judge; we deliberately do not (it needs
    an API key) and report a coarse proxy instead, clearly labelled as such.
    """
    pred, target = P.normalize(prediction), P.normalize(gold)
    if not pred or not target:
        return 0.0
    if target in pred:
        return 1.0
    # multi-clause golds: accept if a decent share of the clauses appear
    clauses = [c.strip() for c in target.replace(";", " and ").split(" and ") if c.strip()]
    hits = sum(1 for c in clauses if c in pred)
    if clauses and hits / len(clauses) >= 0.5:
        return 0.5
    return 0.5 if P.token_f1(prediction, gold) >= 0.5 else 0.0


def evidence_turns(item: dict) -> int:
    return sum(1 for s in item.get("haystack_sessions") or []
               for turn in s if turn.get("has_answer"))


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"missing {data_path}; run _dev/probe_public_bench.py first", flush=True)
        return 1
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    digest = sha256(data_path)

    singles = [d for d in raw
               if str(d.get("question_type", "")).startswith("single-session")]
    singles.sort(key=lambda d: d["question_id"])
    chosen = singles[args.offset: args.offset + args.subset]
    negatives = singles[args.offset + args.subset:
                        args.offset + args.subset + args.negatives]
    print(f"file sha256 {digest[:16]}…  single-session pool {len(singles)}  "
          f"offset {args.offset}  subset {len(chosen)}  negatives {len(negatives)}",
          flush=True)

    cfg = BackboneConfig(
        model_id=resolve_model_path(args.model), n_slots=max(32, args.subset + 1),
        rank=args.rank, alpha=args.alpha, max_new_tokens=args.max_new_tokens,
    )
    print(f"loading {cfg.model_id} (slots={cfg.n_slots}) ...", flush=True)
    bb = Backbone.load(cfg)

    # ---- P1: the answer must not be inside the question we are going to ask --
    p1_failures = [d["question_id"] for d in chosen
                   if not P.assert_evicted(d["question"], str(d["answer"])).evicted]
    print(f"P1 eviction check: {len(chosen) - len(p1_failures)}/{len(chosen)} clean"
          + (f"  (leaking ids: {p1_failures[:3]})" if p1_failures else ""), flush=True)

    # ---- baseline: the frozen model, history absent, memory empty -----------
    bb.erase_all()
    bb.set_read_slots([])
    frozen = {d["question_id"]: bb.answer(d["question"]) for d in chosen}
    frozen_neg = {d["question_id"]: bb.answer(d["question"]) for d in negatives}

    # ---- write every item into its own slot ---------------------------------
    anchors = bb.anchor_logits(GENERIC_ANCHORS) if args.lambda_kl > 0 else {}
    slot_of: dict[str, int] = {}
    writes = []
    t0 = time.perf_counter()
    for slot, item in enumerate(chosen):
        answer_text = str(item["answer"])
        def loss_fn(item=item, slot=slot, answer_text=answer_text):
            bb.set_read_slots([slot])
            loss = bb.write_loss(item["question"], answer_text)
            if anchors:
                bb.set_read_slots([slot])
                loss = loss + args.lambda_kl * bb.kl_to_anchors(anchors)
            return loss

        report = write_slot(bb.wrappers, slot, loss_fn, lr=args.lr, steps=args.steps)
        if not report.frozen_ok:
            raise RuntimeError(f"write isolation violated on slot {slot}")
        with torch.no_grad():
            bb.set_read_slots([slot])
            ce_after = float(bb.write_loss(item["question"], answer_text))
        slot_of[item["question_id"]] = slot
        writes.append({
            "question_id": item["question_id"], "slot": slot,
            "answer_chars": len(answer_text), "evidence_turns": evidence_turns(item),
            "loss_start": report.loss_start, "ce_after": ce_after,
            "seconds": report.seconds,
        })
    write_seconds = time.perf_counter() - t0

    # ---- read back in three separate passes --------------------------------
    # Doing mem_on and mem_off inside one loop would accumulate erased slots as we
    # go, so the router would face a different (progressively deader) key set for
    # each item. Three passes keep the conditions identical across items:
    #   1. mem_on   -- every slot alive, routed
    #   2. negatives -- every slot alive (P3)
    #   3. mem_off  -- erase the item's own slot, ask again (P4)
    slot_keys = {slot_of[d["question_id"]]: bb.query_key(d["question"])
                 for d in chosen}
    rows = []
    for item in chosen:
        qid, slot = item["question_id"], slot_of[item["question_id"]]
        gold = str(item["answer"])
        decision = route(bb.query_key(item["question"]), slot_keys, k=1)
        bb.set_read_slots(decision.slots)
        mem_on = bb.answer(item["question"])
        rows.append({
            "question_id": qid, "routed_slot": decision.top, "own_slot": slot,
            "routing_ok": decision.top == slot,
            "gold": gold, "frozen": frozen[qid], "mem_on": mem_on,
            "contain_frozen": containment(frozen[qid], gold),
            "contain_mem_on": containment(mem_on, gold),
        })

    written_slots = sorted(slot_of.values())
    neg_rows = []
    for item in negatives:
        qid, gold = item["question_id"], str(item["answer"])
        bb.set_read_slots(written_slots)
        neg = bb.answer(item["question"])
        neg_rows.append({"question_id": qid,
                         "contain_frozen": containment(frozen_neg[qid], gold),
                         "contain_mem_on": containment(neg, gold)})

    # pass 3: erase each item's own slot, then ask again
    for row in rows:
        bb.erase_slots([row["own_slot"]])
    rows_by_id = {r["question_id"]: r for r in rows}
    for item in chosen:
        row = rows_by_id[item["question_id"]]
        bb.set_read_slots([])
        row["mem_off"] = bb.answer(item["question"])
        row["contain_mem_off"] = containment(row["mem_off"], row["gold"])
        row["mem_off_equals_frozen"] = row["mem_off"] == row["frozen"]

    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else float("nan")

    def rate(rows, predicate):
        return sum(1 for r in rows if predicate(r)) / len(rows) if rows else float("nan")

    summary = {
        "protocol": "docs/06 section 6.1 (pre-registered)",
        "data_file": str(data_path), "data_sha256": digest,
        "subset_ids": [d["question_id"] for d in chosen],
        "negative_ids": [d["question_id"] for d in negatives],
        "n_subset": len(chosen), "p1_leaking_ids": p1_failures,
        "containment": {
            "frozen": mean(rows, "contain_frozen"),
            "mem_on": mean(rows, "contain_mem_on"),
            "mem_off": mean(rows, "contain_mem_off"),
            "mem_on_full_match": rate(rows, lambda r: r["contain_mem_on"] == 1.0),
        },
        "routing_accuracy": rate(rows, lambda r: r["routing_ok"]),
        "mem_off_equals_frozen_rate": rate(rows, lambda r: r["mem_off_equals_frozen"]),
        "negative_control": {
            "frozen": mean(neg_rows, "contain_frozen"),
            "mem_on": mean(neg_rows, "contain_mem_on"),
        },
        "write": {
            "count": len(writes),
            "mean_ce_after": statistics.mean([w["ce_after"] for w in writes]),
            "max_ce_after": max(w["ce_after"] for w in writes),
            "mean_answer_chars": statistics.mean([w["answer_chars"] for w in writes]),
            "total_seconds": write_seconds,
        },
        "caveats": [
            "oracle indexing: the evidence turn is located with the benchmark's "
            "has_answer flags, so retrieval is not being tested",
            "write query == read query (no paraphrase), so this is an upper bound on "
            "what the parameters can hold",
            "scoring is a containment proxy, not the benchmark's LLM judge",
            "not comparable to published LongMemEval numbers (subset, 0.6B, proxy)",
        ],
        "rows": rows, "writes": writes, "negative_rows": neg_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    print(f"\nwritten {len(writes)} items in {write_seconds/60:.1f} min; "
          f"mean CE after write {summary['write']['mean_ce_after']:.3f} "
          f"(max {summary['write']['max_ce_after']:.3f})", flush=True)
    print(f"{'arm':<10}{'containment':>13}", flush=True)
    for arm in ("frozen", "mem_on", "mem_off"):
        print(f"{arm:<10}{summary['containment'][arm]:>13.3f}", flush=True)
    print(f"routing accuracy {summary['routing_accuracy']:.3f}  "
          f"mem_off==frozen {summary['mem_off_equals_frozen_rate']:.3f}", flush=True)
    print(f"negative control: frozen {summary['negative_control']['frozen']:.3f} -> "
          f"mem_on {summary['negative_control']['mem_on']:.3f}", flush=True)
    print(f"\nfirst {args.manual_samples} cases for manual inspection:", flush=True)
    for row in rows[: args.manual_samples]:
        print(f"  [{row['question_id']}] gold={row['gold'][:60]!r}\n"
              f"      frozen={row['frozen'][:70]!r}\n"
              f"      mem_on={row['mem_on'][:70]!r}", flush=True)
    print(f"\nreport written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
