"""Shared loading, subset selection and scoring for the LongMemEval work.

Two scripts use this module:

* ``experiments/t7d_longmemeval_pilot.py`` -- memory written into the weights
* ``experiments/t7e_context_baselines.py`` -- the same information delivered
  through the context (full-context and RAG arms)

The comparison between "memory in the weights" and "memory in the context" is only
meaningful on the **same questions**, so both go through :func:`select_single_session`
here, and the baseline run additionally *asserts* that its question ids equal the
ones recorded by the weights run rather than assuming it.

Scoring is a crude containment proxy on purpose: the benchmark's own scoring uses an
LLM judge, which needs an API key we deliberately do not use.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .protocol import normalize, token_f1

__all__ = [
    "SINGLE_SESSION_PREFIX",
    "sha256_file",
    "load_oracle",
    "select_single_session",
    "select_multi_session",
    "select_by_type",
    "MULTI_SESSION_PREFIX",
    "evidence_turns",
    "containment",
    "contains_answer",
]

SINGLE_SESSION_PREFIX = "single-session-"
MULTI_SESSION_PREFIX = "multi-session-"


def sha256_file(path: str | Path) -> str:
    """Hash a data file so a result can be tied to the exact bytes it used."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_oracle(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_by_type(
    raw: Iterable[dict[str, Any]], prefix: str, subset: int, offset: int = 0
) -> list[dict[str, Any]]:
    """``subset`` questions at ``offset`` from the pool whose type starts with ``prefix``.

    Same ordering rule as :func:`select_single_session`, generalised so the multi-session
    questions can be measured with the identical pipeline. The paper argued that evidence
    spread across several sessions cannot fit in one slot; running it turns that argument
    into a measurement instead of leaving it as an expectation.
    """
    pool = [d for d in raw if str(d.get("question_type", "")).startswith(prefix)]
    pool.sort(key=lambda d: d["question_id"])
    return pool[offset: offset + subset]


def select_multi_session(
    raw: Iterable[dict[str, Any]], subset: int, offset: int = 0
) -> list[dict[str, Any]]:
    return select_by_type(raw, MULTI_SESSION_PREFIX, subset, offset)


def select_single_session(
    raw: Iterable[dict[str, Any]], subset: int, offset: int = 0
) -> list[dict[str, Any]]:
    """Deterministically take ``subset`` questions at ``offset`` from the
    ``single-session-*`` pool, ordered by ``question_id``.

    Sorting by id (not by file order) is what makes the held-out replication and
    the baselines reproducible from the question list alone.
    """
    singles = [d for d in raw
               if str(d.get("question_type", "")).startswith(SINGLE_SESSION_PREFIX)]
    singles.sort(key=lambda d: d["question_id"])
    return singles[offset: offset + subset]


def evidence_turns(item: dict[str, Any]) -> int:
    """How many turns the benchmark itself flags as carrying the answer."""
    return sum(1 for session in item.get("haystack_sessions") or []
               for turn in session if turn.get("has_answer"))


def containment(prediction: str, gold: str) -> float:
    """1.0 if the gold is present, 0.5 if the two merely overlap, else 0.0."""
    pred, target = normalize(prediction), normalize(gold)
    if not pred or not target:
        return 0.0
    if target in pred:
        return 1.0
    clauses = [c.strip() for c in target.replace(";", " and ").split(" and ")
               if c.strip()]
    hits = sum(1 for clause in clauses if clause in pred)
    if clauses and hits / len(clauses) >= 0.5:
        return 0.5
    return 0.5 if token_f1(prediction, gold) >= 0.5 else 0.0


def contains_answer(prediction: str, gold: str) -> bool:
    """The strict half of :func:`containment` (verbatim containment only)."""
    pred, target = normalize(prediction), normalize(gold)
    return bool(pred) and bool(target) and target in pred
