"""Turning conversation turns into (question, answer) memories.

The T7-d diagnostic writes the benchmark's own ``(question, answer)`` pair into a slot,
which means the memory *is* the gold item: retrieval is oracle-indexed and the written
value is the reference answer. Moving to a real multi-session benchmark means deriving
the memories from the conversation instead, and that is where the noise enters -- an
extraction error is written into the weights and then read back as if it were memory.

So the parsing is kept here as a pure function and tested against the shapes a hosted
model actually returns (fenced JSON, a numbered list, a stray preamble, a truncated
reply), rather than being a regex inside an experiment script. The extraction call
itself is trivial once the prompt and the parser are pinned.

The residual oracle-ness is deliberate and is documented where it is used: memories are
extracted from the sessions the benchmark flags as containing the answer, so the
*retrieval pool* is still narrowed by knowledge of which session matters. What is no
longer oracle is the memory content (it comes from the conversation, not from the
reference answer) and the read query (it is the benchmark's real question, matched
against every extracted memory).
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["EXTRACTION_INSTRUCTIONS", "build_extraction_prompt", "parse_pairs",
           "flatten_turns", "extract_pairs"]

EXTRACTION_INSTRUCTIONS = (
    "You convert conversation excerpts into standalone memory items.\n"
    "For each excerpt that contains a durable fact about the user, produce one item:\n"
    '  {"question": a question the user might later ask, phrased as a question,\n'
    '   "answer": the shortest complete answer, taken from the excerpt}\n'
    "Rules: use only information present in the excerpt; the question must be "
    "self-contained; skip excerpts that contain no durable fact; answer in the same "
    "form the excerpt states it. Return a JSON array and nothing else."
)


def flatten_turns(session: list[dict[str, Any]]) -> list[str]:
    """``[{"role": ..., "content": ...}]`` -> readable, non-empty turn strings."""
    out = []
    for turn in session or []:
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        role = str(turn.get("role", "user")).strip() or "user"
        out.append(f"{role}: {content}")
    return out


def build_extraction_prompt(turns: list[str]) -> str:
    body = "\n".join(turns)
    return f"{EXTRACTION_INSTRUCTIONS}\n\nExcerpt:\n{body}"


def extract_pairs(call, turns: list[str], retries: int = 1, observer=None
                  ) -> list[dict[str, str]]:
    """Ask a model to turn ``turns`` into memory items.

    ``call`` is any ``prompt -> reply`` callable, so this is testable without a network
    and the experiment can hand it the same rate-limited client the judge uses.

    The return value is only the pairs; **an empty list does not distinguish "the model
    found no durable fact" from "every call failed"**. Those two cases look identical in
    a count and have opposite meanings, and the first version of this function merged
    them, so a rate-limited run reported itself as a run that found nothing. Pass
    ``observer`` to receive one dict per attempt (``status``, ``finish``, ``chars``,
    ``error``) and record it: a caller that does not look at the failure mode cannot
    tell the two apart.
    """
    if not turns:
        return []
    prompt = build_extraction_prompt(turns)

    def report(**fields) -> None:
        if observer is not None:
            observer(fields)

    for attempt in range(1, max(1, retries + 1) + 1):
        try:
            reply = call(prompt) or ""
        except Exception as error:                  # transport or rate-limit failure
            # The message matters: "RuntimeError" alone does not say whether this was a
            # rate limit, an authentication problem or a malformed request, and those
            # need different responses.
            report(attempt=attempt, status="error", chars=0,
                   error=f"{type(error).__name__}: {str(error)[:180]}", finish="")
            continue
        pairs = parse_pairs(reply)
        finish = getattr(call, "last_finish_reason", "") if hasattr(call, "last_finish_reason") else ""
        report(attempt=attempt, status="ok" if pairs else "unparseable",
               chars=len(reply), error="", finish=str(finish))
        if pairs:
            return pairs
    return []


def _looks_like_pair(obj: Any) -> bool:
    return (isinstance(obj, dict)
            and isinstance(obj.get("question"), str)
            and isinstance(obj.get("answer"), str)
            and obj["question"].strip()
            and obj["answer"].strip())


def parse_pairs(text: str) -> list[dict[str, str]]:
    """Parse the model's reply into ``[{"question", "answer"}, ...]``.

    Accepts a JSON array, a single JSON object, JSON embedded in a fenced block or in
    prose, and a numbered ``1. Q: ... A: ...`` list, because all four show up in
    practice. Anything unparseable yields an empty list rather than a guess: a silently
    malformed memory would be written into the weights as though it were knowledge.
    """
    if not text or not text.strip():
        return []

    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.S)
    if fence:
        stripped = fence.group(1).strip()

    for candidate in (stripped,
                      stripped[stripped.find("["):stripped.rfind("]") + 1]
                      if "[" in stripped and "]" in stripped else stripped):
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if _looks_like_pair(payload):
            return [{"question": payload["question"].strip(),
                     "answer": payload["answer"].strip()}]
        if isinstance(payload, list):
            pairs = [{"question": item["question"].strip(),
                      "answer": item["answer"].strip()}
                     for item in payload if _looks_like_pair(item)]
            if pairs:
                return pairs

    # numbered / labelled text form
    pairs = []
    pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+[.)]\s*)?Q(?:uestion)?\s*[:\-]\s*(?P<q>.+?)"
        r"\s*(?:\n\s*)?A(?:nswer)?\s*[:\-]\s*(?P<a>.+?)(?=\n\s*(?:\d+[.)]\s*)?Q|$)",
        re.I | re.S)
    for match in pattern.finditer(stripped):
        question = " ".join(match.group("q").split())
        answer = " ".join(match.group("a").split())
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    return pairs
