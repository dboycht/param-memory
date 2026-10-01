"""LLM judge for answer correctness (opt-in; needs an API key).

Why a judge exists here at all: measured 2026-09-29, our containment metric is
biased toward the arm that was **trained** to reproduce the reference string
verbatim -- the context arms answered correctly in their own words and scored zero
(docs/06 section 10.2). Comparing "memory in the weights" with "memory in the
context" therefore needs a semantic judgement, which is also what the benchmark's
own scoring uses.

Credential hygiene: the key is passed in, never logged, never written, masked in
``repr``. The transport is injectable, so prompt building, JSON parsing and verdict
logic are all testable without a network call.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

__all__ = [
    "JudgeVerdict",
    "LLMJudge",
    "STANDARDS",
    "parse_verdicts",
    "parse_single",
    "build_prompt",
    "build_single_prompt",
    "load_api_key",
]

JUDGE_SYSTEM = (
    "You grade answers strictly and reply with JSON only."
)

JUDGE_INSTRUCTIONS = """\
Reference answer: {reference}

Candidate answers:
{candidates}

For EACH candidate decide whether it conveys the same information as the reference
answer. A paraphrase that keeps the substance is CORRECT. An answer that is missing
the key content, contradicts the reference, or is evasive where the reference
states a fact, is INCORRECT.

Reply with a single JSON object and nothing else, of exactly this shape:
{{"<label>": {{"correct": true, "reason": "short"}}, ...}}
"""

# One candidate per call. Measured 2026-09-29: batching several candidates into one
# call made a reasoning model answer about the wrong candidate and invent reasons
# (it marked a verbatim copy of the reference as INCORRECT while citing another
# question's content). Binary, single-candidate grading is slower but much harder
# to get wrong, and a judge that is subtly wrong is worse than no judge at all.
SINGLE_TEMPLATE = """\
Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

{standard}

Answer in exactly this format, nothing else:
Verdict: CORRECT
Reason: <one short sentence>
"""
# (replace CORRECT with INCORRECT when that is your verdict)

# Two standards, because the choice is a real decision and it changes the numbers.
# ``lenient`` is the one the human calibration used (measured 2026-09-29: under it the
# judge agreed 8/10 with the human, with the two misses both explained by the judge
# applying the strict reading instead).
STANDARDS = {
    "lenient": (
        "Decide whether the candidate conveys the substance of the REFERENCE answer -- "
        "the fact or preference that the reference states. Different wording is fine. "
        "Omitting an additional clause -- such as a trailing note about what the user "
        "does *not* want, or trailing examples -- does NOT make the candidate "
        "incorrect, as long as the reference's substance is there. A generic answer "
        "that does not convey that substance is incorrect even if it is a plausible "
        "answer to the question."
    ),
    "strict": (
        "Decide whether the candidate conveys ALL of the information in the reference "
        "answer, including any additional or negative clause. Different wording is "
        "fine, but a candidate that omits part of the reference -- for example the note "
        "about what the user does not want -- is incorrect."
    ),
}


@dataclass
class JudgeVerdict:
    """One grading decision. ``correct`` is None when the reply could not be read."""

    correct: bool | None
    reason: str = ""
    raw: str = ""

    @property
    def usable(self) -> bool:
        return self.correct is not None


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def _coerce_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "correct", "1"):
            return True
        if low in ("false", "no", "incorrect", "0"):
            return False
    return None


def _canonical_label(text) -> str:
    """Normalise a label for matching: ``Item 1`` == ``item1`` == ``"1"``."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def parse_verdicts(text: str, labels: list[str]) -> dict[str, JudgeVerdict]:
    """Pull ``{label: {correct, reason}}`` out of a model reply.

    Tolerates markdown fences, surrounding prose and label spelling drift
    (``Item 1`` vs ``item1``), and records unreadable labels as ``correct=None``
    instead of guessing -- an unreadable verdict must not be silently counted as
    either outcome. A list-of-objects reply is accepted positionally, but only when
    its length matches, since guessing a mapping would be worse than failing.
    """
    verdicts = {label: JudgeVerdict(None, "not returned", text) for label in labels}
    match = _JSON_OBJECT.search(text or "")
    if not match:
        return verdicts
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return verdicts

    if isinstance(data, list):
        if len(data) != len(labels):
            return verdicts
        data = {label: entry for label, entry in zip(labels, data)}
    if not isinstance(data, dict):
        return verdicts

    by_canonical = {_canonical_label(key): value for key, value in data.items()}
    for label in labels:
        entry = data.get(label)
        if entry is None:
            entry = by_canonical.get(_canonical_label(label))
        if isinstance(entry, dict):
            verdicts[label] = JudgeVerdict(
                _coerce_bool(entry.get("correct", entry.get("verdict"))),
                str(entry.get("reason", ""))[:200],
                text,
            )
        elif entry is not None:
            verdicts[label] = JudgeVerdict(_coerce_bool(entry), "", text)
    return verdicts


def build_prompt(reference: str, candidates: dict[str, str]) -> str:
    block = "\n".join(f"{label}: {text}" for label, text in candidates.items())
    return JUDGE_INSTRUCTIONS.format(reference=reference, candidates=block)


def build_single_prompt(question: str, reference: str, candidate: str,
                        standard: str = "lenient") -> str:
    if standard not in STANDARDS:
        raise ValueError(f"unknown standard {standard!r}; expected one of "
                         f"{sorted(STANDARDS)}")
    return SINGLE_TEMPLATE.format(question=question, reference=reference,
                                  candidate=candidate, standard=STANDARDS[standard])


def parse_single(text: str) -> JudgeVerdict:
    """Read a ``Verdict: CORRECT|INCORRECT`` reply.

    ``INCORRECT`` is checked first on purpose -- it contains ``CORRECT``, so the
    naive order would turn every rejection into an acceptance.
    """
    body = text or ""
    upper = body.upper()
    if "INCORRECT" in upper:
        correct: bool | None = False
    elif "CORRECT" in upper:
        correct = True
    else:
        correct = None
    reason = ""
    match = re.search(r"reason\s*:\s*(.+)", body, re.I)
    if match:
        reason = match.group(1).strip().splitlines()[0][:200]
    return JudgeVerdict(correct, reason, body)


def load_api_key(config_path: str | Path) -> str:
    """Read ``llm.api_key`` from a config file. The value is returned, never logged."""
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    key = (data.get("llm") or {}).get("api_key") or ""
    if not key.strip():
        raise ValueError(f"no llm.api_key in {config_path}")
    return key


def load_llm_settings(config_path: str | Path) -> dict[str, Any]:
    """Everything needed to reach the configured model, from one config file.

    The extraction script used to hard-code a provider, so swapping the account in the
    config changed nothing and the calls kept going to the old host. The settings live in
    one place now and the caller only supplies the key.
    """
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    llm = data.get("llm") or {}
    if not (llm.get("base_url") and llm.get("model")):
        raise ValueError(f"config {config_path} has no llm.base_url/llm.model")
    settings: dict[str, Any] = {
        "base_url": str(llm["base_url"]),
        "model": str(llm["model"]),
    }
    # 0 or absent means "no throttle"; a hosted provider may need one, and the operator
    # knows better than we do, so the value is read rather than assumed.
    if isinstance(llm.get("min_interval"), (int, float)) and llm["min_interval"] > 0:
        settings["min_interval"] = float(llm["min_interval"])
    if isinstance(llm.get("timeout"), (int, float)) and llm["timeout"] > 0:
        settings["timeout"] = float(llm["timeout"])
    # Only pass temperature when the config names one AND omitting it is not required by
    # the model: an explicit value that a provider rejects is worse than a default.
    if isinstance(llm.get("temperature"), (int, float)):
        settings["temperature"] = float(llm["temperature"])
    return settings


@dataclass
class LLMJudge:
    """A minimal chat-completions client specialised for grading.

    ``post`` is injectable: it takes ``(url, headers, payload)`` and returns the
    response body as text. Tests pass a stub; production uses the stdlib transport.
    ``sleep`` is injectable for the same reason.
    """

    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout: float = 90.0
    min_interval: float = 0.0
    # None means "do not send the field at all": some models accept only their own
    # fixed value (kimi-k2.6 rejects anything but temperature=1 with HTTP 400), so
    # omitting it is more portable than guessing a "safe" number.
    temperature: float | None = None
    # 2048 was not enough: 12 of 180 verdicts came back with finish_reason=length and
    # had to be recovered from a truncated reasoning field, and the disagreements
    # between arms concentrated on exactly those rows.
    max_tokens: int = 8192
    post: Callable[[str, dict, dict], str] | None = None
    sleep: Callable[[float], None] = time.sleep
    _last_call: float = field(default=0.0, repr=False)
    # Where the last reply's text came from, so an empty ``content`` on a reasoning
    # model is visible instead of silently becoming "unreadable verdict".
    last_source: str = field(default="", repr=False)
    last_finish_reason: str = field(default="", repr=False)

    def __repr__(self) -> str:  # never leak the key
        return (f"LLMJudge(base_url={self.base_url!r}, model={self.model!r}, "
                f"api_key='<set,{len(self.api_key)}>', "
                f"min_interval={self.min_interval})")

    # ------------------------------------------------------------- transport
    def _default_post(self, url: str, headers: dict, payload: dict) -> str:
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8")

    def _chat(self, user_prompt: str, *, attempts: int = 3) -> str:
        """One completion, honouring the configured minimum interval.

        The interval is load-bearing, not politeness: with an organisation limit of
        3 requests/minute (measured 2026-09-29 from an HTTP 429 body), anything below
        ~20s earns rate-limit errors. Retries add backoff, but the interval is what
        keeps the run clean.
        """
        if self.min_interval > 0:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self.sleep(wait)
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        transport = self.post or self._default_post
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                body = transport(url, headers, payload)
                self._last_call = time.monotonic()
                choice = json.loads(body)["choices"][0]
                message = choice["message"]
                # ``length`` means the reply was cut off: a truncated verdict is the
                # most common way a judge silently lies, so keep it visible.
                self.last_finish_reason = str(choice.get("finish_reason", ""))
                content = (message.get("content") or "").strip()
                if content:
                    self.last_source = "content"
                    return content
                # Reasoning models (kimi-k2.6 among them) can return an empty
                # ``content`` with the actual text in ``reasoning_content``; without
                # this fallback the failure looks like "the judge refuses to answer".
                reasoning = (message.get("reasoning_content") or "").strip()
                if reasoning:
                    self.last_source = "reasoning_content"
                    return reasoning
                self.last_source = "empty"
                raise RuntimeError("empty completion: no content and no "
                                   "reasoning_content in the reply")
            except urllib.error.HTTPError as exc:
                # Keep the server's own explanation: "HTTP 400" alone is not
                # debuggable, and the body does not contain the credential.
                try:
                    detail = exc.read().decode("utf-8", "replace")[:400]
                except Exception:  # noqa: BLE001 - the body is a nicety, not required
                    detail = "<no body>"
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                self._last_call = time.monotonic()
                if exc.code in (400, 401, 403, 404):
                    break          # a rejected request will not succeed on retry
            except Exception as exc:  # noqa: BLE001 - retried below, then raised
                last_error = exc
                self._last_call = time.monotonic()
            if attempt < attempts:
                self.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"judge request failed after {attempts} attempts: "
                           f"{type(last_error).__name__}: {last_error}")

    # ---------------------------------------------------------------- grading
    def judge_one(self, question: str, reference: str, candidate: str,
                  standard: str = "lenient") -> JudgeVerdict:
        """Grade a single candidate. One call, binary verdict."""
        reply = self._chat(build_single_prompt(question, reference, candidate,
                                               standard))
        return parse_single(reply)

    def complete(self, user_prompt: str) -> str:
        """Raw completion for the calls that are not judgments.

        The multi-session extraction step needs the same rate limit, retry behaviour and
        truncation reporting as the judge, so it borrows this method instead of
        duplicating the HTTP plumbing.
        """
        return self._chat(user_prompt)

    def judge_batch(self, reference: str, candidates: dict[str, str]) -> dict[str, JudgeVerdict]:
        """Grade several candidate answers against one reference in a single call.

        Batching by question is deliberate: it respects the configured rate limit
        (one call per question instead of one per answer) and keeps the judge's
        standard identical across the arms being compared.
        """
        if not candidates:
            return {}
        reply = self._chat(build_prompt(reference, candidates))
        return parse_verdicts(reply, list(candidates))
