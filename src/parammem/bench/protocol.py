"""Isolation protocol P1-P5 and the statistics used to report it.

The whole point of this module is that "the model answered correctly" is *not*
evidence that the parameters remember something. What counts as evidence:

    P1 eviction check      the value is provably absent from the context
    P2 counterfactual write the answer tracks the value *this run* wrote
    P3 negative control    never-written queries do not improve
    P4 adapter ablation    same context, memory on/off, only the answer changes
    P5 prompt-only control strips the prompt-format artefact

Every function here is a pure function so it can be unit-tested without a model
(see ``tests/test_protocol.py``).
"""

from __future__ import annotations

import math
import random
import re
import unicodedata
from dataclasses import dataclass, field

__all__ = [
    "normalize",
    "numeric_forms",
    "EvictionReport",
    "assert_evicted",
    "compare_counterfactual",
    "NegativeControlReport",
    "negative_control",
    "AttributionReport",
    "attribution_delta",
    "bootstrap_ci",
    "exact_match",
    "token_f1",
]

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_PUNCT = re.compile(r"[^\w\s#]+", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC + case-fold + punctuation-stripped + whitespace-collapsed form.

    This is the P1 check-2 (normalized match) primitive: it makes "Trennick."
    and "  trennick " and "Ｔｒｅｎｎｉｃｋ" compare equal.
    """
    s = unicodedata.normalize("NFKC", text).casefold()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def numeric_forms(text: str) -> set[str]:
    """Canonical numeric/date forms found in ``text`` (P1 check-3 primitive).

    ``"March 3, 2041"``, ``"2041-03-03"``, ``"3 March 2041"`` and ``"03/03/2041"``
    all yield ``{"2041-03-03"}`` so a date cannot hide behind reformatting.
    """
    s = unicodedata.normalize("NFKC", text).casefold()
    found: set[str] = set()

    # ISO 2041-03-03
    for m in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s):
        y, mo, d = (int(g) for g in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            found.add(f"{y:04d}-{mo:02d}-{d:02d}")

    # Month-name first: "march 3, 2041" / "march 3 2041"
    for m in re.finditer(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", s):
        mo = _MONTHS.get(m.group(1))
        if mo:
            found.add(f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(2)):02d}")

    # Day first: "3 march 2041" / "3rd of march 2041"
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\.?,?\s+(\d{4})\b", s):
        mo = _MONTHS.get(m.group(2))
        if mo:
            found.add(f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(1)):02d}")

    # Slashed / dotted: 03/03/2041 or 3.3.2041 (assume m/d/y, US style)
    for m in re.finditer(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b", s):
        mo, d, y = (int(g) for g in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            found.add(f"{y:04d}-{mo:02d}-{d:02d}")

    # Bare numbers (passcodes, ids). Require >= 5 digits: a 4-digit year is far
    # too common to be evidence of a leak, and dates are already covered above.
    for m in re.finditer(r"\b\d{5,}\b", s):
        found.add(str(int(m.group(0))))

    return found


@dataclass
class EvictionReport:
    """Outcome of protocol P1 for one (context, value) pair."""

    evicted: bool
    checks: dict[str, bool] = field(default_factory=dict)
    hits: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # allow `if report:`
        return self.evicted

    def reason(self) -> str:
        if self.evicted:
            return "evicted: no exact/normalized/numeric trace of the value in context"
        return "LEAK: " + ", ".join(self.hits)


def assert_evicted(context: str, value: str, aliases: tuple[str, ...] = ()) -> EvictionReport:
    """Protocol P1: prove the value is absent from the read-time context.

    Three checks, any hit means the sample is NOT attributable and must be
    dropped from the main metric:

    1. exact substring (raw text)
    2. normalized substring (NFKC + case + punctuation + whitespace)
    3. numeric/date canonical form containment

    Also guards the *reverse* leak: for a date-like value we require that no
    canonical numeric form present in the context matches one of the value's.
    """
    forms = (value,) + tuple(aliases)
    checks = {"exact": False, "normalized": False, "numeric": False}
    hits: list[str] = []

    for form in forms:
        if not form:
            continue
        # A single-word value must match on word boundaries: a plain substring
        # test would report "Tren" as present inside "Trennick" and cry wolf.
        # Multi-word values are matched as plain substrings.
        if " " in form:
            pattern = re.escape(form)
        else:
            pattern = r"(?<!\w)" + re.escape(form) + r"(?!\w)"
        if re.search(pattern, context):
            checks["exact"] = True
            hits.append(f"exact:{form!r}")

    ctx_norm = normalize(context)
    ctx_tokens = set(ctx_norm.split())
    for form in forms:
        nf = normalize(form)
        if not nf:
            continue
        # Single-word values must match a whole token, otherwise "Tren" would be
        # reported inside "Trennick" and the check would cry wolf.
        hit = nf in ctx_norm if " " in nf else nf in ctx_tokens
        if hit:
            checks["normalized"] = True
            hits.append(f"normalized:{form!r}")

    ctx_nums = numeric_forms(context)
    if ctx_nums:
        for form in forms:
            shared = ctx_nums & numeric_forms(form)
            if shared:
                checks["numeric"] = True
                hits.append(f"numeric:{sorted(shared)!r}")

    return EvictionReport(evicted=not any(checks.values()), checks=checks, hits=hits)


@dataclass
class CounterfactualReport:
    """Outcome of protocol P2 for one memory item across several seeds."""

    ok: bool
    written_value: str
    other_values: list[str] = field(default_factory=list)
    detail: str = ""


def compare_counterfactual(written_value: str, other_run_values: list[str]) -> CounterfactualReport:
    """Protocol P2: the value written this run must differ from other runs' values.

    If two seeds collide, a correct answer cannot be attributed to *this* write,
    so the item is unusable for attribution.
    """
    w = normalize(written_value)
    collided = [v for v in other_run_values if normalize(v) == w]
    if collided:
        return CounterfactualReport(
            ok=False,
            written_value=written_value,
            other_values=collided,
            detail="value collides across seeds; item not attributable",
        )
    return CounterfactualReport(ok=True, written_value=written_value, other_values=[])


@dataclass
class NegativeControlReport:
    """Outcome of protocol P3: gain on written vs never-written queries."""

    ok: bool
    gain_written: float
    gain_unwritten: float
    detail: str = ""


def negative_control(
    *,
    written_baseline_em: float,
    written_mem_em: float,
    unwritten_baseline_em: float,
    unwritten_mem_em: float,
    format_effect_tolerance: float = 0.05,
    min_gain: float = 0.05,
) -> NegativeControlReport:
    """Protocol P3: improvement must be specific to what was written.

    ``gain_written >> gain_unwritten ~ 0``. A large gain on never-written queries
    means the model is fabricating plausible answers, so the prompt format --
    not the memory -- is driving the score. A gain below ``min_gain`` on the
    written set is not a result either, however clean the negative control is.
    """
    gain_written = written_mem_em - written_baseline_em
    gain_unwritten = unwritten_mem_em - unwritten_baseline_em
    if gain_unwritten > format_effect_tolerance:
        return NegativeControlReport(
            ok=False, gain_written=gain_written, gain_unwritten=gain_unwritten,
            detail="gain on never-written queries too large: prompt-format artefact",
        )
    if gain_written < min_gain:
        return NegativeControlReport(
            ok=False, gain_written=gain_written, gain_unwritten=gain_unwritten,
            detail=f"gain on written queries below the {min_gain:.2f} reporting floor",
        )
    if gain_written <= gain_unwritten:
        return NegativeControlReport(
            ok=False, gain_written=gain_written, gain_unwritten=gain_unwritten,
            detail="no specific gain: memory did not help beyond format effect",
        )
    return NegativeControlReport(
        ok=True, gain_written=gain_written, gain_unwritten=gain_unwritten,
        detail="specific gain confirmed",
    )


@dataclass
class AttributionReport:
    """Outcome of protocols P4/P5: is the answer causally due to the side path?"""

    ok: bool
    em_mem_on: float
    em_mem_off: float
    em_mem_shuffle: float
    em_prompt_only: float
    delta_net: float
    ci_low: float
    ci_high: float
    detail: str = ""


def attribution_delta(
    *,
    em_mem_on: float,
    em_mem_off: float,
    em_shuffle: float,
    em_prompt_only: float,
    ci: tuple[float, float] | None = None,
    leak_tolerance: float = 0.05,
    shuffle_tolerance: float = 0.10,
) -> AttributionReport:
    """Protocols P4 + P5 in one place.

    Requirements for a valid claim:

    * ``MEM_OFF`` must fall back to the prompt-only baseline. The comparison is
      against **prompt-only, not against MEM_ON**: the question is "can the
      answer be recovered without the written slots?", and comparing with MEM_ON
      would pass a run where *everything* is recoverable from the context.
    * ``MEM_SHUFFLE`` must also sit at the prompt-only baseline, otherwise some
      *other* slot is answering the query (cross-talk between memories).
    * the net effect over prompt-only, ``MEM_ON - PromptOnly``, is the number
      that goes in the main table, and its CI must exclude zero.
    """
    delta_net = em_mem_on - em_prompt_only
    lo, hi = ci if ci is not None else (math.nan, math.nan)

    if em_mem_off - em_prompt_only > leak_tolerance:
        detail = (
            "MEM_OFF is far above the prompt-only baseline: the answer is "
            "recoverable without the written slots -> leak, experiment invalid"
        )
        return AttributionReport(False, em_mem_on, em_mem_off, em_shuffle,
                                 em_prompt_only, delta_net, lo, hi, detail)
    if em_shuffle - em_prompt_only > shuffle_tolerance:
        detail = "MEM_SHUFFLE is above the prompt-only baseline: another slot is answering"
        return AttributionReport(False, em_mem_on, em_mem_off, em_shuffle,
                                 em_prompt_only, delta_net, lo, hi, detail)
    if delta_net <= 0.0:
        detail = "net effect over prompt-only is not positive"
        return AttributionReport(False, em_mem_on, em_mem_off, em_shuffle,
                                 em_prompt_only, delta_net, lo, hi, detail)
    if not (math.isnan(lo) or lo > 0.0):
        detail = "net effect CI includes 0: not a reportable claim"
        return AttributionReport(False, em_mem_on, em_mem_off, em_shuffle,
                                 em_prompt_only, delta_net, lo, hi, detail)
    return AttributionReport(True, em_mem_on, em_mem_off, em_shuffle,
                             em_prompt_only, delta_net, lo, hi, "attributable")


def bootstrap_ci(
    values: list[float], *, n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI. No scipy/statsmodels dependency, fully seeded."""
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    k = len(values)
    means = []
    for _ in range(n):
        means.append(sum(values[rng.randrange(k)] for _ in range(k)) / k)
    means.sort()
    lo = means[int(math.floor((alpha / 2) * n))]
    hi = means[min(n - 1, int(math.ceil((1 - alpha / 2) * n)) - 1)]
    return (lo, hi)


def exact_match(prediction: str, value: str, aliases: tuple[str, ...] = ()) -> bool:
    """Normalized exact match against the value or any of its aliases."""
    p = normalize(prediction)
    if not p:
        return False
    for form in (value,) + tuple(aliases):
        if normalize(form) == p:
            return True
        # Tolerate a short answer wrapped in a sentence.
        if normalize(form) and normalize(form) in p.split():
            return True
    # Date answers often come back reformatted; compare canonical numeric forms.
    return bool(numeric_forms(prediction) & numeric_forms(" ".join((value,) + tuple(aliases))))


def token_f1(prediction: str, value: str, aliases: tuple[str, ...] = ()) -> float:
    """Token-level F1 against the best-matching surface form (partial credit)."""
    pred = normalize(prediction).split()
    if not pred:
        return 0.0
    best = 0.0
    for form in (value,) + tuple(aliases):
        gold = normalize(form).split()
        if not gold:
            continue
        common = 0
        remaining = list(gold)
        for tok in pred:
            if tok in remaining:
                remaining.remove(tok)
                common += 1
        if common == 0:
            continue
        precision = common / len(pred)
        recall = common / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best
