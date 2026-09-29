"""Deciding what deserves to become a permanent memory.

In a real conversation most turns are not worth carving into weights. The stream
this module is tested on deliberately mixes two kinds of content:

* things the model **already knows** (writing them wastes a slot, costs a
  gradient write, and moves the frozen backbone for nothing), and
* things it does not know (the actual memory targets).

A criterion has to tell them apart **without labels**. Three are implemented:

``always``
    the upper-bound baseline: write everything writeable. Not a strategy, a
    reference point.
``surprise``
    write only if the model's **pre-write** loss on the content is above a
    threshold -- i.e. "I did not already know this". This is the criterion from
    the literature (surprise-driven updates) and it needs no cooperation from the
    user.
``explicit``
    write only what the user explicitly asked to remember ("remember this",
    "note to self", "save this"). Cheap and precise, but it only works when the
    user cooperates -- which is exactly why it is not sufficient on its own.

**Information gain** (the loss difference on a probe before vs after a tentative
write) is the natural fourth criterion and is deliberately *not* implemented: it
costs a full extra write per candidate, so it has to earn that cost with a
measured gain first. Declaring that here is cheaper than pretending.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "WriteDecision",
    "always_decision",
    "surprise_decision",
    "explicit_decision",
    "selfcheck_decision",
    "EXPLICIT_MARKERS",
    "looks_explicit",
]

# Phrases that mean "carve this into weights". Deliberately plain string matching:
# the point of the criterion is that the request is *stated*, not inferred.
EXPLICIT_MARKERS = (
    "remember",
    "note to self",
    "note down",
    "save this",
    "keep in mind",
    "for your records",
    "let me note",
    "lesson learned",
)


@dataclass
class WriteDecision:
    """The outcome of applying a criterion to one candidate item."""

    should_write: bool
    score: float
    reason: str = ""

    def __bool__(self) -> bool:
        return self.should_write


def always_decision(*, score: float = 0.0) -> WriteDecision:
    return WriteDecision(should_write=True, score=score, reason="always")


def surprise_decision(loss: float, threshold: float) -> WriteDecision:
    """Write when the pre-write loss says the model does not already know it.

    ``loss`` must be measured **before** the write and with the memory switched
    off; otherwise the criterion is measuring its own previous writes.
    """
    if loss != loss:  # NaN guard: a broken measurement must not silently write
        return WriteDecision(False, float("nan"), "surprise: non-finite loss, skipped")
    should = loss >= threshold
    return WriteDecision(
        should, float(loss),
        f"surprise {loss:.3f} {'>=' if should else '<'} threshold {threshold:.3f}",
    )


def looks_explicit(statement: str, markers: tuple[str, ...] = EXPLICIT_MARKERS) -> bool:
    low = statement.lower()
    return any(marker in low for marker in markers)


def explicit_decision(statement: str) -> WriteDecision:
    hit = looks_explicit(statement)
    return WriteDecision(hit, 1.0 if hit else 0.0,
                         "explicit request found" if hit else "no explicit request")


def selfcheck_decision(
    free_answer: str, value: str, aliases: tuple[str, ...] = ()
) -> WriteDecision:
    """Write only if the model cannot already produce the value **on its own**.

    This exists because the loss-based criterion was measured to be mis-specified
    (2026-09-29): forcing the continuation ``" Paris"`` after "What is the capital
    of France?" costs 9.76 nats even though the model answers the question
    perfectly -- it just words the answer as a sentence. A likelihood probe over a
    *fixed* phrasing therefore measures "would the model use exactly this wording",
    not "does the model know this", and it scored known and unknown content almost
    identically (median 6.16 vs 7.72).

    The self-check asks the model the question instead of dictating the answer, and
    it is still label-free: it uses nothing but the model's own behaviour.
    ``free_answer`` must be generated with the memory switched **off**.
    """
    from parammem.bench.protocol import exact_match  # local import: keep this leaf-light

    known = exact_match(free_answer, value, aliases)
    return WriteDecision(
        not known, 1.0 if known else 0.0,
        f"model already answers {value!r}" if known else "model cannot answer this",
    )
