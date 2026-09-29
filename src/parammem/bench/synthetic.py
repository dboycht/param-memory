"""Episode generator for the controlled synthetic benchmark.

An episode is a scripted conversation with four sections::

    [write section]   w_1 .. w_N   one memory item each (fact/pref/lesson/noise)
    [eviction]        the write section is dropped from the context (protocol P1)
    [probe section]   q_1 .. q_M   asks for written items, context has no answers
    [control section] n_1 .. n_L   asks about never-written entities (protocol P3)

Plus an optional inertia block: a wrong premise is stated on topic A, then the
conversation switches to topic B, and we measure whether the wrong premise is
carried over (the "context inertia" metric of docs/02 section 3.3).

Every learnable value is either a word from the fictional world or a value drawn
from a per-seed stream. Nothing is a stock phrase: a lesson such as "clear the
cache before rebuilding" would be answerable from pre-training and would destroy
the attribution the whole benchmark exists to establish.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .entities import FictionalWorld, MemoryValue, _MAX_VALUE_INDEX

# --------------------------------------------------------------------- templates

_FACT_TEMPLATES = {
    "capital": (
        "By the way, {value} is the capital of {entity}.",
        "Let me note down that the capital of {entity} is {value}.",
    ),
    "birthday": (
        "For your records, {entity}'s birthday is {value}.",
        "Remember that {entity} was born on {value}.",
    ),
    "passcode": (
        "The access code for the {entity} locker is {value}.",
        "Save this: the {entity} vault code is {value}.",
    ),
    "dish": (
        "{entity} always orders {value} when we meet.",
        "Keep in mind {entity}'s usual order is {value}.",
    ),
    "accent_color": (
        "{entity} picked {value} as the accent colour for the project.",
        "Note the accent colour {entity} chose: {value}.",
    ),
}

_FACT_QUERIES = {
    "capital": "Which city did I say is the capital of {entity}?",
    "birthday": "What is {entity}'s birthday?",
    "passcode": "What is the access code for the {entity} locker?",
    "dish": "What does {entity} usually order?",
    "accent_color": "Which accent colour did {entity} choose for the project?",
}

# Preferences are keyed by category so that several of them in one episode stay
# distinguishable at read time (otherwise two probes share one question).
_PREF_CATEGORIES = (
    "tea", "notebook", "typeface", "mug", "backpack",
    "keyboard", "lamp", "chair",
)
_PREF_STATEMENTS = (
    "On the subject of {category}, my own preference is {value}.",
    "For {category}, my own preference is {value}, just so you know.",
)
_PREF_QUERY = "What is my preference for {category}?"

# Lessons embed invented nouns for both the trigger and the action, so the whole
# statement is unguessable from pre-training.
_LESSON_TEMPLATE = (
    "Lesson learned: when the {noun_a} alarm fires twice in a row, "
    "run the {noun_b} reset before rebuilding."
)
_LESSON_QUERY = "What did we learn to do when the {noun_a} alarm fires twice in a row?"

_NOISE_STATEMENTS = (
    "Unrelated, but the weather in {entity} was grey all week.",
    "Also, I rewatched an old film about {entity} last night.",
    "Completely off topic: the coffee here is terrible.",
)

# Inertia: a wrong premise on topic A, then a switch to topic B, where the
# correct behaviour is to withhold rather than inherit the premise.
_INERTIA_CASES = (
    {
        "premise": "{country} drives on the left-hand side of the road.",
        "new_task": "Now let's talk about {other}. Which side of the road do cars use there?",
        "inherited_keywords": ("left", "left-hand", "left hand"),
    },
    {
        "premise": "In {country} the standard unit of length is the {unit}.",
        "new_task": "Switching topics: what unit of length should I use for {other}?",
        "inherited_keywords": ("{unit_lower}",),
    },
    {
        "premise": "{country} has a population of exactly 12 million people.",
        "new_task": "Different topic now: what is the population of {other}?",
        "inherited_keywords": ("12 million", "12,000,000", "12000000"),
    },
)


# ------------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class MemoryItem:
    """One unit of information carried by a single write turn."""

    item_id: int
    memory_class: str  # "fact" | "pref" | "lesson" | "noise"
    attribute: str  # e.g. "capital", "pref:tea", "lesson:trigger"
    statement: str
    value: MemoryValue | None = None
    is_noise: bool = False
    is_important: bool = False
    query: str = ""

    @property
    def value_text(self) -> str:
        return "" if self.value is None else self.value.value

    @property
    def aliases(self) -> tuple[str, ...]:
        return () if self.value is None else self.value.aliases

    @property
    def probeable(self) -> bool:
        return (not self.is_noise) and self.value is not None and bool(self.query)


@dataclass(frozen=True)
class Probe:
    """A read-time question. ``item_id is None`` marks a P3 negative control."""

    item_id: int | None
    query: str
    value: str = ""
    aliases: tuple[str, ...] = ()
    is_negative: bool = False


@dataclass(frozen=True)
class InertiaBlock:
    """Topic-A wrong premise -> topic-B switch; measures I1/I2/I3."""

    old_topic: str
    premise: str
    new_task: str
    inherited_keywords: tuple[str, ...]
    old_topic_keywords: tuple[str, ...]


@dataclass
class Episode:
    episode_id: int
    seed: int
    writes: list[MemoryItem] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    negatives: list[Probe] = field(default_factory=list)
    inertia: InertiaBlock | None = None
    capacity_k: int | None = None

    # ------------------------------------------------------------- contexts
    def write_context(self) -> str:
        """The context the writes are learned from (before eviction)."""
        return "\n".join(item.statement for item in self.writes)

    def probe_context(self) -> str:
        """The read-time context: **contains no answer to any probe**.

        This is the structural half of protocol P1; the dynamic half (aliases,
        numeric reformatting) is checked by ``protocol.assert_evicted``.
        """
        lines = [p.query for p in self.probes]
        lines += [p.query for p in self.negatives]
        if self.inertia is not None:
            lines.append(self.inertia.new_task)
        return "\n".join(lines)

    @property
    def probeable_items(self) -> list[MemoryItem]:
        return [i for i in self.writes if i.probeable]


# ------------------------------------------------------------------- generator


def make_episode(
    seed: int,
    *,
    episode_id: int = 0,
    n_facts: int = 3,
    n_prefs: int = 1,
    n_lessons: int = 1,
    n_noise: int = 2,
    n_negatives: int = 2,
    with_inertia: bool = True,
    capacity_k: int | None = None,
    important_fraction: float = 0.5,
    world: FictionalWorld | None = None,
) -> Episode:
    """Build one deterministic episode.

    ``capacity_k`` only *records* the slot budget for the capacity subset; the
    eviction behaviour itself is the job of the memory store under test.
    """
    if n_prefs > len(_PREF_CATEGORIES):
        raise ValueError(
            f"n_prefs must be <= {len(_PREF_CATEGORIES)} (one category per "
            f"preference keeps read-time questions unambiguous)"
        )
    rng = random.Random(f"episode|{seed}|{episode_id}")
    world = world or FictionalWorld(seed)
    ep = Episode(episode_id=episode_id, seed=seed, capacity_k=capacity_k)

    # Entity indices and value indices are separate counters: entities index the
    # name list (<= 256) while values index the much larger encoded value space.
    ecur = 0
    vcur = 0

    def next_entity() -> str:
        nonlocal ecur
        name = world.entity(ecur)
        ecur += 1
        return name

    def next_value(kind: str) -> MemoryValue:
        nonlocal vcur
        value = world.value(kind, vcur)
        vcur += 1
        return value

    # ---- facts -------------------------------------------------------------
    kinds = list(FictionalWorld.KINDS)
    rng.shuffle(kinds)
    for i in range(n_facts):
        kind = kinds[i % len(kinds)]
        entity = next_entity()
        value = next_value(kind)
        tmpl = rng.choice(_FACT_TEMPLATES[kind])
        ep.writes.append(
            MemoryItem(
                item_id=len(ep.writes),
                memory_class="fact",
                attribute=kind,
                statement=tmpl.format(entity=entity, value=value.value),
                value=value,
                query=_FACT_QUERIES[kind].format(entity=entity),
            )
        )

    # ---- preferences -------------------------------------------------------
    categories = list(_PREF_CATEGORIES)
    rng.shuffle(categories)
    for i in range(n_prefs):
        category = categories[i]
        value = next_value("dish" if i % 2 == 0 else "accent_color")
        statement = rng.choice(_PREF_STATEMENTS).format(
            category=category, value=value.value
        )
        ep.writes.append(
            MemoryItem(
                item_id=len(ep.writes),
                memory_class="pref",
                attribute=f"pref:{category}",
                statement=statement,
                value=value,
                query=_PREF_QUERY.format(category=category),
            )
        )

    # ---- lessons -----------------------------------------------------------
    for _ in range(n_lessons):
        noun_a = next_value("capital").value
        noun_b = next_value("capital").value
        action = f"run the {noun_b} reset before rebuilding"
        ep.writes.append(
            MemoryItem(
                item_id=len(ep.writes),
                memory_class="lesson",
                attribute=f"lesson:{noun_a}",
                statement=_LESSON_TEMPLATE.format(noun_a=noun_a, noun_b=noun_b),
                value=MemoryValue("lesson", action, (noun_b,)),
                query=_LESSON_QUERY.format(noun_a=noun_a),
            )
        )

    # ---- noise (distractors, must be evictable) ----------------------------
    for _ in range(n_noise):
        entity = next_entity()
        ep.writes.append(
            MemoryItem(
                item_id=len(ep.writes),
                memory_class="noise",
                attribute="chatter",
                statement=rng.choice(_NOISE_STATEMENTS).format(entity=entity),
                value=None,
                is_noise=True,
            )
        )

    # ---- importance marking ------------------------------------------------
    probeable = ep.probeable_items
    n_important = max(1, int(round(len(probeable) * important_fraction))) if probeable else 0
    important_ids = {i.item_id for i in probeable[:n_important]}
    ep.writes = [
        MemoryItem(
            item_id=w.item_id, memory_class=w.memory_class, attribute=w.attribute,
            statement=w.statement, value=w.value, is_noise=w.is_noise,
            is_important=w.item_id in important_ids, query=w.query,
        )
        for w in ep.writes
    ]

    # ---- written probes ----------------------------------------------------
    for w in ep.writes:
        if w.probeable:
            ep.probes.append(
                Probe(item_id=w.item_id, query=w.query, value=w.value_text, aliases=w.aliases)
            )
    rng.shuffle(ep.probes)

    # ---- negative controls (never written; same surface form) --------------
    for _ in range(n_negatives):
        kind = rng.choice(list(FictionalWorld.KINDS))
        entity = next_entity()
        value = next_value(kind)
        ep.negatives.append(
            Probe(
                item_id=None,
                query=_FACT_QUERIES[kind].format(entity=entity),
                value=value.value,
                aliases=value.aliases,
                is_negative=True,
            )
        )

    # ---- inertia block -----------------------------------------------------
    if with_inertia:
        case = _INERTIA_CASES[rng.randrange(len(_INERTIA_CASES))]
        country = next_entity()
        other = next_entity()
        unit = next_value("capital").value
        premise = case["premise"].format(country=country, unit=unit)
        new_task = case["new_task"].format(other=other)
        keywords = tuple(
            k.format(unit_lower=unit.lower()) for k in case["inherited_keywords"]
        )
        ep.inertia = InertiaBlock(
            old_topic=country,
            premise=premise,
            new_task=new_task,
            inherited_keywords=keywords,
            old_topic_keywords=(country.lower(), unit.lower()),
        )

    if vcur > _MAX_VALUE_INDEX:
        raise AssertionError(
            f"episode used {vcur} value indices, exceeding the world's {_MAX_VALUE_INDEX}"
        )
    return ep


def make_episodes(n: int, *, base_seed: int = 0, **kwargs) -> list[Episode]:
    """``n`` episodes with disjoint worlds (different seeds => different words)."""
    return [make_episode(base_seed + i, episode_id=i, **kwargs) for i in range(n)]
