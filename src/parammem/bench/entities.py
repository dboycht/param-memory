"""Fictional entity / value generator (seeded, collision-free, reproducible).

Two hard requirements from the isolation protocol drive this design:

1. **A written value must not be guessable from pre-training.** Every entity and
   value is a word built from generated syllables, so no real toponym, person
   name or stock phrase can leak an answer into the experiment.
2. **The same item must get a *different* value in a different run** (protocol P2,
   the counterfactual-write control). Random generation only makes that likely;
   here the words are the **bijective encoding of a unique integer**, so values
   for different seeds are *provably* distinct.

Encoding layout (limits are asserted at construction time)::

    entity(seed, i)      -> encode(seed * 4096 + i)                       i < 256
    value(seed, kind, j) -> encode(1e9 + seed * 4096 + j * 8 + k)          j < 512

Because ``j * 8 + k < 4096``, a different seed shifts the encoded integer by a
multiple of 4096 and therefore yields a different word -- for every value kind
that is word-shaped. The remaining kinds (dates, passcodes, colours) are drawn
from a per-``n`` stream; if two runs ever do collide, ``protocol.compare_counterfactual``
detects it and the item is dropped from the attributable set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

_ONSET = ["v", "tr", "k", "m", "z", "br", "l", "n", "dr", "sh", "gl", "p", "f", "th"]
_NUCLEUS = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "ia", "uo"]
# The first entry is the empty coda, so words stay pronounceable; using the full
# option list (not a filtered one) is what keeps the encoding a bijection.
_CODA = ["", "n", "l", "r", "s", "m", "th", "sk", "nd", "rk", "st", "ph"]

_RADIX = len(_ONSET) * len(_NUCLEUS) * len(_CODA)  # 14 * 10 * 12 = 1680
_ENCODE_SYLLABLES = 3
_CAPACITY = _RADIX ** _ENCODE_SYLLABLES  # 4_741_632_000

_MAX_SEED = 2047
_MAX_ENTITY_INDEX = 256
_MAX_VALUE_INDEX = 512
_ENTITY_STRIDE = 4096
_VALUE_BASE = 1_000_000_000

_KINDS = ("capital", "birthday", "passcode", "dish", "accent_color")

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_COLORS = [
    "amber", "cobalt", "crimson", "emerald", "indigo", "marigold",
    "ochre", "saffron", "teal", "vermilion",
]

# Import-time guard: the whole point of the layout is that entity words and value
# words live in disjoint integer ranges and that every value fits the encoding.
_MAX_ENTITY_N = _MAX_SEED * _ENTITY_STRIDE + _MAX_ENTITY_INDEX
_MAX_VALUE_N = _VALUE_BASE + _MAX_SEED * _ENTITY_STRIDE + _MAX_VALUE_INDEX * 8 + len(_KINDS)
assert _MAX_ENTITY_N < _VALUE_BASE, "entity and value integer ranges overlap"
assert _MAX_VALUE_N < _CAPACITY, "value range does not fit the syllable encoding"


def _encode(n: int, syllables: int = _ENCODE_SYLLABLES) -> str:
    """Bijective base-1680 encoding of a non-negative integer into syllables.

    Distinct ``n`` always produce distinct words (for a fixed syllable count),
    which is what makes cross-seed value distinctness provable rather than
    probabilistic.
    """
    if n < 0:
        raise ValueError(f"cannot encode negative integer {n}")
    if n >= _RADIX ** syllables:
        raise ValueError(
            f"integer {n} does not fit in {syllables} syllables "
            f"(max {_RADIX ** syllables - 1})"
        )
    parts = []
    for _ in range(syllables):
        n, digit = divmod(n, _RADIX)
        o_idx, rest = divmod(digit, len(_NUCLEUS) * len(_CODA))
        nu_idx, co_idx = divmod(rest, len(_CODA))
        parts.append(_ONSET[o_idx] + _NUCLEUS[nu_idx] + _CODA[co_idx])
    return "".join(parts).capitalize()


@dataclass(frozen=True)
class MemoryValue:
    """One generated value plus the alternative surface forms it may appear as."""

    kind: str
    value: str
    aliases: tuple[str, ...] = ()

    def surface_forms(self) -> tuple[str, ...]:
        """All textual forms that must be treated as 'the same value'."""
        return (self.value,) + tuple(self.aliases)


class FictionalWorld:
    """Deterministic generator of fictional entities and their attributes.

    The same seed always yields the same world, so an experiment replays
    bit-for-bit *across processes* (never seed with a salted ``hash()``); a
    different seed yields a provably disjoint vocabulary.
    """

    KINDS = _KINDS

    def __init__(self, seed: int, *, names: int = _MAX_ENTITY_INDEX) -> None:
        self.seed = int(seed)
        if not (0 <= self.seed <= _MAX_SEED):
            raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {self.seed}")
        if not (1 <= names <= _MAX_ENTITY_INDEX):
            raise ValueError(f"names must be in [1, {_MAX_ENTITY_INDEX}], got {names}")
        self._entities: list[str] = [
            _encode(self.seed * _ENTITY_STRIDE + i) for i in range(names)
        ]

    # ---------------------------------------------------------------- entities
    def entity(self, index: int) -> str:
        if index >= len(self._entities):
            raise IndexError(
                f"entity index {index} out of range (world has {len(self._entities)})"
            )
        return self._entities[index]

    def entities(self) -> Iterable[str]:
        return tuple(self._entities)

    # ------------------------------------------------------------------ values
    def encoded_n(self, index: int, kind: str = "capital") -> int:
        """The unique integer behind a word-shaped value (also used for lesson
        nouns, so that every invented noun is provably seed-specific)."""
        return self._value_n(kind, index)

    def _value_n(self, kind: str, index: int) -> int:
        if kind not in _KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {_KINDS}")
        if not (0 <= index < _MAX_VALUE_INDEX):
            raise ValueError(
                f"value index must be in [0, {_MAX_VALUE_INDEX}), got {index}"
            )
        return _VALUE_BASE + self.seed * _ENTITY_STRIDE + index * 8 + _KINDS.index(kind)

    def value(self, kind: str, index: int) -> MemoryValue:
        """Generate a value of ``kind`` deterministically from the world seed."""
        n = self._value_n(kind, index)
        # Seeding with a string goes through sha512 in CPython and is therefore
        # stable across processes (unlike hash(), which PYTHONHASHSEED salts).
        rng = random.Random(f"v|{n}")

        if kind == "capital":
            return MemoryValue(kind, _encode(n))
        if kind == "birthday":
            year = 1930 + (n % 146)
            month = 1 + (n // 146) % 12
            day = 1 + (n // (146 * 12)) % 28
            primary = f"{_MONTHS[month - 1]} {day}, {year}"
            aliases = (
                f"{year}-{month:02d}-{day:02d}",
                f"{day} {_MONTHS[month - 1]} {year}",
                f"{month:02d}/{day:02d}/{year}",
            )
            return MemoryValue(kind, primary, aliases)
        if kind == "passcode":
            alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
            code = "".join(rng.choice(alphabet) for _ in range(6))
            return MemoryValue(kind, code)
        if kind == "dish":
            return MemoryValue(kind, f"{rng.choice(_COLORS)} {_encode(n).lower()} stew")
        if kind == "accent_color":
            hexcode = "#" + "".join(rng.choice("0123456789ABCDEF") for _ in range(6))
            return MemoryValue(kind, hexcode, (hexcode.lower(),))
        raise AssertionError("unreachable")
