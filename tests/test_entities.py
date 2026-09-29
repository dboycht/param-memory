"""Tests for the fictional world generator.

The key property under test is *provable* cross-seed distinctness: protocol P2
(counterfactual write) is only meaningful if two runs cannot write the same value
for the same item by chance.
"""

from __future__ import annotations

import pytest

from parammem.bench.entities import FictionalWorld
from parammem.bench.protocol import numeric_forms


def test_deterministic_across_instances():
    assert list(FictionalWorld(42).entities()) == list(FictionalWorld(42).entities())
    assert FictionalWorld(42).value("birthday", 3) == FictionalWorld(42).value("birthday", 3)


def test_cross_seed_vocabulary_is_disjoint():
    for a, b in ((0, 1), (1, 2), (7, 300), (100, 2047)):
        ea, eb = set(FictionalWorld(a).entities()), set(FictionalWorld(b).entities())
        assert not (ea & eb), (a, b)


def test_cross_seed_word_values_are_distinct():
    """Word-shaped values differ across seeds for the same (kind, index)."""
    for kind in ("capital", "dish"):
        for index in (0, 7, 59):
            values = {FictionalWorld(s).value(kind, index).value for s in range(32)}
            assert len(values) == 32, (kind, index)


def test_entities_and_values_never_collide():
    for seed in (0, 5, 999):
        world = FictionalWorld(seed)
        ents = set(world.entities())
        vals = {
            world.value(kind, i).value
            for kind in FictionalWorld.KINDS
            for i in range(8)
        }
        assert not (ents & vals), seed


def test_values_stable_regardless_of_call_order():
    w1 = FictionalWorld(9)
    _ = w1.value("dish", 5)
    v1 = w1.value("capital", 5)
    w2 = FictionalWorld(9)
    v2 = w2.value("capital", 5)
    assert v1 == v2


def test_birthday_aliases_agree_numerically():
    value = FictionalWorld(3).value("birthday", 1)
    canonical = numeric_forms(value.value)
    assert len(canonical) == 1
    for alias in value.aliases:
        assert numeric_forms(alias) == canonical


def test_passcode_shape_and_no_ambiguous_glyphs():
    for index in range(20):
        code = FictionalWorld(4).value("passcode", index).value
        assert len(code) == 6
        assert code.isalnum() and code == code.upper()
        assert not (set(code) & set("IO01"))


def test_accent_colour_is_hex():
    value = FictionalWorld(6).value("accent_color", 2)
    assert value.value.startswith("#") and len(value.value) == 7
    assert value.value[1:].upper() == value.value[1:]


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        FictionalWorld(1).value("not_a_kind", 0)
    with pytest.raises(ValueError):
        FictionalWorld(1).value("capital", 10_000)
    with pytest.raises(ValueError):
        FictionalWorld(99_999)
    with pytest.raises(IndexError):
        FictionalWorld(1, names=4).entity(4)


def test_same_kind_different_index_differs():
    world = FictionalWorld(8)
    seen = {world.value("capital", i).value for i in range(30)}
    assert len(seen) == 30
