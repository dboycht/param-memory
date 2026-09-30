"""Guards for the per-slot stable rank.

The first implementation divided the *sum* of a slot's wrapper energies by the *largest
single* direction, which is not the stable rank of any matrix and returned about 86 for
updates of rank four. The invariant that catches it is simple and worth asserting: the
stable rank of a matrix never exceeds its rank.
"""

from __future__ import annotations

import sys
import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from t2b_composition_mechanism import stable_rank  # noqa: E402

RANK = 4


def _factors(count=3, out=64, inn=32, rank=RANK, seed=0):
    generator = torch.Generator().manual_seed(seed)
    parts, factors = [], []
    for _ in range(count):
        b = torch.randn(out, rank, generator=generator)
        a = torch.randn(rank, inn, generator=generator)
        scale = 0.7
        parts.append(scale * (b @ a))
        factors.append((b, a, scale))
    return parts, factors


def test_stable_rank_never_exceeds_the_rank():
    parts, factors = _factors()
    value = stable_rank(parts, factors=factors)
    assert 0 < value <= RANK + 1e-6, (
        f"stable rank {value} exceeds the rank {RANK}; the ratio is being taken across "
        "matrices instead of within one")


def test_cheap_and_expensive_paths_agree():
    parts, factors = _factors()
    assert stable_rank(parts) == pytest.approx(stable_rank(parts, factors=factors),
                                               rel=1e-4)


def test_energy_in_one_direction_gives_rank_one():
    # A rank-four update whose energy sits in a single direction has stable rank one.
    b = torch.zeros(64, RANK)
    a = torch.zeros(RANK, 32)
    b[:, 0] = torch.randn(64)
    a[0, :] = torch.randn(32)
    parts = [b @ a]
    assert stable_rank(parts, factors=[(b, a, 1.0)]) == pytest.approx(1.0, rel=1e-5)


def test_equal_singular_values_give_the_full_rank():
    # Orthogonal directions of equal size: every direction carries energy.
    q_out, _ = torch.linalg.qr(torch.randn(64, RANK))
    q_in, _ = torch.linalg.qr(torch.randn(32, RANK))
    parts = [q_out @ q_in.t()]
    assert stable_rank(parts) == pytest.approx(RANK, rel=1e-4)
