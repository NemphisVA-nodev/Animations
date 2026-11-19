"""Tests for the Newtonian black hole model."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c as speed_of_light
from scipy.constants import G as gravitational_constant

from core.physics.black_hole import BlackHole

MASS_KG = 1.9885e31  # ~10 solar masses
POSITION = np.array([1.0e6, 2.0e6], dtype=np.float64)
CUTOFF = 1.0e7


def _bh(mass_kg: float = MASS_KG, position=None, cutoff: float = CUTOFF) -> BlackHole:
    return BlackHole(
        mass_kg=mass_kg,
        position=np.array(position if position is not None else POSITION),
        cutoff_radius_m=cutoff,
    )


def test_schwarzschild_radius_formula() -> None:
    bh = _bh()
    expected = 2.0 * gravitational_constant * MASS_KG / speed_of_light**2
    assert bh.schwarzschild_radius_m == pytest.approx(expected, rel=1e-12)


def test_schwarzschild_radius_scales_linearly_with_mass() -> None:
    bh_small = _bh(mass_kg=MASS_KG)
    bh_large = _bh(mass_kg=2.0 * MASS_KG)
    assert bh_large.schwarzschild_radius_m == pytest.approx(
        2.0 * bh_small.schwarzschild_radius_m, rel=1e-12
    )


def test_schwarzschild_radius_positive() -> None:
    bh = _bh()
    assert bh.schwarzschild_radius_m > 0.0


def test_acceleration_points_toward_black_hole() -> None:
    bh = _bh()
    test_point = bh.position + np.array([3.0e7, 4.0e7], dtype=np.float64)
    accel = bh.gravitational_acceleration(test_point)
    displacement = test_point - bh.position
    # Acceleration must be anti-parallel to displacement.
    cross = accel[0] * displacement[1] - accel[1] * displacement[0]
    assert cross == pytest.approx(0.0, abs=1e-9)
    dot = float(np.dot(accel, displacement))
    assert dot < 0.0


def test_acceleration_inverse_square() -> None:
    bh = _bh()
    r1 = bh.position + np.array([1.0e7, 0.0], dtype=np.float64)
    r2 = bh.position + np.array([2.0e7, 0.0], dtype=np.float64)
    a1 = float(np.linalg.norm(bh.gravitational_acceleration(r1)))
    a2 = float(np.linalg.norm(bh.gravitational_acceleration(r2)))
    assert a2 == pytest.approx(a1 / 4.0, rel=1e-12)


def test_acceleration_magnitude_formula() -> None:
    bh = _bh()
    distance = 1.0e7
    point = bh.position + np.array([distance, 0.0], dtype=np.float64)
    accel = bh.gravitational_acceleration(point)
    expected = gravitational_constant * MASS_KG / distance**2
    assert float(np.linalg.norm(accel)) == pytest.approx(expected, rel=1e-12)


def test_translation_invariance() -> None:
    bh = _bh()
    point = bh.position + np.array([1.0e7, 2.0e7], dtype=np.float64)
    accel_base = bh.gravitational_acceleration(point)

    offset = np.array([5.0e6, -3.0e6], dtype=np.float64)
    bh_shifted = BlackHole(
        mass_kg=MASS_KG, position=bh.position + offset, cutoff_radius_m=CUTOFF
    )
    accel_shifted = bh_shifted.gravitational_acceleration(point + offset)
    assert accel_shifted == pytest.approx(accel_base, rel=1e-12)


def test_rejects_nonpositive_mass() -> None:
    with pytest.raises(ValueError):
        BlackHole(mass_kg=0.0, position=POSITION, cutoff_radius_m=CUTOFF)
    with pytest.raises(ValueError):
        BlackHole(mass_kg=-1.0, position=POSITION, cutoff_radius_m=CUTOFF)


def test_rejects_nonfinite_or_wrong_shape_position() -> None:
    with pytest.raises(ValueError):
        BlackHole(
            mass_kg=MASS_KG,
            position=np.array([np.nan, 0.0]),
            cutoff_radius_m=CUTOFF,
        )
    with pytest.raises(ValueError):
        BlackHole(
            mass_kg=MASS_KG,
            position=np.array([1.0, 2.0, 3.0]),
            cutoff_radius_m=CUTOFF,
        )


def test_rejects_invalid_cutoff() -> None:
    with pytest.raises(ValueError):
        BlackHole(mass_kg=MASS_KG, position=POSITION, cutoff_radius_m=0.0)
    schwarzschild = _bh().schwarzschild_radius_m
    with pytest.raises(ValueError):
        BlackHole(
            mass_kg=MASS_KG,
            position=POSITION,
            cutoff_radius_m=schwarzschild,
        )


def test_rejects_singularity_position() -> None:
    bh = _bh()
    with pytest.raises(ValueError):
        bh.gravitational_acceleration(bh.position.copy())
