"""Tests for tidal deformation measurements."""

from __future__ import annotations

import numpy as np
import pytest

from core.physics.integration import TrajectoryResult
from core.physics.measurements import (
    cloud_centres_over_time,
    geometric_center,
    mean_pairwise_separation,
    radial_extent,
    radial_tidal_acceleration,
    reference_trajectory,
    transverse_extent,
    transverse_tidal_acceleration,
)
from core.physics.black_hole import BlackHole

BH = BlackHole(
    mass_kg=1.9885e31,
    position=np.array([0.0, 0.0], dtype=np.float64),
    cutoff_radius_m=1.0e7,
)


def _make_result() -> TrajectoryResult:
    timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    # Three particles: at t=0 all valid; at t=2 particle 1 terminated.
    positions = np.full((3, 3, 2), np.nan, dtype=np.float64)
    velocities = np.zeros((3, 3, 2), dtype=np.float64)
    validity = np.ones((3, 3), dtype=bool)
    positions[0] = np.array([[1.0e8, 0.0], [1.0e8 + 1.0e6, 0.0], [1.0e8, 1.0e6]])
    positions[1] = np.array([[0.95e8, 0.0], [0.95e8 + 1.0e6, 0.0], [0.95e8, 1.0e6]])
    positions[2, 0] = np.array([0.9e8, 0.0])
    positions[2, 1] = np.nan
    positions[2, 2] = np.nan
    validity[2, 1] = False
    validity[2, 2] = False
    return TrajectoryResult(
        timestamps=timestamps,
        positions=positions,
        velocities=velocities,
        validity_mask=validity,
        particle_ids=["p0", "p1", "p2"],
        termination_times=[None, 2.0, 2.0],
        black_hole=BH,
        solver_success=True,
        solver_message="ok",
    )


def test_geometric_center() -> None:
    result = _make_result()
    centre = geometric_center(result, 0)
    assert centre is not None
    expected = np.mean(result.positions[0], axis=0)
    assert centre == pytest.approx(expected, rel=1e-12)
    # No valid particles -> None.
    result2 = _make_result()
    result2.validity_mask[0, :] = False
    assert geometric_center(result2, 0) is None


def test_cloud_centres_over_time_shape() -> None:
    result = _make_result()
    centres = cloud_centres_over_time(result)
    assert centres.shape == (3, 2)
    assert np.all(np.isfinite(centres[0]))
    assert centres[2] == pytest.approx([0.9e8, 0.0])


def test_reference_trajectory() -> None:
    result = _make_result()
    positions, validity = reference_trajectory(result, "p0")
    assert positions.shape == (3, 2)
    assert validity.shape == (3,)
    assert validity.all()
    assert positions[0, 0] == pytest.approx(1.0e8)
    with pytest.raises(ValueError):
        reference_trajectory(result, "nonexistent")


def test_radial_extent() -> None:
    result = _make_result()
    extents = radial_extent(result)
    assert extents.shape == (3,)
    # Radial width is the difference between extreme projections, not distance to BH.
    centre = np.mean(result.positions[0], axis=0)
    radial = centre - BH.position
    radial /= np.linalg.norm(radial)
    projections = (result.positions[0] - centre) @ radial
    assert extents[0] == pytest.approx(np.ptp(projections))
    assert extents[2] == pytest.approx(0.0)


def test_transverse_extent() -> None:
    result = _make_result()
    extents = transverse_extent(result)
    assert extents.shape == (3,)
    # p2 is offset transversely by 1.0e6 at t=0; center has y=1.0e6/3 roughly.
    assert extents[0] > 0.0


def test_mean_pairwise_separation() -> None:
    result = _make_result()
    separations = mean_pairwise_separation(result)
    assert separations.shape == (3,)
    assert separations[0] > 0.0
    # At t=2 only one particle valid -> NaN.
    assert np.isnan(separations[2])


def test_radial_tidal_acceleration() -> None:
    result = _make_result()
    values = radial_tidal_acceleration(result, object_length_m=1.0e6)
    assert values.shape == (3,)
    # Values positive (stretching) where center defined.
    assert np.all(values[0] > 0.0)
    assert np.all(values[1] > 0.0)


def test_transverse_tidal_acceleration_negative() -> None:
    result = _make_result()
    values = transverse_tidal_acceleration(result, object_length_m=1.0e6)
    assert values.shape == (3,)
    assert np.all(values[0] < 0.0)  # compression


def test_transverse_is_minus_half_radial() -> None:
    result = _make_result()
    radial = radial_tidal_acceleration(result, object_length_m=1.0e6)
    transverse = transverse_tidal_acceleration(result, object_length_m=1.0e6)
    assert transverse == pytest.approx(-0.5 * radial, rel=1e-12)


def test_tidal_rejects_bad_length() -> None:
    result = _make_result()
    with pytest.raises(ValueError):
        radial_tidal_acceleration(result, object_length_m=0.0)
    with pytest.raises(ValueError):
        transverse_tidal_acceleration(result, object_length_m=-1.0)


def test_centres_ignore_inactive_particles() -> None:
    result = _make_result()
    centres = cloud_centres_over_time(result)
    for t in range(len(result.timestamps)):
        active = result.positions[t, result.validity_mask[t]]
        assert centres[t] == pytest.approx(np.mean(active, axis=0))

    result.validity_mask[2] = False
    assert np.isnan(cloud_centres_over_time(result)[2]).all()


def test_pairwise_distance_matches_known_triangle() -> None:
    result = _make_result()
    result.positions[0] = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    # Pairwise distances are 3, 4, and 5 metres.
    assert mean_pairwise_separation(result)[0] == pytest.approx(4.0)


def test_tidal_acceleration_matches_scalar_formula() -> None:
    from scipy.constants import G

    result = _make_result()
    length = 1.0e6
    centres = cloud_centres_over_time(result)
    distances = np.linalg.norm(centres - result.black_hole.position, axis=1)
    expected = 2 * G * result.black_hole.mass_kg * length / distances**3
    assert radial_tidal_acceleration(result, length) == pytest.approx(expected)


def test_tidal_acceleration_handles_missing_cloud_center() -> None:
    result = _make_result()
    result.validity_mask[2] = False
    values = radial_tidal_acceleration(result, 1.0e6)
    assert np.isfinite(values[:2]).all()
    assert np.isnan(values[2])
