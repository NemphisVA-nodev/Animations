"""Tests for the test-particle model and cloud initialization."""

from __future__ import annotations

import numpy as np
import pytest

from core.physics.particles import (
    Particle,
    ParticleCloud,
    grid_particles,
    radial_particles,
    ring_particles,
    transverse_particles,
)

CENTER = np.array([1.0e8, 0.0], dtype=np.float64)
VELOCITY = np.array([0.0, 0.0], dtype=np.float64)


def _make_particle(pid: str = "p0") -> Particle:
    return Particle(
        id=pid,
        position=np.array([1.0, 2.0], dtype=np.float64),
        velocity=np.array([0.0, 1.0], dtype=np.float64),
    )


def test_particle_stores_fields() -> None:
    p = _make_particle()
    assert p.id == "p0"
    assert p.position.shape == (2,)
    assert p.velocity.shape == (2,)


@pytest.mark.parametrize("particle_id", ["", "   ", None, 3])
def test_particle_rejects_invalid_id(particle_id) -> None:
    with pytest.raises(ValueError, match="particle id"):
        _make_particle(particle_id)


def test_particle_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        Particle(id="p", position=np.array([1.0, 2.0, 3.0]), velocity=np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        Particle(id="p", position=np.array([1.0, 2.0]), velocity=np.array([1.0]))


def test_particle_rejects_nonfinite() -> None:
    with pytest.raises(ValueError):
        Particle(id="p", position=np.array([np.nan, 2.0]), velocity=np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        Particle(id="p", position=np.array([1.0, 2.0]), velocity=np.array([np.inf, 0.0]))


def test_cloud_count_and_ids() -> None:
    cloud = ParticleCloud(particles=[_make_particle("a"), _make_particle("b")])
    assert cloud.count == 2
    assert cloud.ids == ["a", "b"]


def test_cloud_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError):
        ParticleCloud(particles=[_make_particle("a"), _make_particle("a")])


def test_cloud_positions_velocities_shape() -> None:
    cloud = ParticleCloud(particles=[_make_particle("a"), _make_particle("b")])
    assert cloud.positions.shape == (2, 2)
    assert cloud.velocities.shape == (2, 2)


def test_empty_cloud() -> None:
    cloud = ParticleCloud()
    assert cloud.count == 0
    assert cloud.ids == []
    assert cloud.positions.shape == (0, 2)
    assert cloud.velocities.shape == (0, 2)


def test_radial_particles_deterministic_and_ordered() -> None:
    cloud = radial_particles(
        center=CENTER, count=4, spacing_m=1.0e6, inner_radius_m=0.0,
        common_velocity=VELOCITY,
    )
    assert cloud.count == 4
    ids = cloud.ids
    assert ids == sorted(ids)  # deterministic ordering
    positions = cloud.positions
    # Radii increase monotonically along +x.
    radii = positions[:, 0] - CENTER[0]
    assert np.all(np.diff(radii) > 0.0)
    assert np.all(positions[:, 1] == CENTER[1])  # on the radial axis


def test_transverse_particles_on_axis() -> None:
    cloud = transverse_particles(
        center=CENTER, count=5, spacing_m=1.0e6, half_width_m=2.0e6,
        common_velocity=VELOCITY,
    )
    positions = cloud.positions
    assert np.all(positions[:, 0] == CENTER[0])  # on transverse axis
    offsets = positions[:, 1] - CENTER[1]
    assert np.isclose(offsets[0], -2.0e6)
    assert np.isclose(offsets[-1], 2.0e6)


def test_grid_particles_centred() -> None:
    cloud = grid_particles(
        center=CENTER, nx=3, ny=3, spacing_m=1.0e6, common_velocity=VELOCITY,
    )
    assert cloud.count == 9
    positions = cloud.positions
    assert np.isclose(np.mean(positions[:, 0]), CENTER[0])
    assert np.isclose(np.mean(positions[:, 1]), CENTER[1])


def test_ring_particles_uniform_radius() -> None:
    cloud = ring_particles(
        center=CENTER, radius_m=1.0e6, count=8, common_velocity=VELOCITY,
    )
    positions = cloud.positions
    distances = np.linalg.norm(positions - CENTER, axis=1)
    assert np.all(distances == pytest.approx(1.0e6, rel=1e-12))
    # Uniform angular spacing.
    angles = np.arctan2(positions[:, 1] - CENTER[1], positions[:, 0] - CENTER[0])
    angles = np.sort(angles)
    gaps = np.diff(angles)
    assert np.all(gaps == pytest.approx(2.0 * np.pi / 8, abs=1e-9))


def test_validation_rejects_bad_count_spacing_radius() -> None:
    with pytest.raises(ValueError):
        radial_particles(center=CENTER, count=0, spacing_m=1.0, inner_radius_m=0.0,
                         common_velocity=VELOCITY)
    with pytest.raises(ValueError):
        radial_particles(center=CENTER, count=3, spacing_m=-1.0, inner_radius_m=0.0,
                         common_velocity=VELOCITY)
    with pytest.raises(ValueError):
        ring_particles(center=CENTER, radius_m=-1.0, count=4,
                       common_velocity=VELOCITY)


def test_common_velocity_applied_to_all() -> None:
    velocity = np.array([1.0, -2.0], dtype=np.float64)
    cloud = radial_particles(
        center=CENTER, count=3, spacing_m=1.0e6, inner_radius_m=0.0,
        common_velocity=velocity,
    )
    assert np.all(cloud.velocities == velocity)


def test_particle_copies_and_protects_initial_state() -> None:
    position = np.array([1.0, 2.0])
    velocity = np.array([3.0, 4.0])
    particle = Particle("sample", position, velocity)
    position[0] = 99.0
    velocity[1] = 99.0
    assert np.array_equal(particle.position, [1.0, 2.0])
    assert np.array_equal(particle.velocity, [3.0, 4.0])
    assert not particle.position.flags.writeable
    assert not particle.velocity.flags.writeable


def test_transverse_layout_rejects_inconsistent_width() -> None:
    with pytest.raises(ValueError, match="half_width_m"):
        transverse_particles(
            center=CENTER, count=5, spacing_m=1.0e6,
            half_width_m=1.0e6, common_velocity=VELOCITY,
        )


def test_particle_generators_require_integer_counts() -> None:
    with pytest.raises(ValueError, match="integer"):
        radial_particles(
            center=CENTER, count=2.5, spacing_m=1.0,
            inner_radius_m=1.0, common_velocity=VELOCITY,
        )
    with pytest.raises(ValueError, match="integer"):
        grid_particles(
            center=CENTER, nx=3, ny=2.5,
            spacing_m=1.0, common_velocity=VELOCITY,
        )
