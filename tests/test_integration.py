"""Tests for numerical trajectory integration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import G

from core.physics.integration import integrate_particles
from core.physics.black_hole import BlackHole
from core.physics.particles import Particle, ParticleCloud, radial_particles

MASS_KG = 1.9885e31
CUTOFF = 1.0e7
DURATION = 50.0
STEPS = 500


def _bh() -> BlackHole:
    return BlackHole(
        mass_kg=MASS_KG,
        position=np.array([0.0, 0.0], dtype=np.float64),
        cutoff_radius_m=CUTOFF,
    )


def _single_particle(radius: float, tangential_velocity: float = 0.0) -> ParticleCloud:
    return ParticleCloud(
        particles=[
            Particle(
                id="p0",
                position=np.array([radius, 0.0], dtype=np.float64),
                velocity=np.array([0.0, tangential_velocity], dtype=np.float64),
            )
        ]
    )


def test_radial_infall_accelerates_inward() -> None:
    bh = _bh()
    radius = 1.0e8
    cloud = _single_particle(radius, tangential_velocity=0.0)
    result = integrate_particles(bh, cloud, duration_s=DURATION, num_steps=STEPS)
    assert result.solver_success
    positions = result.positions[:, 0, :]
    valid = result.validity_mask[:, 0]
    # While valid, radial distance should decrease (falling inward).
    distances = np.linalg.norm(positions[valid], axis=1)
    assert distances[-1] < distances[0]
    # Terminated because it reached the cutoff.
    assert result.termination_times[0] is not None


def test_trajectory_shapes() -> None:
    bh = _bh()
    cloud = radial_particles(
        center=np.array([1.0e8, 0.0]), count=3, spacing_m=1.0e6,
        inner_radius_m=0.0, common_velocity=np.array([0.0, 0.0]),
    )
    result = integrate_particles(bh, cloud, duration_s=DURATION, num_steps=STEPS)
    assert result.timestamps.shape == (STEPS,)
    assert result.positions.shape == (STEPS, 3, 2)
    assert result.velocities.shape == (STEPS, 3, 2)
    assert result.validity_mask.shape == (STEPS, 3)
    assert len(result.particle_ids) == 3
    assert len(result.termination_times) == 3


def test_reproducible_results() -> None:
    bh = _bh()
    cloud = _single_particle(1.0e8, tangential_velocity=0.0)
    r1 = integrate_particles(bh, cloud, duration_s=DURATION, num_steps=STEPS)
    r2 = integrate_particles(bh, cloud, duration_s=DURATION, num_steps=STEPS)
    assert np.array_equal(r1.positions, r2.positions, equal_nan=True)
    assert np.array_equal(r1.velocities, r2.velocities, equal_nan=True)
    assert r1.termination_times == r2.termination_times


def test_cutoff_terminates_trajectory() -> None:
    bh = _bh()
    cloud = _single_particle(1.0e8, tangential_velocity=0.0)
    result = integrate_particles(bh, cloud, duration_s=DURATION, num_steps=STEPS)
    assert result.termination_times[0] is not None
    # After termination, particle is invalid and remains invalid.
    validity = result.validity_mask[:, 0]
    first_invalid = int(np.argmax(~validity))
    assert not np.any(validity[first_invalid:])
    # Termination time is within the duration.
    assert 0.0 < result.termination_times[0] <= DURATION


def test_no_continuation_after_termination() -> None:
    bh = _bh()
    cloud = _single_particle(1.0e8, tangential_velocity=0.0)
    result = integrate_particles(bh, cloud, duration_s=DURATION, num_steps=STEPS)
    validity = result.validity_mask[:, 0]
    positions = result.positions[:, 0, :]
    # After the first invalid index, all positions are NaN.
    first_invalid = int(np.argmax(~validity))
    assert np.all(np.isnan(positions[first_invalid:]))


def test_stable_circular_orbit() -> None:
    bh = _bh()
    radius = 1.0e8
    # Circular orbit speed: sqrt(GM / r).
    speed = np.sqrt(G * MASS_KG / radius)
    cloud = _single_particle(radius, tangential_velocity=speed)
    result = integrate_particles(bh, cloud, duration_s=10.0, num_steps=1000)
    assert result.solver_success
    # The particle should not reach the cutoff during a short stable orbit.
    assert result.validity_mask[:, 0].all()
    distances = np.linalg.norm(result.positions[:, 0, :], axis=1)
    # Radius stays close to the initial value (relative drift small).
    assert np.all(distances == pytest.approx(radius, rel=0.02))


def test_energy_conservation_before_cutoff() -> None:
    bh = _bh()
    radius = 1.0e8
    cloud = _single_particle(radius, tangential_velocity=0.0)
    result = integrate_particles(bh, cloud, duration_s=DURATION, num_steps=STEPS)
    positions = result.positions[:, 0, :]
    velocities = result.velocities[:, 0, :]
    valid = result.validity_mask[:, 0]
    # Specific energy: v^2/2 - GM/r.
    r = np.linalg.norm(positions[valid], axis=1)
    v2 = np.sum(velocities[valid] ** 2, axis=1)
    energy = 0.5 * v2 - G * MASS_KG / r
    # Energy should be approximately conserved (relative drift small).
    assert np.all(energy == pytest.approx(energy[0], rel=1e-3))


def test_rejects_invalid_duration() -> None:
    bh = _bh()
    cloud = _single_particle(1.0e8)
    with pytest.raises(ValueError):
        integrate_particles(bh, cloud, duration_s=0.0, num_steps=STEPS)
    with pytest.raises(ValueError):
        integrate_particles(bh, cloud, duration_s=-1.0, num_steps=STEPS)


def test_rejects_too_few_steps() -> None:
    bh = _bh()
    cloud = _single_particle(1.0e8)
    with pytest.raises(ValueError):
        integrate_particles(bh, cloud, duration_s=DURATION, num_steps=1)


def test_multiple_particles_terminate_at_different_times() -> None:
    bh = _bh()
    cloud = radial_particles(
        center=np.array([1.0e8, 0.0]), count=3, spacing_m=2.0e6,
        inner_radius_m=0.0, common_velocity=np.array([0.0, 0.0]),
    )
    result = integrate_particles(bh, cloud, duration_s=DURATION, num_steps=STEPS)
    times = [t for t in result.termination_times if t is not None]
    assert len(times) == 3
    # Nearest particle (largest acceleration) terminates first.
    # radial_particles places particle 0 nearest (smallest radius).
    assert times[0] < times[1] < times[2]


def test_rejects_particle_starting_at_or_inside_cutoff() -> None:
    bh = _bh()
    for radius in (CUTOFF, CUTOFF / 2):
        with pytest.raises(ValueError, match="cutoff"):
            integrate_particles(bh, _single_particle(radius), DURATION, STEPS)


def test_rejects_empty_cloud() -> None:
    with pytest.raises(ValueError, match="empty"):
        integrate_particles(_bh(), ParticleCloud(), DURATION, STEPS)


def test_rejects_nonfinite_duration() -> None:
    with pytest.raises(ValueError):
        integrate_particles(_bh(), _single_particle(1.0e8), np.nan, STEPS)


def test_rejects_noninteger_steps() -> None:
    with pytest.raises(ValueError):
        integrate_particles(_bh(), _single_particle(1.0e8), 1.0, 2.5)


def test_solver_failure_does_not_return_partial_trajectory() -> None:
    # Unsupported methods are errors, not successful simulation results.
    with pytest.raises(ValueError):
        integrate_particles(
            _bh(), _single_particle(1.0e8), 1.0, 10,
            method="not_a_solver",
        )


def test_records_solver_configuration() -> None:
    result = integrate_particles(
        _bh(), _single_particle(1.0e8), 1.0, 10,
        rtol=1e-7, atol=1e-10, method="RK45",
    )
    assert result.solver_method == "RK45"
    assert result.solver_rtol == pytest.approx(1e-7)
    assert result.solver_atol == pytest.approx(1e-10)


def test_failed_solver_result_is_rejected(monkeypatch) -> None:
    from types import SimpleNamespace
    import core.physics.integration as integration

    def failed_solver(*args, **kwargs):
        return SimpleNamespace(
            success=False, status=-1, sol=None,
            message="forced test failure",
        )

    monkeypatch.setattr(integration, "solve_ivp", failed_solver)
    with pytest.raises(RuntimeError, match="forced test failure"):
        integrate_particles(_bh(), _single_particle(1.0e8), 1.0, 10)
