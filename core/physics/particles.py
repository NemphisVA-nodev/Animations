"""Lightweight test-particle model and deterministic cloud initialization.

A particle cloud represents independent, freely falling test particles. No
springs, material elasticity, fluid dynamics, collisions or self-gravity are
introduced. The resulting deformation is not a complete simulation of a real
material body or star.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Particle:
    """A single independent freely falling test particle.

    Attributes
    ----------
    id:
        Stable particle identifier.
    position:
        Initial Cartesian position, shape ``(2,)`` in metres.
    velocity:
        Initial Cartesian velocity, shape ``(2,)`` in m/s.
    """

    id: str
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        pos = np.array(self.position, dtype=np.float64, copy=True)
        vel = np.array(self.velocity, dtype=np.float64, copy=True)
        if pos.shape != (2,):
            raise ValueError(f"particle {self.id!r}: position must have shape (2,)")
        if vel.shape != (2,):
            raise ValueError(f"particle {self.id!r}: velocity must have shape (2,)")
        if not np.all(np.isfinite(pos)):
            raise ValueError(f"particle {self.id!r}: position must contain only finite values")
        if not np.all(np.isfinite(vel)):
            raise ValueError(f"particle {self.id!r}: velocity must contain only finite values")
        pos.setflags(write=False)
        vel.setflags(write=False)
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "velocity", vel)


@dataclass
class ParticleCloud:
    """A collection of independent test particles with deterministic ordering.

    Attributes
    ----------
    particles:
        Ordered list of particles. Particle ids must be unique.
    """

    particles: list[Particle] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids = [p.id for p in self.particles]
        if len(ids) != len(set(ids)):
            raise ValueError("particle ids must be unique")

    @property
    def count(self) -> int:
        """Number of particles in the cloud."""
        return len(self.particles)

    @property
    def ids(self) -> list[str]:
        """Stable particle identifiers in deterministic order."""
        return [p.id for p in self.particles]

    @property
    def positions(self) -> np.ndarray:
        """Initial positions, shape ``(N, 2)`` in metres."""
        if not self.particles:
            return np.empty((0, 2), dtype=np.float64)
        return np.stack([p.position for p in self.particles])

    @property
    def velocities(self) -> np.ndarray:
        """Initial velocities, shape ``(N, 2)`` in m/s."""
        if not self.particles:
            return np.empty((0, 2), dtype=np.float64)
        return np.stack([p.velocity for p in self.particles])


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_count(count: int) -> None:
    if count < 1:
        raise ValueError("count must be at least 1")


def _validate_spacing(spacing_m: float) -> None:
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be a positive finite value")


def radial_particles(
    center: tuple[float, float] | np.ndarray,
    count: int,
    spacing_m: float,
    inner_radius_m: float,
    common_velocity: tuple[float, float] | np.ndarray,
    id_prefix: str = "p",
) -> ParticleCloud:
    """Create particles aligned radially along +x relative to ``center``.

    Particles are placed at ``inner_radius + i * spacing_m`` from the center
    along the outward radial axis.
    """
    _validate_count(count)
    _validate_spacing(spacing_m)
    if not np.isfinite(inner_radius_m) or inner_radius_m < 0.0:
        raise ValueError("inner_radius_m must be a non-negative finite value")

    center_arr = _as_vector(center, "center")
    velocity = _as_vector(common_velocity, "common_velocity")

    particles = []
    for i in range(count):
        radius = inner_radius_m + i * spacing_m
        position = center_arr + np.array([radius, 0.0], dtype=np.float64)
        particles.append(
            Particle(id=f"{id_prefix}_{i:04d}", position=position, velocity=velocity)
        )
    return ParticleCloud(particles=particles)


def transverse_particles(
    center: tuple[float, float] | np.ndarray,
    count: int,
    spacing_m: float,
    half_width_m: float,
    common_velocity: tuple[float, float] | np.ndarray,
    id_prefix: str = "p",
) -> ParticleCloud:
    """Create particles aligned transversely along +y relative to ``center``.

    Particles are placed at offsets from ``-half_width_m`` to
    ``+half_width_m`` along the transverse axis.
    """
    _validate_count(count)
    _validate_spacing(spacing_m)
    if not np.isfinite(half_width_m) or half_width_m < 0.0:
        raise ValueError("half_width_m must be a non-negative finite value")

    center_arr = _as_vector(center, "center")
    velocity = _as_vector(common_velocity, "common_velocity")

    particles = []
    for i in range(count):
        offset = -half_width_m + i * spacing_m
        position = center_arr + np.array([0.0, offset], dtype=np.float64)
        particles.append(
            Particle(id=f"{id_prefix}_{i:04d}", position=position, velocity=velocity)
        )
    return ParticleCloud(particles=particles)


def grid_particles(
    center: tuple[float, float] | np.ndarray,
    nx: int,
    ny: int,
    spacing_m: float,
    common_velocity: tuple[float, float] | np.ndarray,
    id_prefix: str = "p",
) -> ParticleCloud:
    """Create a rectangular grid of particles centred on ``center``.

    ``nx`` columns span x and ``ny`` rows span y, centred so the grid's
    geometric centre coincides with ``center``.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be at least 1")
    _validate_spacing(spacing_m)

    center_arr = _as_vector(center, "center")
    velocity = _as_vector(common_velocity, "common_velocity")

    ox = (nx - 1) * spacing_m / 2.0
    oy = (ny - 1) * spacing_m / 2.0

    particles = []
    index = 0
    for j in range(ny):
        y = center_arr[1] - oy + j * spacing_m
        for i in range(nx):
            x = center_arr[0] - ox + i * spacing_m
            particles.append(
                Particle(
                    id=f"{id_prefix}_{index:04d}",
                    position=np.array([x, y], dtype=np.float64),
                    velocity=velocity,
                )
            )
            index += 1
    return ParticleCloud(particles=particles)


def ring_particles(
    center: tuple[float, float] | np.ndarray,
    radius_m: float,
    count: int,
    common_velocity: tuple[float, float] | np.ndarray,
    id_prefix: str = "p",
    start_angle_rad: float = 0.0,
) -> ParticleCloud:
    """Create a ring of particles at fixed radius around ``center``.

    Particles are distributed uniformly in angle starting at
    ``start_angle_rad``.
    """
    _validate_count(count)
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be a positive finite value")
    if not np.isfinite(start_angle_rad):
        raise ValueError("start_angle_rad must be finite")

    center_arr = _as_vector(center, "center")
    velocity = _as_vector(common_velocity, "common_velocity")

    particles = []
    for i in range(count):
        angle = start_angle_rad + 2.0 * np.pi * i / count
        position = center_arr + radius_m * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float64
        )
        particles.append(
            Particle(id=f"{id_prefix}_{i:04d}", position=position, velocity=velocity)
        )
    return ParticleCloud(particles=particles)
