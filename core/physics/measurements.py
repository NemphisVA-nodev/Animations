"""Tidal deformation measurements from integrated particle trajectories.

All quantities are computed from the integrated particle positions and the
black hole model. Differential acceleration, relative displacement and actual
material deformation are distinct concepts and are reported separately. The
particle cloud is not a physically connected object, so these measurements
describe relative test-particle motion, not real material deformation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .black_hole import BlackHole
from .integration import TrajectoryResult


def _valid_positions_at(
    result: TrajectoryResult, time_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and ids of valid particles at a time index."""
    mask = result.validity_mask[time_index]
    positions = result.positions[time_index][mask]
    ids = np.array(result.particle_ids)[mask]
    return positions, ids


def geometric_center(result: TrajectoryResult, time_index: int) -> Optional[np.ndarray]:
    """Geometric centre of the valid particles at a time index.

    Returns None if no particle is valid at that index.
    """
    positions, _ = _valid_positions_at(result, time_index)
    if positions.size == 0:
        return None
    return np.mean(positions, axis=0)


def cloud_centres_over_time(result: TrajectoryResult) -> np.ndarray:
    """Geometric centre of valid particles at every time, shape ``(T, 2)``.

    Times with no valid particles are NaN.
    """
    timestamps = result.timestamps
    count = len(result.particle_ids)
    centres = np.full((timestamps.size, 2), np.nan, dtype=np.float64)
    for t in range(timestamps.size):
        centre = geometric_center(result, t)
        if centre is not None:
            centres[t] = centre
    return centres


def reference_trajectory(
    result: TrajectoryResult, particle_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Trajectory of a single particle.

    Returns
    -------
    tuple of np.ndarray
        ``(positions, validity)`` where positions has shape ``(T, 2)`` in
        metres (NaN where invalid) and validity has shape ``(T,)`` bool.
    """
    if particle_id not in result.particle_ids:
        raise ValueError(f"unknown particle id: {particle_id!r}")
    index = result.particle_ids.index(particle_id)
    positions = result.positions[:, index, :].copy()
    validity = result.validity_mask[:, index].copy()
    return positions, validity


def radial_extent(result: TrajectoryResult) -> np.ndarray:
    """Maximum distance from the black hole among valid particles, shape ``(T,)``.

    Times with no valid particles are NaN.
    """
    timestamps = result.timestamps
    extents = np.full(timestamps.size, np.nan, dtype=np.float64)
    for t in range(timestamps.size):
        positions, _ = _valid_positions_at(result, t)
        if positions.size == 0:
            continue
        distances = np.linalg.norm(positions - result.black_hole.position, axis=1)
        extents[t] = float(np.max(distances))
    return extents


def transverse_extent(result: TrajectoryResult) -> np.ndarray:
    """Maximum transverse distance from the radial line among valid particles.

    The radial line is defined from the black hole through the cloud's
    geometric centre at each time. Returns shape ``(T,)`` in metres; NaN
    where no valid particles exist.
    """
    timestamps = result.timestamps
    extents = np.full(timestamps.size, np.nan, dtype=np.float64)
    centres = cloud_centres_over_time(result)
    for t in range(timestamps.size):
        positions, _ = _valid_positions_at(result, t)
        if positions.size == 0:
            continue
        centre = centres[t]
        if np.any(np.isnan(centre)):
            continue
        radial_dir = centre - result.black_hole.position
        radial_norm = float(np.linalg.norm(radial_dir))
        if radial_norm <= 0.0:
            continue
        radial_unit = radial_dir / radial_norm
        offsets = positions - centre
        # Transverse distance = norm of offset minus radial component.
        radial_components = np.dot(offsets, radial_unit)[:, None] * radial_unit
        transverse_vectors = offsets - radial_components
        transverse_distances = np.linalg.norm(transverse_vectors, axis=1)
        extents[t] = float(np.max(transverse_distances))
    return extents


def mean_pairwise_separation(result: TrajectoryResult) -> np.ndarray:
    """Mean pairwise separation among valid particles, shape ``(T,)`` in metres.

    NaN where fewer than two particles are valid.
    """
    timestamps = result.timestamps
    separations = np.full(timestamps.size, np.nan, dtype=np.float64)
    for t in range(timestamps.size):
        positions, _ = _valid_positions_at(result, t)
        if positions.shape[0] < 2:
            continue
        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        # Upper triangle without diagonal.
        iu = np.triu_indices(positions.shape[0], k=1)
        separations[t] = float(np.mean(distances[iu]))
    return separations


def radial_tidal_acceleration(
    result: TrajectoryResult, object_length_m: float
) -> np.ndarray:
    """Newtonian radial tidal acceleration estimate ``2 G M L / r^3``.

    Evaluated at the cloud's geometric centre distance from the black hole at
    each time. This is the first-order approximation, not a particle-based
    measurement.

    Parameters
    ----------
    result:
        Integration result.
    object_length_m:
        Characteristic object length ``L`` in metres (must be positive).

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` in m/s^2; NaN where the cloud centre is undefined.
    """
    if not np.isfinite(object_length_m) or object_length_m <= 0.0:
        raise ValueError("object_length_m must be a positive finite value")

    from .tidal import TidalParameters, tidal_difference_approximate

    timestamps = result.timestamps
    values = np.full(timestamps.size, np.nan, dtype=np.float64)
    centres = cloud_centres_over_time(result)
    for t in range(timestamps.size):
        centre = centres[t]
        if np.any(np.isnan(centre)):
            continue
        distance = float(np.linalg.norm(centre - result.black_hole.position))
        if distance <= object_length_m / 2.0:
            continue
        params = TidalParameters(
            attracting_mass_kg=result.black_hole.mass_kg,
            object_length_m=object_length_m,
            center_distance_m=distance,
        )
        values[t] = tidal_difference_approximate(params)
    return values


def transverse_tidal_acceleration(
    result: TrajectoryResult, object_length_m: float
) -> np.ndarray:
    """Newtonian transverse tidal compression estimate ``-G M L / r^3``.

    Evaluated at the cloud's geometric centre distance from the black hole.
    Negative values denote compression.

    Parameters
    ----------
    result:
        Integration result.
    object_length_m:
        Characteristic object length ``L`` in metres (must be positive).

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` in m/s^2; NaN where the cloud centre is undefined.
    """
    if not np.isfinite(object_length_m) or object_length_m <= 0.0:
        raise ValueError("object_length_m must be a positive finite value")

    radial = radial_tidal_acceleration(result, object_length_m)
    return -0.5 * radial
