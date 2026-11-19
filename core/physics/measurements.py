"""Tidal deformation measurements from integrated particle trajectories.

All quantities are computed from the integrated particle positions and the
black hole model. Differential acceleration, relative displacement and actual
material deformation are distinct concepts and are reported separately. The
particle cloud is not a physically connected object, so these measurements
describe relative test-particle motion, not real material deformation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from .integration import TrajectoryResult


def _valid_positions_at(result: TrajectoryResult, time_index: int) -> np.ndarray:
    """Positions of particles that are still active at this time."""
    return result.positions[time_index, result.validity_mask[time_index]]


def geometric_center(result: TrajectoryResult, time_index: int) -> np.ndarray | None:
    """Geometric centre of the valid particles at a time index.

    Returns None if no particle is valid at that index.
    """
    positions = _valid_positions_at(result, time_index)
    if positions.size == 0:
        return None
    return np.mean(positions, axis=0)


def cloud_centres_over_time(result: TrajectoryResult) -> np.ndarray:
    """Mean position of active particles at each timestamp.

    Rows with no active particles contain NaN.
    """
    mask = result.validity_mask
    counts = mask.sum(axis=1)
    centres = np.full((len(result.timestamps), 2), np.nan, dtype=np.float64)
    active_times = counts > 0
    if np.any(active_times):
        positions = np.where(mask[..., None], result.positions, 0.0)
        centres[active_times] = (
            positions[active_times].sum(axis=1) / counts[active_times, None]
        )
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
    """Full cloud width along the local radial direction, in metres.

    The direction points from the black hole toward the centre of the
    remaining valid particles. A single particle has zero extent.
    """
    extents = np.full(result.timestamps.size, np.nan, dtype=np.float64)
    for t in range(result.timestamps.size):
        positions = _valid_positions_at(result, t)
        if positions.size == 0:
            continue
        centre = np.mean(positions, axis=0)
        radial = centre - result.black_hole.position
        radius = float(np.linalg.norm(radial))
        if radius == 0.0:
            continue
        projections = (positions - centre) @ (radial / radius)
        extents[t] = float(np.ptp(projections))
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
        positions = _valid_positions_at(result, t)
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
        positions = _valid_positions_at(result, t)
        if positions.shape[0] < 2:
            continue
        separations[t] = float(pdist(positions).mean())
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
