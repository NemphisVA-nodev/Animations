"""Physical-time-to-frame mapping and trajectory sampling.

Bridges simulation data (physical time in seconds) and animation frames
(presentation time in seconds at a fixed frame rate). Renderer-independent:
no Blender, Matplotlib, GUI, or SciPy is required, so a future Blender
importer can use this with only the Python standard library and NumPy.

Two concerns:

1. ``FrameMapping`` converts a scene's presentation duration and frame rate
   into a deterministic list of frame timestamps and physical simulation
   times.
2. ``sample_particle_positions`` reads particle positions at arbitrary
   physical times from a :class:`SimulationDataset`, interpolating only
   between valid neighbouring samples and never fabricating data after a
   particle's termination.

Timing convention (fixed-rate):

* Effective playback speed = simulation span / presentation duration, in
  simulation seconds per presentation second.
* ``num_frames = max(1, round(duration_s * fps))``.
* Frame ``i`` (0-indexed) presents at ``i / fps`` seconds and maps to
  simulation time ``start_sim_time_s + (i / fps) * effective_playback_speed``.
* The simulation window end (``end_sim_time_s``) is the scene boundary; it
  is not forcibly placed on the final active frame, so the last frame may
  land slightly before the boundary when ``duration_s * fps`` is not an
  integer.
* Hold frames freeze at the last active frame's simulation time and do not
  advance simulation time.

This convention makes adjacent scenes continuous: if scene B starts where
scene A ends (``start_sim_time_s`` equal), scene B's frame 0 begins at that
same physical time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.simulation.dataset import SimulationDataset

__all__ = [
    "FrameMapping",
    "PlaybackSpec",
    "build_frame_mapping",
    "sample_particle_positions",
]

_TIMESTAMPS_KEY = "timestamps"
_POSITIONS_KEY = "positions"
_VALIDITY_MASK_KEY = "validity_mask"
_PARTICLE_IDS_KEY = "particle_ids"


@dataclass(frozen=True)
class PlaybackSpec:
    """Specification for mapping simulation time to presentation frames.

    Attributes
    ----------
    scene_id:
        Unique identifier for the scene within a presentation.
    simulation_id:
        Identifier of the simulation dataset consumed by this scene, or
        ``None`` for conceptual/observational scenes.
    fps:
        Presentation frame rate in frames per second (> 0).
    duration_s:
        Presentation duration of the scene in seconds (> 0).
    start_sim_time_s:
        Start of the physical simulation window consumed, in SI seconds.
    end_sim_time_s:
        End of the physical simulation window consumed, in SI seconds
        (>= start). Treated as the scene boundary.
    hold_s:
        Optional hold period appended after the active frames, in seconds
        (>= 0). Hold frames freeze at the last active frame's simulation
        time.
    """

    scene_id: str
    simulation_id: str | None
    fps: float
    duration_s: float
    start_sim_time_s: float
    end_sim_time_s: float
    hold_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        for name, value in (
            ("fps", self.fps),
            ("duration_s", self.duration_s),
            ("start_sim_time_s", self.start_sim_time_s),
            ("end_sim_time_s", self.end_sim_time_s),
            ("hold_s", self.hold_s),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.end_sim_time_s < self.start_sim_time_s:
            raise ValueError("end_sim_time_s must be >= start_sim_time_s")
        if self.hold_s < 0.0:
            raise ValueError("hold_s must be non-negative")


@dataclass(frozen=True)
class FrameMapping:
    """Deterministic mapping from presentation frames to simulation time.

    Attributes
    ----------
    scene_id:
        Scene identifier.
    simulation_id:
        Simulation dataset identifier, or ``None`` for conceptual scenes.
    fps:
        Frame rate in frames per second.
    duration_s:
        Presentation duration in seconds.
    start_sim_time_s:
        Physical simulation time at frame 0, in seconds.
    end_sim_time_s:
        Physical simulation time at the scene boundary (end of the window).
    num_frames:
        Number of active (simulation-advancing) frames.
    frame_times_s:
        Presentation time of each active frame, shape ``(num_frames,)``.
    sim_times_s:
        Physical simulation time at each active frame, shape
        ``(num_frames,)``. Frame 0 is ``start_sim_time_s``.
    effective_playback_speed:
        Simulation seconds per presentation second, equals
        ``(end_sim_time_s - start_sim_time_s) / duration_s``.
    hold_s:
        Hold duration in seconds.
    endpoint_sim_time_s:
        Physical simulation time of the last active frame; hold frames
        freeze at this value.
    total_frames:
        ``num_frames + hold_frames``.
    """

    scene_id: str
    simulation_id: str | None
    fps: float
    duration_s: float
    start_sim_time_s: float
    end_sim_time_s: float
    num_frames: int
    frame_times_s: np.ndarray
    sim_times_s: np.ndarray
    effective_playback_speed: float
    hold_s: float
    endpoint_sim_time_s: float
    total_frames: int

    @property
    def sim_span_s(self) -> float:
        """Duration of the physical simulation window consumed."""
        return self.end_sim_time_s - self.start_sim_time_s

    @property
    def hold_frames(self) -> int:
        """Number of hold frames (frozen at ``endpoint_sim_time_s``)."""
        return self.total_frames - self.num_frames


def build_frame_mapping(spec: PlaybackSpec) -> FrameMapping:
    """Build a deterministic frame-to-simulation-time mapping.

    Parameters
    ----------
    spec:
        Playback specification.

    Returns
    -------
    FrameMapping
        Deterministic mapping using the fixed-rate convention.
    """
    if not spec.scene_id.strip():
        raise ValueError("scene_id must be non-empty")
    for name, value in (
        ("fps", spec.fps),
        ("duration_s", spec.duration_s),
        ("start_sim_time_s", spec.start_sim_time_s),
        ("end_sim_time_s", spec.end_sim_time_s),
        ("hold_s", spec.hold_s),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if spec.fps <= 0.0:
        raise ValueError("fps must be positive")
    if spec.duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if spec.end_sim_time_s < spec.start_sim_time_s:
        raise ValueError("end_sim_time_s must be >= start_sim_time_s")
    if spec.hold_s < 0.0:
        raise ValueError("hold_s must be non-negative")

    sim_span = spec.end_sim_time_s - spec.start_sim_time_s
    effective_playback_speed = sim_span / spec.duration_s

    # Deterministic frame count.
    num_frames = max(1, int(round(spec.duration_s * spec.fps)))

    # Presentation time of each active frame: 0, 1/fps, 2/fps, ...
    frame_times_s = np.arange(num_frames, dtype=np.float64) / spec.fps

    # Physical simulation time at each active frame (fixed-rate convention).
    # Frame 0 -> start_sim_time_s. The end of the window is the scene
    # boundary; the last frame approximates it and is not forcibly anchored.
    sim_times_s = spec.start_sim_time_s + frame_times_s * effective_playback_speed

    # Hold frames freeze at the last active frame's simulation time.
    endpoint_sim_time_s = float(sim_times_s[-1])
    hold_frames = max(0, int(round(spec.hold_s * spec.fps)))
    total_frames = num_frames + hold_frames

    return FrameMapping(
        scene_id=spec.scene_id,
        simulation_id=spec.simulation_id,
        fps=spec.fps,
        duration_s=spec.duration_s,
        start_sim_time_s=spec.start_sim_time_s,
        end_sim_time_s=spec.end_sim_time_s,
        num_frames=num_frames,
        frame_times_s=frame_times_s,
        sim_times_s=sim_times_s,
        effective_playback_speed=effective_playback_speed,
        hold_s=spec.hold_s,
        endpoint_sim_time_s=endpoint_sim_time_s,
        total_frames=total_frames,
    )


def _last_valid_index(validity_mask: np.ndarray, particle_index: int) -> int:
    """Return the last valid sample index for a particle, or -1 if none."""
    column = validity_mask[:, particle_index]
    valid = np.flatnonzero(column)
    return int(valid[-1]) if valid.size else -1


def _interpolate_at_time(
    timestamps: np.ndarray,
    positions: np.ndarray,
    validity_mask: np.ndarray,
    particle_index: int,
    query_time: float,
) -> tuple[np.ndarray, bool]:
    """Interpolate one particle's position at a single physical time.

    Uses :func:`numpy.searchsorted` for O(log T) lookup. Only interpolates
    between valid neighbouring samples; never crosses invalid samples or
    extrapolates beyond the last valid position.

    Parameters
    ----------
    timestamps:
        Strictly increasing 1-D simulation timestamps, shape ``(T,)``.
    positions:
        Particle positions, shape ``(T, N, 2)``.
    validity_mask:
        Boolean validity, shape ``(T, N)``. Must be a contiguous valid
        prefix per particle (no reactivation).
    particle_index:
        Index of the particle to sample.
    query_time:
        Physical simulation time in seconds.

    Returns
    -------
    tuple of (np.ndarray, bool)
        ``(position, is_valid)`` where ``position`` is shape ``(2,)`` in
        metres. ``is_valid`` is ``False`` when no valid position exists at
        that time.
    """
    nan_position = np.array([np.nan, np.nan], dtype=np.float64)

    last_valid = _last_valid_index(validity_mask, particle_index)
    if last_valid < 0:
        return nan_position, False

    # Valid samples are timestamps[0 : last_valid + 1] (contiguous prefix).
    t_first = float(timestamps[0])
    t_last = float(timestamps[last_valid])
    if query_time < t_first or query_time > t_last:
        return nan_position, False

    # Rightmost valid index whose timestamp is <= query_time.
    pos = int(np.searchsorted(timestamps, query_time, side="right") - 1)
    if pos < 0:
        return nan_position, False
    if pos > last_valid:
        pos = last_valid

    if timestamps[pos] == query_time:
        # Exactly on a valid sample.
        return positions[pos, particle_index].copy(), True

    # Need a right neighbour for interpolation.
    if pos + 1 > last_valid:
        # Beyond the last valid sample; do not fabricate.
        return nan_position, False

    t_left = float(timestamps[pos])
    t_right = float(timestamps[pos + 1])
    frac = (query_time - t_left) / (t_right - t_left)
    pos_left = positions[pos, particle_index]
    pos_right = positions[pos + 1, particle_index]
    position = pos_left + frac * (pos_right - pos_left)
    return position.astype(np.float64, copy=True), True


def sample_particle_positions(
    dataset: SimulationDataset,
    particle_ids: list[str],
    sim_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample particle positions at requested physical simulation times.

    Parameters
    ----------
    dataset:
        Loaded :class:`SimulationDataset` containing trajectory arrays
        (``positions``, ``validity_mask``, ``particle_ids``, ``timestamps``).
    particle_ids:
        Stable identifiers of the particles to sample, in the requested
        output order. Every id must exist in the dataset.
    sim_times:
        Physical simulation times in seconds, shape ``(Q,)``. Must be
        one-dimensional and finite.

    Returns
    -------
    tuple of np.ndarray
        ``(positions, validity)`` where:
        * ``positions`` has shape ``(Q, len(particle_ids), 2)`` in metres.
          Invalid entries are NaN.
        * ``validity`` has shape ``(Q, len(particle_ids))`` and is ``True``
          where a valid interpolated position exists.

    Raises
    ------
    ValueError
        If the dataset lacks required trajectory arrays, has malformed
        validity (e.g. a terminated particle reactivating), if any requested
        particle id is unknown, or if ``sim_times`` is invalid.

    Notes
    -----
    The original NPZ arrays are never modified. Interpolation only occurs
    between valid neighbouring samples; terminated particles and out-of-range
    times yield NaN positions with ``validity = False``. Only NumPy is used,
    so this is safe for a Blender-bundled Python environment.
    """
    arrays = dataset.arrays

    required = (
        _TIMESTAMPS_KEY,
        _POSITIONS_KEY,
        _VALIDITY_MASK_KEY,
        _PARTICLE_IDS_KEY,
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"dataset missing required trajectory arrays: {missing}")

    timestamps = arrays[_TIMESTAMPS_KEY]
    positions = arrays[_POSITIONS_KEY]
    validity_mask = arrays[_VALIDITY_MASK_KEY]
    dataset_ids = arrays[_PARTICLE_IDS_KEY]

    # --- Validate timestamps ---
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if timestamps.size < 2:
        raise ValueError("timestamps must contain at least two samples")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps must contain only finite values")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("timestamps must be strictly increasing")

    n_times = int(timestamps.size)

    # --- Validate positions shape (T, N, 2) ---
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(
            f"array {_POSITIONS_KEY!r} must have shape (T, N, 2), "
            f"got {positions.shape}"
        )
    if positions.shape[0] != n_times:
        raise ValueError(
            f"array {_POSITIONS_KEY!r} first dimension must match "
            "timestamps length"
        )

    # --- Validate validity_mask: boolean, shape (T, N), no reactivation ---
    if validity_mask.dtype.kind != "b":
        raise ValueError("validity_mask must have boolean dtype")
    if validity_mask.ndim != 2:
        raise ValueError(
            f"array {_VALIDITY_MASK_KEY!r} must be two-dimensional, "
            f"got {validity_mask.ndim}D"
        )
    if validity_mask.shape[0] != n_times:
        raise ValueError(
            f"array {_VALIDITY_MASK_KEY!r} first dimension must match "
            "timestamps length"
        )
    n_particles = int(validity_mask.shape[1])
    if positions.shape[1] != n_particles:
        raise ValueError(
            f"array {_POSITIONS_KEY!r} particle dimension must match "
            f"{_VALIDITY_MASK_KEY!r}"
        )

    # Reject validity histories that allow a terminated particle to reactivate.
    # Once a column goes False it must stay False (contiguous valid prefix).
    if n_particles > 0 and n_times > 1:
        transitions = np.diff(validity_mask.astype(np.int8), axis=0)
        # A -1 -> +1 transition (False to True) is reactivation.
        if np.any(transitions < 0) and np.any(transitions > 0):
            raise ValueError(
                "validity_mask must not allow a terminated particle to "
                "reactivate (each particle column must be a contiguous "
                "valid prefix)"
            )

    # --- Validate particle_ids ---
    if dataset_ids.ndim != 1 or dataset_ids.size != n_particles:
        raise ValueError(
            f"array {_PARTICLE_IDS_KEY!r} must be one-dimensional with "
            f"length {n_particles}"
        )

    # Resolve requested particle indices.
    id_to_index = {str(pid): idx for idx, pid in enumerate(dataset_ids.tolist())}
    missing_ids = [pid for pid in particle_ids if pid not in id_to_index]
    if missing_ids:
        raise ValueError(
            f"unknown particle ids: {missing_ids}. Known ids: "
            f"{dataset_ids.tolist()}"
        )
    particle_indices = [id_to_index[pid] for pid in particle_ids]

    # --- Validate sim_times ---
    sim_times = np.asarray(sim_times, dtype=np.float64)
    if sim_times.ndim != 1:
        raise ValueError("sim_times must be one-dimensional")
    if not np.all(np.isfinite(sim_times)):
        raise ValueError("sim_times must contain only finite values")

    n_queries = int(sim_times.size)
    n_requested = len(particle_indices)

    positions_out = np.full(
        (n_queries, n_requested, 2), np.nan, dtype=np.float64
    )
    validity_out = np.zeros((n_queries, n_requested), dtype=bool)

    for q in range(n_queries):
        query_time = float(sim_times[q])
        for p_idx, particle_index in enumerate(particle_indices):
            position, is_valid = _interpolate_at_time(
                timestamps,
                positions,
                validity_mask,
                particle_index,
                query_time,
            )
            positions_out[q, p_idx] = position
            validity_out[q, p_idx] = is_valid

    return positions_out, validity_out
