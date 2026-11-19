"""Map physical simulation time to video frames and sample particle trajectories.

Frames are sampled at regular presentation times (i / fps). The end of a
scene's simulation window is its boundary, not an extra video frame.
Everything here is independent of Blender and SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.simulation.dataset import SimulationDataset


@dataclass(frozen=True)
class PlaybackSpec:
    """Timing for one scene, in seconds and frames per second."""

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

        for name in (
            "fps", "duration_s", "start_sim_time_s", "end_sim_time_s", "hold_s"
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

        if self.fps <= 0 or self.duration_s <= 0:
            raise ValueError("fps and duration_s must be positive")
        if self.end_sim_time_s < self.start_sim_time_s:
            raise ValueError("end_sim_time_s must be >= start_sim_time_s")
        if self.hold_s < 0:
            raise ValueError("hold_s must be non-negative")


@dataclass(frozen=True)
class FrameMapping:
    """Frame timestamps and their corresponding physical simulation times.

    `frame_times_s` and `sim_times_s` describe active frames only.
    Use `all_sim_times_s` to include the frozen frames of a hold.
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
        return self.end_sim_time_s - self.start_sim_time_s

    @property
    def hold_frames(self) -> int:
        return self.total_frames - self.num_frames

    @property
    def all_frame_times_s(self) -> np.ndarray:
        """Presentation times, including holds, sampled at the video FPS."""
        return np.arange(self.total_frames, dtype=np.float64) / self.fps

    @property
    def all_sim_times_s(self) -> np.ndarray:
        """Physical times of every rendered frame, with holds frozen."""
        if not self.hold_frames:
            return self.sim_times_s.copy()
        return np.concatenate((
            self.sim_times_s,
            np.full(self.hold_frames, self.endpoint_sim_time_s),
        ))


def build_frame_mapping(spec: PlaybackSpec) -> FrameMapping:
    """Sample one scene on a fixed-rate video frame grid."""
    sim_span = spec.end_sim_time_s - spec.start_sim_time_s
    speed = sim_span / spec.duration_s
    if not np.isfinite(speed):
        raise ValueError("simulation span and duration produce an invalid speed")

    num_frames = max(1, round(spec.duration_s * spec.fps))
    hold_frames = round(spec.hold_s * spec.fps)
    frame_times = np.arange(num_frames, dtype=np.float64) / spec.fps
    sim_times = spec.start_sim_time_s + frame_times * speed

    # A short scene rounded up to one frame still begins at its start time.
    # Floating point rounding should never push a frame past the boundary.
    sim_times = np.minimum(sim_times, spec.end_sim_time_s)

    return FrameMapping(
        scene_id=spec.scene_id,
        simulation_id=spec.simulation_id,
        fps=spec.fps,
        duration_s=spec.duration_s,
        start_sim_time_s=spec.start_sim_time_s,
        end_sim_time_s=spec.end_sim_time_s,
        num_frames=num_frames,
        frame_times_s=frame_times,
        sim_times_s=sim_times,
        effective_playback_speed=speed,
        hold_s=spec.hold_s,
        endpoint_sim_time_s=float(sim_times[-1]),
        total_frames=num_frames + hold_frames,
    )


def _trajectory_arrays(
    dataset: SimulationDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Check the arrays needed for interpolation and return their contents."""
    arrays = dataset.arrays
    required = ("timestamps", "positions", "velocities", "validity_mask", "particle_ids")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"dataset missing required trajectory arrays: {missing}")

    for name in required:
        if not isinstance(arrays[name], np.ndarray):
            raise ValueError(f"array {name!r} must be a NumPy array")

    times = arrays["timestamps"]
    positions = arrays["positions"]
    velocities = arrays["velocities"]
    mask = arrays["validity_mask"]
    ids = arrays["particle_ids"]

    if (
        times.ndim != 1
        or times.size == 0
        or times.dtype.kind not in ("f", "i", "u")
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("timestamps must be a finite, increasing 1-D array")

    if mask.dtype.kind != "b" or mask.ndim != 2:
        raise ValueError("validity_mask must be a two-dimensional boolean array")

    n_times, n_particles = mask.shape
    if n_times != times.size or n_particles == 0:
        raise ValueError("validity_mask must match timestamps and have particles")

    # Once a particle reaches the cutoff, it cannot become valid again.
    if np.any(~mask[:-1] & mask[1:]):
        raise ValueError("validity_mask must not allow a particle to reactivate")

    for name, value in (("positions", positions), ("velocities", velocities)):
        if value.shape != (n_times, n_particles, 2) or value.dtype.kind != "f":
            raise ValueError(f"{name} must be a real floating-point (T, N, 2) array")
        if not np.all(np.isfinite(value[mask])) or not np.all(np.isnan(value[~mask])):
            raise ValueError(f"{name} must be finite when valid and NaN when invalid")

    if ids.ndim != 1 or ids.size != n_particles or ids.dtype.kind not in ("U", "S"):
        raise ValueError("particle_ids must be a string array of length N")

    try:
        names = [
            item.decode("utf-8") if isinstance(item, bytes) else item
            for item in ids.tolist()
        ]
    except UnicodeError as exc:
        raise ValueError("particle_ids must contain valid UTF-8 strings") from exc

    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("particle_ids must be unique and non-empty")

    return times, positions, mask, names


def sample_particle_positions(
    dataset: SimulationDataset,
    particle_ids: list[str],
    sim_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate valid particle positions at physical times.

    Returns a (Q, P, 2) array in metres and a (Q, P) boolean mask. A particle
    has no position past its last valid simulation sample; it is never
    extrapolated through the black hole cutoff.
    """
    times, positions, mask, names = _trajectory_arrays(dataset)
    indices = {name: i for i, name in enumerate(names)}
    unknown = [name for name in particle_ids if name not in indices]
    if unknown:
        raise ValueError(f"unknown particle ids: {unknown}")

    try:
        query = np.asarray(sim_times, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("sim_times must contain numeric values") from exc
    if query.ndim != 1 or not np.all(np.isfinite(query)):
        raise ValueError("sim_times must be a finite one-dimensional array")

    result = np.full((query.size, len(particle_ids), 2), np.nan)
    valid = np.zeros((query.size, len(particle_ids)), dtype=bool)
    if query.size == 0 or not particle_ids:
        return result, valid

    # Search the time axis once, regardless of how many particles are requested.
    left = np.searchsorted(times, query, side="right") - 1
    bounded = (query >= times[0]) & (query <= times[-1])
    left = np.clip(left, 0, times.size - 1)
    exact = query == times[left]

    for output_index, name in enumerate(particle_ids):
        source_index = indices[name]
        last_valid = np.count_nonzero(mask[:, source_index]) - 1
        if last_valid < 0:
            continue

        on_sample = bounded & exact & (left <= last_valid)
        result[on_sample, output_index] = positions[left[on_sample], source_index]
        valid[on_sample, output_index] = True

        between = bounded & ~exact & (left < last_valid)
        if not np.any(between):
            continue

        i = left[between]
        fraction = (query[between] - times[i]) / (times[i + 1] - times[i])
        before = positions[i, source_index]
        after = positions[i + 1, source_index]
        result[between, output_index] = before + fraction[:, None] * (after - before)
        valid[between, output_index] = True

    return result, valid
