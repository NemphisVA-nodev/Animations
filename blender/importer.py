"""Load and validate trajectory datasets for Blender visualization.

This module is independent of Blender's ``bpy`` module and of SciPy, so it
can be tested with pytest and imported by Blender's bundled Python using only
the Python standard library and NumPy.

Responsibilities
----------------
* Locate and load the ``.meta.json`` / ``.data.npz`` trajectory files.
* Validate the source data before any Blender objects are created.
* Preserve the original SI values (metres, seconds).
* Convert 2-D simulation coordinates (x, y) to Blender's XY plane (z = 0)
  using a user-configurable, constant metres-to-blender-units scale.

This module never modifies the original simulation data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.simulation.dataset import (  # noqa: E402
    SimulationDataset,
    load_dataset,
    VALIDITY_MASK_ARRAY,
    PARTICLE_IDS_ARRAY,
    TRAJECTORY_ARRAYS,
)

__all__ = [
    "ImportError",
    "TrajectoryData",
    "load_trajectory",
    "metres_to_blender",
    "SCALE_DEFAULT",
]

# Default metres-to-blender-units scale. With the default generator settings
# (initial radius 1e8 m), a scale of 1e-6 keeps the scene within roughly
# +/-100 Blender units, which fits comfortably in an orthographic view.
SCALE_DEFAULT = 1.0e-6


class ImportError(Exception):
    """Raised when a trajectory dataset cannot be imported."""


@dataclass(frozen=True)
class TrajectoryData:
    """Validated trajectory data ready for Blender visualization.

    Attributes
    ----------
    timestamps:
        Physical simulation times in seconds, shape ``(T,)``.
    particle_ids:
        Stable particle identifiers, length ``N``.
    positions:
        Particle positions in metres, shape ``(T, N, 2)``. Entries where the
        validity mask is False are NaN.
    velocities:
        Particle velocities in m/s, shape ``(T, N, 2)``. Invalid samples
        remain NaN, just like positions.
    validity_mask:
        Boolean validity per particle per timestamp, shape ``(T, N)``.
    metadata:
        Copy of the dataset's JSON metadata.
    scale:
        Metres-to-blender-units scale used for the coordinate conversion.
    """

    timestamps: np.ndarray
    particle_ids: list[str]
    positions: np.ndarray
    velocities: np.ndarray
    validity_mask: np.ndarray
    metadata: dict
    scale: float


def load_trajectory(
    path: str | Path,
    *,
    scale: float = SCALE_DEFAULT,
) -> TrajectoryData:
    """Load and validate a trajectory dataset for Blender import.

    Parameters
    ----------
    path:
        Path stem of the dataset. Produces ``<stem>.meta.json`` and
        ``<stem>.data.npz``.
    scale:
        Constant metres-to-blender-units scale. Must be positive and finite.

    Returns
    -------
    TrajectoryData
        Validated trajectory data. SI values are preserved; only the scale is
        applied when converting to Blender coordinates.

    Raises
    ------
    ImportError
        If the files are missing, invalid, or the scale is unacceptable.
    """
    path = Path(path)
    meta_path = path.with_name(path.name + ".meta.json")
    data_path = path.with_name(path.name + ".data.npz")

    if not meta_path.exists():
        raise ImportError(f"metadata file not found: {meta_path}")
    if not data_path.exists():
        raise ImportError(f"data archive not found: {data_path}")

    if not np.isfinite(scale) or scale <= 0.0:
        raise ImportError(f"scale must be a positive finite value, got {scale!r}")

    # load_dataset validates schema, units, masks and array shapes.
    dataset: SimulationDataset = load_dataset(meta_path)
    arrays = dataset.arrays

    for name in TRAJECTORY_ARRAYS:
        if name not in arrays or arrays[name].ndim != 3:
            raise ImportError(
                f"dataset missing 3-D trajectory array {name!r}"
            )
    if VALIDITY_MASK_ARRAY not in arrays:
        raise ImportError("dataset missing validity_mask array")
    if PARTICLE_IDS_ARRAY not in arrays:
        raise ImportError("dataset missing particle_ids array")

    positions = arrays["positions"]
    validity = arrays[VALIDITY_MASK_ARRAY]
    particle_ids = arrays[PARTICLE_IDS_ARRAY]
    timestamps = arrays["timestamps"]

    n_times, n_particles, _dims = positions.shape
    if validity.shape != (n_times, n_particles):
        raise ImportError(
            f"validity_mask shape {validity.shape} must match positions "
            f"first two dims ({n_times}, {n_particles})"
        )
    if timestamps.size != n_times:
        raise ImportError(
            "timestamps length must match positions time dimension"
        )
    if particle_ids.size != n_particles:
        raise ImportError(
            "particle_ids length must match positions particle dimension"
        )

    return TrajectoryData(
        timestamps=timestamps.astype(np.float64, copy=False),
        particle_ids=[str(pid) for pid in particle_ids.tolist()],
        positions=positions.astype(np.float64, copy=False),
        velocities=arrays["velocities"].astype(np.float64, copy=False),
        validity_mask=validity.astype(bool, copy=False),
        metadata=dict(dataset.metadata),
        scale=float(scale),
    )


def metres_to_blender(
    positions_m: np.ndarray,
    scale: float = SCALE_DEFAULT,
) -> np.ndarray:
    """Convert ``(T, N, 2)`` metre positions to Blender ``(T, N, 3)``.

    The simulation's 2-D Cartesian plane is mapped to Blender's XY plane with
    Z = 0. All values are multiplied by the constant ``scale`` so that
    astronomical metre values become manageable Blender units.

    Parameters
    ----------
    positions_m:
        Positions in metres, shape ``(T, N, 2)``. NaN entries are preserved.
    scale:
        Metres-to-blender-units scale (positive, finite).

    Returns
    -------
    np.ndarray
        Positions in Blender units, shape ``(T, N, 3)`` with Z = 0.
    """
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be a positive finite value, got {scale!r}")
    if positions_m.ndim != 3 or positions_m.shape[2] != 2:
        raise ValueError(
            f"positions_m must have shape (T, N, 2), got {positions_m.shape}"
        )

    x = positions_m[..., 0] * scale
    y = positions_m[..., 1] * scale
    z = np.zeros_like(x)
    return np.stack((x, y, z), axis=-1)


def python_frame_to_blender_frame(python_index: int) -> int:
    """Convert a zero-based Python frame index to a Blender frame number.

    Blender frames are one-based, so Python frame 0 is Blender frame 1.
    """
    if python_index < 0:
        raise ValueError("python_index must be non-negative")
    return int(python_index) + 1
