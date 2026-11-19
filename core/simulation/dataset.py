"""Renderer-independent simulation data contracts.

Simulation output is stored as a versioned JSON metadata sidecar plus a
compressed NumPy NPZ array archive. Large numerical arrays never live in JSON.
The format is readable with only the Python standard library and NumPy, so a
consumer such as Blender's bundled Python needs no scientific dependency set
(only ``numpy`` and ``json``) to load exported numerical data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from ..physics.integration import TrajectoryResult

SCHEMA_VERSION = "1.0.0"

REQUIRED_METADATA_KEYS = (
    "schema_version",
    "simulation_id",
    "simulation_type",
    "model",
    "model_limitations",
    "units",
    "reference_frame",
    "physical_parameters",
    "created_at",
)

REQUIRED_UNIT_KEYS = ("time", "position")

ALLOWED_UNITS = {
    "time": "s",
    "position": "m",
    "velocity": "m/s",
    "acceleration": "m/s^2",
    "mass": "kg",
    "length": "m",
}

# Arrays that may be multi-dimensional (T, N, D) and contain NaN for
# terminated particles when the nonfinite policy is "masked".
TRAJECTORY_ARRAYS = ("positions", "velocities")
NONFINITE_POLICY_MASKED = "masked"
NONFINITE_POLICY_KEY = "nonfinite_policy"
VALIDITY_MASK_ARRAY = "validity_mask"
PARTICLE_IDS_ARRAY = "particle_ids"

# Trajectory-specific metadata keys (optional; only validated when present).
TRAJECTORY_METADATA_KEYS = (
    "black_hole_mass_kg",
    "schwarzschild_radius_m",
    "integration_cutoff_radius_m",
    "numerical_method",
    "solver_rtol",
    "solver_atol",
    "initial_conditions",
    "termination_status",
)


class SimulationMetadata(TypedDict, total=False):
    """Lightweight, JSON-serializable simulation metadata."""

    schema_version: str
    simulation_id: str
    simulation_type: str
    model: str
    model_limitations: str
    units: dict[str, str]
    reference_frame: dict[str, str]
    physical_parameters: dict[str, Any]
    created_at: str
    array_sources: dict[str, str]
    notes: str


@dataclass
class SimulationDataset:
    """A renderer-independent simulation result.

    Attributes
    ----------
    metadata:
        JSON-serializable metadata describing the simulation.
    arrays:
        Named NumPy arrays holding the numerical data (e.g. timestamps,
        positions, accelerations). Large arrays live here, not in JSON.
    """

    metadata: SimulationMetadata
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


def _empty_metadata() -> SimulationMetadata:
    return SimulationMetadata(
        schema_version=SCHEMA_VERSION,
        simulation_id="",
        simulation_type="",
        model="",
        model_limitations="",
        units={},
        reference_frame={},
        physical_parameters={},
        created_at=datetime.now(timezone.utc).isoformat(),
        array_sources={},
        notes="",
    )


def validate_dataset(dataset: SimulationDataset) -> None:
    """Validate schema, dimensions, units, timestamps and finite values.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    metadata = dataset.metadata
    arrays = dataset.arrays

    for key in REQUIRED_METADATA_KEYS:
        if key not in metadata or metadata[key] is None:
            raise ValueError(f"metadata missing required key: {key}")
        if isinstance(metadata[key], str) and not metadata[key].strip():
            raise ValueError(f"metadata key empty: {key}")

    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version {metadata['schema_version']!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )

    simulation_id = str(metadata["simulation_id"]).strip()
    if not simulation_id:
        raise ValueError("simulation_id must be non-empty")

    units = metadata.get("units", {})
    if not isinstance(units, dict):
        raise ValueError("units must be a dictionary")
    missing_units = [k for k in REQUIRED_UNIT_KEYS if k not in units]
    if missing_units:
        raise ValueError(f"units missing required keys: {missing_units}")
    for key, value in units.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"unit for {key!r} must be a non-empty string")

    reference_frame = metadata.get("reference_frame", {})
    if not isinstance(reference_frame, dict) or not reference_frame:
        raise ValueError("reference_frame must be a non-empty dictionary")

    if not isinstance(metadata.get("physical_parameters"), dict):
        raise ValueError("physical_parameters must be a dictionary")

    if not isinstance(arrays, dict):
        raise ValueError("arrays must be a dictionary")
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise ValueError(f"array {name!r} is not a NumPy array")

    nonfinite_policy = metadata.get(NONFINITE_POLICY_KEY)
    if nonfinite_policy not in (None, NONFINITE_POLICY_MASKED):
        raise ValueError(f"unsupported nonfinite policy: {nonfinite_policy!r}")
    masked = nonfinite_policy == NONFINITE_POLICY_MASKED

    # Trajectory consistency: validity_mask and particle_ids must be present
    # together when any 3-D trajectory array is present. A 1-D array named
    # "positions" or "velocities" is treated as a regular legacy array.
    has_trajectory = any(
        name in TRAJECTORY_ARRAYS and arrays[name].ndim == 3 for name in arrays
    )
    if has_trajectory:
        if not masked:
            raise ValueError("3-D trajectories require the masked nonfinite policy")
        if "timestamps" not in arrays:
            raise ValueError("trajectory arrays require timestamps")
        if VALIDITY_MASK_ARRAY not in arrays:
            raise ValueError(
                f"trajectory arrays require {VALIDITY_MASK_ARRAY!r}"
            )
        if PARTICLE_IDS_ARRAY not in arrays:
            raise ValueError(
                f"trajectory arrays require {PARTICLE_IDS_ARRAY!r}"
            )

    # Check the mask before using it to index trajectory arrays.
    if has_trajectory:
        mask = arrays[VALIDITY_MASK_ARRAY]
        if not isinstance(mask, np.ndarray) or mask.dtype.kind != "b":
            raise ValueError("validity_mask must be a boolean NumPy array")
        if mask.ndim != 2:
            raise ValueError("validity_mask must be two-dimensional")

    lengths: list[int] = []
    for name, value in arrays.items():
        if value.size == 0:
            raise ValueError(f"array {name!r} is empty")

        is_trajectory_candidate = name in TRAJECTORY_ARRAYS
        is_masked_trajectory = (
            is_trajectory_candidate and masked and value.ndim == 3
        )

        # Dimensionality rules.
        if name == "timestamps":
            if value.ndim != 1:
                raise ValueError(
                    f"array {name!r} must be one-dimensional, got {value.ndim}D"
                )
        elif name == VALIDITY_MASK_ARRAY:
            if value.dtype.kind != "b":
                raise ValueError("validity_mask must have boolean dtype")
            if value.ndim != 2:
                raise ValueError(
                    f"array {name!r} must be two-dimensional, got {value.ndim}D"
                )
        elif value.dtype.kind == "b":
            # Boolean masks (e.g. validity_mask) are two-dimensional (T, N).
            if value.ndim != 2:
                raise ValueError(
                    f"array {name!r} must be two-dimensional, got {value.ndim}D"
                )
        elif is_masked_trajectory:
            if value.ndim != 3:
                raise ValueError(
                    f"array {name!r} must have shape (T, N, D), got {value.ndim}D"
                )
        elif value.ndim != 1:
            raise ValueError(
                f"array {name!r} must be one-dimensional, got {value.ndim}D"
            )

        # Finiteness rules.
        if name == "timestamps":
            if not np.all(np.isfinite(value)):
                raise ValueError(f"array {name!r} contains non-finite values")
        elif name == PARTICLE_IDS_ARRAY:
            # String identifiers are not subject to numeric finiteness checks.
            pass
        elif is_masked_trajectory:
            # NaN is permitted only where validity_mask is False; every valid
            # entry must be finite. Never fabricate plausible-looking data.
            validity = arrays.get(VALIDITY_MASK_ARRAY)
            if validity is None or validity.shape != value.shape[:2]:
                raise ValueError(
                    f"{VALIDITY_MASK_ARRAY!r} must match the (T, N) shape of "
                    f"{name!r}"
                )
            valid_values = value[validity]
            if valid_values.size > 0 and not np.all(np.isfinite(valid_values)):
                raise ValueError(
                    f"array {name!r} contains non-finite values where "
                    f"{VALIDITY_MASK_ARRAY!r} is True"
                )
            invalid_values = value[~validity]
            if invalid_values.size > 0 and not np.all(
                np.isnan(invalid_values)
            ):
                raise ValueError(
                    f"array {name!r} must be NaN where "
                    f"{VALIDITY_MASK_ARRAY!r} is False"
                )
        else:
            if not np.all(np.isfinite(value)):
                raise ValueError(f"array {name!r} contains non-finite values")

        if name == PARTICLE_IDS_ARRAY:
            if value.ndim != 1:
                raise ValueError(
                    f"array {name!r} must be one-dimensional, got {value.ndim}D"
                )
            if value.dtype.kind not in ("U", "S"):
                raise ValueError(f"array {name!r} must be a string array")
            # String identifiers have length N, not T; exclude from the
            # shared-length check (validated separately above).
        elif value.ndim == 1:
            lengths.append(value.size)

    # All multi-element 1D arrays must share a length so trajectories align.
    multi_lengths = [n for n in lengths if n > 1]
    if multi_lengths and len(set(multi_lengths)) != 1:
        raise ValueError(
            "all one-dimensional arrays with more than one element must "
            "share the same length"
        )

    # Trajectory dimension consistency.
    if has_trajectory:
        timestamps = arrays["timestamps"]
        validity = arrays[VALIDITY_MASK_ARRAY]
        particle_ids = arrays[PARTICLE_IDS_ARRAY]
        n_times = int(timestamps.size)
        n_particles = int(particle_ids.size)
        for name in TRAJECTORY_ARRAYS:
            if name in arrays and arrays[name].ndim == 3:
                arr = arrays[name]
                if arr.shape != (n_times, n_particles, 2):
                    raise ValueError(
                        f"array {name!r} shape {arr.shape} must be "
                        f"(T={n_times}, N={n_particles}, 2)"
                    )
        if validity.shape != (n_times, n_particles):
            raise ValueError(
                f"{VALIDITY_MASK_ARRAY!r} shape {validity.shape} must be "
                f"(T={n_times}, N={n_particles})"
            )
        if np.any(np.diff(validity.astype(np.int8), axis=0) > 0):
            raise ValueError("terminated particles cannot become valid again")
        identifiers = particle_ids.tolist()
        if not all(identifier.strip() for identifier in identifiers):
            raise ValueError("particle IDs must not be empty")
        if len(set(identifiers)) != n_particles:
            raise ValueError("particle IDs must be unique")

    # Timestamps, when present, must be strictly increasing.
    if "timestamps" in arrays:
        timestamps = arrays["timestamps"]
        if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("timestamps must be strictly increasing")


def trajectory_to_dataset(
    result: TrajectoryResult,
    simulation_id: str,
    *,
    simulation_type: str = "black_hole_trajectory",
    model: str = "newtonian_test_particle",
    model_limitations: str = (
        "Newtonian point-mass test-particle integration. Does not model "
        "general relativistic effects near or inside the Schwarzschild "
        "radius. Terminated particles are marked invalid and their trailing "
        "samples are NaN; no data is fabricated after termination."
    ),
    notes: str = "",
) -> SimulationDataset:
    """Convert a :class:`TrajectoryResult` into a :class:`SimulationDataset`.

    The produced dataset uses the ``nonfinite_policy = "masked"`` convention:
    ``positions`` and ``velocities`` are ``(T, N, 2)`` arrays that may contain
    NaN where the ``validity_mask`` is False. All other arrays remain finite
    and 1-D. This is backward compatible with the existing validator because
    the mask policy is declared explicitly in metadata.

    Parameters
    ----------
    result:
        The integration result to export.
    simulation_id:
        Unique simulation identifier.
    simulation_type:
        Type label stored in metadata.
    model:
        Physical model name stored in metadata.
    model_limitations:
        Honest statement of model limitations.
    notes:
        Free-form notes.

    Returns
    -------
    SimulationDataset
        Dataset ready for :func:`save_dataset`.
    """
    black_hole = result.black_hole
    timestamps = result.timestamps
    positions = result.positions
    velocities = result.velocities
    validity = result.validity_mask
    particle_ids = np.array(result.particle_ids)  # fixed-width unicode array

    arrays = {
        "timestamps": timestamps,
        "positions": positions,
        "velocities": velocities,
        VALIDITY_MASK_ARRAY: validity,
        PARTICLE_IDS_ARRAY: particle_ids,
    }

    units = {
        "time": "s",
        "position": "m",
        "velocity": "m/s",
        "mass": "kg",
        "length": "m",
    }

    reference_frame = {
        "origin": "black hole centre",
        "radial_axis": "positive outward from the black hole",
        "plane": "2D Cartesian (x, y)",
        "gravity_direction": "toward the black hole centre",
    }

    physical_parameters = {
        "black_hole_mass_kg": black_hole.mass_kg,
        "schwarzschild_radius_m": black_hole.schwarzschild_radius_m,
        "integration_cutoff_radius_m": black_hole.cutoff_radius_m,
        "n_particles": int(result.positions.shape[1]),
        "n_timestamps": int(result.timestamps.size),
        "duration_s": float(result.timestamps[-1] - result.timestamps[0]),
    }

    metadata: SimulationMetadata = SimulationMetadata(
        schema_version=SCHEMA_VERSION,
        simulation_id=simulation_id,
        simulation_type=simulation_type,
        model=model,
        model_limitations=model_limitations,
        units=units,
        reference_frame=reference_frame,
        physical_parameters=physical_parameters,
        created_at=datetime.now(timezone.utc).isoformat(),
        array_sources={},
        notes=notes,
        **{
            NONFINITE_POLICY_KEY: NONFINITE_POLICY_MASKED,
            "black_hole_mass_kg": black_hole.mass_kg,
            "schwarzschild_radius_m": black_hole.schwarzschild_radius_m,
            "integration_cutoff_radius_m": black_hole.cutoff_radius_m,
            "numerical_method": result.solver_method,
            "solver_rtol": result.solver_rtol,
            "solver_atol": result.solver_atol,
            "solver_success": bool(result.solver_success),
            "solver_message": str(result.solver_message),
            "termination_times_s": result.termination_times,
            "initial_conditions": {
                "particle_ids": list(result.particle_ids),
                "initial_positions_m": result.positions[0].tolist(),
                "initial_velocities_m_per_s": result.velocities[0].tolist(),
            },
            "termination_status": [
                (
                    "terminated_at_cutoff"
                    if t is not None
                    else "survived_duration"
                )
                for t in result.termination_times
            ],
        },
    )

    return SimulationDataset(metadata=metadata, arrays=arrays)


def save_dataset(
    dataset: SimulationDataset,
    path: str | Path,
    *,
    validate: bool = True,
) -> tuple[Path, Path]:
    """Save a dataset to a JSON metadata sidecar and a compressed NPZ archive.

    Parameters
    ----------
    dataset:
        The SimulationDataset to persist.
    path:
        Output path stem. Produces ``<stem>.meta.json`` and
        ``<stem>.data.npz``.
    validate:
        Whether to validate the dataset before writing.

    Returns
    -------
    tuple of Path
        The written metadata and archive paths.
    """
    if validate:
        validate_dataset(dataset)

    stem = Path(path)
    meta_path = stem.with_name(stem.name + ".meta.json")
    data_path = stem.with_name(stem.name + ".data.npz")

    metadata = dict(dataset.metadata)
    metadata["array_sources"] = {
        name: f"{data_path.name}#{name}" for name in dataset.arrays
    }
    metadata["created_at"] = datetime.now(timezone.utc).isoformat()

    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)

    np.savez_compressed(data_path, **dataset.arrays)
    return meta_path, data_path


def load_dataset(path: str | Path) -> SimulationDataset:
    """Load a dataset previously written with :func:`save_dataset`.

    Accepts either the metadata sidecar (``.meta.json``) or the NPZ archive
    (``.data.npz``) path; the sibling file is located automatically.
    """
    path = Path(path)
    if path.suffix == ".npz":
        data_path = path
        if path.stem.endswith(".data"):
            meta_stem = path.stem[:-len(".data")]
        else:
            meta_stem = path.stem
        meta_path = path.with_name(meta_stem + ".meta.json")
    elif path.suffix == ".json":
        meta_path = path
        if path.stem.endswith(".meta"):
            data_stem = path.stem[: -len(".meta")]
        else:
            data_stem = path.stem
        data_path = path.with_name(data_stem + ".data.npz")
    else:
        data_path = path.with_name(path.name + ".data.npz")
        meta_path = path.with_name(path.name + ".meta.json")

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata file not found: {meta_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data archive not found: {data_path}")

    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    dataset = SimulationDataset(metadata=metadata, arrays=arrays)
    validate_dataset(dataset)
    return dataset


def metadata_only(path: str | Path) -> SimulationMetadata:
    """Read only the JSON metadata sidecar without loading numerical arrays."""
    path = Path(path)
    if path.suffix == ".npz":
        if path.stem.endswith(".data"):
            meta_stem = path.stem[: -len(".data")]
        else:
            meta_stem = path.stem
        meta_path = path.with_name(meta_stem + ".meta.json")
    elif path.suffix == ".json":
        meta_path = path
    else:
        meta_path = path.with_name(path.name + ".meta.json")
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
