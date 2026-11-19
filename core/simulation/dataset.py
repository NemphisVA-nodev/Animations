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

    lengths: list[int] = []
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray):
            raise ValueError(f"array {name!r} is not a NumPy array")
        if value.size == 0:
            raise ValueError(f"array {name!r} is empty")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"array {name!r} contains non-finite values")
        if value.ndim != 1:
            raise ValueError(f"array {name!r} must be one-dimensional, got {value.ndim}D")
        lengths.append(value.size)

    # All multi-element 1D arrays must share a length so trajectories align.
    multi_lengths = [n for n in lengths if n > 1]
    if multi_lengths and len(set(multi_lengths)) != 1:
        raise ValueError(
            "all one-dimensional arrays with more than one element must "
            "share the same length"
        )

    # Timestamps, when present, must be strictly increasing.
    if "timestamps" in arrays:
        timestamps = arrays["timestamps"]
        if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("timestamps must be strictly increasing")


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
    meta_path = stem.with_suffix(".meta.json")
    data_path = stem.with_suffix(".data.npz")

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
        meta_path = path.with_suffix(".meta.json")
    elif path.suffix == ".json":
        meta_path = path
        if path.stem.endswith(".meta"):
            data_stem = path.stem[: -len(".meta")]
        else:
            data_stem = path.stem
        data_path = path.with_name(data_stem + ".data.npz")
    else:
        data_path = path.with_suffix(".data.npz")
        meta_path = path.with_suffix(".meta.json")

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata file not found: {meta_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data archive not found: {data_path}")

    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    with np.load(data_path) as archive:
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
        meta_path = path.with_suffix(".meta.json")
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
