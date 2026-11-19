"""Tests for the simulation data format contract (schema + validation).

Independent of any graphical rendering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.simulation.dataset import (
    SimulationDataset,
    SimulationMetadata,
    SCHEMA_VERSION,
    save_dataset,
    validate_dataset,
)


def _base_metadata() -> SimulationMetadata:
    return {
        "schema_version": SCHEMA_VERSION,
        "simulation_id": "fmt_test",
        "simulation_type": "tidal_acceleration",
        "model": "newtonian_point_mass_tidal",
        "model_limitations": "Newtonian only.",
        "units": {"time": "s", "position": "m", "acceleration": "m/s^2"},
        "reference_frame": {
            "origin": "centre of attracting mass",
            "radial_axis": "positive outward",
        },
        "physical_parameters": {"mass_kg": 5.972e24},
        "created_at": "2026-01-01T00:00:00+00:00",
        "array_sources": {},
        "notes": "",
    }


def _base_dataset(tmp_path: Path) -> SimulationDataset:
    return SimulationDataset(
        metadata=_base_metadata(),
        arrays={
            "timestamps": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "positions": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        },
    )


def test_required_metadata_keys_enforced() -> None:
    meta = _base_metadata()
    for key in (
        "schema_version",
        "simulation_id",
        "simulation_type",
        "model",
        "model_limitations",
        "units",
        "reference_frame",
        "physical_parameters",
        "created_at",
    ):
        bad = dict(meta)
        bad[key] = None
        with pytest.raises(ValueError):
            validate_dataset(SimulationDataset(metadata=bad, arrays={}))


def test_empty_string_metadata_rejected() -> None:
    meta = _base_metadata()
    bad = dict(meta)
    bad["simulation_id"] = "   "
    with pytest.raises(ValueError):
        validate_dataset(SimulationDataset(metadata=bad, arrays={}))


def test_unsupported_schema_version_rejected() -> None:
    meta = _base_metadata()
    bad = dict(meta)
    bad["schema_version"] = "0.9.0"
    with pytest.raises(ValueError):
        validate_dataset(SimulationDataset(metadata=bad, arrays={}))


def test_missing_required_units_rejected() -> None:
    meta = _base_metadata()
    bad = dict(meta)
    bad["units"] = {"position": "m"}  # missing "time"
    with pytest.raises(ValueError):
        validate_dataset(SimulationDataset(metadata=bad, arrays={}))


def test_empty_reference_frame_rejected() -> None:
    meta = _base_metadata()
    bad = dict(meta)
    bad["reference_frame"] = {}
    with pytest.raises(ValueError):
        validate_dataset(SimulationDataset(metadata=bad, arrays={}))


def test_array_must_be_one_dimensional(tmp_path: Path) -> None:
    dataset = _base_dataset(tmp_path)
    bad = SimulationDataset(
        metadata=dataset.metadata,
        arrays={"timestamps": np.zeros((2, 2), dtype=np.float64)},
    )
    with pytest.raises(ValueError):
        validate_dataset(bad)


def test_array_must_be_finite(tmp_path: Path) -> None:
    dataset = _base_dataset(tmp_path)
    bad = SimulationDataset(
        metadata=dataset.metadata,
        arrays={
            "timestamps": np.array([0.0, np.inf, 2.0], dtype=np.float64),
            "positions": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        },
    )
    with pytest.raises(ValueError):
        validate_dataset(bad)


def test_arrays_must_share_length(tmp_path: Path) -> None:
    dataset = _base_dataset(tmp_path)
    bad = SimulationDataset(
        metadata=dataset.metadata,
        arrays={
            "timestamps": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "positions": np.array([1.0, 2.0], dtype=np.float64),
        },
    )
    with pytest.raises(ValueError):
        validate_dataset(bad)


def test_timestamps_must_be_strictly_increasing(tmp_path: Path) -> None:
    dataset = _base_dataset(tmp_path)
    bad = SimulationDataset(
        metadata=dataset.metadata,
        arrays={
            "timestamps": np.array([0.0, 2.0, 1.0], dtype=np.float64),
            "positions": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        },
    )
    with pytest.raises(ValueError):
        validate_dataset(bad)


def test_save_load_roundtrip_preserves_precision(tmp_path: Path) -> None:
    dataset = _base_dataset(tmp_path)
    stem = tmp_path / "fmt"
    save_dataset(dataset, stem)
    from core.simulation.dataset import load_dataset

    loaded = load_dataset(stem)
    for name in dataset.arrays:
        assert np.array_equal(loaded.arrays[name], dataset.arrays[name])
    assert loaded.arrays["timestamps"].dtype == np.float64
    assert loaded.metadata["schema_version"] == SCHEMA_VERSION


def test_load_dataset_from_npz_path(tmp_path: Path) -> None:
    from core.simulation.dataset import load_dataset

    dataset = _base_dataset(tmp_path)
    _, npz_path = save_dataset(dataset, tmp_path / "example")
    loaded = load_dataset(npz_path)
    assert np.array_equal(loaded.arrays["positions"], dataset.arrays["positions"])


def test_rejects_object_dtype_particle_ids() -> None:
    metadata = _base_metadata()
    metadata["nonfinite_policy"] = "masked"
    arrays = {
        "timestamps": np.array([0.0, 1.0]),
        "positions": np.ones((2, 1, 2)),
        "velocities": np.ones((2, 1, 2)),
        "validity_mask": np.ones((2, 1), dtype=bool),
        "particle_ids": np.array(["p0"], dtype=object),
    }
    with pytest.raises(ValueError, match="string array"):
        validate_dataset(SimulationDataset(metadata=metadata, arrays=arrays))


def test_rejects_nonboolean_validity_mask() -> None:
    metadata = _base_metadata()
    metadata["nonfinite_policy"] = "masked"
    arrays = {
        "timestamps": np.array([0.0, 1.0]),
        "positions": np.ones((2, 1, 2)),
        "velocities": np.ones((2, 1, 2)),
        "validity_mask": np.ones((2, 1), dtype=int),
        "particle_ids": np.array(["p0"]),
    }
    with pytest.raises(ValueError, match="boolean"):
        validate_dataset(SimulationDataset(metadata=metadata, arrays=arrays))


def _trajectory_dataset() -> SimulationDataset:
    metadata = _base_metadata()
    metadata["nonfinite_policy"] = "masked"
    return SimulationDataset(
        metadata=metadata,
        arrays={
            "timestamps": np.array([0.0, 1.0]),
            "positions": np.ones((2, 2, 2), dtype=float),
            "velocities": np.zeros((2, 2, 2), dtype=float),
            "validity_mask": np.ones((2, 2), dtype=bool),
            "particle_ids": np.array(["a", "b"]),
        },
    )


def test_rejects_nonarray_before_inspecting_dimensions() -> None:
    dataset = _trajectory_dataset()
    dataset.arrays["positions"] = [[1, 2]]
    with pytest.raises(ValueError, match="NumPy array"):
        validate_dataset(dataset)


def test_rejects_wrong_spatial_dimension() -> None:
    dataset = _trajectory_dataset()
    dataset.arrays["positions"] = np.ones((2, 2, 3))
    with pytest.raises(ValueError, match="shape"):
        validate_dataset(dataset)


@pytest.mark.parametrize("ids", [["a", "a"], ["a", ""]])
def test_rejects_invalid_particle_ids(ids) -> None:
    dataset = _trajectory_dataset()
    dataset.arrays["particle_ids"] = np.array(ids)
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(dataset)


def test_rejects_singleton_time_series_length_mismatch() -> None:
    dataset = _base_dataset(Path("."))
    dataset.arrays["positions"] = np.array([1.0])
    with pytest.raises(ValueError, match="share a length"):
        validate_dataset(dataset)


def test_dotted_output_stem_preserved(tmp_path: Path) -> None:
    from core.simulation.dataset import load_dataset, metadata_only

    dataset = _base_dataset(tmp_path)
    stem = tmp_path / "example.v1"
    meta_path, data_path = save_dataset(dataset, stem)
    assert meta_path.name == "example.v1.meta.json"
    assert data_path.name == "example.v1.data.npz"
    assert load_dataset(stem).metadata["simulation_id"] == "fmt_test"
    assert load_dataset(data_path).metadata["simulation_id"] == "fmt_test"
    assert metadata_only(stem)["simulation_id"] == "fmt_test"


@pytest.mark.parametrize("validity", [
    [False, True, True],
    [True, False, True],
])
def test_dataset_rejects_reactivation_without_prior_cutoff(validity) -> None:
    metadata = _base_metadata()
    metadata["nonfinite_policy"] = "masked"
    mask = np.array(validity, dtype=bool)[:, None]
    values = np.ones((3, 1, 2), dtype=float)
    values[~mask] = np.nan
    dataset = SimulationDataset(
        metadata=metadata,
        arrays={
            "timestamps": np.array([0.0, 1.0, 2.0]),
            "positions": values,
            "velocities": values.copy(),
            "validity_mask": mask,
            "particle_ids": np.array(["p0"]),
        },
    )
    with pytest.raises(ValueError, match="reactivate"):
        validate_dataset(dataset)
