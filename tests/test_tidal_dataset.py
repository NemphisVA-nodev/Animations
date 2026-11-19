"""Tests for the Newtonian tidal acceleration physics and example dataset.

These tests are independent of any graphical rendering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.physics.tidal import (
    GRAVITATIONAL_CONSTANT,
    TidalParameters,
    particle_positions,
    radial_acceleration,
    tidal_difference_approximate,
    tidal_difference_exact,
    transverse_compression,
)
from core.simulation.dataset import (
    SimulationDataset,
    SimulationMetadata,
    load_dataset,
    metadata_only,
    save_dataset,
    validate_dataset,
)

EXAMPLE_META = Path("outputs/tidal_radial_example.meta.json")
EXAMPLE_DATA = Path("outputs/tidal_radial_example.data.npz")


def _build_example_dataset() -> SimulationDataset:
    """Rebuild the example dataset deterministically in memory."""
    mass_kg = 5.972e24
    length_m = 10.0
    r_min_m = 2.0e7
    r_max_m = 1.0e8
    num_points = 200

    distances = np.linspace(r_min_m, r_max_m, num_points).astype(np.float64)
    timestamps = np.arange(num_points, dtype=np.float64)

    near_accel = np.empty(num_points, dtype=np.float64)
    far_accel = np.empty(num_points, dtype=np.float64)
    exact = np.empty(num_points, dtype=np.float64)
    approx = np.empty(num_points, dtype=np.float64)
    transverse = np.empty(num_points, dtype=np.float64)

    for i, r in enumerate(distances):
        params = TidalParameters(mass_kg, length_m, float(r))
        half = length_m / 2.0
        near_accel[i] = radial_acceleration(mass_kg, r - half)
        far_accel[i] = radial_acceleration(mass_kg, r + half)
        exact[i] = tidal_difference_exact(params)
        approx[i] = tidal_difference_approximate(params)
        transverse[i] = transverse_compression(params)

    metadata: SimulationMetadata = {
        "schema_version": "1.0.0",
        "simulation_id": "tidal_radial_newtonian",
        "simulation_type": "tidal_acceleration",
        "model": "newtonian_point_mass_tidal",
        "model_limitations": "Newtonian point-mass approximation.",
        "units": {
            "time": "s",
            "position": "m",
            "acceleration": "m/s^2",
            "mass": "kg",
            "length": "m",
        },
        "reference_frame": {
            "origin": "centre of the attracting mass",
            "radial_axis": "positive outward from the mass",
        },
        "physical_parameters": {
            "attracting_mass_kg": mass_kg,
            "object_length_m": length_m,
            "r_min_m": r_min_m,
            "r_max_m": r_max_m,
            "num_points": num_points,
            "gravitational_constant": GRAVITATIONAL_CONSTANT,
        },
        "created_at": "2026-01-01T00:00:00+00:00",
        "array_sources": {},
        "notes": "test dataset",
    }
    arrays = {
        "timestamps": timestamps,
        "distances": distances,
        "near_acceleration": near_accel,
        "far_acceleration": far_accel,
        "tidal_difference_exact": exact,
        "tidal_difference_approximate": approx,
        "transverse_compression": transverse,
    }
    return SimulationDataset(metadata=metadata, arrays=arrays)


# --- Physics function tests -------------------------------------------------


def test_radial_acceleration_inverse_square() -> None:
    a1 = radial_acceleration(5.972e24, 1.0e7)
    a2 = radial_acceleration(5.972e24, 2.0e7)
    assert a2 == pytest.approx(a1 / 4.0, rel=1e-12)


def test_radial_acceleration_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        radial_acceleration(0.0, 1.0)
    with pytest.raises(ValueError):
        radial_acceleration(5.972e24, -1.0)


def test_exact_tidal_rejects_near_end_at_centre() -> None:
    params = TidalParameters(5.972e24, 10.0, 5.0)  # distance == length / 2
    with pytest.raises(ValueError):
        tidal_difference_exact(params)

    params = TidalParameters(5.972e24, 10.0, 4.0)  # distance < length / 2
    with pytest.raises(ValueError):
        tidal_difference_exact(params)


def test_exact_tidal_positive_for_valid_input() -> None:
    params = TidalParameters(5.972e24, 10.0, 2.0e7)
    assert tidal_difference_exact(params) > 0.0


def test_tidal_approximation_inverse_cube() -> None:
    """Halving distance should yield ~8x the approximate tidal acceleration."""
    base = TidalParameters(5.972e24, 10.0, 2.0e7)
    halved = TidalParameters(5.972e24, 10.0, 1.0e7)
    ratio = tidal_difference_approximate(halved) / tidal_difference_approximate(base)
    assert ratio == pytest.approx(8.0, rel=1e-12)


def test_tidal_approximation_mass_linearity() -> None:
    base = TidalParameters(5.972e24, 10.0, 2.0e7)
    doubled = TidalParameters(2 * 5.972e24, 10.0, 2.0e7)
    assert tidal_difference_approximate(doubled) == pytest.approx(
        2.0 * tidal_difference_approximate(base), rel=1e-12
    )


def test_tidal_approximation_length_linearity() -> None:
    base = TidalParameters(5.972e24, 10.0, 2.0e7)
    doubled = TidalParameters(5.972e24, 20.0, 2.0e7)
    assert tidal_difference_approximate(doubled) == pytest.approx(
        2.0 * tidal_difference_approximate(base), rel=1e-12
    )


def test_exact_and_approximate_agree_when_length_small() -> None:
    """For L << r the exact and approximate tidal values agree closely."""
    mass = 5.972e24
    length = 1.0  # 1 m object
    r = 1.0e7  # much larger than length
    params = TidalParameters(mass, length, r)
    exact = tidal_difference_exact(params)
    approx = tidal_difference_approximate(params)
    assert exact == pytest.approx(approx, rel=1e-6)


def test_transverse_compression_is_negative_half_stretch() -> None:
    params = TidalParameters(5.972e24, 10.0, 2.0e7)
    stretch = tidal_difference_approximate(params)
    compression = transverse_compression(params)
    assert compression < 0.0
    assert compression == pytest.approx(-0.5 * stretch, rel=1e-12)


def test_particle_positions_ordered() -> None:
    params = TidalParameters(5.972e24, 10.0, 2.0e7)
    centre, near, far = particle_positions(params)
    assert near[0] < centre[0] < far[0]
    assert centre[0] == pytest.approx(2.0e7)
    assert near[0] == pytest.approx(2.0e7 - 5.0)
    assert far[0] == pytest.approx(2.0e7 + 5.0)


# --- Dataset shape, units, determinism -------------------------------------


def test_dataset_shape_and_units() -> None:
    dataset = _build_example_dataset()
    validate_dataset(dataset)

    n = dataset.arrays["timestamps"].size
    for name, array in dataset.arrays.items():
        assert array.ndim == 1, f"{name} is not 1D"
        assert array.size == n, f"{name} length mismatch"
        assert np.all(np.isfinite(array)), f"{name} has non-finite values"

    units = dataset.metadata["units"]
    assert units["time"] == "s"
    assert units["position"] == "m"
    assert units["acceleration"] == "m/s^2"


def test_timestamps_strictly_increasing() -> None:
    dataset = _build_example_dataset()
    timestamps = dataset.arrays["timestamps"]
    assert np.all(np.diff(timestamps) > 0.0)


def test_dataset_deterministic() -> None:
    first = _build_example_dataset()
    second = _build_example_dataset()
    assert first.metadata["simulation_id"] == second.metadata["simulation_id"]
    for name in first.arrays:
        assert np.array_equal(first.arrays[name], second.arrays[name])


def test_npz_save_load_roundtrip(tmp_path: Path) -> None:
    dataset = _build_example_dataset()
    stem = tmp_path / "roundtrip"
    save_dataset(dataset, stem)
    loaded = load_dataset(stem)

    assert loaded.metadata["simulation_id"] == dataset.metadata["simulation_id"]
    assert loaded.metadata["model"] == dataset.metadata["model"]
    for name in dataset.arrays:
        assert np.array_equal(loaded.arrays[name], dataset.arrays[name])
    assert np.all(np.isfinite(loaded.arrays["timestamps"]))


def test_load_dataset_rejects_non_increasing_timestamps(tmp_path: Path) -> None:
    dataset = _build_example_dataset()
    bad = SimulationDataset(
        metadata=dict(dataset.metadata),
        arrays=dict(
            dataset.arrays,
            timestamps=np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64),
        ),
    )
    stem = tmp_path / "bad"
    save_dataset(bad, stem, validate=False)
    with pytest.raises(ValueError):
        load_dataset(stem)


def test_load_dataset_rejects_non_finite(tmp_path: Path) -> None:
    dataset = _build_example_dataset()
    bad = SimulationDataset(
        metadata=dict(dataset.metadata),
        arrays=dict(
            dataset.arrays,
            distances=np.array([1.0, np.nan, 3.0], dtype=np.float64),
        ),
    )
    stem = tmp_path / "bad2"
    save_dataset(bad, stem, validate=False)
    with pytest.raises(ValueError):
        load_dataset(stem)


def test_metadata_only_read() -> None:
    from core.simulation.dataset import metadata_only

    if not EXAMPLE_META.exists():
        pytest.skip("example dataset not generated")
    info = metadata_only(EXAMPLE_META)
    assert info["schema_version"] == "1.0.0"
    assert info["simulation_id"] == "tidal_radial_newtonian"
    assert info["model"] == "newtonian_point_mass_tidal"
    assert info["units"]["time"] == "s"
    assert info["units"]["position"] == "m"


def test_example_dataset_on_disk_matches_rebuild() -> None:
    if not EXAMPLE_META.exists() or not EXAMPLE_DATA.exists():
        pytest.skip("example dataset not generated")
    loaded = load_dataset(EXAMPLE_META)
    rebuilt = _build_example_dataset()
    for name in rebuilt.arrays:
        assert np.array_equal(loaded.arrays[name], rebuilt.arrays[name])
    assert loaded.metadata["simulation_id"] == rebuilt.metadata["simulation_id"]


def test_exact_tidal_rejects_nonfinite_or_nonpositive_parameters() -> None:
    for params in (
        TidalParameters(-1.0, 10.0, 20.0),
        TidalParameters(float("nan"), 10.0, 20.0),
        TidalParameters(1.0, -10.0, 20.0),
        TidalParameters(1.0, 10.0, float("inf")),
    ):
        with pytest.raises(ValueError):
            tidal_difference_exact(params)


def test_radial_acceleration_rejects_infinity() -> None:
    with pytest.raises(ValueError):
        radial_acceleration(float("inf"), 1.0)
    with pytest.raises(ValueError):
        radial_acceleration(1.0, float("nan"))
