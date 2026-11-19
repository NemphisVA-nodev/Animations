"""Tests for the Blender trajectory importer data logic.

These tests do not require Blender's ``bpy`` module and run under plain pytest.
They cover coordinate transformation, frame indexing, sampled positions,
validity handling, hold frames, scale validation, stable particle IDs,
velocity preservation, dataset reconstruction, and import repeatability.

The Blender-specific smoke test lives in
``blender/tests/test_blender_integration.py`` and runs inside Blender.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blender.importer import (
    ImportError,
    SCALE_DEFAULT,
    load_trajectory,
    metres_to_blender,
    python_frame_to_blender_frame,
)
from core.playback import (
    PlaybackSpec,
    build_frame_mapping,
    sample_particle_positions,
)
from core.simulation.dataset import (
    SimulationDataset,
    SimulationMetadata,
    load_dataset,
    save_dataset,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory_dataset(
    tmp_path: Path,
    n_times: int = 9,
    n_particles: int = 3,
    duration_s: float = 8.0,
    termination_time: float | None = None,
    stem_name: str = "traj",
) -> Path:
    """Build and save a small deterministic trajectory dataset.

    Particles move linearly: position = [t * (p+1), t * 0.5] with nonzero
    velocities. One particle
    may be terminated at ``termination_time``.
    """
    timestamps = np.linspace(0.0, duration_s, n_times).astype(np.float64)
    positions = np.zeros((n_times, n_particles, 2), dtype=np.float64)
    velocities = np.zeros((n_times, n_particles, 2), dtype=np.float64)
    validity = np.ones((n_times, n_particles), dtype=bool)

    for t in range(n_times):
        for p in range(n_particles):
            positions[t, p, 0] = timestamps[t] * (p + 1)
            positions[t, p, 1] = timestamps[t] * 0.5
            velocities[t, p] = (float(p + 1), 0.5)

    if termination_time is not None:
        term_idx = int(np.searchsorted(timestamps, termination_time, side="right"))
        if term_idx < n_times:
            validity[term_idx:, 0] = False
            positions[term_idx:, 0, :] = np.nan
            velocities[term_idx:, 0, :] = np.nan

    particle_ids = np.array([f"p{i:03d}" for i in range(n_particles)])

    metadata: SimulationMetadata = {
        "schema_version": "1.0.0",
        "simulation_id": "test_traj",
        "simulation_type": "black_hole_trajectory",
        "model": "newtonian_test_particle",
        "model_limitations": "Newtonian test particle.",
        "units": {
            "time": "s",
            "position": "m",
            "mass": "kg",
            "length": "m",
        },
        "reference_frame": {
            "origin": "black hole centre",
            "radial_axis": "positive outward",
        },
        "physical_parameters": {
            "black_hole_mass_kg": 1.9885e30,
            "n_particles": n_particles,
        },
        "created_at": "2026-01-01T00:00:00+00:00",
        "array_sources": {},
        "notes": "test trajectory",
        "nonfinite_policy": "masked",
        "black_hole_mass_kg": 1.9885e30,
        "schwarzschild_radius_m": 2953.383939028493,
        "integration_cutoff_radius_m": 1.0e6,
        "numerical_method": "DOP853",
        "solver_success": True,
        "solver_message": "ok",
        "termination_times_s": [termination_time] + [None] * (n_particles - 1),
        "initial_conditions": {
            "particle_ids": [f"p{i:03d}" for i in range(n_particles)],
            "initial_positions_m": positions[0].tolist(),
            "initial_velocities_m_per_s": np.zeros((n_particles, 2)).tolist(),
        },
        "termination_status": [
            "terminated_at_cutoff"
            if termination_time is not None
            else "survived_duration"
        ]
        + ["survived_duration"] * (n_particles - 1),
    }

    dataset = SimulationDataset(
        metadata=metadata,
        arrays={
            "timestamps": timestamps,
            "positions": positions,
            "velocities": velocities,
            "validity_mask": validity,
            "particle_ids": particle_ids,
        },
    )
    validate_dataset(dataset)
    stem = tmp_path / stem_name
    save_dataset(dataset, stem)
    return stem


# ---------------------------------------------------------------------------
# Coordinate transformation
# ---------------------------------------------------------------------------


def test_metres_to_blender_maps_xy_plane_with_z_zero() -> None:
    positions = np.array(
        [[[1.0e6, 2.0e6], [3.0e6, 4.0e6]]], dtype=np.float64
    )  # (1, 2, 2)
    result = metres_to_blender(positions, scale=1.0e-6)
    assert result.shape == (1, 2, 3)
    assert np.all(result[..., 2] == 0.0)  # Z = 0
    assert result[0, 0, 0] == pytest.approx(1.0)  # 1e6 m * 1e-6 = 1 BU
    assert result[0, 0, 1] == pytest.approx(2.0)
    assert result[0, 1, 0] == pytest.approx(3.0)
    assert result[0, 1, 1] == pytest.approx(4.0)


def test_metres_to_blender_preserves_nan() -> None:
    positions = np.full((1, 1, 2), np.nan, dtype=np.float64)
    result = metres_to_blender(positions, scale=1.0e-6)
    # X and Y carry the NaN; Z is intentionally a real zero (the plane).
    assert np.all(np.isnan(result[..., 0]))
    assert np.all(np.isnan(result[..., 1]))
    assert np.all(result[..., 2] == 0.0)


def test_metres_to_blender_rejects_bad_scale() -> None:
    positions = np.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="scale must be a positive finite"):
        metres_to_blender(positions, scale=0.0)
    with pytest.raises(ValueError, match="scale must be a positive finite"):
        metres_to_blender(positions, scale=-1.0)
    with pytest.raises(ValueError, match="scale must be a positive finite"):
        metres_to_blender(positions, scale=float("nan"))


def test_metres_to_blender_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        metres_to_blender(np.zeros((1, 2)), scale=1.0e-6)


# ---------------------------------------------------------------------------
# Frame indexing
# ---------------------------------------------------------------------------


def test_python_frame_zero_is_blender_frame_one() -> None:
    assert python_frame_to_blender_frame(0) == 1
    assert python_frame_to_blender_frame(1) == 2
    assert python_frame_to_blender_frame(149) == 150


def test_python_frame_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        python_frame_to_blender_frame(-1)


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------


def test_load_trajectory_round_trip(tmp_path: Path) -> None:
    stem = _make_trajectory_dataset(tmp_path, n_times=9, n_particles=3)
    from blender.importer import load_trajectory

    data = load_trajectory(stem, scale=SCALE_DEFAULT)

    assert data.timestamps.shape == (9,)
    assert data.particle_ids == ["p000", "p001", "p002"]
    assert data.positions.shape == (9, 3, 2)
    assert data.validity_mask.shape == (9, 3)
    assert data.scale == SCALE_DEFAULT
    assert data.metadata["units"]["position"] == "m"
    assert data.metadata["units"]["time"] == "s"

    # SI values preserved (not re-scaled inside TrajectoryData).
    assert data.positions[0, 0] == pytest.approx([0.0, 0.0])
    assert data.positions[-1, 2] == pytest.approx([8.0 * 3, 8.0 * 0.5])


def test_load_trajectory_rejects_missing_files(tmp_path: Path) -> None:
    from blender.importer import load_trajectory

    with pytest.raises(ImportError, match="metadata file not found"):
        load_trajectory(tmp_path / "does_not_exist")


def test_load_trajectory_rejects_bad_scale(tmp_path: Path) -> None:
    stem = _make_trajectory_dataset(tmp_path, n_times=3, n_particles=1)
    from blender.importer import load_trajectory

    with pytest.raises(ImportError, match="scale must be a positive finite"):
        load_trajectory(stem, scale=0.0)


# ---------------------------------------------------------------------------
# Sampled positions and validity
# ---------------------------------------------------------------------------


def test_sample_at_valid_times_returns_expected_positions(tmp_path: Path) -> None:
    stem = _make_trajectory_dataset(tmp_path, n_times=9, n_particles=3)
    dataset = load_dataset(stem)

    positions, validity = sample_particle_positions(
        dataset,
        particle_ids=["p000", "p001", "p002"],
        sim_times=np.array([0.0, 4.0, 8.0]),
    )

    assert positions.shape == (3, 3, 2)
    assert np.all(validity)
    # Particle 1 at t=4: [4*2, 4*0.5] = [8, 2]
    assert positions[1, 1] == pytest.approx([8.0, 2.0])


def test_sample_between_samples_interpolates(tmp_path: Path) -> None:
    stem = _make_trajectory_dataset(tmp_path, n_times=9, n_particles=1)
    dataset = load_dataset(stem)
    # At t=1.5, halfway between actual samples at t=1 and t=2.
    positions, validity = sample_particle_positions(
        dataset,
        particle_ids=["p000"],
        sim_times=np.array([1.5]),
    )
    assert validity[0, 0]
    assert positions[0, 0] == pytest.approx([1.5, 0.75])


def test_terminated_particle_invalid_after_cutoff(tmp_path: Path) -> None:
    """No invented motion after cutoff: terminated particle stays invalid."""
    stem = _make_trajectory_dataset(
        tmp_path, n_times=9, n_particles=1, duration_s=8.0, termination_time=4.0
    )
    dataset = load_dataset(stem)

    positions, validity = sample_particle_positions(
        dataset,
        particle_ids=["p000"],
        sim_times=np.array([2.0, 4.0, 6.0]),
    )
    assert validity[0, 0]   # before termination
    assert validity[1, 0]   # exactly at termination time
    assert not validity[2, 0]  # after termination
    assert np.all(np.isnan(positions[2, 0]))


# ---------------------------------------------------------------------------
# Frame mapping / hold frames
# ---------------------------------------------------------------------------


def test_hold_frames_frozen_at_endpoint(tmp_path: Path) -> None:
    stem = _make_trajectory_dataset(tmp_path, n_times=9, n_particles=1)
    dataset = load_dataset(stem)

    spec = PlaybackSpec(
        scene_id="scene",
        simulation_id="test_traj",
        fps=30.0,
        duration_s=2.0,
        start_sim_time_s=0.0,
        end_sim_time_s=8.0,
        hold_s=1.0,
    )
    mapping = build_frame_mapping(spec)

    # 2 s * 30 fps = 60 active frames; 1 s * 30 fps = 30 hold frames.
    assert mapping.num_frames == 60
    assert mapping.hold_frames == 30
    assert mapping.total_frames == 90

    all_sim_times = mapping.all_sim_times_s
    assert all_sim_times.shape == (90,)
    # Hold frames freeze at the endpoint.
    assert np.all(all_sim_times[60:] == all_sim_times[59])


def test_frame_0_maps_to_start_sim_time() -> None:
    spec = PlaybackSpec(
        scene_id="scene",
        simulation_id="sim",
        fps=24.0,
        duration_s=5.0,
        start_sim_time_s=3.0,
        end_sim_time_s=13.0,
    )
    mapping = build_frame_mapping(spec)
    assert mapping.sim_times_s[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Stable particle identifiers
# ---------------------------------------------------------------------------


def test_particle_ids_used_as_stable_names(tmp_path: Path) -> None:
    stem = _make_trajectory_dataset(tmp_path, n_times=5, n_particles=3)
    from blender.importer import load_trajectory

    data = load_trajectory(stem)
    assert data.particle_ids == ["p000", "p001", "p002"]
    # These IDs are the object names used by the animator; they are stable
    # across save/load rounds.
    reloaded = load_dataset(stem)
    ids = [str(p) for p in reloaded.arrays["particle_ids"].tolist()]
    assert ids == data.particle_ids


# ---------------------------------------------------------------------------
# Invalid source data
# ---------------------------------------------------------------------------


def test_load_trajectory_from_npz_path(tmp_path: Path) -> None:
    """load_dataset accepts either the .meta.json or the .data.npz path."""
    stem = _make_trajectory_dataset(tmp_path, n_times=4, n_particles=2)
    meta_path = stem.with_name(stem.name + ".meta.json")
    npz_path = stem.with_name(stem.name + ".data.npz")

    ds_from_meta = load_dataset(meta_path)
    ds_from_npz = load_dataset(npz_path)
    assert ds_from_meta.metadata["simulation_id"] == ds_from_npz.metadata["simulation_id"]
    assert np.array_equal(
        ds_from_meta.arrays["positions"], ds_from_npz.arrays["positions"]
    )


# ---------------------------------------------------------------------------
# Velocity preservation (regression: invalid samples must stay NaN)
# ---------------------------------------------------------------------------


def test_velocities_preserved_for_valid_samples(tmp_path: Path) -> None:
    """Velocities are preserved exactly for valid samples (no zeros substituted)."""
    stem = _make_trajectory_dataset(tmp_path, n_times=5, n_particles=2)
    from blender.importer import load_trajectory

    data = load_trajectory(stem)

    # Velocities should be available on TrajectoryData.
    assert data.velocities is not None
    assert data.velocities.shape == (5, 2, 2)

    # All samples are valid in this dataset; velocities should be finite.
    assert np.all(np.isfinite(data.velocities))

    # Nonzero values ensure a regression to zeros cannot pass this test.
    assert np.any(data.velocities != 0.0)

    # Velocities should match the original dataset exactly.
    dataset = load_dataset(stem)
    assert np.array_equal(data.velocities, dataset.arrays["velocities"])


def test_velocities_nan_for_terminated_particles(tmp_path: Path) -> None:
    """Terminated particle velocities must be NaN, never fabricated."""
    stem = _make_trajectory_dataset(
        tmp_path, n_times=9, n_particles=1, duration_s=8.0, termination_time=4.0
    )
    from blender.importer import load_trajectory

    data = load_trajectory(stem)

    # Samples before/at termination: finite velocities.
    assert np.all(np.isfinite(data.velocities[:5, 0]))

    # Samples after termination: NaN velocities.
    assert np.all(np.isnan(data.velocities[5:, 0]))

    # Positions after termination must also be NaN.
    assert np.all(np.isnan(data.positions[5:, 0]))


def test_trajectory_data_includes_velocities_field(tmp_path: Path) -> None:
    """TrajectoryData dataclass must expose a velocities field."""
    stem = _make_trajectory_dataset(tmp_path, n_times=4, n_particles=2)
    from blender.importer import load_trajectory

    data = load_trajectory(stem)

    # The field exists and is a NumPy array of the right shape.
    assert hasattr(data, "velocities")
    assert isinstance(data.velocities, np.ndarray)
    assert data.velocities.shape == (4, 2, 2)


# ---------------------------------------------------------------------------
# SimulationDataset reconstruction
# ---------------------------------------------------------------------------


def test_reconstruct_simulation_dataset_from_trajectory_data(tmp_path: Path) -> None:
    """_as_simulation_dataset must produce a dataset that passes validation."""
    stem = _make_trajectory_dataset(
        tmp_path, n_times=6, n_particles=2, termination_time=4.0
    )
    from blender.importer import load_trajectory
    from blender.animator import Animator
    from core.playback import PlaybackSpec

    data = load_trajectory(stem)

    spec = PlaybackSpec(
        scene_id="scene",
        simulation_id="test_traj",
        fps=24.0,
        duration_s=2.0,
        start_sim_time_s=0.0,
        end_sim_time_s=8.0,
    )

    # Build an Animator without calling require_bpy (it's a dataclass field).
    # We only need _as_simulation_dataset, which doesn't touch bpy.
    animator = Animator.__new__(Animator)
    animator.data = data
    animator.spec = spec
    animator.scale = SCALE_DEFAULT
    animator.collection_name = "Animations_Preview"
    animator.bh_marker_name = "BlackHole_Marker"

    reconstructed = animator._as_simulation_dataset()

    # Must pass the strict playback validator.
    validate_dataset(reconstructed)

    # Velocities must be preserved (not zeros).
    assert np.array_equal(
        reconstructed.arrays["velocities"], data.velocities, equal_nan=True
    )

    # Invalid samples must actually exist and keep NaN values in both arrays.
    invalid = ~reconstructed.arrays["validity_mask"]
    assert invalid.any()
    assert np.all(np.isnan(reconstructed.arrays["positions"][invalid]))
    assert np.all(np.isnan(reconstructed.arrays["velocities"][invalid]))


# ---------------------------------------------------------------------------
# Import repeatability
# ---------------------------------------------------------------------------


def test_import_repeatability_identical_results(tmp_path: Path) -> None:
    """Loading the same dataset twice must produce identical arrays."""
    stem = _make_trajectory_dataset(tmp_path, n_times=7, n_particles=3)
    from blender.importer import load_trajectory

    data1 = load_trajectory(stem, scale=SCALE_DEFAULT)
    data2 = load_trajectory(stem, scale=SCALE_DEFAULT)

    assert data1.particle_ids == data2.particle_ids
    assert np.array_equal(data1.positions, data2.positions)
    assert np.array_equal(data1.velocities, data2.velocities)
    assert np.array_equal(data1.validity_mask, data2.validity_mask)
    assert np.array_equal(data1.timestamps, data2.timestamps)
    assert data1.scale == data2.scale
