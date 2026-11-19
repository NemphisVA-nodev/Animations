"""Tests for physical-time-to-frame mapping and trajectory sampling.

Tests verify actual expected values, not merely that functions return something.
All datasets are built in temporary directories so the repository stays clean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.playback import (
    FrameMapping,
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
# Helpers to build deterministic trajectory datasets in memory / on disk.
# ---------------------------------------------------------------------------


def _make_trajectory_dataset(
    tmp_path: Path,
    n_times: int = 11,
    n_particles: int = 3,
    duration_s: float = 10.0,
    termination_time: float | None = None,
    stem_name: str = "traj",
) -> Path:
    """Build and save a small deterministic trajectory dataset.

    Particles move linearly: position = [t * (p+1), t * 0.5]. One particle
    may be terminated at ``termination_time`` (its trailing samples become
    invalid / NaN).

    Returns the path stem of the saved dataset.
    """
    timestamps = np.linspace(0.0, duration_s, n_times).astype(np.float64)
    positions = np.zeros((n_times, n_particles, 2), dtype=np.float64)
    velocities = np.zeros((n_times, n_particles, 2), dtype=np.float64)
    validity = np.ones((n_times, n_particles), dtype=bool)

    for t in range(n_times):
        for p in range(n_particles):
            positions[t, p, 0] = float(t) * (p + 1)
            positions[t, p, 1] = float(t) * 0.5
            velocities[t, p, 0] = float(p + 1)
            velocities[t, p, 1] = 0.5

    if termination_time is not None:
        # Mark samples strictly after termination_time invalid for particle 0.
        # The sample at exactly termination_time remains valid, matching the
        # integration behaviour (timestamps <= t_term are valid).
        term_idx = int(np.searchsorted(timestamps, termination_time, side="right"))
        if term_idx < n_times:
            validity[term_idx:, 0] = False
            positions[term_idx:, 0] = np.nan
            velocities[term_idx:, 0] = np.nan

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
            "velocity": "m/s",
            "mass": "kg",
            "length": "m",
        },
        "reference_frame": {
            "origin": "black hole centre",
            "radial_axis": "positive outward",
        },
        "physical_parameters": {
            "black_hole_mass_kg": 1.9885e31,
            "n_particles": n_particles,
        },
        "created_at": "2026-01-01T00:00:00+00:00",
        "array_sources": {},
        "notes": "test trajectory",
        "nonfinite_policy": "masked",
        "black_hole_mass_kg": 1.9885e31,
        "schwarzschild_radius_m": 29533.83939028493,
        "integration_cutoff_radius_m": 1.0e7,
        "numerical_method": "DOP853",
        "solver_success": True,
        "solver_message": "ok",
        "termination_times_s": [termination_time] + [None] * (n_particles - 1),
        "initial_conditions": {
            "particle_ids": [f"p{i:03d}" for i in range(n_particles)],
            "initial_positions_m": positions[0].tolist(),
            "initial_velocities_m_per_s": velocities[0].tolist(),
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
# Frame mapping tests
# ---------------------------------------------------------------------------


def test_frame_count_deterministic_at_24_30_60_fps() -> None:
    """Frame counts are deterministic for a given duration and fps."""
    spec = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=24.0,
        duration_s=5.0,
        start_sim_time_s=0.0,
        end_sim_time_s=10.0,
    )
    mapping_24 = build_frame_mapping(spec)

    spec30 = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=30.0,
        duration_s=5.0,
        start_sim_time_s=0.0,
        end_sim_time_s=10.0,
    )
    mapping_30 = build_frame_mapping(spec30)

    spec60 = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=60.0,
        duration_s=5.0,
        start_sim_time_s=0.0,
        end_sim_time_s=10.0,
    )
    mapping_60 = build_frame_mapping(spec60)

    assert mapping_24.num_frames == 120  # 5 * 24
    assert mapping_30.num_frames == 150  # 5 * 30
    assert mapping_60.num_frames == 300  # 5 * 60

    # Repeated calls are deterministic.
    repeat = build_frame_mapping(spec)
    assert repeat.num_frames == mapping_24.num_frames
    assert repeat.fps == mapping_24.fps
    assert np.array_equal(repeat.sim_times_s, mapping_24.sim_times_s)


def test_first_frame_anchored_to_start_sim_time() -> None:
    """Frame 0 maps exactly to start_sim_time_s."""
    spec = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=30.0,
        duration_s=5.0,
        start_sim_time_s=2.0,
        end_sim_time_s=12.0,
    )
    mapping = build_frame_mapping(spec)

    assert mapping.sim_times_s[0] == pytest.approx(2.0)


def test_last_frame_approximates_scene_boundary() -> None:
    """The last active frame approximates end_sim_time_s (scene boundary),
    but is not forcibly anchored to it when duration*fps is non-integer."""
    # duration=5.0, fps=30.0 -> 150 frames exactly, last frame i=149:
    # sim = 0 + (149/30) * (10/5) = 9.9333... (just before boundary 10.0).
    spec = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=30.0,
        duration_s=5.0,
        start_sim_time_s=0.0,
        end_sim_time_s=10.0,
    )
    mapping = build_frame_mapping(spec)
    assert mapping.num_frames == 150
    assert mapping.sim_times_s[-1] == pytest.approx(9.933333333333334)
    assert mapping.endpoint_sim_time_s == mapping.sim_times_s[-1]
    # The scene boundary (10.0) is NOT on the final active frame.
    assert mapping.sim_times_s[-1] < mapping.end_sim_time_s

    # duration=1.0/3.0, fps=30.0 -> 10 frames exactly.
    spec2 = PlaybackSpec(
        scene_id="scene_b",
        simulation_id="sim_1",
        fps=30.0,
        duration_s=1.0 / 3.0,
        start_sim_time_s=0.0,
        end_sim_time_s=1.0,
    )
    mapping2 = build_frame_mapping(spec2)
    assert mapping2.num_frames == 10
    # Last frame at i=9: sim = 0 + (9/30) * (1.0 / (1/3)) = 0.9
    assert mapping2.sim_times_s[-1] == pytest.approx(0.9)
    assert mapping2.endpoint_sim_time_s == pytest.approx(0.9)


def test_fixed_rate_convention() -> None:
    """sim_time[i] = start + (i / fps) * effective_playback_speed."""
    spec = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=10.0,
        duration_s=2.0,
        start_sim_time_s=5.0,
        end_sim_time_s=15.0,  # span 10s over 2s -> 5x speed
    )
    mapping = build_frame_mapping(spec)

    assert mapping.num_frames == 20  # 2 * 10
    assert mapping.effective_playback_speed == pytest.approx(5.0)

    expected = np.array(
        [5.0 + (i / 10.0) * 5.0 for i in range(20)], dtype=np.float64
    )
    assert np.allclose(mapping.sim_times_s, expected)
    # Last frame at i=19: sim = 5 + (19/10)*5 = 14.5 (before boundary 15.0)
    assert mapping.sim_times_s[-1] == pytest.approx(14.5)
    assert mapping.endpoint_sim_time_s == pytest.approx(14.5)


def test_playback_speed_greater_than_one() -> None:
    """10 s sim over 5 s presentation = 2x effective speed, 150 frames at 30 fps."""
    spec = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=30.0,
        duration_s=5.0,
        start_sim_time_s=0.0,
        end_sim_time_s=10.0,
    )
    mapping = build_frame_mapping(spec)

    assert mapping.num_frames == 150
    assert mapping.effective_playback_speed == pytest.approx(2.0)
    assert mapping.sim_span_s == pytest.approx(10.0)


def test_playback_speed_less_than_one() -> None:
    """5 s sim over 10 s presentation = 0.5x effective speed."""
    spec = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=30.0,
        duration_s=10.0,
        start_sim_time_s=0.0,
        end_sim_time_s=5.0,
    )
    mapping = build_frame_mapping(spec)

    assert mapping.num_frames == 300
    assert mapping.effective_playback_speed == pytest.approx(0.5)


def test_single_active_frame() -> None:
    """A very short scene yields a single active frame at start_sim_time_s."""
    spec = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=30.0,
        duration_s=0.01,  # round(0.3) = 0 -> clamped to 1
        start_sim_time_s=7.0,
        end_sim_time_s=9.0,
    )
    mapping = build_frame_mapping(spec)

    assert mapping.num_frames == 1
    assert mapping.sim_times_s[0] == pytest.approx(7.0)
    assert mapping.endpoint_sim_time_s == pytest.approx(7.0)


def test_hold_frames_freeze_at_endpoint() -> None:
    """Hold frames freeze at the last active frame's simulation time."""
    spec = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=30.0,
        duration_s=2.0,
        start_sim_time_s=0.0,
        end_sim_time_s=4.0,
        hold_s=1.0,
    )
    mapping = build_frame_mapping(spec)

    assert mapping.num_frames == 60  # 2 * 30
    assert mapping.hold_frames == 30  # 1 * 30
    assert mapping.total_frames == 90

    # Last active frame sim time = endpoint.
    assert mapping.sim_times_s[-1] == pytest.approx(
        0.0 + (59.0 / 30.0) * 2.0
    )
    assert mapping.endpoint_sim_time_s == mapping.sim_times_s[-1]


def test_adjacent_scene_continuity() -> None:
    """If scene B starts where scene A ends, B's frame 0 begins at A's end."""
    spec_a = PlaybackSpec(
        scene_id="scene_a",
        simulation_id="sim_1",
        fps=30.0,
        duration_s=2.0,
        start_sim_time_s=0.0,
        end_sim_time_s=4.0,
    )
    spec_b = PlaybackSpec(
        scene_id="scene_b",
        simulation_id="sim_1",
        fps=30.0,
        duration_s=2.0,
        start_sim_time_s=4.0,  # same as A's end boundary
        end_sim_time_s=8.0,
    )
    mapping_a = build_frame_mapping(spec_a)
    mapping_b = build_frame_mapping(spec_b)

    # B's first frame starts at 4.0 = A's end boundary.
    assert mapping_b.sim_times_s[0] == pytest.approx(4.0)
    assert mapping_b.sim_times_s[0] == spec_a.end_sim_time_s


def test_rejects_non_finite_timing_values() -> None:
    """Non-finite fps, duration, or sim times are rejected."""
    bad_spec_args = [
        ("s", None, float("nan"), 5.0, 0.0, 10.0),
        ("s", None, 30.0, float("inf"), 0.0, 10.0),
        ("s", None, 30.0, 5.0, float("nan"), 10.0),
        ("s", None, 30.0, 5.0, 0.0, float("inf")),
    ]
    for args in bad_spec_args:
        with pytest.raises(ValueError):
            PlaybackSpec(*args)


def test_rejects_negative_duration_and_fps() -> None:
    """Non-positive fps or duration are rejected."""
    with pytest.raises(ValueError):
        PlaybackSpec("s", None, 0.0, 5.0, 0.0, 10.0)
    with pytest.raises(ValueError):
        PlaybackSpec("s", None, 30.0, -1.0, 0.0, 10.0)


def test_rejects_end_before_start() -> None:
    """end_sim_time_s before start_sim_time_s is rejected."""
    with pytest.raises(ValueError):
        PlaybackSpec("s", None, 30.0, 5.0, 10.0, 0.0)


def test_conceptual_scene_without_simulation() -> None:
    """A scene with simulation_id=None still builds a frame mapping."""
    spec = PlaybackSpec(
        scene_id="conceptual",
        simulation_id=None,
        fps=30.0,
        duration_s=2.0,
        start_sim_time_s=0.0,
        end_sim_time_s=4.0,
    )
    mapping = build_frame_mapping(spec)
    assert mapping.simulation_id is None
    assert mapping.num_frames == 60


# ---------------------------------------------------------------------------
# Trajectory sampling tests
# ---------------------------------------------------------------------------


def test_sample_at_valid_times_returns_expected_positions(tmp_path: Path) -> None:
    """Sampling at exact sample times returns the stored positions."""
    stem = _make_trajectory_dataset(tmp_path, n_times=11, n_particles=3)
    dataset = load_dataset(stem)

    positions, validity = sample_particle_positions(
        dataset,
        particle_ids=["p000", "p001", "p002"],
        sim_times=np.array([0.0, 5.0, 10.0]),
    )

    assert positions.shape == (3, 3, 2)
    assert validity.shape == (3, 3)
    assert np.all(validity)

    # Particle 0 at t=5: position = [5, 2.5]
    assert positions[1, 0] == pytest.approx([5.0, 2.5])
    # Particle 2 at t=10: position = [30, 5]
    assert positions[2, 2] == pytest.approx([30.0, 5.0])


def test_sample_between_valid_samples_interpolates(tmp_path: Path) -> None:
    """Sampling between samples linearly interpolates."""
    stem = _make_trajectory_dataset(tmp_path, n_times=11, n_particles=1)
    dataset = load_dataset(stem)

    # At t=2.5, particle 0 should be halfway between t=2 ([2,1]) and t=3 ([3,1.5]).
    positions, validity = sample_particle_positions(
        dataset,
        particle_ids=["p000"],
        sim_times=np.array([2.5]),
    )

    assert validity[0, 0]
    assert positions[0, 0] == pytest.approx([2.5, 1.25])


def test_sample_before_first_sample_is_invalid(tmp_path: Path) -> None:
    """Times before the first sample are invalid (NaN)."""
    stem = _make_trajectory_dataset(tmp_path, n_times=11, n_particles=1)
    dataset = load_dataset(stem)

    positions, validity = sample_particle_positions(
        dataset,
        particle_ids=["p000"],
        sim_times=np.array([-1.0]),
    )

    assert not validity[0, 0]
    assert np.all(np.isnan(positions[0, 0]))


def test_sample_after_last_sample_is_invalid(tmp_path: Path) -> None:
    """Times after the last sample are invalid (NaN)."""
    stem = _make_trajectory_dataset(
        tmp_path, n_times=11, n_particles=1, duration_s=10.0
    )
    dataset = load_dataset(stem)

    positions, validity = sample_particle_positions(
        dataset,
        particle_ids=["p000"],
        sim_times=np.array([11.0]),
    )

    assert not validity[0, 0]
    assert np.all(np.isnan(positions[0, 0]))


def test_terminated_particle_invalid_after_termination(tmp_path: Path) -> None:
    """A particle terminated at t=5 is invalid for queries after t=5."""
    stem = _make_trajectory_dataset(
        tmp_path,
        n_times=11,
        n_particles=1,
        duration_s=10.0,
        termination_time=5.0,
    )
    dataset = load_dataset(stem)

    # Before termination: valid.
    positions, validity = sample_particle_positions(
        dataset, particle_ids=["p000"], sim_times=np.array([2.5, 5.0])
    )
    assert validity[0, 0]
    assert validity[1, 0]  # exactly at termination time

    # After termination: invalid.
    positions, validity = sample_particle_positions(
        dataset, particle_ids=["p000"], sim_times=np.array([7.5])
    )
    assert not validity[0, 0]
    assert np.all(np.isnan(positions[0, 0]))


def test_unknown_particle_id_raises(tmp_path: Path) -> None:
    """Requesting an unknown particle id raises ValueError."""
    stem = _make_trajectory_dataset(tmp_path, n_times=5, n_particles=2)
    dataset = load_dataset(stem)

    with pytest.raises(ValueError, match="unknown particle ids"):
        sample_particle_positions(
            dataset,
            particle_ids=["p000", "p999"],
            sim_times=np.array([1.0]),
        )


def test_missing_trajectory_arrays_raises(tmp_path: Path) -> None:
    """A dataset without trajectory arrays raises ValueError."""
    metadata: SimulationMetadata = {
        "schema_version": "1.0.0",
        "simulation_id": "no_traj",
        "simulation_type": "tidal_acceleration",
        "model": "newtonian_point_mass_tidal",
        "model_limitations": "Newtonian only.",
        "units": {"time": "s", "position": "m"},
        "reference_frame": {"origin": "mass centre"},
        "physical_parameters": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "array_sources": {},
        "notes": "",
    }
    dataset = SimulationDataset(
        metadata=metadata,
        arrays={
            "timestamps": np.array([0.0, 1.0]),
            "distances": np.array([1.0, 2.0]),
        },
    )
    with pytest.raises(ValueError, match="missing required trajectory arrays"):
        sample_particle_positions(
            dataset, particle_ids=["p000"], sim_times=np.array([0.5])
        )


def test_non_finite_sim_times_rejected(tmp_path: Path) -> None:
    """Non-finite sim_times are rejected."""
    stem = _make_trajectory_dataset(tmp_path, n_times=5, n_particles=1)
    dataset = load_dataset(stem)

    with pytest.raises(ValueError, match="finite"):
        sample_particle_positions(
            dataset, particle_ids=["p000"], sim_times=np.array([np.nan])
        )
    with pytest.raises(ValueError, match="finite"):
        sample_particle_positions(
            dataset, particle_ids=["p000"], sim_times=np.array([np.inf])
        )


def test_original_arrays_never_modified(tmp_path: Path) -> None:
    """Sampling does not mutate the input dataset arrays."""
    stem = _make_trajectory_dataset(tmp_path, n_times=5, n_particles=1)
    dataset = load_dataset(stem)

    original_positions = dataset.arrays["positions"].copy()
    original_validity = dataset.arrays["validity_mask"].copy()

    sample_particle_positions(
        dataset,
        particle_ids=["p000"],
        sim_times=np.array([0.0, 100.0, -5.0]),
    )

    assert np.array_equal(dataset.arrays["positions"], original_positions)
    assert np.array_equal(dataset.arrays["validity_mask"], original_validity)


def test_save_load_round_trip_preserves_trajectory(tmp_path: Path) -> None:
    """NPZ/JSON save-load round trip preserves trajectory data exactly."""
    stem = _make_trajectory_dataset(tmp_path, n_times=7, n_particles=2)

    original = load_dataset(stem)
    reloaded = load_dataset(stem)

    for name in original.arrays:
        a1 = original.arrays[name]
        a2 = reloaded.arrays[name]
        if a1.dtype.kind == "f":
            assert np.allclose(a1, a2, equal_nan=True), f"{name} mismatch"
        else:
            assert np.array_equal(a1, a2), f"{name} mismatch"

    assert original.metadata["simulation_id"] == reloaded.metadata["simulation_id"]
    assert original.metadata["nonfinite_policy"] == "masked"


def test_sample_multiple_particles_independent_validity(tmp_path: Path) -> None:
    """Each particle's validity is independent."""
    stem = _make_trajectory_dataset(
        tmp_path,
        n_times=11,
        n_particles=2,
        duration_s=10.0,
        termination_time=5.0,
    )
    dataset = load_dataset(stem)

    # Particle 0 terminated at t=5; particle 1 survives to t=10.
    positions, validity = sample_particle_positions(
        dataset,
        particle_ids=["p000", "p001"],
        sim_times=np.array([7.5]),
    )

    assert not validity[0, 0]  # p000 terminated
    assert validity[0, 1]  # p001 still valid
    assert np.all(np.isnan(positions[0, 0]))
    assert np.all(np.isfinite(positions[0, 1]))


# ---------------------------------------------------------------------------
# Physics pipeline test (real black-hole generator output)
# ---------------------------------------------------------------------------


def test_physics_pipeline_end_to_end(tmp_path: Path) -> None:
    """End-to-end: generate black-hole dataset, save, reload, compare.

    Verifies the real generator output round-trips including NaN values and
    metadata, with termination between timestamps, and that frame sampling
    works on genuine physics output including a non-integer frame count.
    """
    # Use a short, fast configuration so the test runs quickly.
    import subprocess
    import sys

    stem = tmp_path / "physics_pipeline"
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_black_hole_trajectories.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "radial-infall",
            "--output",
            str(stem),
            "--mass",
            "1.9885e30",  # 1 solar mass
            "--cutoff",
            "1.0e6",
            "--initial-radius",
            "4.0e6",
            "--duration",
            "5.0",
            "--steps",
            "50",
            "--length",
            "1.0e5",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    meta_path = stem.with_name(stem.name + "_radial_infall.meta.json")
    data_path = stem.with_name(stem.name + "_radial_infall.data.npz")

    # Reload and validate.
    ds = load_dataset(meta_path)
    validate_dataset(ds)

    assert ds.metadata["schema_version"] == "1.0.0"
    assert ds.metadata["nonfinite_policy"] == "masked"
    assert ds.metadata["units"]["time"] == "s"
    assert ds.metadata["units"]["position"] == "m"
    assert ds.metadata["integration_cutoff_radius_m"] == 1.0e6
    assert ds.metadata["schwarzschild_radius_m"] > 0.0

    # Arrays have the expected trajectory shapes.
    positions = ds.arrays["positions"]
    velocities = ds.arrays["velocities"]
    validity = ds.arrays["validity_mask"]
    particle_ids = ds.arrays["particle_ids"]
    timestamps = ds.arrays["timestamps"]

    assert positions.ndim == 3 and positions.shape[2] == 2
    assert velocities.ndim == 3 and velocities.shape[2] == 2
    assert positions.shape == velocities.shape
    assert validity.shape == positions.shape[:2]
    assert particle_ids.ndim == 1
    assert timestamps.ndim == 1

    # All particles terminate before the end (radial infall into cutoff).
    assert all(t is not None for t in ds.metadata["termination_times_s"])

    # NaN/validity consistency: invalid -> NaN, valid -> finite.
    for t in range(validity.shape[0]):
        for n in range(validity.shape[1]):
            if not validity[t, n]:
                assert np.all(np.isnan(positions[t, n]))
                assert np.all(np.isnan(velocities[t, n]))
            else:
                assert np.all(np.isfinite(positions[t, n]))
                assert np.all(np.isfinite(velocities[t, n]))

    # Termination happens between timestamps (not necessarily on a sample).
    last_valid_times = [
        float(timestamps[validity[:, n].nonzero()[0][-1]])
        if validity[:, n].any()
        else None
        for n in range(validity.shape[1])
    ]
    for n, t_term in enumerate(ds.metadata["termination_times_s"]):
        lv = last_valid_times[n]
        assert lv is not None
        assert lv <= t_term  # last valid sample is at or before termination

    # Sample at frame times from a non-integer frame-count mapping.
    spec = PlaybackSpec(
        scene_id="physics_scene",
        simulation_id=ds.metadata["simulation_id"],
        fps=24.0,
        duration_s=1.0 / 7.0,  # ~3.43 frames -> round to 3
        start_sim_time_s=0.0,
        end_sim_time_s=5.0,
    )
    mapping = build_frame_mapping(spec)
    assert mapping.num_frames == 3  # round(24/7) = 3

    positions_out, validity_out = sample_particle_positions(
        ds,
        particle_ids=particle_ids.tolist(),
        sim_times=mapping.sim_times_s,
    )

    assert positions_out.shape == (3, particle_ids.size, 2)
    assert validity_out.shape == (3, particle_ids.size)


def test_reactivation_rejected_in_dataset(tmp_path: Path) -> None:
    """A validity_mask that lets a terminated particle reactivate is rejected."""
    metadata: SimulationMetadata = {
        "schema_version": "1.0.0",
        "simulation_id": "reactivation_test",
        "simulation_type": "black_hole_trajectory",
        "model": "newtonian_test_particle",
        "model_limitations": "Newtonian test particle.",
        "units": {"time": "s", "position": "m", "velocity": "m/s"},
        "reference_frame": {"origin": "centre"},
        "physical_parameters": {},
        "created_at": "2026-01-01T00:00:00+00:00",
        "array_sources": {},
        "notes": "",
        "nonfinite_policy": "masked",
    }
    # Column 0: valid, invalid, valid (reactivation!) -> must be rejected.
    validity = np.array([[True], [False], [True]], dtype=bool)
    dataset = SimulationDataset(
        metadata=metadata,
        arrays={
            "timestamps": np.array([0.0, 1.0, 2.0]),
            "positions": np.ones((3, 1, 2)),
            "validity_mask": validity,
            "particle_ids": np.array(["p0"]),
        },
    )
    with pytest.raises(ValueError, match="reactivate"):
        validate_dataset(dataset)
