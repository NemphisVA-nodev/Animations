"""Unit tests for cinematic layout and animated trail timing (no bpy)."""

import numpy as np
import pytest

from blender.cinematic_paths import (
    camera_plan,
    cinematic_camera_fit,
    decorative_disk_arc,
    estimate_visual_scale,
    select_still_frame,
    trail_reveal_windows,
    trail_width_profile,
    visible_infall_window_end,
)


def test_trail_reveals_only_past_frames() -> None:
    starts, ends, visible = trail_reveal_windows(
        np.ones(6, dtype=bool), tail_frames=2, active_frames=6
    )
    assert visible.tolist() == [False, True, True, True, True, True]
    assert ends == pytest.approx([0, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 1])
    assert starts == pytest.approx([0, 0, 0, 1 / 5, 2 / 5, 3 / 5])
    assert np.all(starts <= ends)


def test_trail_freezes_during_hold_frames() -> None:
    starts, ends, visible = trail_reveal_windows(
        np.ones(7, dtype=bool), tail_frames=2, active_frames=4
    )
    assert np.all(visible[1:])
    assert np.allclose(ends[3:], 1.0)
    assert np.allclose(starts[3:], 1 / 3)


def test_terminated_particle_never_reappears() -> None:
    starts, ends, visible = trail_reveal_windows(
        np.array([True, True, True, False, False]),
        tail_frames=2, active_frames=5,
    )
    assert visible.tolist() == [False, True, True, False, False]
    assert ends[:3] == pytest.approx([0, 0.5, 1])
    assert not np.any(visible[3:])


def test_single_valid_sample_produces_no_trail() -> None:
    starts, ends, visible = trail_reveal_windows(
        np.array([True, False, False]), tail_frames=3, active_frames=3
    )
    assert np.all(starts == 0)
    assert np.all(ends == 0)
    assert not np.any(visible)


@pytest.mark.parametrize("mask", [[False, True], [True, False, True]])
def test_reactivated_particle_rejected(mask) -> None:
    with pytest.raises(ValueError, match="reactivat|become valid"):
        trail_reveal_windows(np.array(mask), tail_frames=2, active_frames=len(mask))


@pytest.mark.parametrize("tail_frames", [0, -1, True, 2.5])
def test_invalid_trail_length_rejected(tail_frames) -> None:
    with pytest.raises(ValueError, match="tail_frames"):
        trail_reveal_windows(
            np.ones(3, dtype=bool), tail_frames=tail_frames, active_frames=3
        )


@pytest.mark.parametrize("active_frames", [0, 4, True, 1.5])
def test_invalid_active_frame_count_rejected(active_frames) -> None:
    with pytest.raises(ValueError, match="active_frames"):
        trail_reveal_windows(
            np.ones(3, dtype=bool), tail_frames=2, active_frames=active_frames
        )


def test_camera_plan_contains_data_and_origin() -> None:
    points = np.array([[[0.0, 0.0, 0.0], [10.0, -2.0, 0.0]]])
    mask = np.array([[True, True]])
    center_x, center_y, half_extent = camera_plan(points, mask)
    assert center_x == pytest.approx(5)
    assert center_y == pytest.approx(-1)
    assert half_extent == pytest.approx(5)


def test_camera_ignores_invalid_nan_points() -> None:
    points = np.array([[[4.0, 0.0, 0.0], [np.nan, np.nan, 0.0]]])
    mask = np.array([[True, False]])
    center_x, center_y, half_extent = camera_plan(points, mask)
    assert center_x == pytest.approx(2)
    assert center_y == pytest.approx(0)
    assert half_extent >= 2.5


def test_camera_rejects_no_valid_positions() -> None:
    points = np.full((2, 1, 3), np.nan)
    with pytest.raises(ValueError, match="no valid"):
        camera_plan(points, np.zeros((2, 1), dtype=bool))


def test_camera_plan_includes_origin_in_extent() -> None:
    """The origin (black hole) must always be inside the camera extent."""
    points = np.array([[[20.0, 5.0, 0.0], [22.0, 7.0, 0.0]]])
    mask = np.array([[True, True]])
    center_x, center_y, half_extent = camera_plan(points, mask)
    # Origin (0,0) must be within half_extent of the center.
    assert abs(center_x) <= half_extent + 1e-6
    assert abs(center_y) <= half_extent + 1e-6


def test_camera_plan_not_far_left() -> None:
    """Camera center should not be biased excessively to the left."""
    # Particles spread to the right of the origin; origin still included.
    points = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]])
    mask = np.array([[True, True, True]])
    center_x, center_y, half_extent = camera_plan(points, mask)
    assert center_x == pytest.approx(10.0)
    assert center_y == pytest.approx(0.0)
    assert half_extent == pytest.approx(10.0)


def test_default_startup_object_names() -> None:
    """The default startup objects to be removed are documented and non-empty."""
    from blender.cinematic_scene import DEFAULT_STARTUP_OBJECTS
    assert "Cube" in DEFAULT_STARTUP_OBJECTS
    assert "Camera" in DEFAULT_STARTUP_OBJECTS
    assert "Light" in DEFAULT_STARTUP_OBJECTS
    assert len(DEFAULT_STARTUP_OBJECTS) == 3


def test_auto_still_prefers_particles_outside_schematic_core() -> None:
    """Select the frame with visible particles, not a later invisible frame."""
    positions = np.zeros((5, 2, 3), dtype=float)
    positions[:, 0, 0] = [3, 2.5, 1.3, 0.4, 0.3]
    positions[:, 1, 0] = [0.2, 1.8, 0.4, 0.3, 0.2]
    valid = np.ones((5, 2), dtype=bool)
    frame, visible = select_still_frame(
        positions, valid, active_frames=5,
    )
    assert frame == 2
    assert visible == 2


def test_auto_still_returns_warning_count_when_scale_too_small() -> None:
    positions = np.zeros((4, 2, 3), dtype=float)
    positions[:, :, 0] = 0.2
    frame, visible = select_still_frame(
        positions, np.ones((4, 2), dtype=bool), active_frames=4,
    )
    assert 1 <= frame <= 4
    assert visible == 0


def test_auto_still_ignores_terminated_nan_particles() -> None:
    positions = np.array(
        [[[3.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[np.nan, np.nan, np.nan]]]
    )
    valid = np.array([[True], [True], [False]])
    frame, visible = select_still_frame(positions, valid, active_frames=3)
    # Both frames 1 and 2 are visible; the early-mid tie-break picks frame 2.
    assert frame == 2
    assert visible == 1


def test_auto_still_excludes_hold_frames() -> None:
    positions = np.zeros((6, 1, 3), dtype=float)
    positions[:, 0, 0] = 3.0
    valid = np.ones((6, 1), dtype=bool)
    frame, visible = select_still_frame(positions, valid, active_frames=3)
    assert frame <= 3
    assert visible == 1


def test_auto_still_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        select_still_frame(np.zeros((2, 3)), np.zeros((2, 1), dtype=bool), active_frames=2)


def test_auto_still_rejects_invalid_radius() -> None:
    with pytest.raises(ValueError, match="core_radius"):
        select_still_frame(
            np.zeros((2, 1, 3)),
            np.ones((2, 1), dtype=bool),
            active_frames=2,
            core_radius=-1,
        )


def test_auto_scale_fits_typical_initial_particle_radius() -> None:
    positions = np.array(
        [[[4e6, 0.0], [0.0, 4e6], [0.0, 1e9]]], dtype=float
    )
    mask = np.ones((1, 3), dtype=bool)
    original = positions.copy()
    scale = estimate_visual_scale(positions, mask)
    assert scale == pytest.approx(1e-6)
    assert np.array_equal(positions, original), "visual scale changed source data"


def test_auto_scale_ignores_invalid_nan_particle() -> None:
    positions = np.array([[[4e6, 0.0], [np.nan, np.nan]]])
    mask = np.array([[True, False]])
    assert estimate_visual_scale(positions, mask) == pytest.approx(1e-6)


def test_auto_scale_does_not_use_later_infall_radius() -> None:
    positions = np.array(
        [[[4e6, 0.0]], [[2e6, 0.0]], [[1e6, 0.0]]], dtype=float
    )
    mask = np.ones((3, 1), dtype=bool)
    assert estimate_visual_scale(positions, mask) == pytest.approx(1e-6)


@pytest.mark.parametrize(
    ("positions", "mask", "message"),
    [
        (np.zeros((2, 2)), np.ones((2, 1), dtype=bool), "shape"),
        (np.zeros((2, 1, 2)), np.ones((1, 1), dtype=bool), "shape"),
        (np.zeros((0, 1, 2)), np.zeros((0, 1), dtype=bool), "frames"),
        (np.zeros((1, 1, 2)), np.zeros((1, 1), dtype=bool), "no valid"),
        (np.zeros((1, 1, 2)), np.ones((1, 1), dtype=bool), "nonzero"),
        (
            np.array([[[np.nan, 0.0]]]),
            np.ones((1, 1), dtype=bool),
            "finite",
        ),
    ],
)
def test_auto_scale_rejects_unusable_initial_conditions(
    positions, mask, message
) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_visual_scale(positions, mask)


@pytest.mark.parametrize("target_radius", [0.0, -3.0, np.nan, np.inf])
def test_auto_scale_rejects_invalid_target_radius(target_radius) -> None:
    with pytest.raises(ValueError, match="target_start_radius"):
        estimate_visual_scale(
            np.array([[[4e6, 0.0]]]),
            np.ones((1, 1), dtype=bool),
            target_start_radius=target_radius,
        )


def test_cinematic_camera_fit_includes_ring_and_distant_particles() -> None:
    positions = np.array([[[4.0, 0.0, 0.0], [7.0, 0.0, 0.0]]])
    valid = np.ones((1, 2), dtype=bool)
    cx, cy, width = cinematic_camera_fit(positions, valid)
    half_width = width / 2
    half_height = width / (2 * (16 / 9))
    assert cx - half_width < -2.12
    assert cx + half_width > 7.0 + 0.115
    assert cy - half_height < -2.12
    assert cy + half_height > 2.12


def test_cinematic_camera_fit_is_symmetric_for_central_dataset() -> None:
    pos = np.array([[[0.0, 0.0, 0.0]]])
    centre_x, centre_y, width = cinematic_camera_fit(
        pos, np.array([[True]])
    )
    assert centre_x == pytest.approx(0.0)
    assert centre_y == pytest.approx(0.0)
    assert width > 2 * 2.12 * 16 / 9


def test_cinematic_camera_fit_ignores_terminated_nan() -> None:
    pos = np.array([[[4., 0., 0.]], [[np.nan, np.nan, np.nan]]])
    valid = np.array([[True], [False]])
    cx, cy, width = cinematic_camera_fit(pos, valid)
    assert np.isfinite([cx, cy, width]).all()


@pytest.mark.parametrize("padding", [0, 1, np.nan, np.inf])
def test_cinematic_camera_fit_rejects_bad_padding(padding) -> None:
    pos = np.zeros((1, 1, 3), dtype=float)
    with pytest.raises(ValueError, match="padding"):
        cinematic_camera_fit(pos, np.ones((1, 1), bool), padding=padding)


def test_cinematic_camera_fit_rejects_no_valid_points() -> None:
    pos = np.zeros((1, 1, 3), dtype=float)
    with pytest.raises(ValueError, match="no valid"):
        cinematic_camera_fit(pos, np.zeros((1, 1), bool))


def test_visible_infall_window_uses_last_visible_source_sample() -> None:
    times = np.arange(7, dtype=float)
    positions = np.array(
        [[[6., 0.]], [[5., 0.]], [[4., 0.]], [[2., 0.]],
         [[0.4, 0.]], [[np.nan, np.nan]], [[np.nan, np.nan]]]
    )
    mask = np.array([[True], [True], [True], [True], [True], [False], [False]])
    before = positions.copy()
    end = visible_infall_window_end(times, positions, mask, scale=1)
    assert end == pytest.approx(4 / 0.95)
    assert np.array_equal(positions, before, equal_nan=True)


def test_visible_infall_window_handles_terminated_particles() -> None:
    times = np.arange(5, dtype=float)
    positions = np.array(
        [[[3., 0.], [6., 0.]], [[2., 0.], [4., 0.]],
         [[1., 0.], [2., 0.]], [[np.nan, np.nan], [0.3, 0.]],
         [[np.nan, np.nan], [np.nan, np.nan]]]
    )
    mask = np.array(
        [[True, True], [True, True], [True, True],
         [False, True], [False, False]]
    )
    # Last centre outside the schematic 0.735-unit outline occurs at t=2.
    assert visible_infall_window_end(
        times, positions, mask, scale=1
    ) == pytest.approx(3 / 0.95)


def test_visible_infall_window_does_not_crop_continuing_motion() -> None:
    times = np.arange(4, dtype=float)
    pos = np.array([[[5., 0.]], [[4., 0.]], [[3., 0.]], [[2., 0.]]])
    assert visible_infall_window_end(
        times, pos, np.ones((4, 1), dtype=bool), scale=1
    ) is None


def test_visible_infall_window_rejects_invisible_source() -> None:
    times = np.arange(4, dtype=float)
    pos = np.zeros((4, 1, 2))
    with pytest.raises(ValueError, match="fewer than two"):
        visible_infall_window_end(
            times, pos, np.ones((4, 1), dtype=bool), scale=1
        )


@pytest.mark.parametrize("bad_scale", [0, -2, np.nan, np.inf])
def test_visible_infall_window_rejects_bad_scale(bad_scale) -> None:
    times = np.arange(3, dtype=float)
    pos = np.ones((3, 1, 2))
    with pytest.raises(ValueError, match="scale"):
        visible_infall_window_end(
            times, pos, np.ones((3, 1), dtype=bool), scale=bad_scale
        )


def test_visible_infall_window_rejects_reversed_time() -> None:
    times = np.array([0., 2., 1.])
    pos = np.ones((3, 1, 2))
    with pytest.raises(ValueError, match="increasing"):
        visible_infall_window_end(
            times, pos, np.ones((3, 1), dtype=bool), scale=1
        )


def test_visible_infall_window_rejects_invalid_valid_positions() -> None:
    times = np.arange(3, dtype=float)
    pos = np.array([[[3., 0.]], [[np.nan, 0.]], [[1., 0.]]])
    with pytest.raises(ValueError, match="finite"):
        visible_infall_window_end(
            times, pos, np.ones((3, 1), dtype=bool), scale=1
        )


def test_visible_infall_pacing_keeps_120_frames_without_fabricating_positions() -> None:
    from core.playback import PlaybackSpec, build_frame_mapping

    times = np.arange(7, dtype=float)
    pos = np.array(
        [[[6., 0.]], [[5., 0.]], [[4., 0.]], [[2., 0.]],
         [[0.4, 0.]], [[np.nan, np.nan]], [[np.nan, np.nan]]]
    )
    mask = np.array([[True], [True], [True], [True], [True], [False], [False]])
    end = visible_infall_window_end(times, pos, mask, scale=1)
    spec = PlaybackSpec(
        scene_id="pacing_contract", simulation_id=None, fps=24.0,
        duration_s=4.0, hold_s=1.0,
        start_sim_time_s=0.0, end_sim_time_s=end,
    )
    mapping = build_frame_mapping(spec)
    assert mapping.num_frames == 96
    assert mapping.hold_frames == 24
    assert mapping.total_frames == 120
    assert mapping.sim_times_s[0] == pytest.approx(0.0)
    assert np.all(np.diff(mapping.sim_times_s) > 0)
    assert mapping.sim_times_s[-1] < end
    assert mapping.all_sim_times_s[-1] == mapping.endpoint_sim_time_s
    assert np.array_equal(mapping.all_sim_times_s[-24:],
                          np.full(24, mapping.endpoint_sim_time_s))


def test_visible_infall_window_refuses_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        visible_infall_window_end(
            np.arange(4, dtype=float),
            np.zeros((4, 2)),
            np.ones((4, 1), dtype=bool),
            scale=1,
        )
    with pytest.raises(ValueError, match="boolean"):
        visible_infall_window_end(
            np.arange(4, dtype=float),
            np.zeros((4, 1, 2)),
            np.ones((4, 1), dtype=int),
            scale=1,
        )


def test_trail_width_profile_decorative_taper() -> None:
    widths = trail_width_profile(20)
    assert widths[0] == pytest.approx(0.42)
    assert widths[-1] == pytest.approx(1.0)
    assert np.all(np.diff(widths) >= 0)
    assert np.all(np.isfinite(widths))


@pytest.mark.parametrize("count", [0, 1, -4, 2.5, True])
def test_trail_width_profile_rejects_invalid_counts(count) -> None:
    with pytest.raises(ValueError, match="point_count"):
        trail_width_profile(count)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(0, 1), (-1, 2), (1.1, 1), (np.nan, 1)]
)
def test_trail_width_profile_rejects_invalid_widths(minimum, maximum) -> None:
    with pytest.raises(ValueError, match="width range"):
        trail_width_profile(10, minimum=minimum, maximum=maximum)


def test_decorative_arc_is_circular_and_fades_at_ends() -> None:
    xyz, widths = decorative_disk_arc(
        1.5, start_angle=0.2, span_angle=0.5, points=25
    )
    assert xyz.shape == (25, 3)
    assert widths.shape == (25,)
    assert np.allclose(np.linalg.norm(xyz[:, :2], axis=1), 1.5)
    assert np.allclose(xyz[:, 2], 0.065)
    assert widths[0] == pytest.approx(widths[-1])
    assert widths[12] > widths[0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"radius": -1.0, "start_angle": 0.0, "span_angle": 0.5}, "radius"),
        ({"radius": 1.0, "start_angle": np.nan, "span_angle": 0.5}, "start_angle"),
        ({"radius": 1.0, "start_angle": 0.0, "span_angle": 0.0}, "span_angle"),
        ({"radius": 1.0, "start_angle": 0.0, "span_angle": 2*np.pi}, "span_angle"),
        ({"radius": 1.0, "start_angle": 0.0, "span_angle": 0.5, "points": 2}, "points"),
    ],
)
def test_decorative_disk_arc_rejects_invalid_parameters(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        decorative_disk_arc(**kwargs)
