"""Unit tests for cinematic layout and animated trail timing (no bpy)."""

import numpy as np
import pytest

from blender.cinematic_paths import camera_plan, trail_reveal_windows


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
