"""Tests for the presentation timeline data model.

Independent of any graphical rendering.
"""

from __future__ import annotations

import pytest

from core.timeline import PresentationTimeline, SceneTimeline


def _make_scene(scene_id: str, sim_id: str | None = "sim_a") -> SceneTimeline:
    return SceneTimeline(
        scene_id=scene_id,
        simulation_id=sim_id,
        duration_s=60.0,
        start_sim_time_s=0.0,
        end_sim_time_s=60.0,
        playback_speed=1.0,
        entry_anchor="anchor_in",
        exit_anchor="anchor_out",
    )


def test_scene_duration_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SceneTimeline(
            scene_id="s1",
            simulation_id=None,
            duration_s=0.0,
            start_sim_time_s=0.0,
            end_sim_time_s=10.0,
            playback_speed=1.0,
            entry_anchor="in",
            exit_anchor="out",
        )


def test_sim_end_must_not_be_before_start() -> None:
    with pytest.raises(ValueError):
        SceneTimeline(
            scene_id="s1",
            simulation_id=None,
            duration_s=60.0,
            start_sim_time_s=10.0,
            end_sim_time_s=0.0,
            playback_speed=1.0,
            entry_anchor="in",
            exit_anchor="out",
        )


def test_playback_speed_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SceneTimeline(
            scene_id="s1",
            simulation_id=None,
            duration_s=60.0,
            start_sim_time_s=0.0,
            end_sim_time_s=10.0,
            playback_speed=0.0,
            entry_anchor="in",
            exit_anchor="out",
        )


def test_sim_span_and_total_presentation() -> None:
    scene = _make_scene("s1")
    assert scene.sim_span_s == 60.0
    assert scene.total_presentation_s == 60.0

    scene_with_hold = SceneTimeline(
        scene_id="s2",
        simulation_id=None,
        duration_s=30.0,
        start_sim_time_s=0.0,
        end_sim_time_s=30.0,
        playback_speed=2.0,
        entry_anchor="in",
        exit_anchor="out",
        hold_s=10.0,
    )
    assert scene_with_hold.sim_span_s == 30.0
    assert scene_with_hold.total_presentation_s == 40.0


def test_presentation_total_duration() -> None:
    tl = PresentationTimeline(
        presentation_id="pres_1", title="Test", scenes=[_make_scene("s1"), _make_scene("s2")]
    )
    assert tl.total_duration_s() == 120.0
    assert tl.scene_ids() == ["s1", "s2"]
    assert tl.simulation_ids() == ["sim_a", "sim_a"]


def test_duplicate_scene_id_rejected() -> None:
    tl = PresentationTimeline(presentation_id="pres_1", title="Test")
    tl.add_scene(_make_scene("s1"))
    with pytest.raises(ValueError):
        tl.add_scene(_make_scene("s1"))


def test_scene_reuse_across_presentations() -> None:
    """The same scene/simulation can be referenced by multiple timelines."""
    scene = _make_scene("shared")
    tl_a = PresentationTimeline(
        presentation_id="pres_a", title="A", scenes=[scene]
    )
    tl_b = PresentationTimeline(
        presentation_id="pres_b", title="B", scenes=[scene]
    )
    assert tl_a.scene_ids() == ["shared"]
    assert tl_b.scene_ids() == ["shared"]


@pytest.mark.parametrize("field,value", [
    ("duration_s", float("nan")),
    ("start_sim_time_s", float("inf")),
    ("end_sim_time_s", float("nan")),
    ("playback_speed", float("inf")),
    ("hold_s", float("nan")),
])
def test_scene_rejects_nonfinite_timing(field, value) -> None:
    values = {
        "scene_id": "s1", "simulation_id": "a", "duration_s": 2.0,
        "start_sim_time_s": 0.0, "end_sim_time_s": 2.0,
        "playback_speed": 1.0, "entry_anchor": "in", "exit_anchor": "out",
    }
    values[field] = value
    with pytest.raises(ValueError, match="finite"):
        SceneTimeline(**values)
