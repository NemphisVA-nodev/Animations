"""Tests for the transition specification data model.

Independent of any graphical rendering or Blender behaviour.
"""

from __future__ import annotations

import pytest

from core.transitions import (
    ALLOWED_TRANSITION_TYPES,
    Transition,
    TransitionPlan,
)


def test_transition_records_start_end_scene_and_anchors() -> None:
    t = Transition(
        transition_id="tr_1",
        start_scene="scene_a",
        end_scene="scene_b",
        transition_type="crossfade",
        duration_s=2.0,
        start_anchor="anchor_out_a",
        end_anchor="anchor_in_b",
        overlap_s=1.0,
    )
    assert t.start_scene == "scene_a"
    assert t.end_scene == "scene_b"
    assert t.start_anchor == "anchor_out_a"
    assert t.end_anchor == "anchor_in_b"
    assert t.overlap_s == 1.0


def test_allowed_transition_types() -> None:
    for transition_type in ALLOWED_TRANSITION_TYPES:
        Transition(
            transition_id="t",
            start_scene="s1",
            end_scene="s2",
            transition_type=transition_type,
            duration_s=1.0,
            start_anchor="a",
            end_anchor="b",
        )


def test_unknown_transition_type_rejected() -> None:
    with pytest.raises(ValueError):
        Transition(
            transition_id="t",
            start_scene="s1",
            end_scene="s2",
            transition_type="warp_drive",
            duration_s=1.0,
            start_anchor="a",
            end_anchor="b",
        )


def test_negative_duration_and_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        Transition(
            transition_id="t",
            start_scene="s1",
            end_scene="s2",
            transition_type="cut",
            duration_s=-1.0,
            start_anchor="a",
            end_anchor="b",
        )
    with pytest.raises(ValueError):
        Transition(
            transition_id="t",
            start_scene="s1",
            end_scene="s2",
            transition_type="cut",
            duration_s=1.0,
            start_anchor="a",
            end_anchor="b",
            overlap_s=-0.5,
        )


def test_transition_plan_validates_scene_references() -> None:
    plan = TransitionPlan()
    plan.add(
        Transition(
            transition_id="tr_1",
            start_scene="scene_a",
            end_scene="scene_b",
            transition_type="crossfade",
            duration_s=2.0,
            start_anchor="a_out",
            end_anchor="b_in",
        )
    )
    plan.validate_against({"scene_a", "scene_b"})

    with pytest.raises(ValueError):
        plan.validate_against({"scene_a"})  # scene_b missing


def test_transition_plan_transition_ids() -> None:
    plan = TransitionPlan()
    plan.add(
        Transition(
            transition_id="tr_1",
            start_scene="s1",
            end_scene="s2",
            transition_type="cut",
            duration_s=0.0,
            start_anchor="a",
            end_anchor="b",
        )
    )
    assert plan.transition_ids() == ["tr_1"]
