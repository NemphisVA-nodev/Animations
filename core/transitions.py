"""Transition specification for scene-to-scene visual alignment.

A transition connects a start scene to an end scene on a shared
presentation timeline. It is pure configuration data; it defines no
rendering or Blender behaviour and does not depend on the physics engine.

Supported transition types are descriptive only. Actual transition
rendering (camera movement, crossfades, zooms) is left to the rendering /
exporting layer and to any future Blender integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALLOWED_TRANSITION_TYPES = {
    "cut",
    "crossfade",
    "zoom",
    "match_cut",
    "object_track",
    "continuous_camera",
}


@dataclass(frozen=True)
class Transition:
    """A visual transition between two scenes.

    Attributes
    ----------
    transition_id:
        Unique identifier for the transition.
    start_scene:
        Scene identifier where the transition begins.
    end_scene:
        Scene identifier where the transition ends.
    transition_type:
        One of ``cut``, ``crossfade``, ``zoom``, ``match_cut``,
        ``object_track``, ``continuous_camera``.
    duration_s:
        Transition duration in presentation seconds (>= 0).
    start_anchor:
        Named camera or subject anchor at the start of the transition.
    end_anchor:
        Named camera or subject anchor at the end of the transition.
    overlap_s:
        Amount of visual overlap with neighbouring scenes on the shared
        timeline, in presentation seconds (>= 0).
    parameters:
        Optional free-form parameters (e.g. zoom factor, camera path
        hints). Pure data; not interpreted here.
    """

    transition_id: str
    start_scene: str
    end_scene: str
    transition_type: str
    duration_s: float
    start_anchor: str
    end_anchor: str
    overlap_s: float = 0.0
    parameters: dict | None = None

    def __post_init__(self) -> None:
        if not self.transition_id.strip():
            raise ValueError("transition_id must be non-empty")
        if not self.start_scene.strip():
            raise ValueError("start_scene must be non-empty")
        if not self.end_scene.strip():
            raise ValueError("end_scene must be non-empty")
        if self.transition_type not in ALLOWED_TRANSITION_TYPES:
            raise ValueError(
                f"unknown transition type {self.transition_type!r}; "
                f"allowed: {sorted(ALLOWED_TRANSITION_TYPES)}"
            )
        if self.duration_s < 0.0:
            raise ValueError("duration_s must be non-negative")
        if self.overlap_s < 0.0:
            raise ValueError("overlap_s must be non-negative")
        if not self.start_anchor.strip():
            raise ValueError("start_anchor must be non-empty")
        if not self.end_anchor.strip():
            raise ValueError("end_anchor must be non-empty")


@dataclass
class TransitionPlan:
    """A collection of transitions for a presentation.

    Transitions reference scenes by identifier; validity against a concrete
    timeline is checked with :meth:`validate_against`.
    """

    transitions: list[Transition] = field(default_factory=list)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def validate_against(self, scene_ids: set[str]) -> None:
        """Ensure every transition references known start and end scenes."""
        for transition in self.transitions:
            if transition.start_scene not in scene_ids:
                raise ValueError(
                    f"transition {transition.transition_id} references "
                    f"unknown start scene {transition.start_scene!r}"
                )
            if transition.end_scene not in scene_ids:
                raise ValueError(
                    f"transition {transition.transition_id} references "
                    f"unknown end scene {transition.end_scene!r}"
                )

    def transition_ids(self) -> list[str]:
        return [t.transition_id for t in self.transitions]
