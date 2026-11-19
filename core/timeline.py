"""Presentation timeline data model.

Distinguishes three notions of time:

1. Physical simulation time — the independent variable of the physics model,
   in SI seconds, independent of animation frame rate.
2. Animation playback time — how quickly simulated seconds are shown on
   screen (simulation seconds per presentation second).
3. Presentation time — the wall-clock duration of a scene within a
   presentation.

This module is pure configuration data. It defines no rendering or Blender
behaviour and does not depend on the physics engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SceneTimeline:
    """Timing and mapping for a single presentation scene.

    Attributes
    ----------
    scene_id:
        Unique identifier for the scene within a presentation.
    simulation_id:
        Identifier of the simulation dataset consumed by this scene, or
        None when the scene is purely conceptual/observational.
    duration_s:
        Presentation time allocated to the scene, in seconds (> 0).
    start_sim_time_s:
        Start of the simulation window consumed, in SI seconds.
    end_sim_time_s:
        End of the simulation window consumed, in SI seconds (>= start).
    playback_speed:
        Simulation seconds advanced per presentation second (> 0). A value
        of 1.0 means real-time; higher values speed the simulation up.
    entry_anchor:
        Named visual anchor used when entering the scene.
    exit_anchor:
        Named visual anchor used when leaving the scene.
    camera_transition:
        Optional metadata describing the camera movement into/within the
        scene (pure data; not implemented here).
    hold_s:
        Optional hold period appended after the scene, in seconds (>= 0).
    """

    scene_id: str
    simulation_id: str | None
    duration_s: float
    start_sim_time_s: float
    end_sim_time_s: float
    playback_speed: float
    entry_anchor: str
    exit_anchor: str
    camera_transition: dict | None = None
    hold_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        if self.end_sim_time_s < self.start_sim_time_s:
            raise ValueError("end_sim_time_s must be >= start_sim_time_s")
        if self.playback_speed <= 0.0:
            raise ValueError("playback_speed must be positive")
        if self.hold_s < 0.0:
            raise ValueError("hold_s must be non-negative")
        if not self.entry_anchor.strip():
            raise ValueError("entry_anchor must be non-empty")
        if not self.exit_anchor.strip():
            raise ValueError("exit_anchor must be non-empty")

    @property
    def sim_span_s(self) -> float:
        """Duration of the simulation window consumed by this scene."""
        return self.end_sim_time_s - self.start_sim_time_s

    @property
    def total_presentation_s(self) -> float:
        """Scene duration plus hold period, in presentation seconds."""
        return self.duration_s + self.hold_s


@dataclass
class PresentationTimeline:
    """An ordered list of scenes forming a presentation timeline.

    Scenes may be reused across presentations because a scene is identified
    by ``scene_id`` and references a simulation by ``simulation_id`` rather
    than by a hard-coded slide number.
    """

    presentation_id: str
    title: str
    scenes: list[SceneTimeline] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.presentation_id.strip():
            raise ValueError("presentation_id must be non-empty")
        seen: set[str] = set()
        for scene in self.scenes:
            if scene.scene_id in seen:
                raise ValueError(f"duplicate scene_id: {scene.scene_id}")
            seen.add(scene.scene_id)

    def add_scene(self, scene: SceneTimeline) -> None:
        if scene.scene_id in {s.scene_id for s in self.scenes}:
            raise ValueError(f"duplicate scene_id: {scene.scene_id}")
        self.scenes.append(scene)

    def total_duration_s(self) -> float:
        """Total presentation time of all scenes plus holds."""
        return sum(scene.total_presentation_s for scene in self.scenes)

    def scene_ids(self) -> list[str]:
        return [scene.scene_id for scene in self.scenes]

    def simulation_ids(self) -> list[str]:
        return [
            scene.simulation_id
            for scene in self.scenes
            if scene.simulation_id is not None
        ]
