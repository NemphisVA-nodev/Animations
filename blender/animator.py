"""Animate precomputed particle trajectories inside Blender.

This module requires Blender's ``bpy`` module and is only usable when run
inside Blender. The data-loading and coordinate-conversion logic lives in
:mod:`blender.importer` and is independent of ``bpy``.

Responsibilities
----------------
* Create one Blender object per particle, named from the stable particle ID.
* Keyframe object locations from sampled physics positions using LINEAR
  interpolation (no Bezier easing).
* Hide particles (viewport + render) when the validity mask is False and keep
  them hidden for all remaining frames, including hold frames.
* Add a central black-hole marker at the simulated origin.
* Build an orthographic camera facing the XY simulation plane.
* Apply a constant metres-to-blender-units scale.

Blender never simulates gravity here; it only visualizes precomputed data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # bpy is only available inside Blender
    import bpy
except ImportError:  # pragma: no cover - exercised inside Blender only
    bpy = None

from core.playback import (  # noqa: E402
    PlaybackSpec,
    build_frame_mapping,
    sample_particle_positions,
)
from core.simulation.dataset import (  # noqa: E402
    SimulationDataset,
    SimulationMetadata,
)
from blender.importer import (  # noqa: E402
    PARTICLE_IDS_ARRAY,
    TrajectoryData,
    VALIDITY_MASK_ARRAY,
    SCALE_DEFAULT,
    metres_to_blender,
    python_frame_to_blender_frame,
)

if TYPE_CHECKING:  # pragma: no cover - type checkers have bpy stubs optionally
    import bpy as _bpy  # noqa: F401

__all__ = [
    "Animator",
    "require_bpy",
    "animate_trajectory_file",
    "SCALE_DEFAULT",
]


def require_bpy() -> None:
    """Raise a clear error if ``bpy`` is unavailable (not inside Blender)."""
    if bpy is None:
        raise ImportError(
            "Blender's bpy module is required to animate trajectories. "
            "Run this code inside Blender (e.g. ``blender --python script.py``)."
        )


@dataclass
class Animator:
    """Create and keyframe Blender objects from a trajectory dataset.

    Attributes
    ----------
    data:
        Validated trajectory data from :func:`blender.importer.load_trajectory`.
    spec:
        Playback timing specification.
    scale:
        Metres-to-blender-units scale.
    collection_name:
        Name of the Blender collection that owns the preview objects.
    bh_marker_name:
        Name of the black-hole marker object.
    """

    data: TrajectoryData
    spec: PlaybackSpec
    scale: float = SCALE_DEFAULT
    collection_name: str = "Animations_Preview"
    bh_marker_name: str = "BlackHole_Marker"

    def __post_init__(self) -> None:
        require_bpy()
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be a positive finite value")

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _preview_collection(self) -> bpy.types.Collection:
        """Get or create the collection that owns preview objects."""
        name = self.collection_name
        if name in bpy.data.collections:
            return bpy.data.collections[name]
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
        return collection

    def _remove_preview_objects(self) -> None:
        """Remove previously created preview objects to avoid duplicates."""
        collection = self._preview_collection()
        owned = {self.bh_marker_name} | set(self.data.particle_ids)
        for obj in list(collection.objects):
            if obj.name in owned:
                for coll in list(obj.users_collection):
                    coll.objects.unlink(obj)
                bpy.data.objects.remove(obj, do_unlink=True)

    @staticmethod
    def _new_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.3
        return material

    def _create_particle_object(
        self, particle_id: str, color: tuple[float, float, float, float]
    ) -> bpy.types.Object:
        """Create a small UV sphere for a particle with a distinct material."""
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0.0, 0.0, 0.0))
        obj = bpy.context.active_object
        obj.name = particle_id

        material = self._new_material(f"{particle_id}_mat", color)
        obj.data.materials.append(material)
        return obj

    def _create_bh_marker(self) -> bpy.types.Object:
        """Create a central marker at the simulated black-hole origin."""
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, 0.0))
        obj = bpy.context.active_object
        obj.name = self.bh_marker_name

        material = self._new_material(
            f"{self.bh_marker_name}_mat", (1.0, 1.0, 0.0, 1.0)
        )  # yellow
        obj.data.materials.append(material)
        return obj

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    def animate(self) -> dict[str, bpy.types.Object]:
        """Build the preview scene and keyframe all particle trajectories.

        Returns a mapping of particle ID (and the BH marker) to Blender
        objects.
        """
        require_bpy()
        self._remove_preview_objects()
        collection = self._preview_collection()

        # Black-hole marker at the simulated origin (visual, not to scale).
        bh_obj = self._create_bh_marker()
        bh_obj.location = (0.0, 0.0, 0.0)
        collection.objects.link(bh_obj)

        # Blender frame mapping: Python frame index i -> Blender frame i + 1.
        mapping = build_frame_mapping(self.spec)
        all_sim_times = mapping.all_sim_times_s  # includes holds, frozen

        # Rebuild a minimal SimulationDataset for the sampler.
        dataset = self._as_simulation_dataset()
        frame_positions, frame_valid = sample_particle_positions(
            dataset,
            self.data.particle_ids,
            all_sim_times,
        )

        blender_positions = metres_to_blender(frame_positions, self.scale)

        created: dict[str, bpy.types.Object] = {self.bh_marker_name: bh_obj}

        base_colors = [
            (0.2, 0.6, 1.0, 1.0),  # blue
            (0.2, 1.0, 0.6, 1.0),  # green
            (1.0, 0.6, 0.2, 1.0),  # orange
            (1.0, 0.2, 1.0, 1.0),  # magenta
            (0.6, 0.2, 1.0, 1.0),  # violet
        ]

        for output_index, particle_id in enumerate(self.data.particle_ids):
            color = base_colors[output_index % len(base_colors)]
            obj = self._create_particle_object(particle_id, color)
            collection.objects.link(obj)
            created[particle_id] = obj

            n_frames = frame_positions.shape[0]
            for frame_index in range(n_frames):
                blender_frame = python_frame_to_blender_frame(frame_index)
                is_valid = bool(frame_valid[frame_index, output_index])

                if is_valid:
                    x, y, z = blender_positions[frame_index, output_index]
                    obj.location = (float(x), float(y), float(z))
                    obj.keyframe_insert(data_path="location", frame=blender_frame)
                    obj.hide_viewport = False
                    obj.hide_render = False
                else:
                    obj.hide_viewport = True
                    obj.hide_render = True

                obj.keyframe_insert(data_path="hide_viewport", frame=blender_frame)
                obj.keyframe_insert(data_path="hide_render", frame=blender_frame)


            # Blender 5.x: use layered Actions for interpolation.
            # Location keyframes use LINEAR; visibility keyframes use CONSTANT
            # (step) so particles snap in/out of view at the right frames.
            if obj.animation_data is None or obj.animation_data.action is None:
                raise RuntimeError(
                    f"no animation data on object {particle_id!r} after "
                    "keyframing; cannot set interpolation"
                )

            action = obj.animation_data.action
            fcurve_count = 0
            for layer in action.layers:
                for strip in layer.strips:
                    if strip.type != "KEYFRAME":
                        continue
                    for channelbag in strip.channelbags:
                        for fcurve in channelbag.fcurves:
                            fcurve_count += 1
                            if fcurve.data_path == "location":
                                interpolation = "LINEAR"
                            else:
                                interpolation = "CONSTANT"
                            for kp in fcurve.keyframe_points:
                                kp.interpolation = interpolation

            if fcurve_count == 0:
                raise RuntimeError(
                    f"no fcurves found in layered Action for object "
                    f"{particle_id!r}; interpolation was not applied"
                )


        return created

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _as_simulation_dataset(self) -> SimulationDataset:
        """Rebuild a minimal SimulationDataset from validated trajectory data.

        :func:`core.playback.sample_particle_positions` expects a full
        :class:`SimulationDataset`; we already validated the arrays in the
        importer, so this is a thin view over them.
        """
        metadata: SimulationMetadata = dict(self.data.metadata)
        metadata["nonfinite_policy"] = "masked"
        return SimulationDataset(
            metadata=metadata,
            arrays={
                "timestamps": self.data.timestamps,
                "positions": self.data.positions,
                "velocities": self.data.velocities,
                VALIDITY_MASK_ARRAY: self.data.validity_mask,
                PARTICLE_IDS_ARRAY: np.array(self.data.particle_ids),
            },
        )


def animate_trajectory_file(
    path: str | Path,
    *,
    scale: float = SCALE_DEFAULT,
    fps: float = 24.0,
    duration_s: float = 5.0,
    start_sim_time_s: float = 0.0,
    end_sim_time_s: float | None = None,
    hold_s: float = 0.0,
    collection_name: str = "Animations_Preview",
    bh_marker_name: str = "BlackHole_Marker",
) -> dict[str, bpy.types.Object]:
    """Convenience: load a dataset file and animate it in Blender.

    Parameters
    ----------
    path:
        Path stem of the trajectory dataset.
    scale:
        Metres-to-blender-units scale.
    fps:
        Presentation frame rate (default 24).
    duration_s:
        Presentation duration of the active preview (default 5 s).
    start_sim_time_s:
        Start of the simulation window consumed.
    end_sim_time_s:
        End of the simulation window (scene boundary). Defaults to the
        dataset's last timestamp.
    hold_s:
        Optional hold duration appended after the active frames.
    collection_name:
        Blender collection that owns the preview objects.
    bh_marker_name:
        Name of the black-hole marker object.
    """
    require_bpy()
    from core.simulation.dataset import load_dataset

    dataset = load_dataset(path)
    if end_sim_time_s is None:
        end_sim_time_s = float(dataset.arrays["timestamps"][-1])

    from blender.importer import load_trajectory

    data = load_trajectory(path, scale=scale)

    spec = PlaybackSpec(
        scene_id=str(dataset.metadata.get("simulation_id", "preview")),
        simulation_id=dataset.metadata.get("simulation_id"),
        fps=fps,
        duration_s=duration_s,
        start_sim_time_s=start_sim_time_s,
        end_sim_time_s=float(end_sim_time_s),
        hold_s=hold_s,
    )

    animator = Animator(
        data=data,
        spec=spec,
        scale=scale,
        collection_name=collection_name,
        bh_marker_name=bh_marker_name,
    )
    return animator.animate()
