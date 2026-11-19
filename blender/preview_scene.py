"""Build a simple radial-infall preview scene inside Blender.

This script requires Blender's ``bpy`` module. It:

* Creates (or replaces) the ``Animations_Preview`` collection.
* Adds three test particles animated from a real trajectory dataset.
* Adds a central black-hole marker at the simulated origin.
* Sets a dark background, an orthographic camera facing the XY plane, and
  basic lighting.

The black-hole marker and particle sizes are visual representations only —
they are not physical-to-scale models. Use a constant metres-to-blender-units
scale to make astronomical values viewable.

Run inside Blender::

    blender --python blender/preview_scene.py -- --dataset outputs/... --fps 24

or call :func:`build_preview_scene` from Blender's Python console.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # bpy is only available inside Blender
    import bpy
except ImportError:  # pragma: no cover - exercised inside Blender only
    bpy = None

from blender.animator import (  # noqa: E402
    SCALE_DEFAULT,
    animate_trajectory_file,
    require_bpy,
)
from blender.output_paths import (  # noqa: E402
    default_blend_output,
    next_available_blend_path,
)

__all__ = [
    "build_preview_scene",
    "SCALE_DEFAULT",
]

# Particle start radii for the preview radial-infall demonstration.
PREVIEW_INITIAL_RADIUS_M = 4.0e6
PREVIEW_CUTOFF_M = 1.0e6
PREVIEW_MASS_KG = 1.9885e30  # 1 solar mass


def _set_dark_background() -> None:
    """Set the world background to near-black."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("Preview_World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.02, 0.02, 0.03, 1.0)
        bg.inputs["Strength"].default_value = 0.5


def _add_orthographic_camera() -> bpy.types.Object:
    """Add an orthographic camera facing the XY simulation plane."""
    bpy.ops.object.camera_add(location=(8.0, -8.0, 6.0))
    camera = bpy.context.active_object
    camera.name = "Preview_Camera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 12.0

    # Point the camera at the origin.
    direction = -camera.location
    rot = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot.to_euler()
    bpy.context.scene.camera = camera
    return camera


def _add_key_light() -> bpy.types.Object:
    """Add a basic key light so particles are visible."""
    bpy.ops.object.light_add(type="POINT", location=(5.0, 5.0, 8.0))
    light = bpy.context.active_object
    light.name = "Preview_KeyLight"
    light.data.energy = 500.0
    return light


def build_preview_scene(
    dataset_path: str | Path,
    *,
    scale: float = SCALE_DEFAULT,
    fps: float = 24.0,
    duration_s: float = 5.0,
    start_sim_time_s: float = 0.0,
    end_sim_time_s: float | None = None,
    hold_s: float = 0.0,
    collection_name: str = "Animations_Preview",
) -> dict[str, bpy.types.Object]:
    """Build the complete preview scene in Blender.

    Parameters
    ----------
    dataset_path:
        Path stem of the trajectory dataset.
    scale:
        Metres-to-blender-units scale.
    fps:
        Presentation frame rate (default 24).
    duration_s:
        Active preview duration in seconds (default 5).
    start_sim_time_s:
        Start of the simulation window consumed.
    end_sim_time_s:
        End of the simulation window (scene boundary).
    hold_s:
        Optional hold duration appended after the active frames.
    collection_name:
        Blender collection that owns the preview objects.

    Returns
    -------
    dict
        Mapping of object name to Blender object for the created preview.
    """
    require_bpy()
    objects = animate_trajectory_file(
        dataset_path,
        scale=scale,
        fps=fps,
        duration_s=duration_s,
        start_sim_time_s=start_sim_time_s,
        end_sim_time_s=end_sim_time_s,
        hold_s=hold_s,
        collection_name=collection_name,
    )

    # Scene environment (only created if not already present).
    _set_dark_background()
    if "Preview_Camera" not in bpy.data.objects:
        _add_orthographic_camera()
    if "Preview_KeyLight" not in bpy.data.objects:
        _add_key_light()

    return objects


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Blender's ``--python`` mode."""
    require_bpy()
    parser = argparse.ArgumentParser(
        description="Build a Blender preview scene from a trajectory dataset."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path stem of the trajectory dataset (.meta.json / .data.npz).",
    )
    parser.add_argument(
        "--scale", type=float, default=SCALE_DEFAULT,
        help="Metres-to-blender-units scale (default 1e-6).",
    )
    parser.add_argument(
        "--fps", type=float, default=24.0,
        help="Presentation frame rate (default 24).",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Active preview duration in seconds (default 5).",
    )
    parser.add_argument(
        "--hold", type=float, default=0.0,
        help="Optional hold duration in seconds appended after active frames.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .blend path (default: dataset name + preview + FPS).",
    )

    if argv is None:
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1:]
        else:
            argv = []

    args = parser.parse_args(argv)

    build_preview_scene(
        args.dataset,
        scale=args.scale,
        fps=args.fps,
        duration_s=args.duration,
        hold_s=args.hold,
    )

    # Preserve previous exports, even when --output is reused.
    requested_path = args.output or default_blend_output(
        args.dataset, args.fps, kind="preview"
    )
    output_path = next_available_blend_path(requested_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    if not output_path.is_file():
        raise RuntimeError(f"Blender did not save the scene to {output_path}")

    print(f"Preview scene built for dataset: {args.dataset}")
    print(f"Saved Blender scene: {output_path}")

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
