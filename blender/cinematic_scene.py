"""Step 5C: a lightweight cinematic Blender scene using verified physics data.

The source trajectories and animation timing are inherited from Step 5B.
Accretion-like rings, glow and camera treatment are SCHEMATIC, not simulated
gravitational lensing, general relativity, or a realistic accretion flow.

Run under Blender 5.3 from the repository root:
    blender --background --python-exit-code 1 --python blender/cinematic_scene.py -- \\
        --dataset outputs/preview_radial_infall_radial_infall --scale 5e-8 --fps 24

Add --render-still to create an individual PNG preview without exporting video.
The draft quality setting is deliberately light enough for modest hardware.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import bpy
    from mathutils import Vector
except ImportError:  # Outside Blender: only pure helpers are unit tested.
    bpy = None
    Vector = None

from blender.animator import Animator, require_bpy  # noqa: E402
from blender.cinematic_paths import camera_plan, trail_reveal_windows  # noqa: E402
from blender.importer import (  # noqa: E402
    SCALE_DEFAULT,
    load_trajectory,
    metres_to_blender,
)
from blender.output_paths import (  # noqa: E402
    default_blend_output,
    next_available_blend_path,
    next_available_png_path,
)
from core.playback import (  # noqa: E402
    PlaybackSpec,
    build_frame_mapping,
    sample_particle_positions,
)

VISUAL_COLLECTION = "Cinematic_Visuals"
RING_COLORS = (
    (1.0, 0.28, 0.06, 1.0),
    (1.0, 0.65, 0.18, 1.0),
    (0.20, 0.52, 1.0, 1.0),
)
PARTICLE_COLORS = (
    (0.20, 0.65, 1.00, 1.0),
    (0.45, 0.95, 0.80, 1.0),
    (1.00, 0.72, 0.28, 1.0),
    (0.90, 0.45, 1.00, 1.0),
)

# Names of objects that Blender's default startup file creates. These are
# removed from the cinematic scene so they never occlude the black hole or
# appear as an unexplained gray block in the render.
DEFAULT_STARTUP_OBJECTS = {"Cube", "Camera", "Light"}

__all__ = ["build_cinematic_scene", "main"]


def _remove_default_startup_objects() -> None:
    """Remove Blender startup-file objects that are not part of the cinematic scene.

    Blender's default startup file contains a Cube, Camera and Light. The
    Cube sits at the origin and would occlude the black-hole center; the
    default Camera and Light are superseded by the cinematic camera and fill
    light. Objects already claimed by the Step 5B preview collection are left
    untouched.
    """
    for name in DEFAULT_STARTUP_OBJECTS:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        # Never remove objects that belong to the Step 5B preview collection.
        if any(c.name == "Animations_Preview" for c in obj.users_collection):
            continue
        bpy.data.objects.remove(obj, do_unlink=True)


def _visual_collection():
    """Get a dedicated, disposable collection for cinematic-only objects."""
    scene = bpy.context.scene
    collection = bpy.data.collections.get(VISUAL_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(VISUAL_COLLECTION)
    if collection.name not in scene.collection.children:
        scene.collection.children.link(collection)

    # Do not touch the Step 5B particle collection.
    for obj in list(collection.objects):
        data = obj.data
        object_type = obj.type
        materials = list(data.materials) if object_type in {"MESH", "CURVE"} else []
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if object_type == "MESH":
                bpy.data.meshes.remove(data)
            elif object_type == "CURVE":
                bpy.data.curves.remove(data)
            elif object_type == "CAMERA":
                bpy.data.cameras.remove(data)
            elif object_type == "LIGHT":
                bpy.data.lights.remove(data)
        for material in materials:
            if material is not None and material.users == 0:
                bpy.data.materials.remove(material)
    return collection


def _move_into_collection(obj, collection) -> None:
    if collection not in obj.users_collection:
        collection.objects.link(obj)
    for other in list(obj.users_collection):
        if other != collection:
            other.objects.unlink(obj)


def _emission_material(name: str, color: tuple[float, ...], strength: float):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = strength
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    material.diffuse_color = color
    return material


def _darken_black_hole(obj) -> None:
    """Make the black-hole marker a readable dark schematic core.

    A pure black sphere on a near-black background is invisible. Instead the
    marker uses a very dark cool color with a faint self-illumination so its
    circular silhouette reads against the background. It is explicitly a
    schematic marker, not a relativistic event horizon.
    """
    if obj.data.materials:
        material = obj.data.materials[0]
    else:
        material = bpy.data.materials.new(name="Cinematic_BlackHole_Surface")
        obj.data.materials.append(material)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.02, 0.025, 0.045, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.98
    bsdf.inputs["Emission Strength"].default_value = 0.15
    bsdf.inputs["Emission Color"].default_value = (0.05, 0.07, 0.12, 1.0)
    material.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    obj["visual_classification"] = "SCHEMATIC black-hole marker, not to scale"


def _make_glow_ring(name, radius, thickness, height, color, collection):
    """Add one tilted schematic accretion-style ring.

    The ring is rotated about X by ~18 degrees so it reads as a disk seen at
    an angle rather than a flat stack of circles. Thickness and glow are kept
    modest for Eevee draft performance.
    """
    bpy.ops.mesh.primitive_torus_add(
        major_segments=96,
        minor_segments=8,
        major_radius=radius,
        minor_radius=thickness,
        location=(0.0, 0.0, height),
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = (0.30, 0.0, 0.0)  # ~18 degree tilt toward camera
    _move_into_collection(obj, collection)
    obj.data.materials.append(_emission_material(name + "_Mat", color, 2.2))
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj["visual_classification"] = "SCHEMATIC glowing ring; no lensing simulation"
    return obj


def _make_trail(pid, sampled_xyz, validity, active_frames, tail_frames, color, collection):
    """Build a time-revealed curve that never shows future or invalid positions."""
    start, end, visible = trail_reveal_windows(
        validity, tail_frames=tail_frames, active_frames=active_frames
    )
    count = int(np.count_nonzero(validity[:active_frames]))
    if count < 2:
        return None

    curve = bpy.data.curves.new(pid + "_TrailGeometry", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = 0.018
    curve.bevel_resolution = 2
    # POLY curve: RESOLUTION maps each equally spaced sample to one segment.
    curve.bevel_factor_mapping_start = "RESOLUTION"
    curve.bevel_factor_mapping_end = "RESOLUTION"
    spline = curve.splines.new("POLY")
    spline.points.add(count - 1)
    for idx, point in enumerate(sampled_xyz[:count]):
        spline.points[idx].co = (
            float(point[0]), float(point[1]), 0.14, 1.0
        )

    obj = bpy.data.objects.new(pid + "_Trail", curve)
    collection.objects.link(obj)
    curve.materials.append(_emission_material(
        pid + "_TrailMat", color, 1.6
    ))
    obj["visual_classification"] = "CALCULATED positions; SCHEMATIC trailing highlight"

    for idx in range(validity.size):
        frame = idx + 1
        curve.bevel_factor_start = float(start[idx])
        curve.bevel_factor_end = float(end[idx])
        curve.keyframe_insert(data_path="bevel_factor_start", frame=frame)
        curve.bevel_factor_end = float(end[idx])
        curve.keyframe_insert(data_path="bevel_factor_end", frame=frame)
        obj.hide_viewport = not bool(visible[idx])
        obj.hide_render = not bool(visible[idx])
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        obj.keyframe_insert(data_path="hide_render", frame=frame)

    # Blender 5.3 layered actions: ensure no smooth easing or visual overshoot.
    for datablock in (curve, obj):
        action = datablock.animation_data.action
        for layer in action.layers:
            for strip in layer.strips:
                if strip.type != "KEYFRAME":
                    continue
                for bag in strip.channelbags:
                    for fcurve in bag.fcurves:
                        mode = (
                            "CONSTANT" if fcurve.data_path.startswith("hide_")
                            else "LINEAR"
                        )
                        for key in fcurve.keyframe_points:
                            key.interpolation = mode
    return obj


def _configure_camera(center_x, center_y, half_extent, collection):
    """Place the camera for a balanced 16:9 cinematic composition.

    The black hole sits near the left third of the frame so the particle
    trails have room to extend into the right side without excessive empty
    space. The camera looks down from upper-right toward the origin.
    """
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "Cinematic_Camera"
    _move_into_collection(camera, collection)
    camera.data.type = "ORTHO"
    # Ortho scale fits both the black hole and the particle trail region.
    camera.data.ortho_scale = max(8.0, half_extent * 2.6)

    # Rule of thirds: offset the subject (origin / black hole) toward the
    # left side so the right side carries the trailing particle motion.
    offset_x = center_x - half_extent * 0.28
    offset_y = center_y - half_extent * 0.42
    camera.location = (
        offset_x + half_extent * 0.62,
        offset_y - half_extent * 0.62,
        half_extent * 2.1,
    )
    target = Vector((center_x, center_y, 0.0))
    camera.rotation_euler = (target - camera.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()
    bpy.context.scene.camera = camera
    return camera


def _configure_scene(quality: str) -> None:
    scene = bpy.context.scene
    # Blender 5.x renamed the Eevee engine back to BLENDER_EEVEE.
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 960 if quality == "draft" else 1920
    scene.render.resolution_y = 540 if quality == "draft" else 1080
    scene.render.resolution_percentage = 75 if quality == "draft" else 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("Cinematic_World")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.006, 0.009, 0.018, 1.0)
        background.inputs["Strength"].default_value = 0.30
    scene["physics_classification"] = "CALCULATED Newtonian particle trajectories"
    scene["visual_classification"] = "SCHEMATIC rings, glow, camera, marker"


def build_cinematic_scene(
    dataset_path: str | Path,
    *,
    scale: float = SCALE_DEFAULT,
    fps: float = 24.0,
    duration_s: float = 5.0,
    hold_s: float = 0.0,
    trail_frames: int = 14,
    quality: str = "draft",
) -> dict[str, object]:
    """Build a self-contained scene from Step 5B's validated particle positions.

    Returns the original particle ID/object map. Cinematic decorations belong
    to a separate collection, and neither the input dataset nor its physics
    arrays are edited.
    """
    require_bpy()
    if quality not in ("draft", "final"):
        raise ValueError("quality must be draft or final")
    if isinstance(trail_frames, bool) or not isinstance(trail_frames, int) or trail_frames < 1:
        raise ValueError("trail_frames must be a positive integer")

    # Remove default Blender startup objects before adding anything else, so
    # the startup Cube never occludes the black-hole center.
    _remove_default_startup_objects()

    data = load_trajectory(dataset_path, scale=scale)
    spec = PlaybackSpec(
        scene_id="cinematic_black_hole",
        simulation_id=data.metadata.get("simulation_id"),
        fps=fps,
        duration_s=duration_s,
        start_sim_time_s=float(data.timestamps[0]),
        end_sim_time_s=float(data.timestamps[-1]),
        hold_s=hold_s,
    )
    # The animator provides all source-of-truth particle keyframes.
    animator = Animator(data=data, spec=spec, scale=scale)
    particles = animator.animate()
    mapping = build_frame_mapping(spec)
    sampled, mask = sample_particle_positions(
        animator._as_simulation_dataset(),
        data.particle_ids,
        mapping.all_sim_times_s,
    )
    scaled_xyz = metres_to_blender(sampled, scale)

    collection = _visual_collection()
    _darken_black_hole(particles[animator.bh_marker_name])
    particles[animator.bh_marker_name].scale = (0.62, 0.62, 0.62)
    for number, (radius, thickness, height) in enumerate((
        (1.15, 0.050, -0.020),
        (1.50, 0.085, -0.075),
        (2.05, 0.022, -0.110),
    )):
        _make_glow_ring(
            f"Cinematic_AccretionRing_{number + 1}",
            radius, thickness, height, RING_COLORS[number], collection
        )

    for index, pid in enumerate(data.particle_ids):
        color = PARTICLE_COLORS[index % len(PARTICLE_COLORS)]
        # Reuse the per-particle material but switch to emission for visibility.
        material = particles[pid].data.materials[0]
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = color
        emission.inputs["Strength"].default_value = 4.0
        material.node_tree.links.new(
            emission.outputs["Emission"], output.inputs["Surface"]
        )
        # Slightly larger particles for readability at draft resolution.
        particles[pid].scale = (0.85, 0.85, 0.85)
        _make_trail(
            pid,
            scaled_xyz[:, index],
            mask[:, index],
            mapping.num_frames,
            trail_frames,
            color,
            collection,
        )

    center_x, center_y, extent = camera_plan(scaled_xyz, mask)
    _configure_camera(center_x, center_y, extent, collection)
    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 7.0))
    light = bpy.context.active_object
    light.name = "Cinematic_FillLight"
    _move_into_collection(light, collection)
    light.data.energy = 700.0
    light.data.shape = "DISK"
    light.data.size = 7.0
    _configure_scene(quality)

    bpy.context.scene.frame_set(1)
    return particles


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for --background Blender execution."""
    require_bpy()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Trajectory dataset stem.")
    parser.add_argument("--scale", type=float, default=SCALE_DEFAULT)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--hold", type=float, default=0.0)
    parser.add_argument("--trail-frames", type=int, default=14)
    parser.add_argument("--quality", choices=("draft", "final"), default="draft")
    parser.add_argument("--output", default=None, help="Optional .blend output path.")
    parser.add_argument(
        "--render-still", action="store_true",
        help="Also render one PNG at roughly the middle of the active animation.",
    )
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    build_cinematic_scene(
        args.dataset,
        scale=args.scale,
        fps=args.fps,
        duration_s=args.duration,
        hold_s=args.hold,
        trail_frames=args.trail_frames,
        quality=args.quality,
    )

    requested = args.output or default_blend_output(
        args.dataset, args.fps, kind="cinematic"
    )
    blend_path = next_available_blend_path(requested)
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if not blend_path.is_file():
        raise RuntimeError(f"Blender failed to save {blend_path}")
    print(f"Cinematic scene saved: {blend_path}")

    if args.render_still:
        scene = bpy.context.scene
        frame = min(scene.frame_end, 1 + round(
            (scene.frame_end - 1) * 0.45
        ))
        scene.frame_set(frame)
        png_path = next_available_png_path(
            blend_path.with_name(blend_path.stem + "_still.png")
        )
        scene.render.filepath = str(png_path)
        bpy.ops.render.render(write_still=True)
        if not png_path.is_file():
            raise RuntimeError(f"Blender failed to render {png_path}")
        print(f"Cinematic still rendered: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
