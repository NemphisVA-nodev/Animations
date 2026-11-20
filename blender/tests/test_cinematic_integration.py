"""Blender 5.3 smoke test for the Step 5C cinematic preview.

Run this test with Blender's --python option. No pytest installation inside
Blender is necessary. It deliberately does not render an image; use
cinematic_scene.py --render-still to test actual Eevee output separately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # Allow ordinary pytest to import the module without Blender installed.
    import bpy
except ImportError:
    bpy = None

from blender.animator import require_bpy
from blender.cinematic_paths import (
    select_still_frame, trail_reveal_windows, visible_infall_window_end
)
from blender.cinematic_scene import build_cinematic_scene, VISUAL_COLLECTION
from blender.importer import metres_to_blender
from blender.output_paths import default_blend_output, next_available_blend_path
from core.playback import PlaybackSpec, build_frame_mapping, sample_particle_positions
from core.simulation.dataset import load_dataset


def run_cinematic_test(
    dataset_path: str | Path,
    *,
    scale: float,
    fps: float,
    trail_frames: int,
    output: str | Path,
) -> None:
    require_bpy()
    dataset = load_dataset(dataset_path)
    objects = build_cinematic_scene(
        dataset_path, fps=fps, scale=scale, trail_frames=trail_frames
    )
    scene = bpy.context.scene
    assert scene.render.engine == "BLENDER_EEVEE"
    assert scene.camera is not None and scene.camera.data.type == "ORTHO"
    assert scene.render.resolution_x == 960
    assert "BlackHole_Marker" in objects
    assert objects["BlackHole_Marker"]["visual_classification"].startswith("SCHEMATIC")

    # Regression: no default Blender startup cube should remain in the scene.
    assert "Cube" not in bpy.data.objects, "startup Cube was not removed"

    # Regression: black hole marker must be dark/schematic (low emission).
    bh = objects["BlackHole_Marker"]
    bh_mat = bh.data.materials[0]
    bh_mat.use_nodes = True
    bsdf = bh_mat.node_tree.nodes.get("Principled BSDF")
    assert bsdf is not None, "black hole should use Principled BSDF"
    base_color = bsdf.inputs["Base Color"].default_value[:3]
    assert all(c < 0.15 for c in base_color), (
        f"black hole base color too bright: {base_color}"
    )
    assert bsdf.inputs["Emission Strength"].default_value < 0.5, (
        "black hole should not be brightly emissive"
    )

    collection = bpy.data.collections[VISUAL_COLLECTION]
    ring_names = {
        f"Cinematic_AccretionRing_{index}" for index in range(1, 4)
    }
    assert ring_names.issubset({obj.name for obj in collection.objects})
    assert scene["visual_classification"].startswith("SCHEMATIC")
    assert scene["physics_classification"].startswith("CALCULATED")

    # Regression: rings must be tilted (rotation about X > 0).
    for ring_name in ring_names:
        ring = bpy.data.objects[ring_name]
        assert abs(ring.rotation_euler.x) > 0.05, (
            f"{ring_name} should be tilted, got x={ring.rotation_euler.x}"
        )

    # Regression: camera framing should not be excessively far left.
    cam = scene.camera
    assert cam is not None
    assert cam.data.sensor_fit == "HORIZONTAL"

    # Project actual ring mesh vertices to camera space. A visual crop
    # regression can pass every pure math test but still ruin the screenshot.
    from bpy_extras.object_utils import world_to_camera_view
    for ring_name in ring_names:
        ring = bpy.data.objects[ring_name]
        for vertex in ring.data.vertices:
            coord = world_to_camera_view(
                scene, cam, ring.matrix_world @ vertex.co
            )
            assert 0.0 < coord.x < 1.0 and 0.0 < coord.y < 1.0, (
                f"{ring_name} is outside the rendered camera frame: "
                f"({coord.x:.4f}, {coord.y:.4f})"
            )

    ids = [pid for pid in objects if pid != "BlackHole_Marker"]
    spec = PlaybackSpec(
        scene_id="cinematic_test",
        simulation_id=dataset.metadata.get("simulation_id"),
        fps=fps,
        duration_s=5.0,
        start_sim_time_s=float(dataset.arrays["timestamps"][0]),
        end_sim_time_s=float(dataset.arrays["timestamps"][-1]),
    )
    mapping = build_frame_mapping(spec)
    positions, valid = sample_particle_positions(
        dataset, ids, mapping.all_sim_times_s
    )
    expected = metres_to_blender(positions, scale)
    assert scene.frame_end == mapping.total_frames
    recommended, visible = select_still_frame(
        expected, valid, active_frames=mapping.num_frames
    )
    assert scene["recommended_still_frame"] == recommended
    assert scene["recommended_still_visible_particles"] == visible
    assert scene["visual_scale_m_to_blender"] == scale
    assert 1 <= recommended <= mapping.num_frames
    # Blender stores Object.scale as float32; exact float64 equality is brittle.
    assert np.allclose(
        tuple(objects[ids[0]].scale), (0.23, 0.23, 0.23),
        rtol=0.0, atol=1e-6,
    ), f"unexpected particle marker scale: {tuple(objects[ids[0]].scale)}"

    frames = sorted({0, mapping.total_frames // 2, mapping.total_frames - 1})
    for index, pid in enumerate(ids):
        if np.count_nonzero(valid[:mapping.num_frames, index]) < 2:
            continue
        trail = bpy.data.objects[pid + "_Trail"]
        starts, ends, shown = trail_reveal_windows(
            valid[:, index],
            tail_frames=trail_frames,
            active_frames=mapping.num_frames,
        )
        for frame_idx in frames:
            scene.frame_set(frame_idx + 1)
            assert trail.hide_render == (not bool(shown[frame_idx]))
            assert abs(trail.data.bevel_factor_start - starts[frame_idx]) < 1e-5
            assert abs(trail.data.bevel_factor_end - ends[frame_idx]) < 1e-5
            if valid[frame_idx, index]:
                actual = np.array(objects[pid].location[:], dtype=np.float64)
                assert np.allclose(
                    actual, expected[frame_idx, index], rtol=1e-5, atol=1e-5
                ), f"trajectory mismatch at frame {frame_idx + 1} for {pid}"

    # A second run must replace, not duplicate, cinematic decorations.
    old_names = {obj.name for obj in collection.objects}
    build_cinematic_scene(
        dataset_path, fps=fps, scale=scale, trail_frames=trail_frames
    )
    new_names = {obj.name for obj in collection.objects}
    assert new_names == old_names, "cinematic rerun left duplicate objects"
    assert len(new_names) == len(collection.objects)

    scene.frame_set(1)
    output_path = next_available_blend_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    assert output_path.is_file()
    print("Cinematic integration PASSED")
    print(f"  particles: {ids}")
    print(f"  frames: {mapping.total_frames} (fps={fps})")
    print(f"  decorative objects: {len(new_names)}")
    print(f"  saved: {output_path}")


def run_visible_motion_pacing_test(
    dataset_path: str | Path, *, scale: float, fps: float
) -> None:
    """Verify the retimed scene still contains real sampled SI trajectories."""
    require_bpy()
    source = load_dataset(dataset_path)
    arr = source.arrays
    expected_end = visible_infall_window_end(
        arr["timestamps"], arr["positions"], arr["validity_mask"], scale=scale
    )
    if expected_end is None:
        raise AssertionError(
            "this smoke dataset does not have an early visible infall ending"
        )

    objects = build_cinematic_scene(
        dataset_path, scale=scale, fps=fps, duration_s=5.0,
        fit_visible_motion=True, motion_fraction=0.8
    )
    scene = bpy.context.scene
    assert scene["motion_fit_enabled"]
    assert abs(scene["motion_fit_end_sim_time_s"] - expected_end) < 1e-5
    assert scene["motion_fit_active_seconds"] == 4.0
    assert scene["motion_fit_hold_seconds"] == 1.0
    assert scene.frame_end == round(5.0 * fps)

    spec = PlaybackSpec(
        scene_id="paced_smoke",
        simulation_id=source.metadata.get("simulation_id"),
        fps=fps,
        duration_s=4.0,
        hold_s=1.0,
        start_sim_time_s=float(arr["timestamps"][0]),
        end_sim_time_s=float(expected_end),
    )
    mapping = build_frame_mapping(spec)
    ids = [pid for pid in objects if pid != "BlackHole_Marker"]
    positions, valid = sample_particle_positions(
        source, ids, mapping.all_sim_times_s
    )
    expected_xyz = metres_to_blender(positions, scale)
    for index in sorted({0, mapping.num_frames // 2, mapping.num_frames - 1}):
        scene.frame_set(index + 1)
        for j, pid in enumerate(ids):
            assert objects[pid].hide_render == (not bool(valid[index, j]))
            if valid[index, j]:
                assert np.allclose(
                    objects[pid].location[:], expected_xyz[index, j],
                    rtol=1e-5, atol=1e-5
                ), f"retimed Blender position differs from source at frame {index + 1}"
    print(
        "Motion pacing integration PASSED: "
        f"{mapping.num_frames} active, {mapping.hold_frames} hold frames"
    )



def run_animated_disk_test(
    dataset_path: str | Path, *, scale: float, fps: float
) -> None:
    """Check decorative ring motion and independence from particle keyframes."""
    require_bpy()
    objects = build_cinematic_scene(
        dataset_path, scale=scale, fps=fps, animated_disk=True,
        fit_visible_motion=True,
    )
    scene = bpy.context.scene
    assert scene["schematic_disk_highlights_enabled"]
    inner = bpy.data.objects["Cinematic_Disk_Inner_Highlight"]
    outer = bpy.data.objects["Cinematic_Disk_Outer_Highlight"]
    plane = bpy.data.objects["Cinematic_Disk_Highlight_Plane"]
    assert inner.parent == plane and outer.parent == plane
    assert inner["visual_classification"].startswith("SCHEMATIC")
    assert len(inner.data.splines[0].points) == 25
    assert len(objects) >= 2

    scene.frame_set(1)
    start_angle = float(inner.rotation_euler.z)
    scene.frame_set(scene.frame_end // 2)
    middle_angle = float(inner.rotation_euler.z)
    scene.frame_set(scene.frame_end)
    end_angle = float(inner.rotation_euler.z)
    assert start_angle < middle_angle < end_angle
    assert abs(end_angle - 2 * np.pi) < 1e-4
    assert plane.rotation_euler.x > 0.05

    # Particle scene is still driven by exactly the same source data.
    source = load_dataset(dataset_path)
    arr = source.arrays
    ids = [pid for pid in objects if pid != "BlackHole_Marker"]
    spec = PlaybackSpec(
        scene_id="disk_test", simulation_id=source.metadata.get("simulation_id"),
        fps=fps, duration_s=4.0, hold_s=1.0,
        start_sim_time_s=float(arr["timestamps"][0]),
        end_sim_time_s=float(scene["motion_fit_end_sim_time_s"]),
    )
    mapping = build_frame_mapping(spec)
    coords, valid = sample_particle_positions(
        source, ids, mapping.all_sim_times_s
    )
    scaled = metres_to_blender(coords, scale)
    for frame_idx in sorted({0, mapping.num_frames // 2, mapping.num_frames - 1}):
        scene.frame_set(frame_idx + 1)
        for j, pid in enumerate(ids):
            if valid[frame_idx, j]:
                assert np.allclose(
                    objects[pid].location[:], scaled[frame_idx, j],
                    rtol=1e-5, atol=1e-5
                ), f"decorative disk altered {pid} trajectory frame {frame_idx + 1}"
    print("Animated schematic disk integration PASSED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cinematic Blender smoke test")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scale", type=float, default=1e-6)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--trail-frames", type=int, default=14)
    parser.add_argument(
        "--animated-disk-test", action="store_true",
        help="Validate optional decorative ring highlights.",
    )
    parser.add_argument(
        "--fit-visible-motion-test", action="store_true",
        help="Test physics-preserving retimed visible infall instead of the standard scene.",
    )
    parser.add_argument("--output", default=None)
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    output = args.output or default_blend_output(
        args.dataset, args.fps, kind="cinematic_smoke"
    )
    if args.animated_disk_test:
        run_animated_disk_test(
            args.dataset, scale=args.scale, fps=args.fps
        )
        return 0
    if args.fit_visible_motion_test:
        run_visible_motion_pacing_test(
            args.dataset, scale=args.scale, fps=args.fps
        )
        return 0
    run_cinematic_test(
        args.dataset,
        scale=args.scale,
        fps=args.fps,
        trail_frames=args.trail_frames,
        output=output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
