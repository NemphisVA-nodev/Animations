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
from blender.cinematic_paths import trail_reveal_windows
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

    collection = bpy.data.collections[VISUAL_COLLECTION]
    ring_names = {
        f"Cinematic_AccretionRing_{index}" for index in range(1, 4)
    }
    assert ring_names.issubset({obj.name for obj in collection.objects})
    assert scene["visual_classification"].startswith("SCHEMATIC")
    assert scene["physics_classification"].startswith("CALCULATED")

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cinematic Blender smoke test")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scale", type=float, default=1e-6)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--trail-frames", type=int, default=14)
    parser.add_argument("--output", default=None)
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    output = args.output or default_blend_output(
        args.dataset, args.fps, kind="cinematic_smoke"
    )
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
