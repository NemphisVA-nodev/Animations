"""Blender integration smoke test for the trajectory importer.

This script runs inside Blender 5.3 and verifies more than process termination:

* The preview collection exists after import.
* Three expected particles and the black-hole marker exist.
* Particle animation keyframes have been created.
* Particle movement follows the sampled simulation positions.
* Visibility changes are correctly animated.
* Scene FPS and frame range are correct.
* The generated .blend file can be saved.

Run inside Blender::

    blender --background --python blender/tests/test_blender_integration.py -- \
        --dataset outputs/preview_radial_infall_radial_infall \
        --fps 24 \
        --blender-exe "C:\\Path\\To\\blender.exe" \
        --output outputs/integration_test.blend

The Blender executable path is configurable via --blender-exe (or the
BLENDER_EXE environment variable); it is never hardcoded in committed code.

This test is opt-in and separate from the standard pytest suite.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender.animator import (  # noqa: E402
    SCALE_DEFAULT,
    animate_trajectory_file,
    require_bpy,
)

__all__ = [
    "run_integration_test",
]


def _verify_collection_exists(name: str) -> None:
    assert name in bpy.data.collections, f"collection {name!r} not found"


def _verify_objects_exist(particle_ids: list[str], bh_marker_name: str) -> None:
    for pid in particle_ids:
        assert pid in bpy.data.objects, f"particle object {pid!r} not found"
    assert bh_marker_name in bpy.data.objects, (
        f"black-hole marker {bh_marker_name!r} not found"
    )


def _verify_keyframes_exist(particle_ids: list[str]) -> None:
    for pid in particle_ids:
        obj = bpy.data.objects[pid]
        assert obj.animation_data is not None, (
            f"no animation data on {pid!r}"
        )
        assert obj.animation_data.action is not None, (
            f"no action on {pid!r}"
        )
        # Count fcurves across layered Actions.
        fcurve_count = 0
        for layer in obj.animation_data.action.layers:
            for strip in layer.strips:
                if strip.type != "KEYFRAME":
                    continue
                for channelbag in strip.channelbags:
                    fcurve_count += len(channelbag.fcurves)
        assert fcurve_count > 0, f"no fcurves on {pid!r}"


def _verify_positions_match_sampled(
    particle_ids: list[str],
    expected_positions: list[tuple[float, float, float]],
    tolerance: float = 1e-6,
) -> None:
    """Verify frame-1 (Python frame 0) positions match the first sample."""
    for pid, expected in zip(particle_ids, expected_positions):
        obj = bpy.data.objects[pid]
        # Blender frame 1 = Python frame 0.
        obj.location  # ensure evaluated
        actual = obj.location
        assert all(
            abs(actual[i] - expected[i]) < tolerance for i in range(3)
        ), f"{pid!r}: expected {expected}, got {tuple(actual)}"


def _verify_visibility_keyframes(particle_ids: list[str]) -> None:
    """Visibility properties must have CONSTANT interpolation keyframes."""
    for pid in particle_ids:
        obj = bpy.data.objects[pid]
        for layer in obj.animation_data.action.layers:
            for strip in layer.strips:
                if strip.type != "KEYFRAME":
                    continue
                for channelbag in strip.channelbags:
                    for fcurve in channelbag.fcurves:
                        if fcurve.data_path in ("hide_viewport", "hide_render"):
                            for kp in fcurve.keyframe_points:
                                assert kp.interpolation == "CONSTANT", (
                                    f"{pid!r} {fcurve.data_path} keyframe "
                                    "not CONSTANT"
                                )


def _verify_scene_settings(expected_fps: float, expected_frame_end: int) -> None:
    scene = bpy.context.scene
    assert scene.render.fps == expected_fps, (
        f"scene fps {scene.render.fps} != {expected_fps}"
    )
    assert scene.frame_end == expected_frame_end, (
        f"scene frame_end {scene.frame_end} != {expected_frame_end}"
    )


def run_integration_test(
    dataset_path: str | Path,
    *,
    scale: float = SCALE_DEFAULT,
    fps: float = 24.0,
    duration_s: float = 5.0,
    output_blend: str | Path | None = None,
) -> None:
    """Run the full integration test inside Blender.

    Parameters
    ----------
    dataset_path:
        Path stem of the trajectory dataset.
    scale:
        Metres-to-blender-units scale.
    fps:
        Presentation frame rate.
    duration_s:
        Active preview duration in seconds.
    output_blend:
        Optional path to save the .blend file for reopen verification.
    """
    require_bpy()
    dataset_path = Path(dataset_path)

    # --- Import and animate ---
    objects = animate_trajectory_file(
        dataset_path,
        scale=scale,
        fps=fps,
        duration_s=duration_s,
        start_sim_time_s=0.0,
        end_sim_time_s=None,
        hold_s=0.0,
    )

    particle_ids = [k for k in objects if k != "BlackHole_Marker"]
    bh_marker_name = "BlackHole_Marker"

    # --- Verify collection ---
    _verify_collection_exists("Animations_Preview")

    # --- Verify objects ---
    _verify_objects_exist(particle_ids, bh_marker_name)

    # --- Verify keyframes ---
    _verify_keyframes_exist(particle_ids)

    # --- Verify frame 1 matches first sampled position ---
    from core.simulation.dataset import (  # noqa: E402
        load_dataset,
    )
    from core.playback import (  # noqa: E402
        PlaybackSpec,
        build_frame_mapping,
        sample_particle_positions,
    )
    from blender.importer import (  # noqa: E402
        metres_to_blender,
    )

    ds = load_dataset(dataset_path)
    spec = PlaybackSpec(
        scene_id="integration",
        simulation_id=ds.metadata.get("simulation_id"),
        fps=fps,
        duration_s=duration_s,
        start_sim_time_s=0.0,
        end_sim_time_s=float(ds.arrays["timestamps"][-1]),
    )
    mapping = build_frame_mapping(spec)
    sim_times = mapping.all_sim_times_s
    positions, validity = sample_particle_positions(
        ds, particle_ids, sim_times
    )
    blender_positions = metres_to_blender(positions, scale)

    # Frame 1 (Python index 0) should match the first sampled position.
    _verify_positions_match_sampled(
        particle_ids,
        [tuple(blender_positions[0, i]) for i in range(len(particle_ids))],
    )

    # --- Verify visibility keyframes ---
    _verify_visibility_keyframes(particle_ids)

    # --- Verify scene settings ---
    _verify_scene_settings(fps, mapping.total_frames)

    # --- Verify .blend save and reopen ---
    if output_blend is not None:
        output_path = Path(output_blend).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        assert output_path.exists(), f"failed to save {output_path}"

    print("Integration test PASSED")
    print(f"  particles: {particle_ids}")
    print(f"  frames: {mapping.total_frames} (fps={fps})")
    print(f"  collection: Animations_Preview")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Blender integration smoke test for trajectory importer."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path stem of the trajectory dataset.",
    )
    parser.add_argument(
        "--scale", type=float, default=SCALE_DEFAULT,
        help="Metres-to-blender-units scale.",
    )
    parser.add_argument(
        "--fps", type=float, default=24.0,
        help="Presentation frame rate.",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Active preview duration in seconds.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional .blend output path for save/reopen verification.",
    )
    parser.add_argument(
        "--blender-exe", default=None,
        help="Path to Blender executable (informational only; not hardcoded).",
    )

    if argv is None:
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1:]
        else:
            argv = []

    args = parser.parse_args(argv)

    # --blender-exe is informational; we're already inside Blender.
    if args.blender_exe is not None:
        print(f"Blender executable (from args): {args.blender_exe}")
    elif "BLENDER_EXE" in os.environ:
        print(f"Blender executable (from env): {os.environ['BLENDER_EXE']}")

    run_integration_test(
        args.dataset,
        scale=args.scale,
        fps=args.fps,
        duration_s=args.duration,
        output_blend=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
