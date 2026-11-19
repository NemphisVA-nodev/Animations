"""Blender integration smoke test for the trajectory importer.

This script runs inside Blender 5.3 and verifies more than process termination:

* The preview collection exists after import.
* Three expected particles and the black-hole marker exist.
* Particle animation keyframes have been created.
* Particle positions match first, middle, and final sampled frames.
* Visibility changes are correctly animated.
* Scene FPS and frame range are correct, including fractional FPS.
* A second import removes stale preview objects.
* The verified .blend file is saved under a unique filename.

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
from blender.output_paths import (  # noqa: E402
    default_blend_output,
    next_available_blend_path,
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
        # Validate position curves in Blender's layered Action structure.
        location_curves = 0
        for layer in obj.animation_data.action.layers:
            for strip in layer.strips:
                if strip.type != "KEYFRAME":
                    continue
                for channelbag in strip.channelbags:
                    for curve in channelbag.fcurves:
                        if curve.data_path == "location":
                            location_curves += 1
                            assert all(
                                kp.interpolation == "LINEAR"
                                for kp in curve.keyframe_points
                            ), f"{pid!r}: non-linear location curve"
        assert location_curves == 3, (
            f"{pid!r}: expected 3 XYZ location curves, got {location_curves}"
        )


def _verify_positions_match_sampled(
    particle_ids: list[str],
    expected_positions: list[tuple[float, float, float]],
    tolerance: float = 1e-6,
) -> None:
    """Verify object locations at the currently evaluated frame."""
    for pid, expected in zip(particle_ids, expected_positions):
        obj = bpy.data.objects[pid]
        # Blender frame 1 = Python frame 0.
        actual = obj.location
        assert all(
            abs(actual[i] - expected[i]) <= tolerance + 1e-6 * abs(expected[i])
            for i in range(3)
        ), f"{pid!r}: expected {expected}, got {tuple(actual)}"


def _verify_visibility_keyframes(particle_ids: list[str]) -> None:
    """Visibility properties must have CONSTANT interpolation keyframes."""
    for pid in particle_ids:
        obj = bpy.data.objects[pid]
        found = set()
        for layer in obj.animation_data.action.layers:
            for strip in layer.strips:
                if strip.type != "KEYFRAME":
                    continue
                for channelbag in strip.channelbags:
                    for fcurve in channelbag.fcurves:
                        if fcurve.data_path in ("hide_viewport", "hide_render"):
                            found.add(fcurve.data_path)
                            for kp in fcurve.keyframe_points:
                                assert kp.interpolation == "CONSTANT", (
                                    f"{pid!r} {fcurve.data_path} keyframe "
                                    "not CONSTANT"
                                )
        assert found == {"hide_viewport", "hide_render"}, (
            f"{pid!r}: missing visibility curves: "
            f"{sorted({'hide_viewport', 'hide_render'} - found)}"
        )


def _verify_scene_settings(expected_fps: float, expected_frame_end: int) -> None:
    scene = bpy.context.scene
    effective_fps = scene.render.fps / scene.render.fps_base
    assert abs(effective_fps - expected_fps) < 1e-5, (
        f"scene effective fps {effective_fps} != {expected_fps}"
    )
    assert scene.frame_start == 1, f"scene frame_start is {scene.frame_start}"
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
        Optional path to save the .blend file after the checks.
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

    # Check positions and visibility after evaluating three separate frames.
    for frame_index in sorted({0, mapping.total_frames // 2, mapping.total_frames - 1}):
        bpy.context.scene.frame_set(frame_index + 1)
        active_ids = []
        active_positions = []
        for particle_index, pid in enumerate(particle_ids):
            expected_valid = bool(validity[frame_index, particle_index])
            obj = bpy.data.objects[pid]
            assert bool(obj.hide_viewport) == (not expected_valid)
            assert bool(obj.hide_render) == (not expected_valid)
            if expected_valid:
                active_ids.append(pid)
                active_positions.append(
                    tuple(blender_positions[frame_index, particle_index])
                )
        _verify_positions_match_sampled(active_ids, active_positions)

    # --- Verify visibility keyframes ---
    _verify_visibility_keyframes(particle_ids)

    # --- Verify scene settings ---
    _verify_scene_settings(fps, mapping.total_frames)

    # Re-import with a stale preview object: only the fresh objects should remain.
    collection = bpy.data.collections["Animations_Preview"]
    stale = bpy.data.objects.new("Stale_Preview_Object", None)
    collection.objects.link(stale)
    refreshed = animate_trajectory_file(
        dataset_path, scale=scale, fps=fps, duration_s=duration_s,
    )
    assert "Stale_Preview_Object" not in bpy.data.objects
    assert set(refreshed) == set(objects)
    assert {obj.name for obj in collection.objects} == set(refreshed)

    bpy.context.scene.frame_set(1)

    # --- Save the verified .blend without overwriting an earlier run ---
    output_path = None
    if output_blend is not None:
        output_path = next_available_blend_path(output_blend)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        assert output_path.is_file(), f"failed to save {output_path}"

    print("Integration test PASSED")
    print(f"  particles: {particle_ids}")
    print(f"  frames: {mapping.total_frames} (fps={fps})")
    print("  collection: Animations_Preview")
    if output_path is not None:
        print(f"  saved: {output_path}")


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
        help="Output .blend path (default: dataset name + integration + FPS).",
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
        output_blend=args.output or default_blend_output(
            args.dataset, args.fps, kind="integration"
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
