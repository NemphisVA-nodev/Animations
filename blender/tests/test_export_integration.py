"""Blender 5.3 integration test for frame-sequence rendering with resume.

This test runs inside Blender (``blender --background --python
blender/tests/test_export_integration.py -- ...``). It:

1. Creates a minimal .blend scene with a camera and a rotating cube.
2. Saves the scene to a temporary .blend file.
3. Calls render_frames to render a short frame range.
4. Verifies that PNG frames are produced.
5. Calls render_frames again with resume and verifies that already-rendered
   frames are skipped.

No pytest installation inside Blender is necessary. The module imports cleanly
under ordinary pytest (bpy = None) and is skipped there.
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

try:  # pytest is optional; only needed when running under pytest, not Blender.
    import pytest
except ImportError:
    pytest = None

from core.exporting.manifest import RenderManifest
from scripts.render_frames import render_frames  # noqa: E402

if pytest is not None:
    pytestmark = pytest.mark.skipif(
        bpy is None,
        reason="Blender integration test requires bpy (run inside Blender)",
    )


def _make_minimal_scene(blend_path: Path) -> None:
    """Create a simple .blend scene with a camera and a cube."""
    import bpy

    # Clear default objects.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Add a camera.
    bpy.ops.object.camera_add(location=(3.0, -3.0, 2.0))
    camera = bpy.context.active_object
    camera.data.type = "PERSP"
    camera.rotation_euler = (1.1, 0.0, 0.785)
    bpy.context.scene.camera = camera

    # Add a cube.
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))

    # Animate the cube moving slightly so frames differ.
    cube = bpy.data.objects["Cube"]
    cube.location = (0.0, 0.0, 0.0)
    cube.keyframe_insert(data_path="location", frame=1)
    cube.location = (1.0, 0.0, 0.0)
    cube.keyframe_insert(data_path="location", frame=5)

    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))


def test_render_frames_produces_pngs(tmp_path: Path) -> None:
    """Render a short frame range and verify PNG outputs."""
    import bpy

    blend_path = tmp_path / "test_scene.blend"
    frame_dir = tmp_path / "frames"
    manifest_path = tmp_path / "render_manifest.json"

    _make_minimal_scene(blend_path)

    result = render_frames(
        blend_path=blend_path,
        frame_dir=frame_dir,
        frame_start=1,
        frame_end=5,
        fps=24.0,
        fps_base=1.0,
        resolution_x=960,
        resolution_y=540,
        manifest_path=manifest_path,
        stem="frame",
    )

    # All 5 frames should be rendered.
    assert result["frames_rendered"] == 5
    assert result["frames_skipped"] == 0
    assert result["total_frames"] == 5

    # Verify each PNG exists.
    for frame in range(1, 6):
        png = frame_dir / f"frame_{frame:04d}.png"
        assert png.is_file(), f"missing rendered frame: {png}"
        assert png.stat().st_size > 0, f"empty rendered frame: {png}"

    # Verify manifest was written and records all frames.
    manifest = RenderManifest.load(manifest_path)
    assert manifest.completed_frames == [1, 2, 3, 4, 5]
    assert manifest.missing_frames(1, 5) == []


def test_render_frames_resume_skips_completed(tmp_path: Path) -> None:
    """Render part of a range, then resume and verify skipped frames."""
    import bpy

    blend_path = tmp_path / "test_scene_resume.blend"
    frame_dir = tmp_path / "frames_resume"
    manifest_path = tmp_path / "render_manifest_resume.json"

    _make_minimal_scene(blend_path)

    # First render: frames 1-3 only.
    result1 = render_frames(
        blend_path=blend_path,
        frame_dir=frame_dir,
        frame_start=1,
        frame_end=3,
        fps=24.0,
        fps_base=1.0,
        resolution_x=960,
        resolution_y=540,
        manifest_path=manifest_path,
        stem="frame",
    )
    assert result1["frames_rendered"] == 3
    assert result1["frames_skipped"] == 0

    # Verify frames 1-3 exist, 4-5 do not.
    for frame in (1, 2, 3):
        assert (frame_dir / f"frame_{frame:04d}.png").is_file()
    for frame in (4, 5):
        assert not (frame_dir / f"frame_{frame:04d}.png").exists()

    # Second render: resume, should skip 1-3 and render 4-5.
    result2 = render_frames(
        blend_path=blend_path,
        frame_dir=frame_dir,
        frame_start=1,
        frame_end=5,
        fps=24.0,
        fps_base=1.0,
        resolution_x=960,
        resolution_y=540,
        manifest_path=manifest_path,
        stem="frame",
    )
    assert result2["frames_rendered"] == 2
    assert result2["frames_skipped"] == 3

    # All 5 frames now exist.
    for frame in range(1, 6):
        png = frame_dir / f"frame_{frame:04d}.png"
        assert png.is_file(), f"missing rendered frame after resume: {png}"

    manifest = RenderManifest.load(manifest_path)
    assert manifest.completed_frames == [1, 2, 3, 4, 5]


def test_render_frames_incompatible_manifest_raises(tmp_path: Path) -> None:
    """Render with one config, then attempt resume with a different config."""
    import bpy

    blend_path = tmp_path / "test_scene_incompat.blend"
    frame_dir = tmp_path / "frames_incompat"
    manifest_path = tmp_path / "render_manifest_incompat.json"

    _make_minimal_scene(blend_path)

    # First render at 24 fps.
    render_frames(
        blend_path=blend_path,
        frame_dir=frame_dir,
        frame_start=1,
        frame_end=5,
        fps=24.0,
        fps_base=1.0,
        resolution_x=960,
        resolution_y=540,
        manifest_path=manifest_path,
        stem="frame",
    )

    # Second render at 30 fps with same manifest -> should raise.
    try:
        render_frames(
            blend_path=blend_path,
            frame_dir=frame_dir,
            frame_start=1,
            frame_end=5,
            fps=30.0,
            fps_base=1.0,
            resolution_x=960,
            resolution_y=540,
            manifest_path=manifest_path,
            stem="frame",
        )
    except Exception as exc_info:
        assert "incompatible" in str(exc_info).lower()
    else:
        raise AssertionError("Expected ManifestError for incompatible config")


def test_render_frames_no_resume_renders_all(tmp_path: Path) -> None:
    """--no-resume must not overwrite previously rendered frames."""
    import bpy

    blend_path = tmp_path / "test_scene_noresume.blend"
    frame_dir = tmp_path / "frames_noresume"
    manifest_path = tmp_path / "render_manifest_noresume.json"

    _make_minimal_scene(blend_path)

    # First render.
    render_frames(
        blend_path=blend_path,
        frame_dir=frame_dir,
        frame_start=1,
        frame_end=3,
        fps=24.0,
        fps_base=1.0,
        resolution_x=960,
        resolution_y=540,
        manifest_path=manifest_path,
        stem="frame",
    )

    # Re-rendering into the same directory could overwrite valuable images.
    try:
        render_frames(
            blend_path=blend_path,
            frame_dir=frame_dir,
            frame_start=1,
            frame_end=3,
            fps=24.0,
            fps_base=1.0,
            resolution_x=960,
            resolution_y=540,
            manifest_path=manifest_path,
            stem="frame",
            resume=False,
        )
    except Exception as exc:
        assert "new output directory" in str(exc)
    else:
        raise AssertionError("Expected output protection for --no-resume")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Blender's --background --python mode."""
    if bpy is None:
        print("This script must run inside Blender.")
        return 1

    parser = argparse.ArgumentParser(
        description="Run Blender frame-render integration tests."
    )
    parser.add_argument(
        "--tmp-dir", default=None,
        help="Directory for temporary test files (default: system temp).",
    )
    if argv is None:
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1 :]
        else:
            argv = []
    args = parser.parse_args(argv)

    import tempfile

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp())

    test_render_frames_produces_pngs(tmp_dir)
    test_render_frames_resume_skips_completed(tmp_dir)
    test_render_frames_incompatible_manifest_raises(tmp_dir)
    test_render_frames_no_resume_renders_all(tmp_dir)
    test_render_frames_restores_deleted_frame(tmp_dir)
    test_fractional_fps_preserved_inside_blender(tmp_dir)
    test_render_frames_changed_blend_rejects_resume(tmp_dir)
    test_render_frames_corrupt_png_refuses_resume(tmp_dir)

    print("All Blender frame-render integration tests PASSED")
    return 0



def test_render_frames_restores_deleted_frame(tmp_path: Path) -> None:
    """Resume recreates missing frames instead of incorrectly skipping them."""
    blend = tmp_path / "test_scene_deleted.blend"
    folder = tmp_path / "deleted_frames"
    _make_minimal_scene(blend)
    render_frames(
        blend_path=blend, frame_dir=folder, frame_start=1, frame_end=3,
        resolution_x=320, resolution_y=180
    )
    (folder / "frame_0002.png").unlink()
    result = render_frames(
        blend_path=blend, frame_dir=folder, frame_start=1, frame_end=3,
        resolution_x=320, resolution_y=180
    )
    assert result["frames_rendered"] == 1
    assert result["frames_skipped"] == 2
    assert (folder / "frame_0002.png").is_file()


def test_fractional_fps_preserved_inside_blender(tmp_path: Path) -> None:
    blend = tmp_path / "fractional_scene.blend"
    folder = tmp_path / "fractional_frames"
    _make_minimal_scene(blend)
    render_frames(
        blend_path=blend, frame_dir=folder, frame_start=1, frame_end=1,
        fps=23.976, fps_base=1, resolution_x=320, resolution_y=180,
    )
    import bpy
    effective = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    assert abs(effective - 23.976) < 1e-5




def test_render_frames_changed_blend_rejects_resume(tmp_path: Path) -> None:
    """Same path with changed scene bytes must never reuse old frames."""
    blend = tmp_path / "test_scene_modified.blend"
    folder = tmp_path / "modified_frames"
    _make_minimal_scene(blend)
    render_frames(
        blend_path=blend, frame_dir=folder, frame_start=1, frame_end=1,
        resolution_x=320, resolution_y=180,
    )
    with blend.open("ab") as handle:
        handle.write(b"changed scene source")
    try:
        render_frames(
            blend_path=blend, frame_dir=folder, frame_start=1, frame_end=1,
            resolution_x=320, resolution_y=180,
        )
    except Exception as exc:
        assert "incompatible" in str(exc).lower()
    else:
        raise AssertionError("Expected incompatible source .blend hash")


def test_render_frames_corrupt_png_refuses_resume(tmp_path: Path) -> None:
    """A corrupt existing image is never silently overwritten or skipped."""
    blend = tmp_path / "test_scene_corrupt.blend"
    folder = tmp_path / "corrupt_frames"
    _make_minimal_scene(blend)
    render_frames(
        blend_path=blend, frame_dir=folder, frame_start=1, frame_end=1,
        resolution_x=320, resolution_y=180,
    )
    image = folder / "frame_0001.png"
    image.write_bytes(b"corrupted-image-bytes")
    try:
        render_frames(
            blend_path=blend, frame_dir=folder, frame_start=1, frame_end=1,
            resolution_x=320, resolution_y=180,
        )
    except Exception as exc:
        assert "differs" in str(exc).lower()
    else:
        raise AssertionError("Expected PNG checksum mismatch")


if __name__ == "__main__":
    raise SystemExit(main())

