"""Render a frame sequence from an existing Blender .blend scene.

This script runs inside Blender (``blender --background --python
scripts/render_frames.py -- ...``). It:

1. Loads the specified .blend scene.
2. Configures render settings (PNG output, FPS, resolution, frame range).
3. Renders each frame in the range, skipping frames already recorded in the
   render manifest (resume support).
4. Updates the manifest after each successfully rendered frame.

The manifest is a JSON sidecar that records render configuration and completed
frames. If the configuration changes (different FPS, resolution, or frame
range), a new manifest is started.

Output files are never overwritten: each render uses a dedicated frame output
directory, and the manifest ensures idempotent re-runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # bpy is only available inside Blender
    import bpy
except ImportError:  # pragma: no cover - exercised inside Blender only
    bpy = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.exporting.manifest import RenderManifest, ManifestError  # noqa: E402

__all__ = ["main"]


def require_bpy() -> None:
    """Raise a clear error if ``bpy`` is unavailable (not inside Blender)."""
    if bpy is None:
        raise ImportError(
            "Blender's bpy module is required to render frames. "
            "Run this script inside Blender (e.g. ``blender --background "
            "--python scripts/render_frames.py -- ...``)."
        )


def _output_pattern(frame_dir: Path, stem: str = "frame") -> str:
    """Build a Blender output filename pattern with zero-padded frame numbers.

    Blender replaces ``####`` with the zero-padded frame number in output
    paths. We use four digits by default, which supports up to 9999 frames.
    """
    return str(frame_dir / f"{stem}_####.png")


def _render_frame(
    scene: bpy.types.Scene,
    frame_number: int,
    output_dir: Path,
    stem: str = "frame",
) -> Path:
    """Render a single frame and return the output file path.

    Blender's render operator writes to ``scene.render.filepath`` with the
    frame number substituted. We set the filepath per-frame so that each
    frame lands at a predictable location.
    """
    scene.frame_set(frame_number)
    output_path = output_dir / f"{stem}_{frame_number:04d}.png"
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    if not output_path.is_file():
        raise RuntimeError(
            f"Blender render did not produce expected output: {output_path}"
        )
    return output_path


def render_frames(
    *,
    blend_path: str | Path,
    frame_dir: str | Path,
    frame_start: int,
    frame_end: int,
    fps: float = 24.0,
    fps_base: float = 1.0,
    resolution_x: int = 960,
    resolution_y: int = 540,
    resolution_percentage: int = 100,
    manifest_path: str | Path | None = None,
    stem: str = "frame",
    resume: bool = True,
) -> dict[str, object]:
    """Render a frame sequence from an existing .blend scene.

    Parameters
    ----------
    blend_path:
        Path to the source .blend file.
    frame_dir:
        Directory where rendered PNG frames will be written.
    frame_start:
        First frame to render (inclusive).
    frame_end:
        Last frame to render (inclusive).
    fps:
        Render frame rate numerator.
    fps_base:
        Render frame rate denominator. Effective FPS is ``fps / fps_base``.
    resolution_x:
        Render width in pixels.
    resolution_y:
        Render height in pixels.
    resolution_percentage:
        Render resolution percentage (1-100).
    manifest_path:
        Optional path to the render manifest JSON. If not given, defaults to
        ``frame_dir / "render_manifest.json"``.
    stem:
        Filename stem for rendered frames (default ``"frame"``).
    resume:
        If True (default), an existing compatible manifest is used to skip
        already-rendered frames. If False, rendering starts from
        ``frame_start`` regardless of existing manifest state.

    Returns
    -------
    dict
        Summary information: frames rendered, frames skipped, output dir,
        manifest path.

    Raises
    ------
    ImportError
        If bpy is unavailable (not running inside Blender).
    ManifestError
        If an existing manifest is incompatible with the current request.
    """
    require_bpy()

    blend_path = Path(blend_path).resolve()
    if not blend_path.is_file():
        raise FileNotFoundError(f"blend file not found: {blend_path}")

    frame_dir = Path(frame_dir).resolve()
    frame_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path is None:
        manifest_path = frame_dir / "render_manifest.json"
    manifest_path = Path(manifest_path)

    # Load or create manifest.
    manifest = None
    if resume and manifest_path.exists():
        try:
            manifest = RenderManifest.load(manifest_path)
        except ManifestError:
            manifest = None

    if manifest is not None:
        if not manifest.config_matches(
            blend_path=blend_path,
            fps=fps,
            fps_base=fps_base,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            resolution_percentage=resolution_percentage,
        ):
            raise ManifestError(
                f"existing manifest at {manifest_path} is incompatible with "
                "the current render configuration (different FPS, "
                "resolution, or blend file). Delete the manifest or use a "
                "different output directory to start a new render."
            )
    else:
        manifest = RenderManifest(
            output_dir=str(frame_dir),
            blend_path=str(blend_path),
            fps=fps,
            fps_base=fps_base,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            resolution_percentage=resolution_percentage,
            frame_start=frame_start,
            frame_end=frame_end,
        )

    # Load the .blend scene.
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    scene = bpy.context.scene

    # Configure render settings.
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = resolution_percentage
    # Blender's render.fps expects an int; store the original float in the
    # manifest for accurate config matching.
    scene.render.fps = int(fps)
    scene.render.fps_base = fps_base
    scene.frame_start = frame_start
    scene.frame_end = frame_end

    # Determine which frames still need rendering.
    completed = set(manifest.completed_frames)
    in_range = set(range(frame_start, frame_end + 1))
    pending = [f for f in range(frame_start, frame_end + 1) if f not in completed]
    skipped = len(in_range & completed)

    rendered = 0

    for frame_number in pending:
        _render_frame(scene, frame_number, frame_dir, stem=stem)
        manifest.mark_complete(
            frame_number, output_file=frame_dir / f"{stem}_{frame_number:04d}.png"
        )
        manifest.save(manifest_path)
        rendered += 1

    return {
        "frames_rendered": rendered,
        "frames_skipped": skipped,
        "total_frames": frame_end - frame_start + 1,
        "output_dir": str(frame_dir),
        "manifest_path": str(manifest_path),
        "blend_path": str(blend_path),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Blender's --background --python mode."""
    require_bpy()
    parser = argparse.ArgumentParser(
        description="Render a frame sequence from a Blender .blend scene."
    )
    parser.add_argument(
        "--blend", required=True,
        help="Path to the source .blend file.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for rendered PNG frames.",
    )
    parser.add_argument(
        "--frame-start", type=int, required=True,
        help="First frame to render (inclusive).",
    )
    parser.add_argument(
        "--frame-end", type=int, required=True,
        help="Last frame to render (inclusive).",
    )
    parser.add_argument(
        "--fps", type=float, default=24.0,
        help="Render frame rate numerator (default 24).",
    )
    parser.add_argument(
        "--fps-base", type=float, default=1.0,
        help="Render frame rate denominator (default 1).",
    )
    parser.add_argument(
        "--resolution-x", type=int, default=960,
        help="Render width in pixels (default 960).",
    )
    parser.add_argument(
        "--resolution-y", type=int, default=540,
        help="Render height in pixels (default 540).",
    )
    parser.add_argument(
        "--resolution-percentage", type=int, default=100,
        help="Render resolution percentage 1-100 (default 100).",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Path to render manifest JSON (default: <output-dir>/render_manifest.json).",
    )
    parser.add_argument(
        "--stem", default="frame",
        help="Filename stem for rendered frames (default 'frame').",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing manifest and render all frames from scratch.",
    )

    if argv is None:
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1 :]
        else:
            argv = []

    args = parser.parse_args(argv)

    result = render_frames(
        blend_path=args.blend,
        frame_dir=args.output_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        fps=args.fps,
        fps_base=args.fps_base,
        resolution_x=args.resolution_x,
        resolution_y=args.resolution_y,
        resolution_percentage=args.resolution_percentage,
        manifest_path=args.manifest,
        stem=args.stem,
        resume=not args.no_resume,
    )

    print(f"Render complete: {result['frames_rendered']} rendered, "
          f"{result['frames_skipped']} skipped of "
          f"{result['total_frames']} total frames.")
    print(f"Output dir: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
