"""Export rendered PNG frames to an H.264 MP4 video using FFmpeg.

This script runs outside Blender (ordinary Python). It:

1. Locates FFmpeg (on PATH or via --ffmpeg-path).
2. Builds an FFmpeg command line with H.264 / yuv420p for PowerPoint
   compatibility.
3. Encodes sequentially numbered PNG frames into an MP4 file.

The output path is never overwritten: if it already exists, a collision-safe
suffix (_001, _002, ...) is added automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.exporting.manifest import RenderManifest, ManifestError  # noqa: E402
from core.exporting.video import (  # noqa: E402
    FFmpegNotFoundError,
    export_to_mp4,
    find_ffmpeg,
    ffmpeg_available,
)
__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for FFmpeg MP4 export."""
    parser = argparse.ArgumentParser(
        description="Encode PNG frame sequence to H.264 MP4 video."
    )
    parser.add_argument(
        "--frame-dir",
        help="Directory containing sequentially numbered PNG frames.",
    )
    parser.add_argument(
        "--output",
        help="Destination MP4 file path.",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Input FPS (default: render manifest, else 24).",
    )
    parser.add_argument(
        "--fps-base", type=float, default=None,
        help="FPS denominator (default: render manifest, else 1).",
    )
    parser.add_argument(
        "--ffmpeg-path", default=None,
        help="Explicit path to the ffmpeg executable (default: search PATH).",
    )
    parser.add_argument(
        "--input-pattern", default=None,
        help="Input filename pattern (default: infer from manifest).",
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="libx264 quality (lower = better, default 18).",
    )
    parser.add_argument(
        "--preset", default="medium",
        help="libx264 encoding preset (default 'medium').",
    )
    parser.add_argument(
        "--check-ffmpeg", action="store_true",
        help="Only check that FFmpeg is available and exit.",
    )

    if argv is None:
        # Blender-style invocations use '--'; ordinary Python/uv invocations
        # pass options directly after the script name.
        argv = (
            sys.argv[sys.argv.index("--") + 1:]
            if "--" in sys.argv else sys.argv[1:]
        )

    args = parser.parse_args(argv)

    if args.check_ffmpeg:
        if ffmpeg_available(args.ffmpeg_path):
            print(f"FFmpeg available: {find_ffmpeg(args.ffmpeg_path)}")
            return 0
        print("FFmpeg not found. Install FFmpeg or pass --ffmpeg-path.")
        return 1

    if not args.frame_dir or not args.output:
        parser.error("--frame-dir and --output are required for encoding")

    try:
        manifest_path = Path(args.frame_dir) / "render_manifest.json"
        manifest = (
            RenderManifest.load(manifest_path) if manifest_path.exists() else None
        )
        if manifest is not None:
            if args.fps is not None and args.fps != manifest.fps:
                parser.error("--fps disagrees with the render manifest")
            if args.fps_base is not None and args.fps_base != manifest.fps_base:
                parser.error("--fps-base disagrees with the render manifest")
        fps = args.fps if args.fps is not None else (manifest.fps if manifest else 24.0)
        fps_base = (
            args.fps_base if args.fps_base is not None
            else (manifest.fps_base if manifest else 1.0)
        )
        pattern = args.input_pattern or (
            f"{manifest.stem}_%04d.png" if manifest else "frame_%04d.png"
        )
        output_path = export_to_mp4(
            frame_dir=args.frame_dir,
            output_path=args.output,
            fps=fps,
            fps_base=fps_base,
            ffmpeg_path=args.ffmpeg_path,
            input_pattern=pattern,
            crf=args.crf,
            preset=args.preset,
        )
    except FFmpegNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    except (RuntimeError, ValueError, ManifestError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"MP4 exported: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
