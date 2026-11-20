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

from core.exporting.video import (  # noqa: E402
    FFmpegNotFoundError,
    export_to_mp4,
    find_ffmpeg,
    ffmpeg_available,
)
from blender.output_paths import (  # noqa: E402
    next_available_blend_path,
)

__all__ = ["main"]


def _next_available_mp4_path(requested: str | Path) -> Path:
    """Return a free MP4 output path, adding _001, _002, ... after collisions."""
    path = Path(requested).expanduser().resolve()
    if path.suffix.lower() != ".mp4":
        raise ValueError("output path must end in .mp4")
    if not path.exists():
        return path

    for number in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{number:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"no available output filename near {path}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for FFmpeg MP4 export."""
    parser = argparse.ArgumentParser(
        description="Encode PNG frame sequence to H.264 MP4 video."
    )
    parser.add_argument(
        "--frame-dir", required=True,
        help="Directory containing sequentially numbered PNG frames.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination MP4 file path.",
    )
    parser.add_argument(
        "--fps", type=float, default=24.0,
        help="Input frame rate (frames per second, default 24).",
    )
    parser.add_argument(
        "--fps-base", type=float, default=1.0,
        help="Frame rate denominator (default 1; effective FPS is fps/fps_base).",
    )
    parser.add_argument(
        "--ffmpeg-path", default=None,
        help="Explicit path to the ffmpeg executable (default: search PATH).",
    )
    parser.add_argument(
        "--input-pattern", default="frame_%04d.png",
        help="Input frame filename pattern (default 'frame_%04d.png').",
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
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1 :]
        else:
            argv = []

    args = parser.parse_args(argv)

    if args.check_ffmpeg:
        if ffmpeg_available(args.ffmpeg_path):
            print(f"FFmpeg available: {find_ffmpeg(args.ffmpeg_path)}")
            return 0
        print("FFmpeg not found. Install FFmpeg or pass --ffmpeg-path.")
        return 1

    try:
        output_path = export_to_mp4(
            frame_dir=args.frame_dir,
            output_path=args.output,
            fps=args.fps,
            fps_base=args.fps_base,
            ffmpeg_path=args.ffmpeg_path,
            input_pattern=args.input_pattern,
            crf=args.crf,
            preset=args.preset,
        )
    except FFmpegNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"MP4 exported: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
