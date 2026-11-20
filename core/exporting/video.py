"""FFmpeg detection and H.264 MP4 export for rendered frame sequences.

This module is independent of Blender and SciPy. It depends only on the Python
standard library and NumPy (used minimally, if at all). FFmpeg itself is an
external dependency: this module detects it on PATH or accepts an explicit
executable path, but never installs it.

The exported MP4 uses H.264 video with the yuv420p pixel format, which is
widely supported by Microsoft PowerPoint.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

__all__ = [
    "FFmpegNotFoundError",
    "find_ffmpeg",
    "ffmpeg_available",
    "build_ffmpeg_command",
    "export_to_mp4",
]


class FFmpegNotFoundError(RuntimeError):
    """Raised when FFmpeg is not available and no explicit path is given."""


def find_ffmpeg(explicit_path: str | Path | None = None) -> str:
    """Locate the FFmpeg executable.

    Parameters
    ----------
    explicit_path:
        Optional path to a specific ffmpeg executable. If given, it is
        validated for existence and executability and returned directly.

    Returns
    -------
    str
        Resolved path to the ffmpeg executable.

    Raises
    ------
    FFmpegNotFoundError
        If no explicit path is given and ffmpeg is not on PATH, or if the
        explicit path does not exist or is not executable.
    """
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FFmpegNotFoundError(
                f"explicit FFmpeg path does not exist: {path}"
            )
        if not _is_executable(path):
            raise FFmpegNotFoundError(
                f"explicit FFmpeg path is not executable: {path}"
            )
        return str(path)

    found = shutil.which("ffmpeg")
    if found is None:
        raise FFmpegNotFoundError(
            "ffmpeg not found on PATH. Install FFmpeg or pass --ffmpeg-path "
            "to the export script."
        )
    return found


def ffmpeg_available(explicit_path: str | Path | None = None) -> bool:
    """Return True if FFmpeg is available (detected or at explicit path)."""
    try:
        find_ffmpeg(explicit_path)
    except FFmpegNotFoundError:
        return False
    return True


def _is_executable(path: Path) -> bool:
    """Return True if the file exists and is executable by the current user.

    On Windows, :func:`os.access` with ``os.X_OK`` returns True for any
    existing file, so we check for common executable extensions. Extensionless
    files are also accepted (e.g. ``ffmpeg`` without ``.exe`` can be executed
    directly on Windows). On POSIX, we check the execute bit.
    """
    import os

    if not path.is_file():
        return False
    if os.name == "nt":
        ext = path.suffix.lower()
        return ext in (".exe", ".bat", ".cmd", ".ps1", ".com") or ext == ""
    return os.access(path, os.X_OK)


def build_ffmpeg_command(
    *,
    input_pattern: str | Path,
    output_path: str | Path,
    fps: float,
    fps_base: float = 1.0,
    ffmpeg_path: str | Path = "ffmpeg",
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
    crf: int = 18,
    preset: str = "medium",
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    """Build an FFmpeg command line to encode PNG frames into MP4.

    Parameters
    ----------
    input_pattern:
        Glob pattern for input PNG files (e.g. ``frame_%04d.png``). FFmpeg
        interprets %04d as a zero-padded frame number.
    output_path:
        Destination MP4 file path.
    fps:
        Input frame rate (frames per second). Combined with fps_base to form
        the effective rate ``fps / fps_base``.
    fps_base:
        Frame rate denominator. Effective input FPS is ``fps / fps_base``.
    ffmpeg_path:
        Path to the ffmpeg executable.
    codec:
        Video codec name (default ``libx264``).
    pix_fmt:
        Pixel format (default ``yuv420p`` for PowerPoint compatibility).
    crf:
        Constant rate factor for libx264 quality (lower = better, 18 is
        visually near-lossless).
    preset:
        libx264 encoding speed/quality preset.
    extra_args:
        Additional FFmpeg options appended before the output path.

    Returns
    -------
    list[str]
        Command-line argument list ready for :func:`subprocess.run`.
    """
    fps_val = float(fps)
    fps_base_val = float(fps_base)
    if fps_base_val == 0.0:
        raise ValueError(
            "effective FPS must be positive and finite (fps_base is zero)"
        )
    effective_fps = fps_val / fps_base_val
    if effective_fps <= 0 or effective_fps != effective_fps:  # NaN check
        raise ValueError(
            f"effective FPS must be positive and finite, got {effective_fps}"
        )

    cmd: list[str] = [str(ffmpeg_path)]
    cmd.extend(["-y"])  # overwrite output without prompting
    cmd.extend(["-framerate", f"{effective_fps:.6f}"])
    cmd.extend(["-i", str(input_pattern)])
    cmd.extend(["-c:v", codec])
    cmd.extend(["-pix_fmt", pix_fmt])
    cmd.extend(["-crf", str(int(crf))])
    cmd.extend(["-preset", preset])
    if extra_args:
        cmd.extend(str(a) for a in extra_args)
    cmd.append(str(output_path))
    return cmd


def export_to_mp4(
    *,
    frame_dir: str | Path,
    output_path: str | Path,
    fps: float,
    fps_base: float = 1.0,
    ffmpeg_path: str | Path | None = None,
    input_pattern: str = "frame_%04d.png",
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Render a sequence of PNG frames to an H.264 MP4 video.

    Parameters
    ----------
    frame_dir:
        Directory containing the sequentially numbered PNG frames.
    output_path:
        Destination MP4 file path.
    fps:
        Input frame rate (frames per second).
    fps_base:
        Frame rate denominator. Effective input FPS is ``fps / fps_base``.
    ffmpeg_path:
        Optional path to the ffmpeg executable. If not given, FFmpeg is
        located on PATH.
    input_pattern:
        Filename pattern for input frames (default ``frame_%04d.png``).
    crf:
        libx264 quality parameter (default 18).
    preset:
        libx264 encoding preset (default ``medium``).

    Returns
    -------
    Path
        Path to the written MP4 file.

    Raises
    ------
    FFmpegNotFoundError
        If FFmpeg cannot be located.
    FileNotFoundError
        If the frame directory does not exist or contains no matching frames.
    RuntimeError
        If FFmpeg returns a non-zero exit code.
    """
    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"frame directory not found: {frame_dir}")

    # Resolve ffmpeg path first so we fail fast if it's unavailable.
    resolved_ffmpeg = find_ffmpeg(ffmpeg_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify at least one input frame matches the pattern.
    import re

    # Convert %04d-style pattern to a regex for matching existing files.
    pattern_regex = re.escape(str(input_pattern)).replace(r"%04d", r"(\d{4})")
    pattern_regex = pattern_regex.replace(r"%03d", r"(\d{3})")
    pattern_regex = pattern_regex.replace(r"%02d", r"(\d{2})")
    pattern_regex = pattern_regex.replace(r"%d", r"(\d+)")

    matching = list(frame_dir.glob("*.png"))
    if not matching:
        raise FileNotFoundError(
            f"no PNG frames found in {frame_dir} matching pattern "
            f"{input_pattern!r}"
        )

    cmd = build_ffmpeg_command(
        input_pattern=frame_dir / input_pattern,
        output_path=output_path,
        fps=fps,
        fps_base=fps_base,
        ffmpeg_path=resolved_ffmpeg,
        crf=crf,
        preset=preset,
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed with exit code {result.returncode}:\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}"
        )

    if not output_path.is_file():
        raise RuntimeError(
            f"FFmpeg reported success but output file was not created: "
            f"{output_path}"
        )

    return output_path
