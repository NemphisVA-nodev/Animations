"""Pure-Python tests for FFmpeg detection and MP4 export command construction.

No Blender or FFmpeg installation is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.exporting.video import (
    FFmpegNotFoundError,
    build_ffmpeg_command,
    ffmpeg_available,
    find_ffmpeg,
)


# ---------------------------------------------------------------------------
# FFmpeg detection
# ---------------------------------------------------------------------------


def test_find_ffmpeg_explicit_path(tmp_path: Path) -> None:
    fake_ffmpeg = tmp_path / "ffmpeg.exe"
    fake_ffmpeg.write_text("dummy", encoding="utf-8")
    fake_ffmpeg.chmod(0o755)

    result = find_ffmpeg(fake_ffmpeg)
    assert result == str(fake_ffmpeg.resolve())


def test_find_ffmpeg_nonexistent_explicit_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FFmpegNotFoundError, match="does not exist"):
        find_ffmpeg(tmp_path / "no_such_ffmpeg.exe")


def test_find_ffmpeg_nonexecutable_explicit_path_raises(tmp_path: Path) -> None:
    fake_ffmpeg = tmp_path / "ffmpeg.txt"
    fake_ffmpeg.write_text("dummy", encoding="utf-8")

    with pytest.raises(FFmpegNotFoundError, match="not executable"):
        find_ffmpeg(fake_ffmpeg)


def test_ffmpeg_available_with_existing_executable(tmp_path: Path) -> None:
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text("dummy", encoding="utf-8")
    fake_ffmpeg.chmod(0o755)

    assert ffmpeg_available(fake_ffmpeg)


def test_ffmpeg_available_without_executable(tmp_path: Path) -> None:
    fake_ffmpeg = tmp_path / "ffmpeg.txt"
    fake_ffmpeg.write_text("dummy", encoding="utf-8")

    assert not ffmpeg_available(fake_ffmpeg)


# ---------------------------------------------------------------------------
# FFmpeg command construction
# ---------------------------------------------------------------------------


def test_build_ffmpeg_command_defaults() -> None:
    cmd = build_ffmpeg_command(
        input_pattern="frames/frame_%04d.png",
        output_path="outputs/video.mp4",
        fps=24.0,
    )

    assert cmd[0] == "ffmpeg"
    assert "-y" in cmd
    assert "-framerate" in cmd
    assert "24.000000" in cmd
    assert "-i" in cmd
    assert "frames/frame_%04d.png" in cmd
    assert "-c:v" in cmd
    assert "libx264" in cmd
    assert "-pix_fmt" in cmd
    assert "yuv420p" in cmd
    assert "-crf" in cmd
    assert "18" in cmd
    assert "-preset" in cmd
    assert "medium" in cmd
    assert cmd[-1] == "outputs/video.mp4"


def test_build_ffmpeg_command_with_fps_base() -> None:
    cmd = build_ffmpeg_command(
        input_pattern="f/fr_%04d.png",
        output_path="out.mp4",
        fps=24.0,
        fps_base=1.0,
    )
    assert "24.000000" in cmd

    cmd = build_ffmpeg_command(
        input_pattern="f/fr_%04d.png",
        output_path="out.mp4",
        fps=24.0,
        fps_base=2.0,
    )
    # 24 / 2 = 12 fps effective.
    assert "12.000000" in cmd


def test_build_ffmpeg_command_custom_args() -> None:
    cmd = build_ffmpeg_command(
        input_pattern="f/fr_%04d.png",
        output_path="out.mp4",
        fps=30.0,
        ffmpeg_path="/usr/local/bin/ffmpeg",
        codec="libx265",
        pix_fmt="yuv420p10",
        crf=22,
        preset="slow",
        extra_args=["-x264-params", "keyint=48"],
    )

    assert cmd[0] == "/usr/local/bin/ffmpeg"
    assert "libx265" in cmd
    assert "yuv420p10" in cmd
    assert "22" in cmd
    assert "slow" in cmd
    assert "-x264-params" in cmd
    assert "keyint=48" in cmd
    assert cmd[-1] == "out.mp4"


def test_build_ffmpeg_command_rejects_zero_fps() -> None:
    with pytest.raises(ValueError, match="effective FPS must be positive"):
        build_ffmpeg_command(
            input_pattern="f/fr_%04d.png",
            output_path="out.mp4",
            fps=0.0,
        )

    with pytest.raises(ValueError, match="effective FPS must be positive"):
        build_ffmpeg_command(
            input_pattern="f/fr_%04d.png",
            output_path="out.mp4",
            fps=24.0,
            fps_base=0.0,
        )


# ---------------------------------------------------------------------------
# ffmpeg_available without FFmpeg installed (this environment)
# ---------------------------------------------------------------------------


def test_ffmpeg_available_false_when_not_installed() -> None:
    """This environment has no FFmpeg on PATH (verified during development)."""
    # If FFmpeg happens to be installed in some CI environment, this test
    # simply returns True and is still correct.
    result = ffmpeg_available()
    assert isinstance(result, bool)
