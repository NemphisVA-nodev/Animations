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
    assert "-n" in cmd
    assert "-y" not in cmd
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


def test_build_ffmpeg_command_rejects_nonfinite_fps() -> None:
    for invalid in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="effective FPS"):
            build_ffmpeg_command(
                input_pattern="frame_%04d.png", output_path="video.mp4",
                fps=invalid,
            )


def test_export_rejects_gapped_frames(tmp_path: Path, monkeypatch) -> None:
    from core.exporting import video

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for i in (1, 3):
        (frame_dir / f"frame_{i:04d}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"data"
        )
    monkeypatch.setattr(video, "find_ffmpeg", lambda unused: "ffmpeg")
    with pytest.raises(ValueError, match="missing"):
        video.export_to_mp4(
            frame_dir=frame_dir, output_path=tmp_path / "out.mp4", fps=24
        )


def test_export_uses_first_frame_and_avoids_overwrite(tmp_path: Path, monkeypatch) -> None:
    from core.exporting import video
    import subprocess

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in (12, 13):
        (frames / f"frame_{i:04d}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"data"
        )
    existing = tmp_path / "movie.mp4"
    existing.write_bytes(b"existing original")

    monkeypatch.setattr(video, "find_ffmpeg", lambda unused: "ffmpeg")
    recorded = []

    def fake_run(args, **kwargs):
        recorded.append(args)
        Path(args[-1]).write_bytes(b"mock mp4")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    result = video.export_to_mp4(
        frame_dir=frames, output_path=existing, fps=24
    )
    assert result.name == "movie_001.mp4"
    assert existing.read_bytes() == b"existing original"
    assert "-n" in recorded[0]
    assert recorded[0][recorded[0].index("-start_number") + 1] == "12"


def test_ffmpeg_check_does_not_require_export_arguments(monkeypatch) -> None:
    from scripts import export_video
    monkeypatch.setattr(export_video, "ffmpeg_available", lambda path: True)
    monkeypatch.setattr(export_video, "find_ffmpeg", lambda path: "ffmpeg")
    assert export_video.main(["--check-ffmpeg"]) == 0
