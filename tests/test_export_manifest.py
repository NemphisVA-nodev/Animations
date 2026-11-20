"""Pure-Python tests for the render manifest module.

No Blender or FFmpeg is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.exporting.manifest import ManifestError, RenderManifest


def _make_manifest(
    tmp_path: Path,
    frame_start: int = 1,
    frame_end: int = 10,
    completed: list[int] | None = None,
    fps: float = 24.0,
    fps_base: float = 1.0,
) -> Path:
    """Create and save a manifest, return its path."""
    manifest = RenderManifest(
        output_dir=str(tmp_path / "frames"),
        blend_path=str(tmp_path / "scene.blend"),
        fps=fps,
        fps_base=fps_base,
        resolution_x=960,
        resolution_y=540,
        resolution_percentage=100,
        frame_start=frame_start,
        frame_end=frame_end,
        completed_frames=completed or [],
    )
    path = tmp_path / "render_manifest.json"
    manifest.save(path)
    return path


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_manifest_save_and_load_round_trip(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, completed=[1, 3, 5])
    loaded = RenderManifest.load(path)

    assert loaded.output_dir == str(tmp_path / "frames")
    assert loaded.blend_path == str(tmp_path / "scene.blend")
    assert loaded.fps == 24.0
    assert loaded.fps_base == 1.0
    assert loaded.resolution_x == 960
    assert loaded.resolution_y == 540
    assert loaded.frame_start == 1
    assert loaded.frame_end == 10
    assert loaded.completed_frames == [1, 3, 5]
    assert loaded.created_at
    assert loaded.updated_at


def test_manifest_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="manifest not found"):
        RenderManifest.load(tmp_path / "nonexistent.json")


def test_manifest_load_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json ", encoding="utf-8")
    with pytest.raises(ManifestError, match="invalid manifest JSON"):
        RenderManifest.load(bad)


# ---------------------------------------------------------------------------
# Completed frame tracking
# ---------------------------------------------------------------------------


def test_is_complete_and_missing_frames(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, frame_start=1, frame_end=5, completed=[1, 3])
    manifest = RenderManifest.load(path)

    assert manifest.is_complete(1)
    assert manifest.is_complete(3)
    assert not manifest.is_complete(2)
    assert not manifest.is_complete(4)
    assert not manifest.is_complete(5)

    missing = manifest.missing_frames(1, 5)
    assert missing == [2, 4, 5]


def test_mark_complete_adds_and_sorts(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, frame_start=1, frame_end=5, completed=[3])
    manifest = RenderManifest.load(path)

    manifest.mark_complete(1, "frames/frame_0001.png")
    manifest.mark_complete(5, "frames/frame_0005.png")
    manifest.mark_complete(3, "frames/frame_0003.png")  # already present

    assert manifest.completed_frames == [1, 3, 5]
    assert manifest.is_complete(5)


# ---------------------------------------------------------------------------
# Config compatibility
# ---------------------------------------------------------------------------


def test_config_matches_when_identical(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, fps=24.0, fps_base=1.0)
    manifest = RenderManifest.load(path)

    assert manifest.config_matches(
        blend_path=str(tmp_path / "scene.blend"),
        fps=24.0,
        fps_base=1.0,
        resolution_x=960,
        resolution_y=540,
        resolution_percentage=100,
    )


def test_config_mismatch_detected(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, fps=24.0, fps_base=1.0)
    manifest = RenderManifest.load(path)

    # Different fps.
    assert not manifest.config_matches(
        blend_path=str(tmp_path / "scene.blend"),
        fps=30.0,
        fps_base=1.0,
        resolution_x=960,
        resolution_y=540,
        resolution_percentage=100,
    )

    # Different resolution.
    assert not manifest.config_matches(
        blend_path=str(tmp_path / "scene.blend"),
        fps=24.0,
        fps_base=1.0,
        resolution_x=1920,
        resolution_y=1080,
        resolution_percentage=100,
    )

    # Different blend path.
    assert not manifest.config_matches(
        blend_path=str(tmp_path / "other.blend"),
        fps=24.0,
        fps_base=1.0,
        resolution_x=960,
        resolution_y=540,
        resolution_percentage=100,
    )


def test_config_matches_with_fps_base(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, fps=24.0, fps_base=1.0)
    manifest = RenderManifest.load(path)

    # 24/1 == 48/2 as effective FPS but the stored values differ.
    assert not manifest.config_matches(
        blend_path=str(tmp_path / "scene.blend"),
        fps=48.0,
        fps_base=2.0,
        resolution_x=960,
        resolution_y=540,
        resolution_percentage=100,
    )

    assert manifest.config_matches(
        blend_path=str(tmp_path / "scene.blend"),
        fps=24.0,
        fps_base=1.0,
        resolution_x=960,
        resolution_y=540,
        resolution_percentage=100,
    )


# ---------------------------------------------------------------------------
# JSON structure
# ---------------------------------------------------------------------------


def test_manifest_json_contains_all_fields(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, completed=[1, 2])
    data = json.loads(path.read_text(encoding="utf-8"))

    expected_keys = {
        "output_dir", "blend_path", "fps", "fps_base",
        "resolution_x", "resolution_y", "resolution_percentage",
        "frame_start", "frame_end", "completed_frames",
        "blend_sha256", "stem", "frame_checksums",
        "created_at", "updated_at",
    }
    assert set(data.keys()) == expected_keys
    assert data["completed_frames"] == [1, 2]
    assert isinstance(data["created_at"], str)
    assert isinstance(data["updated_at"], str)


def test_manifest_detects_modified_blend_fingerprint(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path)
    manifest = RenderManifest.load(path)
    manifest.blend_sha256 = "abc"
    assert manifest.config_matches(
        blend_path=tmp_path / "scene.blend", blend_sha256="abc",
        stem="frame", output_dir=tmp_path / "frames",
        fps=24, fps_base=1, resolution_x=960, resolution_y=540,
        resolution_percentage=100,
    )
    assert not manifest.config_matches(
        blend_path=tmp_path / "scene.blend", blend_sha256="def",
        stem="frame", output_dir=tmp_path / "frames",
        fps=24, fps_base=1, resolution_x=960, resolution_y=540,
        resolution_percentage=100,
    )


def test_manifest_verifies_png_checksum_and_missing_frame(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, completed=[])
    manifest = RenderManifest.load(path)
    directory = tmp_path / "frames"
    directory.mkdir()
    image = directory / "frame_0001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"image-data")
    manifest.record_frame_checksum(1, image)
    assert manifest.verified_frames(directory) == {1}
    image.unlink()
    assert manifest.verified_frames(directory) == set()


def test_manifest_refuses_modified_png(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, completed=[])
    manifest = RenderManifest.load(path)
    directory = tmp_path / "frames"
    directory.mkdir()
    image = directory / "frame_0001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"image-data")
    manifest.record_frame_checksum(1, image)
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"other-data")
    with pytest.raises(ManifestError, match="differs"):
        manifest.verified_frames(directory)


def test_manifest_save_is_atomic(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path)
    reloaded = RenderManifest.load(path)
    reloaded.completed_frames = [1, 2]
    reloaded.save(path)
    assert RenderManifest.load(path).completed_frames == [1, 2]
    assert not list(tmp_path.glob(".render_manifest.json.*.tmp"))
