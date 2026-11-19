"""Regression coverage for collision-safe Blender output filenames.

No Blender installation is needed to test path selection.
"""

from pathlib import Path

import pytest

from blender.output_paths import default_blend_output, next_available_blend_path


def test_default_names_preserve_dataset_and_frame_rate() -> None:
    dataset = "outputs/preview_radial_infall_radial_infall"
    fps24 = default_blend_output(dataset, 24, kind="integration")
    fps30 = default_blend_output(dataset, 30, kind="integration")
    assert fps24.name == "preview_radial_infall_radial_infall_integration_24fps.blend"
    assert fps30.name == "preview_radial_infall_radial_infall_integration_30fps.blend"
    assert fps24 != fps30


def test_default_names_distinguish_preview_and_integration() -> None:
    dataset = "outputs/example.meta.json"
    preview = default_blend_output(dataset, 24, kind="preview")
    integration = default_blend_output(dataset, 24, kind="integration")
    assert preview.name == "example_preview_24fps.blend"
    assert integration.name == "example_integration_24fps.blend"


def test_default_names_accept_npz_path_and_fractional_fps() -> None:
    path = default_blend_output("outputs/example.data.npz", 23.976, kind="preview")
    assert path.name == "example_preview_23p976fps.blend"


def test_unused_path_is_selected_unchanged(tmp_path: Path) -> None:
    requested = tmp_path / "scene.blend"
    assert next_available_blend_path(requested) == requested.resolve()


def test_existing_scene_never_overwritten(tmp_path: Path) -> None:
    original = tmp_path / "scene.blend"
    original.write_bytes(b"original animation")

    second = next_available_blend_path(original)
    assert second.name == "scene_001.blend"
    second.write_bytes(b"second animation")

    third = next_available_blend_path(original)
    assert third.name == "scene_002.blend"
    assert original.read_bytes() == b"original animation"
    assert second.read_bytes() == b"second animation"


@pytest.mark.parametrize("filename", ["scene.mp4", "scene", "scene.blend1"])
def test_rejects_non_blend_extensions(tmp_path: Path, filename: str) -> None:
    with pytest.raises(ValueError, match=r"end in \.blend"):
        next_available_blend_path(tmp_path / filename)
