"""Choose descriptive output paths without overwriting earlier Blender artifacts.

Filename reservation is designed for sequential local runs. Concurrent
processes should use separate output folders.
"""

from __future__ import annotations

from pathlib import Path


def default_blend_output(dataset: str | Path, fps: float, *, kind: str) -> Path:
    """Build a descriptive .blend filename from dataset, purpose and FPS."""
    name = Path(dataset).name
    for suffix in (".meta.json", ".data.npz"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    fps_label = f"{fps:g}".replace(".", "p")
    return Path("outputs") / f"{name}_{kind}_{fps_label}fps.blend"


def _next_available_output_path(requested: str | Path, extension: str) -> Path:
    """Return a free output path, adding _001, _002, ... after collisions."""
    path = Path(requested).expanduser().resolve()
    if path.suffix.lower() != extension:
        raise ValueError(f"output path must end in {extension}")
    if not path.exists():
        return path

    for number in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{number:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"no available output filename near {path}")


def next_available_blend_path(requested: str | Path) -> Path:
    """Choose a free .blend filename without overwriting a prior scene."""
    return _next_available_output_path(requested, ".blend")


def next_available_png_path(requested: str | Path) -> Path:
    """Choose a free .png filename without overwriting a prior still render."""
    return _next_available_output_path(requested, ".png")
