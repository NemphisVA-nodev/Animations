"""Choose descriptive Blender output names without silently replacing earlier files.

These helpers use only the standard library, so normal pytest can cover them
without starting Blender. They avoid collisions for sequential local runs;
simultaneous writers to the same directory should use separate output folders.
"""

from __future__ import annotations

from pathlib import Path


def default_blend_output(dataset: str | Path, fps: float, *, kind: str) -> Path:
    """Build a descriptive output name from the dataset, purpose and frame rate."""
    name = Path(dataset).name
    for suffix in (".meta.json", ".data.npz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    fps_label = f"{fps:g}".replace(".", "p")
    return Path("outputs") / f"{name}_{kind}_{fps_label}fps.blend"


def next_available_blend_path(requested: str | Path) -> Path:
    """Use the requested path, or append _001, _002, ... if it already exists.

    Never delete, truncate or overwrite an existing .blend. The resolved
    output path is returned so the caller can print the exact saved filename.
    """
    path = Path(requested).expanduser().resolve()
    if path.suffix.lower() != ".blend":
        raise ValueError("Blender output path must end in .blend")
    if not path.exists():
        return path

    for number in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{number:03d}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise FileExistsError(f"no available Blender output filename near {path}")
