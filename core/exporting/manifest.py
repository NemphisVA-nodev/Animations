"""Render manifest for frame-sequence rendering with resume support.

A manifest is a JSON sidecar stored next to a frame output directory. It
records the render configuration (FPS, resolution, frame range, source .blend
path) and the set of successfully completed frames. On resume, the manifest is
validated against the current render request; if the configuration matches,
rendering continues from the first missing frame. If it does not match, a new
manifest is started (or an error is raised, depending on caller preference).

The manifest never modifies rendered frames; it only records which ones exist
and were produced by a compatible configuration.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ManifestError(Exception):
    """Raised when a manifest is invalid, incompatible, or cannot be loaded."""


@dataclass
class RenderManifest:
    """State record for a frame-sequence render.

    Attributes
    ----------
    output_dir:
        Directory containing rendered PNG frames.
    blend_path:
        Path to the source .blend file.
    fps:
        Render frame rate (numerator).
    fps_base:
        Render frame rate denominator (Blender stores effective FPS as
        fps / fps_base).
    resolution_x:
        Render width in pixels.
    resolution_y:
        Render height in pixels.
    resolution_percentage:
        Render resolution percentage (1-100).
    frame_start:
        First frame to render (inclusive).
    frame_end:
        Last frame to render (inclusive).
    completed_frames:
        Sorted list of frame numbers known to be fully rendered.
    created_at:
        ISO-8601 timestamp when the manifest was first created.
    updated_at:
        ISO-8601 timestamp when the manifest was last modified.
    """

    output_dir: str
    blend_path: str
    fps: float
    fps_base: float
    resolution_x: int
    resolution_y: int
    resolution_percentage: int
    frame_start: int
    frame_end: int
    completed_frames: list[int] = field(default_factory=list)
    # These identify the exact scene and file naming contract for safe resume.
    blend_sha256: str = ""
    stem: str = "frame"
    # Completed frame numbers are verified against their actual PNG bytes.
    frame_checksums: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "output_dir": self.output_dir,
            "blend_path": self.blend_path,
            "fps": self.fps,
            "fps_base": self.fps_base,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y,
            "resolution_percentage": self.resolution_percentage,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "completed_frames": sorted(self.completed_frames),
            "blend_sha256": self.blend_sha256,
            "stem": self.stem,
            "frame_checksums": dict(sorted(self.frame_checksums.items())),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderManifest":
        """Create a Manifest from a dictionary (e.g. loaded from JSON)."""
        return cls(
            output_dir=str(data["output_dir"]),
            blend_path=str(data["blend_path"]),
            fps=float(data["fps"]),
            fps_base=float(data["fps_base"]),
            resolution_x=int(data["resolution_x"]),
            resolution_y=int(data["resolution_y"]),
            resolution_percentage=int(data["resolution_percentage"]),
            frame_start=int(data["frame_start"]),
            frame_end=int(data["frame_end"]),
            completed_frames=[int(f) for f in data.get("completed_frames", [])],
            blend_sha256=str(data.get("blend_sha256", "")),
            stem=str(data.get("stem", "frame")),
            frame_checksums={
                str(k): str(v) for k, v in data.get("frame_checksums", {}).items()
            },
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )

    def save(self, path: str | Path) -> None:
        """Write the manifest as JSON to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import datetime

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now
        payload = self.to_dict()
        text = json.dumps(payload, indent=2)
        # Atomic replacement avoids truncating the only resume record on crash.
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "RenderManifest":
        """Load a manifest from a JSON file.

        Raises
        ------
        ManifestError
            If the file does not exist or contains invalid JSON.
        """
        path = Path(path)
        if not path.exists():
            raise ManifestError(f"manifest not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"invalid manifest JSON in {path}: {exc}") from exc
        try:
            if not isinstance(data, dict):
                raise ValueError("manifest root must be a JSON object")
            return cls.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"invalid manifest structure in {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Compatibility and resume logic
    # ------------------------------------------------------------------
    def config_matches(
        self,
        *,
        blend_path: str | Path,
        fps: float,
        fps_base: float,
        resolution_x: int,
        resolution_y: int,
        resolution_percentage: int,
        blend_sha256: str | None = None,
        stem: str | None = None,
        output_dir: str | Path | None = None,
    ) -> bool:
        """Return True if this manifest's config matches the given request.

        The output directory, completed frames, and frame range are not part
        of the config comparison; only the render parameters that affect
        pixel output are compared. This allows resuming a render with a
        different frame range (e.g. extending the range or rendering a subset
        of remaining frames).
        """
        return (
            str(Path(blend_path).resolve()) == str(Path(self.blend_path).resolve())
            and float(fps) == self.fps
            and float(fps_base) == self.fps_base
            and int(resolution_x) == self.resolution_x
            and int(resolution_y) == self.resolution_y
            and int(resolution_percentage) == self.resolution_percentage
            and (blend_sha256 is None or self.blend_sha256 == blend_sha256)
            and (stem is None or self.stem == stem)
            and (
                output_dir is None
                or Path(self.output_dir).resolve() == Path(output_dir).resolve()
            )
        )

    def is_complete(self, frame_number: int) -> bool:
        """Return True if ``frame_number`` is recorded as completed."""
        return int(frame_number) in set(self.completed_frames)

    def missing_frames(
        self, frame_start: int, frame_end: int
    ) -> list[int]:
        """Return frame numbers in [frame_start, frame_end] not yet completed."""
        completed = set(self.completed_frames)
        return [
            f
            for f in range(int(frame_start), int(frame_end) + 1)
            if f not in completed
        ]

    def mark_complete(self, frame_number: int, output_file: str | Path) -> None:
        """Record a frame as completed.

        Parameters
        ----------
        frame_number:
            The Blender frame number that was rendered.
        output_file:
            Path to the rendered output file (recorded for reference but not
            validated on load).
        """
        frame_number = int(frame_number)
        if frame_number not in self.completed_frames:
            self.completed_frames.append(frame_number)
            self.completed_frames.sort()


    def verified_frames(self, frame_dir: str | Path) -> set[int]:
        """Verify claimed completed frames and return those still present.

        A deleted frame is safe to re-render. An existing frame with missing
        checksum, an empty file, or mismatching bytes is *not* safe to silently
        reuse or overwrite; report the problem instead.
        """
        import hashlib

        folder = Path(frame_dir)
        verified: set[int] = set()
        for frame in self.completed_frames:
            candidate = folder / f"{self.stem}_{frame:04d}.png"
            if not candidate.exists():
                continue
            expected = self.frame_checksums.get(str(frame))
            if not expected:
                raise ManifestError(
                    f"frame {frame} has no checksum; cannot safely resume. "
                    "Use a new output directory."
                )
            if candidate.stat().st_size <= 8:
                raise ManifestError(
                    f"frame {frame} is empty or truncated: {candidate}"
                )
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if actual != expected:
                raise ManifestError(
                    f"frame {frame} differs from the recorded PNG: {candidate}. "
                    "Use a new output directory."
                )
            verified.add(frame)
        return verified

    def record_frame_checksum(self, frame_number: int, output_file: str | Path) -> None:
        """Record a completed frame's exact SHA-256, alongside its frame number."""
        import hashlib

        path = Path(output_file)
        if not path.is_file() or path.stat().st_size <= 8:
            raise ManifestError(f"rendered frame is missing or empty: {path}")
        self.frame_checksums[str(int(frame_number))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        self.mark_complete(frame_number, path)
