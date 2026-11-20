"""Exporting system: saving frames and encoding video output.

This package provides:

* :mod:`core.exporting.manifest` -- render manifest for frame-sequence
  rendering with resume support.
* :mod:`core.exporting.video` -- FFmpeg detection and H.264 MP4 export.

Blender-specific frame rendering lives in :mod:`scripts.render_frames`.
"""

from core.exporting.manifest import RenderManifest, ManifestError
from core.exporting.video import (
    FFmpegNotFoundError,
    find_ffmpeg,
    ffmpeg_available,
    build_ffmpeg_command,
    export_to_mp4,
)

__all__ = [
    "RenderManifest",
    "ManifestError",
    "FFmpegNotFoundError",
    "find_ffmpeg",
    "ffmpeg_available",
    "build_ffmpeg_command",
    "export_to_mp4",
]
