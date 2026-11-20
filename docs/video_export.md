# Step 5C.3 — Resumable Blender frame rendering and MP4 export

This exporter uses the **existing** cinematic Blender scene; it never
recomputes the Newtonian simulation or modifies particle trajectories.
Black-hole rings, materials, lighting, and camera remain **schematic**.

## Separate worktrees

Keep the worktrees separate until integration and tests pass.
The exporter review lives on `review/video-export-safety`, based on
`feat/video-export-pipeline`. No exporter changes have yet been merged into
`main` at the time this review document was authored.

## Render a small preview (PowerShell, Windows)

Run the following in your Animations worktree. Choose an existing .blend
rendered from the latest cinematic script:

```powershell
$blender = "C:\Users\BVBITNB10Test\Desktop\Blender\blender.exe"
$scene = "outputs/preview_radial_infall_radial_infall_cinematic_24fps_008.blend"
$frames = "outputs/preview_frames_1to5"

& $blender --background --python-exit-code 1 --python scripts/render_frames.py -- --blend $scene --output-dir $frames --frame-start 1 --frame-end 5 --fps 24 --resolution-x 960 --resolution-y 540
```

Check the output for `frame_0001.png` to `frame_0005.png`,
plus `render_manifest.json`. The engine keeps `fps / fps_base` accurate.

To resume or extend the same exact source/render settings, re-run with the
**same frame directory**, for example `--frame-end 10`. Completed files
are checked against their SHA-256 manifest entries and are skipped;
deleted frames are rendered again. A modified scene, corrupted image or
different rendering configuration is rejected.

**Do not use `--no-resume` in an existing frame directory.** This option now
requires a new frame directory to avoid overwriting valuable renders.

## Export MP4 after the full sequence exists

```powershell
uv run python scripts/export_video.py --check-ffmpeg

uv run python scripts/export_video.py --frame-dir $frames --output "outputs/cinematic_preview.mp4"
```

The command uses the render manifest's FPS, FPS base and filename stem.
It checks the frame range and verifies PNG checksums before encoding.
FFmpeg must be installed or supplied via `--ffmpeg-path`; it is not
installed automatically. H.264, `yuv420p` and CRF 18 are the defaults.
Existing MP4s get a numbered suffix; FFmpeg refuses to overwrite them.

If FFmpeg is unavailable, you can still render PNG frames but not MP4s
until it is provided.

## Verification before merge

```powershell
uv run pytest
& $blender --background --python-exit-code 1 --python blender/tests/test_export_integration.py
```

A real FFmpeg end-to-end encode test and watching the rendered MP4 are
still necessary before declaring the pipeline presentation-ready.
