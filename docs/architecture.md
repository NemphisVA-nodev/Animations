# Architecture

## Separation of responsibilities

The framework separates three distinct responsibilities:

### Physics engine (`core/physics`)

Responsible for numerical calculations, physical equations, simulation state, time integration, physical constants, and units.

- Uses SI units for physical calculations unless explicitly documented otherwise.
- Simulation calculations are independent of animation frame rate.
- Deterministic simulations support reproducible results.
- Must remain independent of rendering. Physics modules must be testable without importing any rendering code.

### Rendering engine (`core/rendering`)

Responsible for converting simulation data into animation frames, visual appearance, camera positioning, graphical effects, frame resolution, and frame rate.

- Consumes simulation data produced by the physics engine rather than calculating physics itself.
- Supports configurable resolution and FPS.

### Exporting system (`core/exporting`)

Responsible for saving rendered frames, video encoding, export configuration, and output file management.

- Intended primary video format: MP4 using H.264, suitable for Microsoft PowerPoint.
- FFmpeg may be used as an external dependency for video encoding. The code detects whether FFmpeg is installed but does not automatically install it and does not require administrator access.

## Planned simulation-to-video workflow

1. The physics engine advances the simulation state in SI units, producing time-stamped simulation data independent of frame rate.
2. The rendering engine consumes that simulation data and draws each animation frame at the configured resolution and FPS.
3. The exporting system saves the rendered frames and encodes them into an output video file (MP4/H.264 via FFmpeg when available).

## Why physics calculations should not depend on FPS

Simulation time integration must advance by a fixed physical time step (e.g. seconds), not by animation frames. Decoupling physics from FPS means:

- Changing the output frame rate does not change the simulated physics.
- Simulations remain deterministic and reproducible across different render settings.
- The physics engine can be tested and run headless without any rendering.

## How new animations should be added

1. Create a new package under `animations/<topic>/` with an `__init__.py`.
2. Implement the physics in `core/physics` (or animation-specific physics modules) using SI units and frame-rate-independent time integration.
3. Implement the renderer in `core/rendering` that consumes the simulation data.
4. Implement export configuration in `core/exporting` as needed.
5. Add tests under `tests/` that verify imports and behavior without rendering.

## Scientific accuracy versus cinematic visualization

The physics engine prioritizes scientific accuracy using SI units and established physical equations. The rendering engine is responsible for cinematic presentation (camera, visual effects, appearance) and may apply visualization choices that aid understanding, but it must not alter the underlying simulation data. Where artistic choices affect scientific accuracy, they must be explicitly documented.

## Why pre-rendering is preferred for presentations

Animations are pre-rendered into video files (MP4/H.264) rather than rendered live in PowerPoint because:

- Pre-rendered video plays reliably on any machine without requiring Python or the framework to be installed.
- Playback performance is consistent and independent of the host hardware.
- H.264 MP4 is broadly supported by Microsoft PowerPoint.
- Complex simulations can be computed once and reviewed many times.

## Status

The physics engine is partially implemented: Newtonian tidal acceleration
(`core/physics/tidal.py`), an idealized black hole model
(`core/physics/black_hole.py`), test-particle clouds
(`core/physics/particles.py`), numerical integration with event-based cutoff
(`core/physics/integration.py`), and tidal measurements
(`core.physics.measurements`) are implemented and tested. The rendering engine
and exporting/FFmpeg video encoding are not yet implemented. No physics
simulations are rendered to video yet. No claim is made that any of the
described cinematic functionality already works.
