# Animations

A modular Python framework for developing scientifically accurate, visually appealing physics animations for educational presentations.

## Project goals

- Provide a clean, extensible foundation for physics-based animations.
- Separate physics calculation, rendering, and exporting into independent layers.
- Pre-render animations into video files (MP4/H.264) suitable for Microsoft PowerPoint.
- Support reproducible, deterministic simulations using SI units.
- Keep physics modules testable without any rendering dependency.

## Planned animation topics

- Spaghettification near a black hole (first planned animation)
- Gravitational lensing
- Orbital mechanics
- Chaotic systems
- Other scientific topics as needed

## Technology stack

- Python 3.15+
- [uv](https://github.com/astral-sh/uv) for environment and dependency management
- NumPy — numerical calculations
- SciPy — scientific computing and integration
- Matplotlib — rendering of animation frames
- Pillow — image handling
- pytest — testing
- FFmpeg (optional, external) — video encoding (detected, not installed automatically)

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv). No global Python packages are installed and no administrator privileges are required.

```bash
# Clone the repository
git clone https://github.com/NemphisVA-nodev/Animations.git
cd Animations

# Create and synchronize the local .venv environment
uv sync
```

This installs the runtime dependencies (NumPy, SciPy, Matplotlib, Pillow) and the development dependency (pytest) into the project-local `.venv`.

## Running tests

```bash
uv run pytest
```

## Folder structure

```text
Animations/
├── animations/            # Animation modules (spaghettification, lensing, orbits, chaos)
├── core/                  # Core framework (physics, rendering, exporting)
├── assets/                # Static assets (placeholder, kept via .gitkeep)
├── outputs/               # Generated frames and videos (ignored by Git, .gitkeep preserved)
├── tests/                 # pytest test suite
├── docs/                  # Documentation
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── pyproject.toml         # Project metadata and dependencies
└── uv.lock                # Locked dependency versions
```

## Current development status

**Presentation-driven simulation foundation.** The development environment,
project architecture, package structure, and test suite are in place. The
following Python-side (no Blender) foundation has been implemented and tested:

- Development environment: `uv`-managed local `.venv`, NumPy, SciPy, Matplotlib,
  Pillow, and pytest (including physics, dataset and playback regression tests).
- Package structure: `core/` (physics, rendering, exporting, simulation),
  `animations/` (spaghettification, lensing, orbits, chaos), `tests/`, `docs/`,
  `assets/`, `outputs/`, `scripts/`.
- Newtonian tidal acceleration physics (`core/physics/tidal.py`) in SI units,
  frame-rate independent.
- Black hole model (`core/physics/black_hole.py`): mass, position, Schwarzschild
  radius `r_s = 2GM/c^2`, vector gravitational acceleration, configurable
  integration cutoff outside `r_s`.
- Test-particle model (`core.physics.particles`): deterministic clouds (radial,
  transverse, grid, ring) with validation.
- Numerical integration (`core.physics.integration`): SciPy `solve_ivp` (DOP853),
  event-based cutoff termination, per-particle validity, reproducible output.
- Tidal deformation measurements (`core.physics.measurements`).
- Renderer-independent simulation data contracts: JSON metadata + compressed
  NPZ arrays, versioned schema `1.0.0`, extended for multi-particle
  trajectories with an explicit validity mask (readable with `json` + `numpy`).
- Presentation timeline (`core/timeline.py`) and transition
  (`core/transitions.py`) specifications.
- Renderer-independent frame mapping and masked trajectory sampling
  (`core/playback.py`), documented in `docs/simulation_to_frame_mapping.md`.
- Deterministic example dataset generators:
  `scripts/generate_tidal_dataset.py` and
  `scripts/generate_black_hole_trajectories.py` (radial-infall + tidal-cloud
  presets).
- Documentation: storyboard, data format, transition architecture, black hole
  simulation, and spaghettification physics.

**Not yet implemented:** Blender integration, the rendering engine, exporting/FFmpeg video
encoding, graphical user interface, animation demos, and Blender integration.
No physics simulations are rendered to video yet. See `docs/` for the planned
architecture, storyboard, data format, and transition design.

## Copyright and licensing

Copyright © 2026 Fabian Göken. All rights reserved.

This project is proprietary. No permission is granted to copy, modify, redistribute, or reuse the original source code or assets without prior written permission, except as permitted by law or GitHub's Terms of Service. See [LICENSE](LICENSE) for details.

**Previous versions:** This project was previously published under the MIT License. Rights validly granted for those earlier versions are not revoked by this change. Third-party dependencies remain subject to their own licenses.
