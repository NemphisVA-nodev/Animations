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

- Newtonian tidal acceleration physics (`core/physics/tidal.py`) in SI units,
  frame-rate independent.
- Renderer-independent simulation data contracts: JSON metadata + compressed
  NPZ arrays with versioned schema and validation (`core/simulation/`).
- Presentation timeline model distinguishing simulation time, playback time,
  and presentation time (`core/timeline.py`).
- Transition specification for scene-to-scene visual alignment
  (`core/transitions.py`).
- Deterministic example tidal-acceleration dataset generator
  (`scripts/generate_tidal_dataset.py`) with output in `outputs/`.
- Test suite covering physics, data format, timeline, and transitions.

**Not yet implemented:** the rendering engine, exporting/FFmpeg video
encoding, graphical user interface, animation demos, and Blender integration.
No physics simulations are rendered to video yet. See `docs/` for the planned
architecture, storyboard, data format, and transition design.
