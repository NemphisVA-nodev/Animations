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

**Initialization phase.** The project architecture and development environment have been set up. No physics simulations, rendering, graphical user interface, or animation demos have been implemented yet. The next step is to implement the physics, rendering, and exporting engines described in `docs/architecture.md`.
