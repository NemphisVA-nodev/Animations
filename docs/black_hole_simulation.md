# Black Hole Simulation

## Purpose

This document describes the Python-side Newtonian test-particle simulation used
to study black hole spaghettification. It simulates independent, freely falling
test particles around an idealized, nonrotating black hole and exports the
trajectories in a renderer-independent format that a future Blender adapter can
consume.

The simulation demonstrates the central presentation question:

**"How can the same physics that moves Earth's oceans tear apart a star?"**

by showing that tidal acceleration — the difference in gravitational pull
between the near and far ends of an object — grows rapidly as the object
approaches a compact mass, eventually overwhelming any material strength.

## Newtonian equations of motion

The black hole is modelled as a point mass ``M`` at position ``r_BH``. A test
particle at position ``r`` experiences the Newtonian gravitational acceleration

``a(r) = -G M (r - r_BH) / |r - r_BH|^3``

where ``G`` is the gravitational constant. The equations of motion for each
independent particle are

``d^2 r / dt^2 = a(r)``

integrated in time with SciPy ``solve_ivp`` (DOP853 method by default).

All quantities use SI units: metres (m), seconds (s), kilograms (kg),
metres per second squared (m/s^2).

## The Schwarzschild radius

The Schwarzschild radius is

``r_s = 2 G M / c^2``

It is used as:

1. **Metadata** documenting the scale at which Newtonian gravity breaks down.
2. **A simulation boundary:** the integration cutoff radius must be strictly
   greater than ``r_s``.

The Schwarzschild radius is **not** a relativistic correction to the Newtonian
acceleration. This simulator does not model general relativistic motion near or
inside the event horizon, and trajectories are never simulated at or inside
``r_s``.

## Initial conditions

Each test particle is defined by:

- A stable identifier.
- An initial Cartesian position ``(x, y)`` in metres.
- An initial Cartesian velocity ``(vx, vy)`` in m/s.

The `core.physics.particles` module provides deterministic cloud initializers:

- **radial_particles:** particles aligned along the inward radial line.
- **transverse_particles:** particles aligned along the transverse axis.
- **grid_particles:** a centred rectangular grid.
- **ring_particles:** particles uniformly distributed on a ring.

All particles in a cloud share a common starting velocity. Ordering is
deterministic, so the same inputs always produce the same particle ordering.

## Numerical integration

Each particle is integrated independently with SciPy ``solve_ivp`` using the
DOP853 Runge-Kutta method. The state vector is
``[x, y, vx, vy]`` and the right-hand side is

``d/dt [x, y, vx, vy] = [vx, vy, a_x, a_y]``.

Integration is independent of animation frame rate: the solver advances in
physical simulation time (seconds), not frames.

### Tolerances

Configurable absolute (``atol``) and relative (``rtol``) solver tolerances
control accuracy. Defaults are ``rtol=1e-9`` and ``atol=1e-12``.

### Cutoff handling (event-based)

Particles stop being simulated when they reach the configured cutoff radius.
The cutoff is handled with **event detection**: SciPy monitors
``|r - r_BH| - cutoff_radius`` and terminates the integration the instant a
particle crosses the boundary (direction: inward, terminal). This is more
accurate than checking the boundary only after a full integration step.

Particles reaching the cutoff at different times are handled correctly: each
particle has its own termination time. Results are interpolated onto a common
time grid, and an explicit ``validity_mask`` marks which particles are still
being simulated at each timestamp. Terminated particles have NaN entries after
termination; no data is fabricated after cutoff.

No gravitational softening is added, so trajectories approaching the black hole
centre remain singular and are bounded by the cutoff.

## Tidal stretching and compression

The tidal field stretches particles radially (along the line to the black hole)
and compresses them transversely (perpendicular to that line).

- **Radial stretching:** the near end of an object is pulled more strongly
  than the far end, increasing radial separation.
- **Transverse compression:** off-axis particles are pulled toward the radial
  line through the cloud centre.

The `core.physics.measurements` module computes reusable quantities from a
trajectory result: geometric centre, reference-particle trajectory, radial and
transverse extents, mean pairwise separation, and Newtonian tidal acceleration
estimates (``2 G M L / r^3`` radial, ``-G M L / r^3`` transverse).

## Independent particles vs real deformation

The simulated particles are **independent, freely falling test particles** with
no springs, elasticity, collisions, self-gravity, or material bonding. They do
not represent a physically connected star or human body. Consequently:

- The measured "stretching" is **relative test-particle motion** (differential
  acceleration producing relative displacement), not actual material
  deformation.
- Real tidal disruption of a star requires fluid dynamics and material
  strength, which are not modelled here.
- Any exaggerated visual deformation in a future animation is cinematic
  (SCHEMATIC) and must be labelled as distinct from the calculated particle
  motion.

## Dataset structure

Simulation output follows the versioned contract in
`docs/simulation_data_format.md` (schema `1.0.0`), extended for trajectories:

- `<stem>.meta.json` — JSON metadata.
- `<stem>.data.npz` — compressed NumPy array archive.

Trajectory arrays use the ``nonfinite_policy = "masked"`` convention:

| Array | Shape | Units | Notes |
|-------|-------|-------|-------|
| `timestamps` | ``(T,)`` | s | Strictly increasing, always finite. |
| `positions` | ``(T, N, 2)`` | m | NaN where validity_mask is False. |
| `velocities` | ``(T, N, 2)`` | m/s | NaN where validity_mask is False. |
| `validity_mask` | ``(T, N)`` | — | True while the particle is simulated. |
| `particle_ids` | ``(N,)`` | — | Fixed-width string identifiers. |

Where ``T`` is the number of timestamps and ``N`` the number of particles.
Particles that terminate at different times are marked invalid after their
termination time.

Metadata describes the black hole mass, Schwarzschild radius, integration
cutoff, numerical method, solver tolerances, initial conditions, units,
reference frame, model assumptions, scientific limitations, and per-particle
termination status.

## Generating example datasets

Use the CLI script:

```bash
# Scenario A: radial infall (3 particles fall toward the black hole)
uv run python scripts/generate_black_hole_trajectories.py radial-infall \
    --output outputs/bh_radial_example

# Scenario B: tidal particle cloud (5x5 grid shows stretch + compression)
uv run python scripts/generate_black_hole_trajectories.py tidal-cloud \
    --output outputs/bh_tidal_example
```

Common options: ``--mass``, ``--cutoff``, ``--initial-radius``, ``--duration``,
``--steps``, ``--length``, ``--output``.

Scenario A also accepts ``--spacing``; Scenario B accepts ``--nx``, ``--ny``,
``--spacing``.

Both produce `<stem>.meta.json` and `<stem>.data.npz` in the `outputs/`
directory (gitignored generated data).

## Consumption by a future Blender adapter

The exported files require only the Python standard library and NumPy to read,
so Blender's bundled Python can load them without installing the full
scientific dependency set:

```python
import json
import numpy as np
from pathlib import Path

stem = Path("outputs/bh_radial_example_radial_infall")
meta = json.loads(stem.with_suffix(".meta.json").read_text())
data = np.load(stem.with_suffix(".data.npz"))
arrays = {name: data[name] for name in data.files}
```

A Blender adapter would:

1. Load `positions` (``(T, N, 2)`` m), `velocities`, and `validity_mask`.
2. Apply a visual coordinate transform (scale metres to scene units, re-centre
   on the falling object, map axes to Blender's Z-up convention).
3. Use `validity_mask` to hide particles after termination.
4. Drive camera motion via the timeline/transition specification
   (`docs/transition_architecture.md`).

Original SI values are retained as the source of truth; the transform is
purely visual.

## Scientific limitations

- Newtonian point-mass gravity only; no general relativity near/inside the
  Schwarzschild radius.
- Independent test particles only; no fluid dynamics, material strength,
  collisions, self-gravity, or artificial bonding.
- Not a simulation of a real star or human body.
- No relativistic correction to acceleration; the Schwarzschild radius is a
  metadata/boundary value only.
