# Physics — Spaghettification

Documentation of what is scientifically calculated, what is an educational
approximation, and what is purely cinematic for the spaghettification
animation. Classification labels: **OBSERVED**, **CALCULATED**, **SCHEMATIC**.

## Newtonian tidal model (CALCULATED)

The Python-side physics engine (`core/physics/tidal.py`) computes, in SI
units and independent of animation frame rate:

- Gravitational acceleration magnitude: `g = G M / r²` (CALCULATED).
- Exact near-minus-far tidal acceleration for an object of length `L`
  centred at distance `r` from a mass `M`:
  `Δa_exact = G M / (r - L/2)² - G M / (r + L/2)²` (CALCULATED).
- First-order inverse-cube approximation for `L << r`:
  `Δa ≈ 2 G M L / r³` (CALCULATED; educational approximation).
- Transverse tidal compression: magnitude `G M L / r³`, opposite sign to
  the radial stretch (CALCULATED; approximation).

`G = 6.67430e-11 m³ kg⁻¹ s⁻²` (CODATA 2022).

All calculations are Newtonian and deterministic.

## What is NOT simulated (limitations)

- **General relativity is not modelled.** The engine does not simulate
  relativistic motion near or inside a black hole event horizon. The
  Schwarzschild radius `r_s = 2 G M / c²` is presented as a reference only
  (Scene 6), with an explicit note that the Newtonian engine is invalid
  near/inside it.
- **Real coastal tides** depend on coastline shape, ocean depth, Earth's
  rotation, and the Sun; the Earth–Moon tidal model (Scene 3) is idealized
  and cannot predict real tides.
- No fluid dynamics, material strength, or structural failure of the
  falling object is computed.

## Calculated vs schematic vs observed

| Aspect | Label | Basis |
|--------|-------|-------|
| Tidal acceleration vs distance (`Δa`) | CALCULATED | Newtonian formulas above, SI units. |
| Particle cloud radial stretch / transverse compression | CALCULATED | Particle positions from the tidal field. |
| Event horizon reference `r_s` | CALCULATED | `2 G M / c²`; Newtonian engine invalid near it. |
| Exaggerated artistic object deformation | SCHEMATIC | Illustrative only; distinct from calculated particle motion. |
| Shoemaker–Levy 9, stellar tidal disruption events | OBSERVED | Real astronomical evidence; no fabrication. |
| Earth–Moon tidal bulges (idealized) | SCHEMATIC / CALCULATED | Qualitative illustration of differential gravity. |

## How the Blender preview uses simulation data

1. Physics is computed once in Python (SI units, frame-rate independent).
2. Results are exported as `*.meta.json` + `*.data.npz` (see
   `docs/simulation_data_format.md`), readable with only `json` + `numpy`.
3. The current Blender adapter loads the validated arrays, converts the
   simulation XY plane to Blender XY with Z = 0, and applies a constant scale.
   Original SI values are retained.
4. The preview keyframes particle positions and termination visibility.
   Cinematic camera movement and visual deformation driven by transitions
   (`docs/transition_architecture.md`) are planned, not yet implemented.

## Scientific accuracy versus cinematic visualization

The physics engine prioritizes scientific accuracy in SI units. The rendering
layer handles visual appearance and may apply artistic choices (e.g. camera
framing, exaggerated stretch for clarity), but it must not alter the
underlying simulation data, and any departure from the calculated values must
be explicitly labelled SCHEMATIC.
