# Step 5C.1 — Cinematic Blender preview

## Goal and scientific boundaries

Step 5B's importer already converts validated Newtonian test-particle
positions into actual Blender keyframes. This first Step 5C increment leaves
that physics and the original `blender/preview_scene.py` intact.

The new `blender/cinematic_scene.py` builds an optional cinematic scene using:

- A dark black-hole marker and three emission-shader rings
- An automatically framed orthographic camera and a simple area light
- Emissive particle materials with restrained sizes
- Trailing curves derived from the same frame-sampled particle positions
- A Blender 5.3 Eevee rendering preset for affordable draft stills

**CALCULATED:** particle positions, timing, validity/termination.

**SCHEMATIC:** ring appearance, particle sizes, glow, trail thickness, camera,
and lighting. These are *not* general relativistic lensing, a realistic
accretion disk, ray tracing, or hydrodynamics. No source trajectories are
changed to make a shot prettier.

## Data flow

```text
.meta.json + .data.npz
        |
        v
load_trajectory()  [validated metre/second data]
        |
        +-----> Animator.animate()  [verified Step 5B particle keyframes]
        |
        +-----> build_frame_mapping() + sample_particle_positions()
                         |
                         v
               metres_to_blender()  [visual scale only]
                         |
                         +------> polyline trails, hidden after termination
                         +------> camera framing over valid samples
        |
        v
schematic emissive rings, world and light
        |
        v
uniquely named .blend; optional PNG still
```

A trail exposes **only current and preceding valid frames**; the camera
helper ignores NaN positions on terminated particles. Its maximum visible
length is configurable in frames. During hold frames, the trail freezes.
We use equal-point-spacing on a Blender POLY curve, so progression aligns
with the sampled frame index, not a fabricated orbital path.

## Local Windows commands

From the repository root, using a previously generated dataset:

```powershell
$blender = "C:\Users\BVBITNB10Test\Desktop\Blender\blender.exe"
$dataset = "outputs/preview_radial_infall_radial_infall"

uv run pytest
& $blender --background --python-exit-code 1 --python blender/tests/test_cinematic_integration.py -- --dataset $dataset --scale 1e-6 --fps 24
Write-Host "Blender exit code: $LASTEXITCODE"

& $blender --background --python-exit-code 1 --python blender/cinematic_scene.py -- --dataset $dataset --scale 1e-6 --fps 24 --quality draft
Write-Host "Blender exit code: $LASTEXITCODE"
```

Open the saved `outputs/*cinematic_24fps.blend` in Blender to inspect the
lighting, placement, timing and trails. Use `--render-still` on the last
command to render one PNG near the middle of the shot:

```powershell
& $blender --background --python-exit-code 1 --python blender/cinematic_scene.py -- --dataset $dataset --scale 1e-6 --fps 24 --quality draft --render-still
```

Both `.blend` and PNG use collision-safe filenames; repeated runs produce
`_001`, `_002`, etc. and do not overwrite earlier versions. Unlike
commit-tracked code, generated outputs stay local under `outputs/` and
should be backed up independently.

## Quality/performance

| Preset | Output settings | Use |
|---|---|---|
| `draft` | 960×540 at 75% resolution | Quick previews on weaker computers |
| `final` | 1920×1080 at 100% resolution | High-quality stills for presentation |

Only a few rings and one curve per particle are created. The frame-by-frame
keyframes and static curve points scale approximately with
`number_of_particles × number_of_frames`. Avoid thousands of particles in
the initial Eevee preview.

## Implementation stages

- **5C.1 (this branch):** first cinematic preview, trails, lighting, PNG still
- **5C.2 (later):** inspect actual render and tune composition, materials, glow
- **5C.3 (later):** frame sequence/video export, edit/presentation integration

**Verification:** Standard pytest covers the pure camera/trail calculations
and collision-safe file naming. `blender/tests/test_cinematic_integration.py`
is a Blender-only smoke test and must run in the actual Blender installation
before merging. A passed Python suite alone is not evidence of rendered
visual correctness.


## Still-frame selection (trajectory visibility polish)

`--render-still` now selects a frame with the largest number of **valid
calculated particle centres outside the schematic black-hole core**. This is
only a preview frame selection: no physical positions or times are changed.

For a specific frame, pass `--still-frame 12 --render-still` (Blender frame
numbers start at 1). It is useful for comparing early infall to late infall.
If all particles are hidden inside the decorative core at the chosen
`--scale`, the script prints a warning and suggests increasing the *visual*
scale (for instance, from `5e-8` to `1e-6`). A visual scale change does not
change the trajectory dataset.


### Automatic visual scale

Add `--auto-scale` to pick a constant metres-to-Blender conversion from the
median initial distance of **valid, nonzero** particle positions. By default,
the typical starting distance is placed at approximately 4 Blender units,
outside the schematic rings. This is helpful when different datasets use
very different astronomical distances:

```powershell
& $blender --background --python-exit-code 1 --python blender/cinematic_scene.py -- --dataset $dataset --auto-scale --fps 24 --quality draft --render-still
```

`--auto-scale` and `--scale` are mutually exclusive. For scientific
comparison, the selected constant scale is saved on the Blender scene as
`visual_scale_m_to_blender`; source positions and timestamps remain untouched.
If the dataset has no valid, nonzero starting radius, automatic scaling fails
with a clear error and you can still choose an explicit `--scale`.

**Limitation:** this is a framing aid, not a guarantee that particles are
visible at every later frame. Infall can still take them inside the core or
behind schematic rings. Use `--still-frame` for explicit inspection.
