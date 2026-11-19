# Simulation-to-Frame Mapping Guide

This guide explains how to generate a black-hole trajectory dataset, reload it,
map physical simulation time to animation frames, and sample particle positions
for those frames — all without using Blender.

## 1. Generate and reload a dataset

Use the existing CLI generator to produce a deterministic Newtonian trajectory
dataset:

```bash
# Scenario A: radial infall (3 particles fall toward the black hole)
uv run python scripts/generate_black_hole_trajectories.py radial-infall \
    --output outputs/my_radial_infall

# Scenario B: tidal particle cloud (grid shows stretch + compression)
uv run python scripts/generate_black_hole_trajectories.py tidal-cloud \
    --output outputs/my_tidal_cloud
```

Each command writes two files:

* `<stem>.meta.json` — JSON metadata (schema, units, solver info, termination).
* `<stem>.data.npz` — compressed NumPy arrays (timestamps, positions,
  velocities, validity_mask, particle_ids).

Reload and validate with the project's own loader:

```python
from core.simulation.dataset import load_dataset, validate_dataset

ds = load_dataset("outputs/my_radial_infall_radial_infall.meta.json")
validate_dataset(ds)

print(ds.arrays["positions"].shape)   # (T, N, 2) in metres
print(ds.arrays["validity_mask"].shape)  # (T, N) boolean
print(ds.arrays["particle_ids"])      # (N,) string identifiers
```

## 2. Map simulation time to video frames

Use `core.playback.build_frame_mapping` to convert a scene's presentation
duration and frame rate into a deterministic list of frame timestamps and the
corresponding physical simulation times.

### Timing convention (fixed-rate)

* **Effective playback speed** = simulation span / presentation duration, in
  simulation seconds per presentation second.
* **`num_frames = max(1, round(duration_s * fps))`**.
* **Frame `i`** (0-indexed) presents at `i / fps` seconds and maps to
  simulation time
  ``start_sim_time_s + (i / fps) * effective_playback_speed``.
* The simulation window end (``end_sim_time_s``) is the **scene boundary**;
  it is not forcibly placed on the final active frame. When
  ``duration_s * fps`` is not an integer, the last frame lands slightly before
  the boundary.
* **Hold frames** (after the active range) freeze at the last active frame's
  simulation time and do not advance simulation time.

This convention makes adjacent scenes continuous: if scene B's
``start_sim_time_s`` equals scene A's ``end_sim_time_s``, scene B's frame 0
begins at that same physical time.

### Example

```python
from core.playback import PlaybackSpec, build_frame_mapping

# 10 seconds of simulation played over 5 seconds at 30 FPS = 2x speed,
# 150 frames.
spec = PlaybackSpec(
    scene_id="spaghettification",
    simulation_id="black_hole_radial_infall",
    fps=30.0,
    duration_s=5.0,
    start_sim_time_s=0.0,
    end_sim_time_s=10.0,
)
mapping = build_frame_mapping(spec)

print(mapping.num_frames)                 # 150
print(mapping.effective_playback_speed)   # 2.0
print(mapping.sim_times_s[0])             # 0.0 (first frame)
print(mapping.sim_times_s[-1])            # ~9.93 (last active frame, before boundary)
print(mapping.end_sim_time_s)             # 10.0 (scene boundary)
```

Note that with 150 frames at 30 FPS over 5 seconds, the final active frame
(index 149) maps to `149/30 * 2.0 ≈ 9.93 s`, just before the 10 s scene
boundary. This is expected: the boundary defines the scene extent, while the
fixed-rate frame grid determines the actual sample times.

## 3. Sample particle positions at frame times

Use `core.playback.sample_particle_positions` to get particle positions at the
mapped simulation times:

```python
import numpy as np
from core.playback import sample_particle_positions

# Sim times for each frame (from build_frame_mapping).
sim_times = mapping.sim_times_s  # shape (150,)

positions, validity = sample_particle_positions(
    ds,
    particle_ids=["radial_0000", "radial_0001", "radial_0002"],
    sim_times=sim_times,
)

# positions: shape (150, 3, 2) in metres
# validity:  shape (150, 3) boolean
```

### Invalid position handling

* Times before the first sample or after a particle's termination →
  `validity = False`, position = `NaN`.
* Terminated particles never have positions fabricated after termination.
* Interpolation only occurs between valid neighbouring samples; gaps of
  invalid samples are never crossed.
* The original NPZ arrays are never modified.

## 4. Data the future Blender importer will consume

The Blender importer (future work) will read the same `.meta.json` + `.data.npz`
files using only `json` and `numpy` — no SciPy, Matplotlib, or Pillow needed:

```python
import json
import numpy as np
from pathlib import Path

stem = Path("outputs/my_radial_infall_radial_infall")
meta = json.loads(stem.with_name(stem.name + ".meta.json").read_text())
with np.load(stem.with_name(stem.name + ".data.npz"), allow_pickle=False) as data:
    arrays = {name: data[name] for name in data.files}
```

Arrays it will consume:
| Array | Shape | Units |
|-------|-------|-------|
| `timestamps` | `(T,)` | s |
| `positions` | `(T, N, 2)` | m |
| `velocities` | `(T, N, 2)` | m/s |
| `validity_mask` | `(T, N)` | — |
| `particle_ids` | `(N,)` | — |

## 5. Scientific vs illustrative

* **CALCULATED:** Particle positions, velocities, and validity masks come from
  Newtonian test-particle integration around a point mass. SI units throughout.
* **SCHEMATIC:** Any exaggerated visual deformation, camera motion, or artistic
  timing applied downstream is illustrative and must be labelled as such. The
  particles are independent test particles — they do not represent real material
  deformation of a star or planet.
* **Not modelled:** General relativity near/inside the Schwarzschild radius,
  fluid dynamics, material strength, collisions, or self-gravity.

## Executable usage example

```python
"""End-to-end: generate, load, map to frames, sample positions."""
from pathlib import Path
import numpy as np

from core.playback import PlaybackSpec, build_frame_mapping, sample_particle_positions
from core.simulation.dataset import load_dataset, validate_dataset

# 1. Load a previously generated dataset.
ds = load_dataset("outputs/my_radial_infall_radial_infall.meta.json")
validate_dataset(ds)

# 2. Map 20 seconds of simulation to 6 seconds of video at 24 FPS.
spec = PlaybackSpec(
    scene_id="radial_infall",
    simulation_id=ds.metadata["simulation_id"],
    fps=24.0,
    duration_s=6.0,
    start_sim_time_s=0.0,
    end_sim_time_s=20.0,
)
mapping = build_frame_mapping(spec)

# 3. Sample positions for all particles at every frame.
positions, validity = sample_particle_positions(
    ds,
    particle_ids=ds.arrays["particle_ids"].tolist(),
    sim_times=mapping.sim_times_s,
)

print(f"Frames: {mapping.num_frames} (24 fps x 6 s)")
print(f"Effective speed: {mapping.effective_playback_speed:.2f}x")
print(f"Valid samples: {int(validity.sum())} / {validity.size}")
```
