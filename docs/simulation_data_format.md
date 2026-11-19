# Simulation Data Format

Renderer-independent representation of scientific simulation output. Blender
(or any consumer) can read these files without recomputing the physics and
without installing the full scientific dependency set — only the Python
standard library and NumPy are required.

## Design goals

- **Renderer independent:** the file contains no rendering, camera, or
  Blender-specific information.
- **Blender ready:** readable by Blender's bundled Python with NumPy only.
- **Deterministic:** the same inputs always produce the same output.
- **Validated:** schema version, array dimensions, units, timestamp ordering
  and finite values are checked on load.

## File layout

A simulation is stored as two files sharing a path stem:

```
<stem>.meta.json     lightweight JSON metadata
<stem>.data.npz      compressed NumPy array archive
```

Large numerical arrays live only in the NPZ archive, never in JSON.

## Metadata (`<stem>.meta.json`)

JSON, versioned schema `1.0.0`. Required keys:

| Key | Type | Description |
|-----|------|-------------|
| `schema_version` | string | Must be `"1.0.0"`. |
| `simulation_id` | string | Unique simulation identifier (non-empty). |
| `simulation_type` | string | e.g. `"tidal_acceleration"`. |
| `model` | string | Physical model name, e.g. `"newtonian_point_mass_tidal"`. |
| `model_limitations` | string | Honest statement of what the model does not cover. |
| `units` | object | Mapping of quantity -> unit string (SI). `time` and `position` required. |
| `reference_frame` | object | Origin, axes, and direction conventions (non-empty). |
| `physical_parameters` | object | Model inputs (mass, length, distances, constants). |
| `created_at` | string | ISO-8601 UTC timestamp. |
| `array_sources` | object | array name -> `"<archive>#<array>"` references. |
| `notes` | string | Free-form notes. |

All quantities use SI units unless explicitly documented otherwise.

## Numerical arrays (`<stem>.data.npz`)

Compressed NumPy archive. Each member is a 1-D `float64` array.

Recommended members for a tidal dataset:

| Member | Units | Description |
|--------|-------|-------------|
| `timestamps` | s | Physical simulation time, strictly increasing. |
| `distances` | m | Centre distance of the object from the mass. |
| `near_acceleration` | m/s² | Acceleration at the near end. |
| `far_acceleration` | m/s² | Acceleration at the far end. |
| `tidal_difference_exact` | m/s² | Exact near-minus-far tidal acceleration. |
| `tidal_difference_approximate` | m/s² | `2 G M L / r³` approximation. |
| `transverse_compression` | m/s² | Transverse tidal compression (negative). |

All multi-element arrays must share the same length so trajectories align.
Every value must be finite.

## Classification labels

Every derived visual is labelled:

- **OBSERVED** — real astronomical observation or measurement.
- **CALCULATED** — numerical result from a physical model.
- **SCHEMATIC** — illustrative, not a quantitative simulation.

The data files themselves contain CALCULATED values. Any exaggerated artistic
deformation applied downstream is SCHEMATIC and must be labelled as such.

## Reading the format (Blender bundled Python)

```python
import json
import numpy as np
from pathlib import Path

def load_simulation(stem: str | Path):
    stem = Path(stem)
    meta = json.loads((stem.with_suffix(".meta.json")).read_text())
    data = np.load(stem.with_suffix(".data.npz"))
    arrays = {name: data[name] for name in data.files}
    return meta, arrays
```

Only `json` and `numpy` are required. No SciPy, Matplotlib, or Pillow.

## Validity checks (enforced on load)

- All required metadata keys present and non-empty.
- `schema_version` == `1.0.0`.
- `units` contains `time` and `position`.
- `reference_frame` is a non-empty object.
- Every array is 1-D, non-empty, finite, and (when multi-element) shares a
  common length with the others.
- `timestamps`, when present, is strictly increasing.
