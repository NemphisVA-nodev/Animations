# Transition Architecture

Specification for continuous cinematic transitions between scientific
simulations. Pure configuration data — no rendering, camera animation, or
Blender-specific behaviour is implemented here. The physics engine remains
independent of the transition system.

## Transition definition

A transition connects a start scene to an end scene on a shared presentation
timeline.

| Field | Type | Description |
|-------|------|-------------|
| `transition_id` | string | Unique transition identifier. |
| `start_scene` | string | Scene identifier where the transition begins. |
| `end_scene` | string | Scene identifier where the transition ends. |
| `transition_type` | string | One of: `cut`, `crossfade`, `zoom`, `match_cut`, `object_track`, `continuous_camera`. |
| `duration_s` | number | Transition duration in presentation seconds (>= 0). |
| `start_anchor` | string | Named camera or subject anchor at the start. |
| `end_anchor` | string | Named camera or subject anchor at the end. |
| `overlap_s` | number | Visual overlap with neighbouring scenes on the timeline (>= 0). |
| `parameters` | object | Optional free-form hints (e.g. zoom factor, camera path). |

A transition has a defined start and end scene. Two scenes may overlap
visually on the shared presentation timeline via `overlap_s`.

Scenes are identified by `scene_id`, so a scene can be reused across
different presentations without hard-coded slide numbers.

## Data model

Defined in `core/transitions.py`:
- `Transition` — a single start-to-end transition.
- `TransitionPlan` — a collection; `validate_against(scene_ids)` confirms
  every transition references known start and end scenes.

## Timing contract

Scene durations, simulation windows, and playback speed are currently
descriptive fields. The models reject non-finite timing values, but they do
not yet derive a frame mapping or require
`end_sim_time_s - start_sim_time_s == duration_s * playback_speed`.

Before the Blender adapter uses these fields, choose one source of truth for
playback timing and define how holds and transitions affect the final frame
count. A video renderer must not silently guess when these values disagree.

Transition overlap is bounded by its own duration. Transition identifiers
must be unique within a plan, and transitions must reference known scenes.

## Physical-to-visual coordinate transformation

Exported simulation data retains original SI values (metres, seconds,
m/s²) with origin at the attracting mass and the radial axis positive
outward. Blender-friendly coordinates are a purely visual convenience and
must never overwrite the SI values.

A documented transform may be applied downstream by the rendering/Blender
layer:

1. **Scale:** convert metres to a scene-appropriate unit (e.g. kilometres or
   a normalised unit) using a constant scale factor.
2. **Origin shift:** re-centre the camera frame on the falling object rather
   than the mass, by subtracting the object's centre position.
3. **Axis mapping:** map the framework's radial axis to Blender's chosen
   up/right axes (documented per scene; Blender uses Z-up by default).

The original SI arrays remain the source of truth; the transform is applied
only for visual placement and is reversible.

## Constraints

- The physics engine does not import or depend on the transition system.
- Transitions describe visual alignment only; they do not alter simulation
  data.
- No camera animation, Blender scene files, materials, or video rendering
  are implemented here.
