# Presentation Storyboard — Spaghettification

**Presentation title:** "How can the same physics that moves our oceans tear apart a star?"
**Target duration:** ~12 minutes (within the 10–15 minute limit).
**Central research question:** How can the same physics that moves our oceans tear apart a star?
**Structure:** Four sections — A: Everyday gravity and tidal forces, B: Black holes and spaghettification, C: Real astronomical evidence, D: Our animation and its scientific limitations.

The exact number of PowerPoint slides may exceed the number of animation scenes. This storyboard documents the animation scenes; slides are not built here.

Classification labels used throughout this project:
- **OBSERVED** — real astronomical observation or measurement.
- **CALCULATED** — numerical result from a physical model.
- **SCHEMATIC** — illustrative, not a quantitative simulation.

## Storyboard table

| Scene | Section | Research question | Educational objective | Physics model | Proposed visual | Duration | Scientific limitations | Transition | Priority |
|-------|---------|-------------------|----------------------|---------------|-----------------|----------|------------------------|------------|----------|
| S1 | A (Intro) | Can gravity stretch an object like spaghetti? | Introduce the central question and the idea that gravity can deform objects. | None (conceptual) | Black hole silhouette / future rendered environment; an approaching object; title question. | 45 s | Conceptual only; no calculation. | Fade in from black. | High |
| S2 | A | Why does gravitational acceleration change with distance? | Show that gravity is stronger closer to a mass: g = GM/r². | Newtonian point mass, g = GM/r² | Two objects and an attracting mass; arrows showing gravitational acceleration; distance changing. | 60 s | Point-mass approximation; ignores object extent and rotation. | Cut to Earth–Moon. | High |
| S3 | A | Why does the Earth experience tides? | Connect differential gravity to ocean tides (qualitative). | Differential gravity (Earth–Moon) | Earth and Moon; near-side and far-side differential acceleration; idealized tidal bulges. | 90 s | A simple gravity-only model cannot accurately predict real coastal tides (coastline, ocean depth, Earth's rotation, Sun all matter). | Crossfade to extended object. | High |
| S4 | B | Why are the near and far ends pulled differently? | Quantify the near/far acceleration difference for an extended object. | Δa ≈ 2GML/r³ (small-object approximation) | Extended object; near and far measurement points; acceleration vectors; numerical acceleration difference. | 75 s | Small-object approximation (L << r); Newtonian only. | Crossfade to distance sweep. | High |
| S5 | B | Why does tidal acceleration increase so rapidly with decreasing distance? | Demonstrate inverse-cube distance dependence: halving distance ≈ eightfold tidal acceleration. | Δa ∝ 1/r³ | Same object at different distances; live numerical values; optional inverse-cube graph. | 60 s | Approximation breaks down as L approaches r; Newtonian only. | Zoom toward black hole. | High |
| S6 | B | What makes a black hole different? | Introduce the event horizon and the scale at which Newtonian gravity fails. | Schwarzschild radius r_s = 2GM/c² | Black hole; event-horizon reference; approaching object. | 60 s | The Newtonian engine does NOT simulate relativistic motion near or inside the horizon. | Cut to particle cloud. | High |
| S7 | B | Why does an object stretch lengthwise and compress sideways? | Show radial stretching and transverse compression of a freely falling particle cloud. | Newtonian tidal field (radial stretch + transverse compression) | Freely falling particle cloud; radial stretching; transverse compression; tracking of relative positions. | 90 s | Real calculated particle motion distinguished from artistically exaggerated deformation. | Fade to evidence. | High |
| S8 | C | Have astronomers observed tidal disruption? | Present real evidence: Shoemaker–Levy 9 and stellar tidal disruption events. | None (OBSERVED content) | Annotated scientific images or data; no expensive simulation required. | 90 s | Distinguish observations from reconstructions; no claim of direct imaging of spaghettification. | Crossfade to comparison. | Medium |
| S9 | D | How are Earth's tides and spaghettification related? | Reinforce the common physical principle across scales. | Same differential-gravity principle (Δa ≈ 2GML/r³) | Return to simplified tidal-acceleration diagram; connect Earth–Moon with black-hole tidal physics. | 60 s | Simplified diagram; omits real-world complexity of coastal tides. | Fade out. | High |

**Total scene time:** 10 min 30 s (630 s), within the 10–15 minute limit. Allow additional time for speaker pauses or questions.

## Scientific limitations (presentation-wide)

- The Python-side physics engine is Newtonian. It does not model general relativistic effects near or inside a black hole event horizon (Scene 6 clarifies this).
- Real coastal tides are driven by more than Earth–Moon differential gravity (Scene 3); the presentation uses an idealized model.
- Object deformation in Scene 7: calculated particle positions are shown as CALCULATED; any exaggerated artistic deformation is SCHEMATIC and must be labeled as such.
- Scene 8 relies on OBSERVED astronomical evidence; no observations are fabricated.

## Implementation priority order

1. S2–S5 (Newtonian tidal acceleration, inverse-cube dependence) — first usable dataset (Phase 6).
2. S6 (event-horizon reference, Newtonian-limitation note).
3. S7 (particle cloud radial stretch / transverse compression).
4. S1, S3, S8, S9 (conceptual / observational / comparative scenes).
5. S4 numerical near/far difference supports S5.
