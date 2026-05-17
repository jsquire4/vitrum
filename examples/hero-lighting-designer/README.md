# hero-lighting-designer — Interactive GI Lighting Designer

Demonstrates real-time global illumination via `walkaround-hybrid` (WebGPU)
with live, per-slider light updates. A procedural architectural interior
(room + furniture) hosts 3 adjustable point lights with independent RGB
color, intensity, and XYZ position controls.

## What it demonstrates

- `attachVitrum({ prefer: 'realtime' })` for real-time GI at ~60fps.
- `sceneFromThreeJS()` converting a procedural THREE.js scene to a vitrum Scene.
- `engine.setScene()` + `engine.reset()` (debounced 50ms) to push slider
  changes through the GI pipeline without spamming the engine.
- `engine.onFrame()` telemetry forwarded to an in-HUD FPS counter.

## How to run

```bash
# from repo root
npm install
cd examples/hero-lighting-designer
npm run dev
# open http://localhost:5177
```

Drag the sliders in the right panel to change light color, intensity, and
position. The GI re-converges within ~30 frames after each change.

## Architecture note

The scene is rebuilt from scratch on each slider-change batch (50ms debounce)
via `sceneFromThreeJS`. This is acceptable for a static-topology scene;
if the engine had a wired `updateEmitter()` path the slider could go
directly without a full scene rebuild. See `plan/archive/animation-support-status.md`
for the incremental-update roadmap.

## Assets

Procedural only — no external files required.
