# hero-product-viz — Progressive PT Product Visualizer

A glass-on-pedestal scene rendered with the `pt-webgl` path tracer. Adjust
material parameters (roughness, metallic, transmission, IOR) via sliders and
watch the scene re-converge. Save a high-resolution PNG when you're satisfied.

## What it demonstrates

- `attachVitrum({ prefer: 'quality' })` for progressive path tracing (SPP accumulation).
- `engine.onProgress()` (`kind: 'pt-spp'`) for a live progress bar.
- `engine.setScene()` + `engine.reset()` (debounced 80ms) for material slider updates.
- `createEngine()` on an offscreen `<canvas>` for a separate high-res render pass.
- Procedural scene in <30 lines: shelf pedestal + glass sphere + glass icosahedron + studio lights.

## How to run

```bash
# from repo root
npm install
cd examples/hero-product-viz
npm run dev
# open http://localhost:5178
```

Drag the material sliders to change the glass properties. The path tracer
re-converges after each change (50ms debounce). When happy with the result,
click "Save high-res PNG" — this renders a separate 4K frame offline and
downloads it. **Be patient: a 4K render at 512 SPP typically takes 20–60 seconds.**

## High-res save architecture note

The save button creates a temporary offscreen canvas, spins up a second
`pt-webgl` engine via `createEngine()`, runs RAF until convergence, then
calls `canvas.toDataURL('image/png')`. The main canvas engine is paused
during the save to avoid GPU contention. After saving the offscreen engine
and canvas are disposed.

`preserveDrawingBuffer` is not set on the main canvas (the factory default
avoids the performance cost); the offscreen canvas doesn't need it since we
read immediately after convergence via `toDataURL`.

## Assets

Procedural only — no external files required. Scene: shelf pedestal
(CylinderGeometry), glass sphere (SphereGeometry), glass icosahedron
(IcosahedronGeometry), diffuse floor (PlaneGeometry), two RectAreaLights.

**TODO:** swap in a polished CC0 hero asset from
[Sketchfab](https://sketchfab.com/features/free-3d-models) and cite source.
