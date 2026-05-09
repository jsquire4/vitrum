# Cornell box — vitrum example

The classic radiometric validation scene. Two diffuse walls (red, green), three diffuse surfaces (white floor/ceiling/back), one rect-area emitter on the ceiling, two diffuse-only objects inside the box.

## Why this example

Cornell box is the standard scene for verifying a renderer's energy-conservation, color bleeding, and indirect-light behavior. If this scene renders correctly under both `@vitrum/pt-webgl` and `@vitrum/walkaround-hybrid`, the engine's foundation is sound. Specific things to check:

1. **Color bleeding**: green wall tints adjacent floor; red wall tints adjacent floor. Should converge to ~5–10% chromatic mix at the floor-wall interface.
2. **Indirect light intensity**: surfaces facing AWAY from the area light should still be lit by indirect bounces. The back-of-the-cube test.
3. **Energy conservation**: total emitted energy = total absorbed energy + escaped (none escapes a closed Cornell box). Any "free energy" indicates the BSDF or the path integrator is broken.
4. **Convergence rate**: PT_PREVIEW (192 samples) should produce a recognizable image; PT_FINAL (2112 samples) should be near-noise-free.

## Construction

The scene is constructed via `@vitrum/core` types directly, NOT through `@vitrum/three-bindings`. This keeps the example dependency-free and makes it usable as a pure type-system test.

See `index.ts` (to be authored).

## Reference numbers

These will be filled in once `@vitrum/pt-webgl` is wired and the example actually renders. Reference values are part of the regression test:

- Brightest pixel luminance at 192 samples: TBD
- Mean floor luminance at 192 samples: TBD
- Color-bleeding ratio at 1cm from green wall (G:R ratio): TBD
- Convergence frame at which mean per-pixel variance drops below 0.001: TBD

## Status

Scaffolded. The actual scene-construction code lands when `@vitrum/pt-webgl` reaches the point of accepting a `Scene` and producing a non-throwing `renderFrame`. Until then, this directory is a placeholder.
