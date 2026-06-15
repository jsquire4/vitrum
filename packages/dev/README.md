# @vitrum/dev

Dev-only debug overlay components for `@vitrum` engines. Add as a **devDependency**; never ship in production.

## Components

- `<FrameTimeHUD>` — per-frame ms histogram + rolling FPS readout.
- `<DenoiserABToggle>` — keyboard `D` / badge toggles `engine.debug.setDenoiserEnabled` on `@vitrum/walkaround-hybrid` (bypasses the denoiser pass; raw HDR).
- `<DDGIAtlasViewer>` — live irradiance + visibility atlas readback via GPU copyTextureToBuffer + canvas blit at ~10 Hz (HybridEngine only).
- `<BVHVisualizer>` — BVH depth histogram + node-count / leaf-count stats panel, polled at 2 Hz via `engine.debug.bvhNodes()`. (A camera-projected AABB overlay would need view/proj matrices on the debug surface — future work.)
- `<GISignalSplit>` — 2×2 quadrant view of direct / indirect / AO / total HDR textures via the shared `startGpuTextureBlit` helper.
- `<MaterialInspector>` — UI panel that live-edits the selected primitive's MaterialSpec via `engine.updatePrimitive`. Pass the render `canvas` to enable built-in click-to-pick when the engine exposes `engine.debug.pickPrimitive`; hosts can also drive selection explicitly with `selectedPrimitiveId`.

## Vanilla harness

`attachDebugOverlays(engine, container, { overlays: [...] })` mounts the JS-side equivalents without React. Useful for non-React hosts.

## Capability gating

Each component duck-types the specific `engine.debug.<method>` it needs and renders a one-line "not available on this engine" badge if it's absent. The coarse `engine.capabilities.debugSurface` flag exists on the contract but isn't consulted here — the method-level check is finer-grained (the surface can expose `bvhNodes` without `pickPrimitive`, for example).

## Status

Pre-1.0. New overlays added as engine debug surface grows.
