# @vitrum/dev

Dev-only debug overlay components for `@vitrum` engines. Add as a **devDependency**; never ship in production.

## Components

- `<FrameTimeHUD>` — per-frame ms histogram + rolling FPS readout.
- `<DenoiserABToggle>` — UI scaffold for A/B comparator (requires `engine.debug.setDenoiserEnabled` — not yet implemented by any backend; renders a warning until that lands).
- `<DDGIAtlasViewer>` — live irradiance + visibility atlas readback via GPU copyTextureToBuffer + canvas blit at ~10 Hz (HybridEngine only).
- `<BVHVisualizer>` — BVH depth histogram + node-count / leaf-count stats panel, polled at 2 Hz via `engine.debug.bvhNodes()`. (A camera-projected AABB overlay would need view/proj matrices on the debug surface — future work.)
- `<GISignalSplit>` — 2×2 quadrant view of direct / indirect / AO / total HDR textures via the shared `startGpuTextureBlit` helper.
- `<MaterialInspector>` — UI panel that live-edits the selected primitive's MaterialSpec via `engine.updatePrimitive`. Click-to-pick is a future addition (requires `engine.debug.pickPrimitive`); hosts wire the selection state by passing `selectedPrimitiveId` as a prop.

## Vanilla harness

`attachDebugOverlays(engine, container, { overlays: [...] })` mounts the JS-side equivalents without React. Useful for non-React hosts.

## Capability gating

Each component reads `engine.capabilities.debugSurface` and the corresponding `engine.debug.<method>` to decide whether to render. If the underlying engine doesn't expose the data the component needs, it renders a one-line "not available on this engine" badge instead of failing.

## Status

Pre-1.0. New overlays added as engine debug surface grows.
