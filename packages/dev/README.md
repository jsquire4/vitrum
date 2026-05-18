# @vitrum/dev

Dev-only debug overlay components for `@vitrum` engines. Add as a **devDependency**; never ship in production.

## Components

- `<FrameTimeHUD>` — per-frame ms histogram + rolling FPS readout.
- `<DenoiserABToggle>` — quick A/B comparator for denoiser modes.
- `<DDGIAtlasViewer>` — live irradiance + visibility atlas readback (HybridEngine only).
- `<BVHVisualizer>` — wireframe overview of scene BVH AABBs.
- `<GISignalSplit>` — side-by-side direct radiance / indirect radiance / AO channels.
- `<MaterialInspector>` — click-to-pick: ray-cast on pointer-down and display the hit primitive's material.

## Vanilla harness

`attachDebugOverlays(engine, container, { overlays: [...] })` mounts the JS-side equivalents without React. Useful for non-React hosts.

## Capability gating

Each component reads `engine.capabilities.debugSurface` and the corresponding `engine.debug.<method>` to decide whether to render. If the underlying engine doesn't expose the data the component needs, it renders a one-line "not available on this engine" badge instead of failing.

## Status

Pre-1.0. New overlays added as engine debug surface grows.
