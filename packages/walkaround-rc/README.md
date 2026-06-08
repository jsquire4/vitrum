# @vitrum/walkaround-rc

Radiance Cascades subsystem: cascade pyramid layout, raw WebGPU dispatch, and
raw WGSL shader strings. Optional THREE/TSL receiver helpers live at
`@vitrum/walkaround-rc/three`.

Reference: Sannikov 2023, "Radiance Cascades: A Novel Approach to Calculating
Global Illumination."

## Status

Pre-1.0. Hoisted out of `@vitrum/walkaround-hybrid/src/rc/` on 2026-05-18
(W8 follow-up). Composition with DDGI / ReSTIR-GI happens in
`@vitrum/walkaround-hybrid` via `HybridEngineRC`; this package owns the RC
algorithm itself.

## Root Surface

- `RCDispatcher` - raw-WebGPU dispatch driver for the cascade compute passes.
- `CASCADE_DIMS` / `CASCADE_COUNT` / `CascadeDim` - default cascade pyramid
  sizes. Hosts can override per-engine via
  `HybridEngineOptions.cascadeDims: readonly CascadeDim[]`.
- `PROBE_RAY_CAST_WGSL` / `CASCADE_MERGE_WGSL` - raw WGSL strings for host
  inspection or headless WGSL-compile testing.
- `computeOctahedralSolidAngles` - pure CPU helper used by cascade-merge math.

## THREE Bridge Surface

Import these from `@vitrum/walkaround-rc/three`:

- `allocateCascades` / `disposeCascades` / `fillCascadeDebug` - cascade
  allocation helpers backed by `StorageBufferAttribute`.
- `CascadeBufferManager` - per-engine cascade buffer ownership for TSL hosts.
- `GIReceiver` - `NodeMaterial` wrapper that samples cascade-0 from the TSL side.
- `buildWalkaroundLightingNode` - TSL diffuse-lighting node for hosts composing
  the receiver into a larger material graph.

The bridge requires `three/webgpu` and `three/tsl`; the package root does not.

## Design Principles

The package is pure RC; it does not know about DDGI, ReSTIR, or SVGF.
Composition with those subsystems is the consumer's responsibility. The
`@vitrum/walkaround-hybrid` package does that through `HybridEngineRC.ts` on the
raw-WebGPU path.

## Testing

- `__tests__/rcSolidAngles.test.ts` - CPU-only unit tests for the octahedral
  solid-angle table.
- `__tests__/rc-bindings.test.ts` - host-side dispatcher and bridge structural
  smoke tests with a fake GPU device.
