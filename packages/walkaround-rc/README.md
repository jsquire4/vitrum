# @vitrum/walkaround-rc

Radiance Cascades subsystem — the cascade pyramid layout, BVH compute,
dispatch state machine, buffer manager, receiver material wrapper, and
raw WGSL shader strings.

Reference: Sannikov 2023, "Radiance Cascades: A Novel Approach to
Calculating Global Illumination."

## Status

Pre-1.0. Hoisted out of `@vitrum/walkaround-hybrid/src/rc/` on
2026-05-18 (W8 follow-up). Composition with DDGI / ReSTIR-GI (and the
Track-A balance-heuristic MIS) happens in `@vitrum/walkaround-hybrid`
via `HybridEngineRC`; this package is the algorithm itself.

## Public surface

- `RCDispatcher` — raw-WebGPU dispatch driver for the cascade compute
  passes (cast → merge). The `dispatchFrameRaw` entry exposes the
  pre-compiled pipelines so a host can wire them into a larger compute
  graph without bringing the TSL-side material wrapper.
- `CascadeBufferManager` — per-engine cascade ping-pong buffer ownership.
- `buildRCSceneBVH` / `SceneBVH` — `StorageBufferAttribute`-typed adapter
  over `@vitrum/shared-bvh`.
- `CASCADE_DIMS` / `CASCADE_COUNT` — cascade pyramid sizes (Cornell-tuned
  defaults; future revisions will derive from scene AABB).
- `GIReceiver` — `NodeMaterial` wrapper that samples cascade-0 from the
  TSL side. Requires `three/webgpu` + `three/tsl`.
- `buildWalkaroundLightingNode` — TSL diffuse-lighting node used by
  walkaround-style hosts that compose the receiver into a larger
  material graph. Requires `three/tsl`.
- `PROBE_RAY_CAST_WGSL` / `CASCADE_MERGE_WGSL` — raw WGSL strings for
  host inspection or headless WGSL-compile testing.
- `computeOctahedralSolidAngles` — pure CPU helper used by the WGSL
  cascade-merge math.

## Design principles

The package is pure RC; it doesn't know about DDGI, ReSTIR, or SVGF.
Composition with those subsystems is the consumer's responsibility:
`@vitrum/walkaround-hybrid` does it via `HybridEngineRC.ts` (raw-WebGPU
path) and `applyDDGIShading` (TSL path).

## Testing

- `__tests__/rcSolidAngles.test.ts` — CPU-only unit tests for the
  octahedral solid-angle table.
- `__tests__/rc-bindings.test.ts` — host-side dispatcher + buffer
  manager structural smoke tests with a fake GPU device.
