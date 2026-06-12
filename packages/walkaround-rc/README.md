# @vitrum/walkaround-rc

Radiance Cascades subsystem: cascade pyramid layout, raw WebGPU dispatch, and
raw WGSL shader strings.

Reference: Sannikov 2023, "Radiance Cascades: A Novel Approach to Calculating
Global Illumination."

## Status

GPU-validated (2026-06-07). The cascade-zero light-model gaps (RC only sampled
sun + emissive + env, never the abstract rect-area emitter list) were fully fixed:
- **PART A (`596c341`):** `shade.wgsl` gates RC confidence weight `cRc` on
  RC-has-energy — an empty cascade forces `cRc = 0` so ReSTIR-GI retains full
  weight.
- **PART B (`1e893fa`):** RC probe-ray hits now run one-sample-per-emitter NEE
  (`rcEmitterNEE`) against the world-space rect-area emitter tris.

Behavioral validation: CPU-brute-force oracle (tree-shape-invariant, 100% vs
ground truth) for merged-BVH traversal; GPU-proven HARM → NEUTRAL → BENEFICIAL
arc for the emitter-NEE fix; two-scene RC acceptance gate
(`tools/reference-renders/rc-gate-2026-06-07/`).

## Root Surface

- `RCDispatcher` — raw-WebGPU dispatch driver for the cascade compute passes.
- `CASCADE_DIMS` / `CASCADE_COUNT` / `CascadeDim` — default cascade pyramid
  sizes. Hosts can override per-engine via
  `HybridEngineOptions.cascadeDims: readonly CascadeDim[]`.
- `validateCascadeDims` — runtime guard for custom cascade overrides (positive
  probe grids, square ray counts, 2x ray-grid steps, valid intervals).
- `PROBE_RAY_CAST_WGSL` / `CASCADE_MERGE_WGSL` — raw WGSL strings for host
  inspection or headless WGSL-compile testing.
- `computeOctahedralSolidAngles` — pure CPU helper used by cascade-merge math.

There is no THREE bridge subpath. The package root is raw-runtime safe; it has
no `three` or `three/webgpu` peer dependency. TSL-side receiver helpers
(`GIReceiver`, `buildWalkaroundLightingNode`) remain in
`@vitrum/walkaround-hybrid` — composition happens at the engine level, not in
this subsystem package.

## Design Principles

The package is pure RC; it does not know about DDGI, ReSTIR, or SVGF.
Composition with those subsystems is the consumer's responsibility. The
`@vitrum/walkaround-hybrid` package does that through `HybridEngineRC.ts` on the
raw-WebGPU path.

## Testing

- `__tests__/rcSolidAngles.test.ts` — CPU-only unit tests for the octahedral
  solid-angle table.
- `__tests__/rcKernelMath.test.ts` — pure math invariants (cascade geometry,
  uniform packing).
- `__tests__/rcLightEvalWgsl.test.ts` — direct-light WGSL contract for
  glass-skip shadow visibility.
- `__tests__/cascadeDimsOverride.test.ts` — per-engine cascade dimension override.
- `__tests__/cascadeDispatchInvalidation.test.ts` — raw dispatcher bind-group
  cache invalidation across binding-relevant input changes.
- `__tests__/packageRootBoundary.test.ts` — verifies no three/webgpu import
  leaks from the package root.
- `__tests__/rcBehavior.gpu.test.ts` — GPU-gated behavioral smoke (requires
  WebGPU; skipped in CI without adapter).
