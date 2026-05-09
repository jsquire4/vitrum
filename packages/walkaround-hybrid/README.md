# @vitrum/walkaround-hybrid

**Stability:** pre-alpha — `HybridEngine` and shader/pipeline APIs may change until the extraction milestones in `plan/generalized-library-milestones.md` are closed.

WebGPU layered DDGI + RC + ReSTIR DI engine.

Provides a class-based `Engine` implementation (`HybridEngine`) that composes:
- **DDGI** (Dynamic Diffuse Global Illumination) — probe-atlas irradiance, updated via compute each frame.
- **RC** (Radiance Cascades) — multi-resolution cascade GI for the standalone walkaround engine path.
- **ReSTIR DI** (Reservoir-based Spatiotemporal Importance Resampling) — direct illumination with temporal + spatial reuse.

## Peer dependencies

- `three >= 0.160.0`
- `three/webgpu` — required for DDGI path (`StorageTexture`, `MeshPhysicalNodeMaterial`)
- `three/tsl` — required for the RC material-wrapper path (`applyDDGIShading`, `giReceiver`, `walkaroundDiffuseLighting`)

ReSTIR-only usage does not trigger the DDGI or RC-material-wrapper paths.

## Architecture

```
src/
  HybridEngine.ts        — Engine implementation (de-React-ified useHybridLayeredGI)
  ddgi/                  — DDGI subsystem (probe grid, update pass, atlas layout)
  rc/                    — RC subsystem (cascade pyramid, dispatch, material wrappers)
    cascadePyramid.ts    — storage layout + allocation
    bvhCompute.ts        — RC BVH builder (StorageBufferAttribute adapter)
    cascadeBuffers.ts    — CascadeBufferManager (de-React-ified useCascadeBuffers)
    cascadeDispatch.ts   — RCDispatcher (raw WebGPU compute; converted from TSL)
    giReceiver.ts        — GIReceiver class (TSL-preserved NodeMaterial wrapper)
    walkaroundDiffuseLighting.ts — TSL node for C0 cascade sampling
    applyDDGIShading.ts  — TSL-based DDGI outputNode injection
    wgsl/
      probeRayCast.wgsl.ts — Assembled raw WGSL for probe ray-cast compute kernel
      cascadeMerge.wgsl.ts — Assembled raw WGSL for cascade merge compute kernel
    TSL_TO_RAW_MAPPING.md — Documents every TSL primitive → raw WebGPU mapping
  pipeline/              — ReSTIR pipeline (7-way split of WalkaroundGPUPipeline)
  restir/                — ReSTIR BVH builder
  shaders/               — ReSTIR WGSL shader strings
  lib/                   — Shared utilities (nodeMaterialUpgrade)
```

## Known Issues

### RC subsystem: TSL→raw WebGPU port not GPU-verified

The RC (Radiance Cascades) subsystem in `src/rc/` was ported from a TSL-based
implementation to raw WebGPU during Phase 4 Step 4 of the extraction plan,
under a "maximum-diligence-without-GPU-verification" protocol per the
extraction plan's RD-12.

**What was done**:
- Compute kernels (`cascadeDispatch.ts`) converted from TSL `compute()`/`storage()`
  to raw `GPUComputePipeline` + `passEncoder.dispatchWorkgroups()`. WGSL kernel
  source captured verbatim from the TSL `wgslFn()` arguments and assembled into
  complete modules in `rc/wgsl/`.
- Resource binding layouts derived from the TSL declarations and unit-tested for
  structural conformance.
- Material-wrapping files (`applyDDGIShading.ts`, `giReceiver.ts`,
  `walkaroundDiffuseLighting.ts`) preserved as TSL — these are not compute
  kernels, they're Three.js NodeMaterial customization hooks.
- `cascadePyramid.ts` and `bvhCompute.ts` retain `StorageBufferAttribute` from
  `three/webgpu` because the C0 cascade buffer is consumed by the TSL
  `walkaroundDiffuseLighting.ts` node.

**Residual risk**:
- Workgroup sizing, dispatch dimensions, cascade indexing, and merge-pass color
  space have NOT been visually verified against the original.
- The `StorageBufferAttribute.__gpuBuffer` accessor used in `cascadeDispatch.ts`
  is a Three.js WebGPU renderer internal property. If Three.js changes this
  internal API, `RCDispatcher.initialize()` will throw.
- Library consumers running RC for the first time should A/B against a
  known-good reference before reporting visual discrepancies as bugs.
- Issues filed against the RC path should be triaged with this in mind.

**Verification status**: structural (TypeScript compile + binding-shape unit
tests), not behavioral (no GPU render comparison).

**Affected files**: `src/rc/cascadeDispatch.ts`, `src/rc/cascadePyramid.ts`,
`src/rc/cascadeBuffers.ts`, `src/rc/wgsl/probeRayCast.wgsl.ts`,
`src/rc/wgsl/cascadeMerge.wgsl.ts`.

### DDGI path: `three/webgpu` renderer internals coupling

`probeUpdatePass.ts` accesses `renderer.backend.device` (raw `GPUDevice`) and imports
`StorageTexture` from `three/webgpu`. This is an accepted known cost — `@vitrum/walkaround-hybrid`
requires `three/webgpu` as a peer dep on the DDGI path. ReSTIR-only consumers can
create the `GPUDevice` themselves and avoid this dependency by not calling DDGI APIs.
