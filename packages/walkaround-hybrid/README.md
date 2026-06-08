# @vitrum/walkaround-hybrid

**Stability:** production-grade for the shipped walkaround pipeline (DDGI, ReSTIR-DI/GI, GTAO, SVGF, RC, PPG, neural). Public API surface is still evolving with host-contract work in `@vitrum/core`.

WebGPU **ReSTIR DI + ReSTIR-GI** walkaround engine with **DDGI** probe updates and atlas sampling in the shade pass, **GTAO** ambient occlusion (half-res + bilateral upsample), per-channel **SVGF / à-trous-variance** denoising on direct + indirect, and opt-in **PPG** path guiding, **neural U-Net** denoiser, and **Radiance Cascades** (W8 shipped). RC is opt-in via `HybridEngineOptions.rcEnabled` and dispatches cascades each frame against the engine's own raw `GPUBuffer` allocation (no THREE WebGPU renderer dependency); cascade-0 sampling + MIS composition with DDGI / ReSTIR-GI are documented in [plan/archive/w8-rc-mis-composition-archived-2026-05-30.md](../../plan/archive/w8-rc-mis-composition-archived-2026-05-30.md).

Provides a class-based `Engine` implementation (`HybridEngine`) that composes:
- **DDGI** (Dynamic Diffuse Global Illumination) — probe-atlas irradiance, updated via compute each frame.
- **RC** (Radiance Cascades, Sannikov 2023) — opt-in via `HybridEngineOptions.rcEnabled`; W8 Phase 2 dispatches the 5-cascade pyramid per frame against a raw-GPUBuffer BVH. Sampling in shade.wgsl (`sampleCascadeC0`) + balance-heuristic MIS (`rcWeight`) shipped (W8 Phase 3).
- **ReSTIR DI** (Reservoir-based Spatiotemporal Importance Resampling) — direct illumination with temporal + spatial reuse.
- **ReSTIR-GI** (Ouyang et al. 2021) — indirect-illumination reservoirs with RIS + temporal + spatial reuse (Sprints 16–17).
- **GTAO** (Jiménez 2016) — half-resolution ground-truth-based ambient occlusion with bilateral upsample (Sprint 15).
- **Denoisers** (selectable via `EngineOptions.denoiser`): `'atrous'`, `'atrous-variance'` (default), `'svgf-real'` (per-channel SVGF on direct + indirect, Sprint 18), `'neural'` (opt-in U-Net; requires preloaded weights — see `tools/neural-denoiser-training/README.md`).
- **PPG** path guiding (Müller et al. 2017) — opt-in via `EngineOptions.ppgEnabled`; sTree + dTree on CPU with WGSL update training plus inline gi-ris guided sampling under `src/ppg/`.

## Denoisers

The post-shade denoise chain is selectable via `HybridEngineOptions.denoiser`.
All modes share the same engine surface — only the post-shade pass changes.

| Mode               | Default | Description                                                                          |
|--------------------|---------|--------------------------------------------------------------------------------------|
| `'atrous'`         |         | Legacy 3-iteration à-trous (no variance weighting).                                  |
| `'atrous-variance'`| ✓       | Welford temporal accumulator + variance lookup + 3-iter à-trous. Current production. |
| `'svgf-real'`      |         | Real Schied 2017 SVGF (T2.H1) — reprojection, moments, 7×7 filter, 5-tap à-trous.    |
| `'bmfr'`           |         | Blockwise Multi-Order Feature Regression (Koskela et al. 2019) — Householder-QR feature regression. |
| `'neural'`         |         | U-Net neural denoiser (T2.H2 / W10). Requires `neuralWeights` — see below.           |
| `'oidn-final'`     |         | Intel OIDN via ONNX Runtime Web (async; requires `extensions.oidnModelUrl`).         |

### Neural denoiser — weights interface

`denoiser: 'neural'` requires `neuralWeights: ModelWeights` to be passed to
the engine constructor; missing weights produce a clear validation error at
construction time.

```ts
import {
  createWalkaroundEngine_Hybrid,
  loadWeightsFromArrayBuffer,
} from '@vitrum/walkaround-hybrid';

const weightsBytes = await (await fetch('/vi-neural-weights.bin')).arrayBuffer();
const neuralWeights = loadWeightsFromArrayBuffer(weightsBytes);

const engine = await createWalkaroundEngine_Hybrid({
  // …other engine options…
  denoiser:      'neural',
  neuralWeights,
});
```

The binary format is `.vitrum-model` (magic `0xDEAF1984`, little-endian) —
mirrored by the Python exporter at `tools/neural-denoiser-training/export_weights.py`
and the TypeScript serialiser `serializeWeightsToArrayBuffer`.

The canonical trained checkpoint ships as `vi-neural-weights.bin` (target ~2.1 MB
at f32 for the default ~535k-parameter spec). The repo does NOT ship a trained
checkpoint — see `tools/neural-denoiser-training/README.md` for training.

**Smoke-test path (no trained weights):** `buildRandomWeightsForSpec(spec, seed)`
synthesises deterministic He-init random weights. The pipeline runs end-to-end
but the denoised output will NOT be visually clean — this is only for wiring
verification (used by `examples/neural-denoiser/`).

### Example app

`examples/neural-denoiser/` demonstrates all three modes side-by-side with a
URL toggle (`?denoiser=atrous-variance|svgf-real|neural`); `npm run dev
--workspace @vitrum-examples/neural-denoiser`.

## Peer dependencies

`three >=0.167.0 <0.190` is an optional peer. The package root is safe to import
for Three-free constants, types, GI-state helpers, and neural-weight utilities;
calling `createWalkaroundEngine_Hybrid()` dynamically loads the concrete engine,
which still requires Three for its legacy raw-`threeScene` fallback. The
`@vitrum/walkaround-hybrid/three` subpath contains the explicit TSL/Node-material
bridge exports (`applyDDGIShading`, `upgradeToNodeMaterial`) for hosts that
already use `three/webgpu` and `three/tsl`.

`@vitrum/three-bindings` is not a production dependency of this package; it is
kept as a dev/test dependency for core-vs-Three equivalence tests and host
adapter checks. `three` and `@types/three` remain in devDependencies for tests
and typecheck.

## Architecture

```
src/
  HybridEngine.ts        — Engine implementation (orchestrator after W4 decomp)
  HybridEngineRC.ts      — RCSubsystem owning per-engine cascade allocation +
                           dispatch; bridges @vitrum/walkaround-rc into the hybrid frame.
  HybridEnginePrimitiveUpdates.ts — Dispatcher for engine.updatePrimitive (transform-
                           refit / positions-refit / topology-rebuild branches).
  hostScene/             — Three-side scene adapters consumed by HybridEngine
  ddgi/                  — DDGI subsystem (probe grid, update pass, atlas layout)
    applyDDGIShading.ts  — TSL-based DDGI outputNode injection (the only RC-adjacent
                           file that stayed here — DDGI-specific, not part of RC).
  pipeline/              — Declarative Pass / PassRegistry, denoiser registry, FrameResources
    Pass.ts, PassRegistry.ts — Pass interface + registry (W1-R1)
    passes/              — One file per pass (RIS, RIS-GI, Temporal[GI], Spatial[GI],
                           Shade, IndirectCombine, IndirectTemporalAccum, AtrousIndirect,
                           GTAO, GTAOUpsample, Composite, Resolve, SampleBudget,
                           PPGUpdate) + declarative passOrder
    denoisers/           — Denoiser registry (atrous, atrous-variance, svgf-real,
                           neural, oidn-final, none)
    pipelineCompiler.ts  — WGSL include-graph (declarative `requires:`; W1-R6)
    WalkaroundGPUPipeline.ts — Iterates PASS_ORDER each frame
  restir/                — ReSTIR BVH + emitter list builders
  shaders/               — WGSL shader strings: ris, risGi, temporal, temporalGi,
                           spatial, spatialGi, shade, indirectCombine,
                           indirectTemporalAccum, gtao, gtaoUpsample,
                           composite, resolve, sampleBudget, welfordTemporal
  ppg/                   — Practical Path Guiding (Müller 2017): sTree + dTree on CPU,
                           ppgUpdate WGSL + gi-ris inline guiding (opt-in via ppgEnabled)
  neural/                — U-Net denoiser (Chaitanya 2017): InferenceGraph,
                           inputPacker, unetArchitecture, weights loader (opt-in
                           via denoiser: 'neural'); WGSL kernels under neural/wgsl/
  lib/                   — Shared utilities (nodeMaterialUpgrade)
```

The Radiance Cascades subsystem (`cascadePyramid`, `cascadeDispatch`, `cascadeBuffers`,
`giReceiver`, `walkaroundDiffuseLighting`, `bvhCompute`, raw WGSL) was extracted
2026-05-18 into [`@vitrum/walkaround-rc`](../walkaround-rc/). `walkaround-hybrid`
re-exports the public surface for back-compat.

## Known Issues

### RC subsystem: TSL→raw WebGPU port not GPU-verified

The RC (Radiance Cascades) subsystem moved to `@vitrum/walkaround-rc` on
2026-05-18 (W8 follow-up). The public API is re-exported here for
back-compat; the residual-risk notes below still apply to the now-extracted
package.

The RC subsystem (now in `@vitrum/walkaround-rc/src/`) was ported from a TSL-based
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
- Material-wrapping files under explicit `/three` bridge subpaths
  (`applyDDGIShading.ts`, `giReceiver.ts`, `walkaroundDiffuseLighting.ts`) are
  preserved as TSL hooks; package roots stay raw-runtime safe.
- RC cascade storage that needs `StorageBufferAttribute` lives in
  `@vitrum/walkaround-rc/three`; the raw RC runtime and walkaround-hybrid BVH
  ingestion paths use host-neutral typed-array/GPU-buffer contracts.

**Residual risk**:
- Workgroup sizing, dispatch dimensions, cascade indexing, and merge-pass color
  space have NOT been visually verified against the original.
- The `StorageBufferAttribute.__gpuBuffer` reach-through that this section
  used to flag was dropped 2026-05-18 along with the THREE-tied
  `RCDispatcher.dispatchFrame` entry point; only the raw-GPU
  `dispatchFrameRaw` path remains, and `RCSubsystem` allocates its own
  `GPUBuffer` handles via `device.createBuffer`. The TSL-side material
  wrappers (`GIReceiver`, `buildWalkaroundLightingNode`) still consume
  `StorageBufferAttribute` because three.js's TSL only binds three.js-
  native data types; that coupling is intrinsic to TSL, not a reach-
  through.
- Library consumers running RC for the first time should A/B against a
  known-good reference before reporting visual discrepancies as bugs.
- Issues filed against the RC path should be triaged with this in mind.

**Verification status**: structural (TypeScript compile + binding-shape unit
tests), not behavioral (no GPU render comparison).

**Affected files** (all now in `@vitrum/walkaround-rc`): `src/cascadeDispatch.ts`,
`src/cascadePyramid.ts`, `src/cascadeBuffers.ts`, `src/wgsl/probeRayCast.wgsl.ts`,
`src/wgsl/cascadeMerge.wgsl.ts`.

### DDGI path: `three/webgpu` renderer internals coupling

`probeUpdatePass.ts` accesses `renderer.backend.device` (raw `GPUDevice`) and imports
`StorageTexture` from `three/webgpu`. This is an accepted known cost — `@vitrum/walkaround-hybrid`
requires `three/webgpu` as a peer dep on the DDGI path. ReSTIR-only consumers can
create the `GPUDevice` themselves and avoid this dependency by not calling DDGI APIs.
