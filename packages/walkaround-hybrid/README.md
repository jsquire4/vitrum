# @vitrum/walkaround-hybrid

**Stability:** pre-alpha — `HybridEngine` and shader/pipeline APIs may change until the extraction milestones in `plan/generalized-library-milestones.md` are closed.

WebGPU **ReSTIR DI + ReSTIR-GI** walkaround engine with **DDGI** probe updates and atlas sampling in the shade pass, **GTAO** ambient occlusion (half-res + bilateral upsample), per-channel **SVGF / à-trous-variance** denoising on direct + indirect, and opt-in **PPG** path guiding, **neural U-Net** denoiser, and **Radiance Cascades** (W8 sprint in flight). RC is currently behind `HybridEngineOptions.rcEnabled` and dispatches cascades each frame against the engine's own raw `GPUBuffer` allocation (no THREE WebGPU renderer dependency); cascade-0 sampling in the shade pass + MIS composition with DDGI / ReSTIR-GI is W8 Phase 3 work (see [plan/w8-rc-mis-composition.md](../../plan/w8-rc-mis-composition.md)).

Provides a class-based `Engine` implementation (`HybridEngine`) that composes:
- **DDGI** (Dynamic Diffuse Global Illumination) — probe-atlas irradiance, updated via compute each frame.
- **RC** (Radiance Cascades, Sannikov 2023) — opt-in via `HybridEngineOptions.rcEnabled`; W8 Phase 2 dispatches the 5-cascade pyramid per frame against a raw-GPUBuffer BVH. Sampling in shade.wgsl + MIS composition lands in Phase 3 (see plan doc).
- **ReSTIR DI** (Reservoir-based Spatiotemporal Importance Resampling) — direct illumination with temporal + spatial reuse.
- **ReSTIR-GI** (Ouyang et al. 2021) — indirect-illumination reservoirs with RIS + temporal + spatial reuse (Sprints 16–17).
- **GTAO** (Jiménez 2016) — half-resolution ground-truth-based ambient occlusion with bilateral upsample (Sprint 15).
- **Denoisers** (selectable via `EngineOptions.denoiser`): `'atrous'`, `'atrous-variance'` (default), `'svgf-real'` (per-channel SVGF on direct + indirect, Sprint 18), `'neural'` (opt-in U-Net; requires preloaded weights — see `tools/neural-denoiser-training/README.md`).
- **PPG** path guiding (Müller et al. 2017) — opt-in via `EngineOptions.ppgEnabled`; sTree + dTree on CPU with WGSL update/guide kernels under `src/ppg/`.

## Denoisers

The post-shade denoise chain is selectable via `HybridEngineOptions.denoiser`.
All modes share the same engine surface — only the post-shade pass changes.

| Mode               | Default | Description                                                                          |
|--------------------|---------|--------------------------------------------------------------------------------------|
| `'atrous'`         |         | Legacy 3-iteration à-trous (no variance weighting).                                  |
| `'atrous-variance'`| ✓       | Welford temporal accumulator + variance lookup + 3-iter à-trous. Current production. |
| `'svgf-real'`      |         | Real Schied 2017 SVGF (T2.H1) — reprojection, moments, 7×7 filter, 5-tap à-trous.    |
| `'neural'`         |         | U-Net neural denoiser (T2.H2 / W10). Requires `neuralWeights` — see below.           |

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

The canonical trained checkpoint ships as `vi-neural-weights.bin` (target ~1.7 MB
at f32 for the default ~426k-parameter spec). The repo does NOT ship a trained
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

- `three >= 0.160.0`
- `three/webgpu` — required for DDGI path (`StorageTexture`, `MeshPhysicalNodeMaterial`)
- `three/tsl` — required for the RC material-wrapper path (`applyDDGIShading`, `giReceiver`, `walkaroundDiffuseLighting`)

ReSTIR-only usage does not trigger the DDGI or RC-material-wrapper paths.

## Architecture

```
src/
  HybridEngine.ts        — Engine implementation (de-React-ified useHybridLayeredGI)
  hostScene/             — Three-side scene adapters consumed by HybridEngine
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
  pipeline/              — Declarative Pass / PassRegistry, denoiser registry, FrameResources
    Pass.ts, PassRegistry.ts — Pass interface + registry (W1-R1)
    passes/              — One file per pass (RIS, RIS-GI, Temporal[GI], Spatial[GI],
                           Shade, IndirectCombine, IndirectTemporalAccum, AtrousIndirect,
                           GTAO, GTAOUpsample, Composite, Resolve, SampleBudget,
                           PPGGuide, PPGUpdate) + declarative passOrder
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
                           ppgUpdate/ppgGuide WGSL kernels (opt-in via ppgEnabled)
  neural/                — U-Net denoiser (Chaitanya 2017): InferenceGraph,
                           inputPacker, unetArchitecture, weights loader (opt-in
                           via denoiser: 'neural'); WGSL kernels under neural/wgsl/
  lib/                   — Shared utilities (nodeMaterialUpgrade)
```

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

**Affected files** (all now in `@vitrum/walkaround-rc`): `src/cascadeDispatch.ts`,
`src/cascadePyramid.ts`, `src/cascadeBuffers.ts`, `src/wgsl/probeRayCast.wgsl.ts`,
`src/wgsl/cascadeMerge.wgsl.ts`.

### DDGI path: `three/webgpu` renderer internals coupling

`probeUpdatePass.ts` accesses `renderer.backend.device` (raw `GPUDevice`) and imports
`StorageTexture` from `three/webgpu`. This is an accepted known cost — `@vitrum/walkaround-hybrid`
requires `three/webgpu` as a peer dep on the DDGI path. ReSTIR-only consumers can
create the `GPUDevice` themselves and avoid this dependency by not calling DDGI APIs.
