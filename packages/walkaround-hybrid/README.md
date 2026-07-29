# @vitrum/walkaround-hybrid

**Stability:** production-grade for the shipped default walkaround pipeline (DDGI, ReSTIR-DI/GI, GTAO, à-trous/SVGF/BMFR denoisers, and RC). The U-Net runtime is fully implemented and checkpoint-gated; explicit selection requires production-ready v2 metadata, while output quality remains a property of the host-supplied checkpoint. PPG and NRC remain opt-in systems with scene-dependent quality. Public API surface is still evolving with host-contract work in `@vitrum/core`.

WebGPU **ReSTIR DI + ReSTIR-GI** walkaround engine with **DDGI** probe updates and atlas sampling in the shade pass, **GTAO** ambient occlusion (half-res + bilateral upsample), per-channel **SVGF / à-trous-variance** denoising on direct + indirect, and opt-in **PPG** path guiding, **neural U-Net** denoiser, and **Radiance Cascades** (W8 shipped). RC is opt-in via `HybridEngineOptions.rcEnabled` and dispatches cascades each frame against the engine's own raw `GPUBuffer` allocation (no THREE WebGPU renderer dependency); cascade-0 sampling + MIS composition with DDGI / ReSTIR-GI are documented in [plan/archive/w8-rc-mis-composition-archived-2026-05-30.md](../../plan/archive/w8-rc-mis-composition-archived-2026-05-30.md).

Provides a class-based `Engine` implementation (`HybridEngine`) that composes:
- **DDGI** (Dynamic Diffuse Global Illumination) — probe-atlas irradiance, updated via compute each frame.
- **RC** (Radiance Cascades, Sannikov 2023) — opt-in via `HybridEngineOptions.rcEnabled`; GPU-validated 2026-06-07 (cascade energy, emitter NEE, oracle-matched). Dispatches the 5-cascade pyramid per frame against a raw-GPUBuffer BVH. Sampling in shade.wgsl (`sampleCascadeC0`) + balance-heuristic MIS (`rcWeight`) shipped (W8 Phase 3).
- **ReSTIR DI** (Reservoir-based Spatiotemporal Importance Resampling) — direct illumination with temporal + spatial reuse.
- **ReSTIR-GI** (Ouyang et al. 2021) — indirect-illumination reservoirs with RIS + temporal + spatial reuse (Sprints 16–17).
- **GTAO** (Jiménez 2016) — half-resolution ground-truth-based ambient occlusion with bilateral upsample (Sprint 15).
- **Denoisers** (selectable via `EngineOptions.denoiser`): `'auto'` (resolves from production-ready host neural metadata or an OIDN model URL, otherwise to the preset/default), `'atrous'`, `'atrous-variance'` (default), `'svgf-real'` (per-channel SVGF on direct + indirect, Sprint 18), `'bmfr'`, `'oidn-final'`, and `'neural'` (checkpoint-gated U-Net; requires preloaded weights — see `tools/neural-denoiser-training/README.md`).
- **PPG** path guiding (Müller et al. 2017) — opt-in/supported via `EngineOptions.ppgEnabled`; GPU-side training atomics feed CPU sTree/dTree updates, `splitOverflowLeaves` adaptively splits high-sample spatial cells, and inline gi-ris guided sampling consumes the learned distribution under `src/ppg/`.
- **NRC** (Neural Radiance Caching) — stable opt-in biased estimator via `nrcEnabled:true`; the default is off by product policy and construction emits a structured bias disclosure.
- **Approximate material lobes** — atlas-backed scalar `specular`, `clearcoat`, `sheen`, `anisotropy`, and `iridescence` controls feed shade-owned direct, analytic, sun, glossy-indirect, ReSTIR-DI, and GI suffix material paths; readable specular, clearcoat factor/roughness/normal, sheen color/roughness, anisotropy, and iridescence factor/thickness maps multiply, perturb, or thin-film-modify the scalar controls. These rows remain approximate because GI receiver/reuse targeting and validation are not rich-lobe-complete.

## Denoisers

The post-shade denoise chain is selectable via `HybridEngineOptions.denoiser`.
All modes share the same engine surface — only the post-shade pass changes.

| Mode               | Default | Description                                                                          |
|--------------------|---------|--------------------------------------------------------------------------------------|
| `'auto'`           |         | Resolves at construction to host `neuralWeights`, then host `extensions['walkaround-hybrid'].oidnModelUrl`, then the preset/default denoiser. Emits a structured resolution warning and does not imply bundled production weights. |
| `'atrous'`         |         | Legacy 3-iteration à-trous (no variance weighting).                                  |
| `'atrous-variance'`| ✓       | Welford temporal accumulator + variance lookup + 3-iter à-trous. Current production. |
| `'svgf-real'`      |         | Real Schied 2017 SVGF (T2.H1) — reprojection, moments, 7×7 filter, 5-tap à-trous.    |
| `'bmfr'`           |         | Explicit full-tier mode: overlapping 32×32 Blockwise Multi-Order Feature Regression (Koskela et al. 2019), using cooperative direct Householder TSQR plus deterministic overlap resolution. Never selected by `'auto'` or allowed on the lite tier. The persistent pass exposes `bmfr-fit` and `bmfr-resolve` timestamp-query labels; no device-independent real-time budget is claimed. |
| `'neural'`         |         | U-Net neural denoiser (T2.H2 / W10). Requires `neuralWeights` — see below.           |
| `'oidn-final'`     |         | Intel OIDN via ONNX Runtime Web (async; requires `extensions['walkaround-hybrid'].oidnModelUrl`). |

### Neural denoiser — weights interface

`denoiser: 'neural'` is an opt-in, production-implemented runtime. It requires a
production-ready v2 `neuralWeights: ModelWeights` checkpoint; missing, legacy,
shape-invalid, or preprocessing-mismatched weights fail during construction.

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

The repo contains limited research checkpoints under
`tools/neural-denoiser-training/checkpoints/` for local experiments, but the
`@vitrum/walkaround-hybrid` package does **not** ship production neural weights.
Those checkpoints are small CPU-trained artifacts, not a production default.
Their exact bytes and research/production classification are pinned by
`tools/neural-denoiser-training/checkpoints/manifest.json` and checked by
`npm run learned-systems-proof-check`.
See `tools/neural-denoiser-training/README.md` for training and export guidance.

## Host Contract

The package root accepts `@vitrum/core` scene data and a raw `GPUDevice` / canvas
context supplied through the engine options. The DDGI, ReSTIR, RC, and denoiser
paths run on raw WebGPU resources; `walkaround-hybrid` has no direct
`three` / `three/webgpu` dependency.

Hosts should convert their scene graph to the `@vitrum/core` `Scene` contract before
constructing `HybridEngine`. The Three.js adapter (`sceneFromThreeJS`) was removed with
the THREE decouple (`e14000c`); host adapters belong outside this package.

This realtime backend does not progressively accumulate toward an SPP target.
Construction-time `maxSamplesPerPixel` and per-frame
`FrameInput.quality.samplesTarget` are therefore rejected instead of being
accepted as no-ops. `FrameInput.quality.bounces` remains the supported per-frame
path-depth control.

## Architecture

```
src/
  HybridEngine.ts        — Engine implementation (orchestrator after W4 decomp)
  HybridEngineRC.ts      — RCSubsystem owning per-engine cascade allocation +
                           dispatch; bridges @vitrum/walkaround-rc into the hybrid frame.
  HybridEnginePrimitiveUpdates.ts — Dispatcher for engine.updatePrimitive (transform-
                           refit / positions-refit / topology-rebuild branches).
  ddgi/                  — DDGI subsystem (probe grid, L2-SH irradiance atlas, update pass)
  pipeline/              — Declarative Pass / PassRegistry, denoiser registry, FrameResources
    Pass.ts, PassRegistry.ts — Pass interface + registry (W1-R1)
    passes/              — One file per pass (RIS, RIS-GI, Temporal[GI], Spatial[GI],
                           Shade, IndirectCombine, IndirectTemporalAccum, AtrousIndirect,
                           GTAO, GTAOUpsample, Composite, Resolve, SampleBudget,
                           PPGUpdate, ReGIRBuild, MotionVectors) + declarative passOrder
    denoisers/           — Denoiser registry (atrous, atrous-variance, svgf-real,
                           bmfr, neural, oidn-final, none; auto resolves before registry lookup)
    WalkaroundGPUPipeline.ts — Iterates PASS_ORDER each frame
  restir/                — ReSTIR BVH + emitter list builders (bvhCore.ts, emitterList.ts)
  shaders/               — WGSL shader strings: ris, risGi, temporal, temporalGi,
                           spatial, spatialGi, shade, indirectCombine,
                           indirectTemporalAccum, gtao, gtaoUpsample,
                           composite, resolve, sampleBudget, welfordTemporal
  ppg/                   — Practical Path Guiding (Müller 2017): CPU sTree/dTree
                           state, GPU training/readback merge, adaptive
                           splitOverflowLeaves, and gi-ris inline guiding
                           (opt-in/supported via ppgEnabled)
  neural/                — checkpoint-gated U-Net denoiser (Chaitanya 2017):
                           InferenceGraph, CPU oracle, liveness planner, v2 weights
                           loader, preprocessing contract, and WGSL kernels
```

The Radiance Cascades subsystem (`cascadePyramid`, `cascadeDispatch`, `cascadeBuffers`,
raw WGSL) was extracted
2026-05-18 into [`@vitrum/walkaround-rc`](../walkaround-rc/). `walkaround-hybrid`
re-exports the public surface for back-compat.

## GRIS reuse boundary

The default ReSTIR-GI spatial and temporal passes use the compact 20-u32
reservoir and retain their realtime-biased reuse rules. `grisReuse: true` is a
construction-time choice that selects a 30-u32 reservoir and a separate shader
graph for a narrower estimator: one cosine-sampled direction whose surface
suffix radiance comes from DDGI, or an environment direction.

The GRIS variant stores the native receiver state, evaluates the bounded
all-technique transformed-density matrix, uses exact surface-shift Jacobians and
environment identity shifts, traces reconnection visibility, folds exact attempt
counts, and rejects history from an earlier scene/lighting mutation epoch.
Receiver material response is applied later in shading. Glossy/specular path
reuse and full path-prefix transport are outside this mode.

This mode does not promise an unbiased path-tracing result: DDGI is a cached
irradiance approximation and finite `restirGiIrrClamp` / `restirGiWCap` controls
remain active. It can cost substantially more than compact reuse because viable
spatial candidates require visibility rays and an O(K^2) density matrix for
K <= 6 domains. The layout is fixed at engine creation.

`restirPtReuse` remains only as a deprecated migration alias for `grisReuse`.
Conflicting values are rejected, and supplying the alias emits a structured
construction warning.

## Known Issues

### DDGI path: raw WebGPU ownership

DDGI now uses the same raw-device ownership model as the rest of the hybrid
engine. Hosts provide lifecycle-owned GPU resources through the engine
constructor; the package no longer imports `three/webgpu` or depends on Three.js
renderer internals for probe updates.

### Shader portability

The walkaround-hybrid and `@vitrum/walkaround-rc` production shaders use the
shared-bvh module-scope value-return loader seam. They do not require the
non-core `unrestricted_pointer_parameters` capability. The shader gate compiles
the exact emitted WGSL on naga and treats every portability failure as fatal;
the production-pipeline gate creates pipelines from that same unmodified source.

### RC subsystem: GPU-validated 2026-06-07

The RC (Radiance Cascades) subsystem was extracted to `@vitrum/walkaround-rc` on
2026-05-18 (W8 follow-up). `walkaround-hybrid` re-exports its public API for back-compat.

**Verification status (2026-06-07):** GPU-validated. RC cascade-zero energy, emitter
NEE wiring (`cRc` gate + probe-cast emitter), and two-scene gate all GPU-proven with
oracle-matched results. DDGI irradiance migrated to L2 SH (3×3 cells, seam-free,
2026-06-07). See `plan/archive/` for the full audit trail and `@vitrum/walkaround-rc`
for the subsystem source.

**Open items:** glossy and metal surfaces now receive specular indirect GI via
`lo_indirectSpecular` (GGX lobe re-weighting of the same ReSTIR-GI reservoir — B1
done, Wave A). DDGI's diffuse `lo_indirect` still gates on `isGlass || isMetal` (the
reservoir p̂ is Lambertian); metals use `lo_indirectSpecular` instead. Smooth/rough
glass uses the current one-interface refracted GI path; multi-interface transparent
transport remains tracked in road-to-100. HDRI is fully directional-IBL-capable
(importance-sampled DI NEE candidate in the RIS loop + DDGI probe misses, 2026-06-10,
`caab499`; ledger grade `native`). `updateEnvironment` rebuilds directional CDFs at
runtime; scalar-tint fallback active when no env map is loaded. See `plan/road-to-100.md`.

**Directional lighting (sun NEE):** as of 2026-06-10, the `primaryLightDir` /
`primaryLightIntensity` directional emitter is wired to `lo_sunNEE` in `shade.wgsl`
— a deterministic shadow ray from each opaque surface toward the sun direction,
evaluated with the full `evalGGX` BRDF (diffuse + GGX specular). This is
**default-ON** (no flag required); sharp sun shadows now appear on generic scenes
without the stained-glass opt-in. Prior to this, direct sun at opaque surfaces was
available only via the `SG_FLAG_SUN_CAUSTIC` path in `lo_sg_caustic` (the
stained-glass extension, default OFF). The DDGI-indirect term (`lo_indirect`) carries
sun-bounce indirect light (sun → bounce wall → receiver) which is disjoint from the
direct term (no double-count — see `lo_sunNEE` comment in `shade.wgsl.ts`). The
stained-glass caustic (lo_sg_caustic) remains flag-gated and handles tinted-glass
transmittance; the two terms are non-overlapping (lo_sg_caustic uses
`bvhTraceTintedVisibility`, lo_sunNEE uses binary opaque-only shadow ray).
