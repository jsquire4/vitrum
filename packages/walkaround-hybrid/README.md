# @vitrum/walkaround-hybrid

**Stability:** production-grade for the shipped walkaround pipeline (DDGI, ReSTIR-DI/GI, GTAO, SVGF, RC, PPG, neural). Public API surface is still evolving with host-contract work in `@vitrum/core`.

WebGPU **ReSTIR DI + ReSTIR-GI** walkaround engine with **DDGI** probe updates and atlas sampling in the shade pass, **GTAO** ambient occlusion (half-res + bilateral upsample), per-channel **SVGF / à-trous-variance** denoising on direct + indirect, and opt-in **PPG** path guiding, **neural U-Net** denoiser, and **Radiance Cascades** (W8 shipped). RC is opt-in via `HybridEngineOptions.rcEnabled` and dispatches cascades each frame against the engine's own raw `GPUBuffer` allocation (no THREE WebGPU renderer dependency); cascade-0 sampling + MIS composition with DDGI / ReSTIR-GI are documented in [plan/archive/w8-rc-mis-composition-archived-2026-05-30.md](../../plan/archive/w8-rc-mis-composition-archived-2026-05-30.md).

Provides a class-based `Engine` implementation (`HybridEngine`) that composes:
- **DDGI** (Dynamic Diffuse Global Illumination) — probe-atlas irradiance, updated via compute each frame.
- **RC** (Radiance Cascades, Sannikov 2023) — opt-in via `HybridEngineOptions.rcEnabled`; GPU-validated 2026-06-07 (cascade energy, emitter NEE, oracle-matched). Dispatches the 5-cascade pyramid per frame against a raw-GPUBuffer BVH. Sampling in shade.wgsl (`sampleCascadeC0`) + balance-heuristic MIS (`rcWeight`) shipped (W8 Phase 3).
- **ReSTIR DI** (Reservoir-based Spatiotemporal Importance Resampling) — direct illumination with temporal + spatial reuse.
- **ReSTIR-GI** (Ouyang et al. 2021) — indirect-illumination reservoirs with RIS + temporal + spatial reuse (Sprints 16–17).
- **GTAO** (Jiménez 2016) — half-resolution ground-truth-based ambient occlusion with bilateral upsample (Sprint 15).
- **Denoisers** (selectable via `EngineOptions.denoiser`): `'atrous'`, `'atrous-variance'` (default), `'svgf-real'` (per-channel SVGF on direct + indirect, Sprint 18), `'neural'` (opt-in U-Net; requires preloaded weights — see `tools/neural-denoiser-training/README.md`).
- **PPG** path guiding (Müller et al. 2017) — opt-in/experimental via `EngineOptions.ppgEnabled`; sTree + dTree on CPU with WGSL update training plus inline gi-ris guided sampling under `src/ppg/`. Directional flux training is live; spatial sTree never splits (single global cell — road-to-100).
- **Approximate material lobes** — atlas-backed scalar `specular`, `clearcoat`, and `sheen` controls feed shade-owned direct, analytic, sun, and glossy-indirect lighting; readable specular color/intensity maps multiply the scalar specular controls. These rows remain approximate because upstream ReSTIR candidate PDFs/payloads and the remaining lobe texture maps are not lobe-complete.

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

`denoiser: 'neural'` is **opt-in / experimental**. It requires `neuralWeights: ModelWeights`
to be passed to the engine constructor; missing weights produce a clear validation error at
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
at f32 for the default ~535k-parameter spec). **The repo does NOT ship a trained
checkpoint** — see `tools/neural-denoiser-training/README.md` for training.

**Smoke-test path (no trained weights):** `buildRandomWeightsForSpec(spec, seed)`
synthesises deterministic He-init random weights. The pipeline runs end-to-end
but the denoised output will NOT be visually clean — wiring verification only.

## Host Contract

The package root accepts `@vitrum/core` scene data. The DDGI path (`probeUpdatePass.ts`)
accesses `renderer.backend.device` and imports `StorageTexture` from `three/webgpu` —
`three/webgpu` is therefore a peer dependency **only if you use DDGI** (which is the
default production path). ReSTIR-only consumers that supply their own raw `GPUDevice`
without calling DDGI APIs do not need the Three.js peer dep.

Hosts should convert their scene graph to the `@vitrum/core` `Scene` contract before
constructing `HybridEngine`. The Three.js adapter (`sceneFromThreeJS`) was removed with
the THREE decouple (`e14000c`); host adapters belong outside this package.

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
                           neural, oidn-final, none)
    WalkaroundGPUPipeline.ts — Iterates PASS_ORDER each frame
  restir/                — ReSTIR BVH + emitter list builders (bvhCore.ts, emitterList.ts)
  shaders/               — WGSL shader strings: ris, risGi, temporal, temporalGi,
                           spatial, spatialGi, shade, indirectCombine,
                           indirectTemporalAccum, gtao, gtaoUpsample,
                           composite, resolve, sampleBudget, welfordTemporal
  ppg/                   — Practical Path Guiding (Müller 2017): sTree + dTree on CPU,
                           ppgUpdate WGSL + gi-ris inline guiding (opt-in/experimental
                           via ppgEnabled — spatial sTree split is road-to-100)
  neural/                — U-Net denoiser (Chaitanya 2017): InferenceGraph,
                           inputPacker, unetArchitecture, weights loader (opt-in/
                           experimental via denoiser: 'neural'); WGSL kernels under neural/wgsl/
```

The Radiance Cascades subsystem (`cascadePyramid`, `cascadeDispatch`, `cascadeBuffers`,
`giReceiver`, `walkaroundDiffuseLighting`, `bvhCompute`, raw WGSL) was extracted
2026-05-18 into [`@vitrum/walkaround-rc`](../walkaround-rc/). `walkaround-hybrid`
re-exports the public surface for back-compat.

## Bias & the unbiased GRIS variant

The default ReSTIR-GI spatial and temporal reuse (`restirPtReuse: false`) is
**intentionally biased** for the realtime frame budget. The unbiased GRIS
reconnection-shift variant (Lin et al. 2022, SIGGRAPH) is available as opt-in
via `HybridEngineOptions.restirPtReuse: true`.

### Default (OFF) bias sources

| # | Source | File:line | Manifestation |
|---|--------|-----------|---------------|
| B1 | **Jacobian clamp `[0.1, 10]`** | `shaders/jacobianShift.wgsl.ts` — `return clamp(J, 0.1, 10.0)` | Systematic over/under-weighting of neighbours with extreme solid-angle ratios (depth-discontinuity edges in the indirect channel). Stationary — does not grow with frame count. |
| B2 | **No reconnection-visibility ray** | `shaders/spatialGi.wgsl.ts` (OFF, `SPATIAL_GI_WGSL`) and `shaders/temporalGi.wgsl.ts` (OFF, `TEMPORAL_GI_WGSL`) — reuse weight applied without occlusion test | Indirect light bleeds through geometry at depth discontinuities; occluded neighbours contribute energy they should not. Partially suppressed by the normal/depth consistency test but not eliminated. |
| B3 | **No full GBH MIS** | `spatialGi.wgsl.ts` and `temporalGi.wgsl.ts` (OFF variants) — weight `w_q = pHatZ * rQ.W * Mq * J` with no MIS denominator | Contribution weights do not sum to 1 (Lin 2022 GBH sense); slightly over-energised indirect where M counts differ across pixels. |
| B4 | **Centroid p̂** (shared ON and OFF) | `shaders/restirPHat.wgsl.ts` — `let centroid = (e.vA + e.vB + e.vC) / 3.0` | DI target distribution uses emitter centroid instead of sampled point xi; small error for compact lights, larger for wide-area sources at close range. Not fixed by enabling GRIS. |

### Cost of enabling `restirPtReuse: true`

The unbiased path adds per accepted spatial/temporal reuse candidate:
- one BVH shadow ray (reconnection-visibility test), and
- the full generalized-balance MIS cross-evaluation over all K_SPATIAL_GI = 5
  gathered neighbours (O(K²) target evaluations in the spatial pass).

The GI spatial + temporal passes are the **realtime frame-time bottleneck** for
typical walkaround usage — enabling GRIS roughly doubles their cost on scenes
where many neighbours pass the geometric-consistency test. This is the regime
argument for keeping `false` as the default.

### When to enable

Enable `restirPtReuse: true` when: the scene is rendered in a converged or
offline-ish mode (sustained still camera, progressive accumulation, A/B
validation); you observe energy bleed or indirect light leak at geometry
boundaries; or you are running the V19 GPU unbiasedness validation
(`HARDWARE-VALIDATION-NEEDS.md`).

The gate is COMPILE-TIME (fixed at engine creation — changes the GI pipeline
layout and shader variant). Same pattern as `rcEnabled`, `ppgEnabled`, `regir`.

## Known Issues

### DDGI path: `three/webgpu` renderer internals coupling

`probeUpdatePass.ts` accesses `renderer.backend.device` (raw `GPUDevice`) and imports
`StorageTexture` from `three/webgpu`. This is an accepted known cost — `@vitrum/walkaround-hybrid`
requires `three/webgpu` as a peer dep on the DDGI path. ReSTIR-only consumers can
create the `GPUDevice` themselves and avoid this dependency by not calling DDGI APIs.

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
reservoir p̂ is Lambertian); metals use `lo_indirectSpecular` instead. Glass refracted
GI remains out of scope (empty reservoir, tracked in road-to-100). HDRI is fully directional-IBL-capable
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
