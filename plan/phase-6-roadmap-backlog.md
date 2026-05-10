# Phase 6 roadmap — implementation backlog

**Canonical narrative**: [`plan/phase-6-roadmap.md`](./phase-6-roadmap.md) (goals, risks, mode-scope matrix, decision log).

**Purpose**: single checklist of *definition-of-done* items pulled from the roadmap so progress can be tracked without rereading full sprint prose. Each sprint has a matching benchmark stub: [`plan/sprint-<N>-benchmark.md`](./sprint-1-benchmark.md).

**Legend**: unchecked = not done in-repo; this file is updated as sprints land.

---

## Sprint 1 — PT preview speed wins

- [x] **1.1** Outdoor HDRI presets load with HTTP 200 (no silent 404). *Library slice*: `@vitrum/pt-webgl` exports working preset URLs + optional RGBE loader helper; hosts wire IBL. Cornell: `?hdri=<presetId>`.
- [x] **1.2** Preview bounce count = 3 via `PT_PREVIEW_BOUNCES` / `PT_PREVIEW_OPTIONS.bounces`.
- [x] **1.3** Preview uses `resolutionFactor: 0.5` in `PT_PREVIEW_OPTIONS` (0.5× render resolution vs viewport — bilinear resolve).
- [x] **1.4** Skip expensive post-process for first N samples after reset. `FrameOutput.suggestSkipPostProcess` now emitted by `@vitrum/pt-webgl` using `PT_POSTPROCESS_WARMUP_SAMPLES = 8`.
- [x] **1.5** OrbitControls damping factor 0.15 in Cornell example (interactive orbit).

## Sprint 2 — Per-cell luminance precompute

- [x] **2.1** Walkaround: `cellPower` buffer populated/uploaded in `buildSceneBVH` + `WalkaroundGPUPipeline.initialize`.
- [x] **2.2** PT fork: `light.power` in lights texture (`three-gpu-pathtracer` commit `5388ef0`, packed in `s2.a` and read in GLSL `Light` struct).
- [x] **2.3** Round-trip: doubling `Le[i]` doubles `cellPower[i]` (walkaround unit test in `sprint2-cellPower.test.ts`).
- [x] **2.4** No intentional visual delta (foundation-only CPU/GPU buffer prep in walkaround path).

## Sprint 3 — Sampling theory upgrade (PT)

- [ ] **3.1** Mixture PDF (BSDF + env + light) replaces binary branch (shader verified).
- [ ] **3.2** Light-tree CDF on scene CPU; binary search in GLSL.
- [ ] **3.3** Back-face NEE resample up to 4×.
- [ ] **3.4** Variance benchmark: ≥3× floor-pixel stddev reduction vs baseline at 192 spp (`plan/sprint-3-benchmark.md`).

## Sprint 4 — BSDF cost reduction (PT)

- [ ] **4.1** `lobeMask` bitfield at `getSurfaceRecord`.
- [ ] **4.2** Sheen / clearcoat / iridescence gated on mask bits.
- [ ] **4.3** `liteMode` for indirect (`depth > 1`): Lambertian + GGX-only.
- [ ] **4.4** Material LOD: skip heavy texture reads past depth threshold.
- [ ] **4.5** Profile: ≥40% ms/sample reduction on glass+came scene.

## Sprint 5 — Analytic came + MRT scaffold (PT)

- [x] **5.1** Came UBO populated (≤500 segments) via `packCameUBO` + tests.
- [ ] **5.2** `intersectCameSegment` / `intersectCameNode` in shader; closest hit vs BVH.
- [ ] **5.3** Synthetic `SurfaceHit` for came (material, normal, UV).
- [ ] **5.4** Profile: ≥30% BVH node-visit reduction.
- [x] **5.5** Tier fallback: `maxFragmentUniformVectors < 256` disables analytic came (mesh fallback) in `@vitrum/pt-webgl` capabilities/init path.
- [ ] **5.6** MRT: color + normal/depth + albedo outputs as per Decision 12.
- [x] **5.7** Mesh came remains in BVH as fallback (analytic path is additive and capability-gated).

## Sprint 6 — Visible-quality wins

- [ ] **6.1** Rough refraction lobe (GGX on refracted ray).
- [ ] **6.2** Edge-stopping spatial filter first in post chain (PT preview).
- [ ] **6.3** Spatial filter auto-off when `samples > 24`.
- [ ] **6.4** A/B: hammered glass / preview speckle vs clean.

## Sprint 7 — Volume + SSS (PT)

- [ ] **7.1** Single-scatter SSS for flagged translucent materials.
- [ ] **7.2** Homogeneous volume medium (density, albedo, g).
- [ ] **7.3** Equi-angular / Kulla–Conty-style volume NEE.
- [ ] **7.4** UI hooks (host) for haze + SSS overrides.
- [ ] **7.5** A/B: god rays + opalescent milky glow.

## Sprint 8 — RGB-as-3λ spectral + Jakob+Hanika rider (PT)

- [ ] **8.1** Cauchy IOR per RGB wavelength triplet.
- [ ] **8.2** `dispersionStrength` / material wiring for bevel glass.
- [x] **8.3** Jakob+Hanika spectral upsampling implemented in `@vitrum/shared-samplers` (`rgbToSpectralCoefficients`, `evaluateSpectrum`) with tests.
- [ ] **8.4** A/B: smooth prism edges vs tri-band fan.

## Sprint 9 — Convergence (walkaround)

- [x] **9.1** Welford variance buffer (RG32Float) allocated by walkaround `createVarianceBuffer`.
- [x] **9.2** Versioned `WelfordVariance` struct in `common.wgsl` with structural tests.
- [ ] **9.3** Per-pixel sample tier from variance.
- [ ] **9.4** Checkerboard temporal upsampling + reprojection.
- [ ] **9.5** A/B: ≥30% noise reduction at 16 samples.
- [ ] **9.6** Motion: ghosting within acceptable bound.

## Sprint 10a — SVGF (walkaround + PT preview)

- [ ] **10a.1** Walkaround à-trous replaced by SVGF.
- [ ] **10a.2** PT preview spatial path uses SVGF-class filter.
- [ ] **10a.3** A/B: 8 samples ≈ 64-sample reference on diffuse.

## Sprint 10b — OIDN final (PT final)

- [x] **10b.1** ORT-Web lazy loading + session cache implemented in `@vitrum/shared-denoisers/src/oidnBridge.ts`.
- [ ] **10b.2** Denoise action on converged buffer + G-buffers.
- [ ] **10b.3** Export denoised image alongside raw.
- [ ] **10b.4** &lt;2 s for 2K typical GPU.
- [x] **10b.5** EPs default to `['webnn','webgpu','wasm']` in OIDN bridge.

## Sprint 10c — BDPT (PT final, gated)

- [ ] **10c.1** Light subpath storage to MRT/SSBO (N≤3).
- [ ] **10c.2** Eye–light connections + joint PDF + MIS.
- [ ] **10c.3** A/B: floor caustic ~256 vs ~1024+ spp.
- [ ] **10c.4** Trigger review after Sprint 7 hero comparison.

## Sprint 11 — PPG (walkaround)

- [x] **11.1** PPG grid/leaf/sample buffers (~10K cap) implemented and tested in walkaround resource manager.
- [x] **11.2** Directional bins (16) + WGSL sample/update logic implemented (`ppgSample.wgsl`, `ppgUpdate.wgsl`) with structural tests.
- [ ] **11.3** Online update + sampling from learned PDF.
- [ ] **11.4** A/B: 30 vs 90 samples indirect scene; bail-out if &lt;30 fps.

## Sprint 12 — Hero-wavelength spectral (PT, gated)

- [ ] **12.1** Wavelength + scalar throughput path payload.
- [ ] **12.2** Wavelength-aware BSDF sites.
- [ ] **12.3** Spectral accumulation + CIE reconstruction.
- [ ] **12.4** A/B vs Sprint 8; trigger on material list sign-off.

## Sprint 13 — Custom WebGPU neural denoiser (walkaround, gated)

- [ ] **13.1** Training pipeline for noisy/clean pairs.
- [ ] **13.2** UNet-ish graph on WebGPU compute.
- [ ] **13.3** ~10–50 ms inference typical GPU.
- [ ] **13.4** A/B vs OIDN at low spp; Decision 14 triggers only.

---

## External RFE / contract stubs

Tracked in roadmap §3 footnote and [`plan/cursor-recommended-plan.md`](./cursor-recommended-plan.md) §13.2 (separate from sprint table above).

---

## Cross-links

- Sprint 0 DoD: [`plan/sprint-0-api-contract.md`](./sprint-0-api-contract.md)
- Package map: [`plan/library-architecture.md`](./library-architecture.md)
