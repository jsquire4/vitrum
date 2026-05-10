# Phase 6 Status — vitrum library-side completion

**Date**: 2026-05-09
**Branch**: main
**Commits**: 7bda9c4..0cebaf9 (6 Phase 6 sprint commits; 28 total on main)

---

## What's done (vitrum library-side)

### Baseline sprints (1–9): COMPLETE

- **Sprint 1** — PT preview speed wins. `PT_PREVIEW_BOUNCES = 3` constant landed in
  `@vitrum/pt-webgl/src/constants.ts`; `PT_PREVIEW_OPTIONS` export. Host-side
  checklist (HDRI 404, DPR halving, EffectComposer skip, OrbitControls damping)
  documented in `plan/sprint-1-host-checklist.md`. Commit range: `7bda9c4`.
  Mode scope: PT preview.

- **Sprint 2** — Per-cell luminance precompute. `userData.cellPower` populated on
  every `THREE.Light` produced by `vitrumSceneToThree()` in `@vitrum/three-bindings`.
  Fork patch for `lights_struct.glsl.js` + `LightsInfoUniformStruct.js` documented
  in `plan/sprint-2-pt-fork-patch.md`; fork patch deferred to Sprint 3 application
  window. Commit range: `7bda9c4`. Tests: 8 (walkaround cellPower suite). Mode
  scope: both.

- **Sprint 3** — Sampling theory upgrade. Light tree CDF builder (`buildLightTreeCDF`)
  + mixture PDF MIS heuristics (`mixturePDF`, `sampleMixturePDF`) landed in
  `@vitrum/shared-samplers`. Back-face NEE resample logic documented. Fork patch
  (`direct_lighting.glsl.js`, new `light_tree.glsl.js`) in
  `plan/sprint-3-pt-fork-patch.md`. Commit range: `ef31201`. Tests: included in
  shared-samplers suite (54 total). Mode scope: PT.

- **Sprint 4** — BSDF cost reduction. `lobeMask` bitfield helper + `liteMode`
  material flag authored in `@vitrum/shared-samplers`. Fork patch
  (`bsdf_functions.glsl.js`, `surface_record_struct.glsl.js`,
  `get_surface_record_function.glsl.js`) documented in `plan/sprint-4-pt-fork-patch.md`.
  Commit range: `ef31201`. Mode scope: PT.

- **Sprint 5** — Analytic came geometry + MRT G-buffer scaffold. Came segment
  types + `cameUniformPacker` in `@vitrum/pt-webgl`. MRT G-buffer layout locked
  (Decision 12): `gColor` (loc 0), `gNormalDepth` (loc 1), `gAlbedo` (loc 2),
  all RGBA16F or RGBA8. Fork patch documented in `plan/sprint-5-pt-fork-patch.md`;
  MRT layout spec in `plan/sprint-5-mrt-gbuffer-spec.md`. Commit range: `8f5e6d3`.
  Mode scope: PT.

- **Sprint 6** — Visible-quality wins. Rough refraction lobe authored as GLSL
  snippet in `@vitrum/pt-webgl` (`roughRefractionGLSL`). Edge-stopping spatial
  filter 37-tap hexagonal kernel WGSL in `@vitrum/shared-denoisers`
  (`spatialFilter.wgsl.ts`). Fork patch in `plan/sprint-6-pt-fork-patch.md`.
  Commit range: `8f5e6d3`. Mode scope: PT preview (spatial filter); both (rough
  refraction).

- **Sprint 7** — Volumetric + SSS. Phase function utilities (Henyey-Greenstein,
  equi-angular PDF) + SSS material flag helpers in `@vitrum/shared-samplers`.
  Fork patch (`path_tracer.glsl.js` loop restructure, new `volume_march.glsl.js`,
  HG phase function) in `plan/sprint-7-pt-fork-patch.md`. Commit range: `8f5e6d3`.
  Mode scope: PT.

- **Sprint 8** — RGB-as-3λ spectral + Jakob+Hanika rider. Cauchy dispersion
  formula helper + `dispersionStrength` material field + Jakob+Hanika
  3-coefficient polynomial spectrum utilities in `@vitrum/shared-samplers`.
  Fork patch (`bsdf_functions.glsl.js` per-channel IOR) in
  `plan/sprint-8-pt-fork-patch.md`. Commit range: `8f5e6d3`. Mode scope: PT.

- **Sprint 9** — Convergence efficiency + Welford struct rider. `WelfordVariance`
  struct + `welfordUpdate` / `welfordVariance` helpers in `common.wgsl.ts` (version
  1, Decision 13). `sampleBudgetKernel` compute shader + `resolveKernel` (checkerboard
  upsampling) authored. `createVarianceBuffer` helper exported from `resourceManager.ts`.
  Dispatch integration deferred (see below). Full integration spec in
  `plan/sprint-9-walkaround-integration.md`. Commit range: `f5bfde0`. Tests: 63
  (Welford suite). Mode scope: walkaround.

### Frontier sprints

#### Sprint 10a — SVGF spatiotemporal denoising: COMPLETE (vitrum-side)

`@vitrum/shared-denoisers` ships `svgf.wgsl.ts` with two entry points
(`svgfVarianceMain`, `svgfAtrousMain`), `svgfBindings.ts` with TypeScript
descriptor types (`SVGFVarianceBindGroupLayout`, `SVGFAtrousBindGroupLayout`,
`SVGF_DEFAULT_UNIFORMS`), and packing helpers (`packSVGFVarianceUniforms`,
`packSVGFUniforms`). Tests: included in shared-denoisers suite (69 total).
Walkaround wiring deferred — see `plan/sprint-10a-walkaround-integration.md`.
PT preview wiring (replacing Sprint 6 hexagonal filter) deferred — see
`plan/sprint-10a-pt-fork-patch.md`. Commit range: `4aee481`.

#### Sprint 10b — OIDN ONNX bridge: COMPLETE (vitrum-side)

`@vitrum/shared-denoisers` ships `oidnBridge.ts` — lazy ONNX Runtime Web
wrapper exporting `denoiseFinal`, `preloadOIDNModel`, `clearOIDNCache`.
Execution providers configured as `['webnn', 'webgpu', 'wasm']` per Decision 11.
Session caching by model URL. `onnxruntime-web` declared as optional peer
dependency. Host-side checklist (install ORT-Web, bundle OIDN ONNX model,
"Denoise" button, float32 readback, model pre-warm, denoised PNG save) in
`plan/sprint-10b-host-checklist.md`. Commit range: `4aee481`.

#### Sprint 10c — BDPT for caustics: DEFERRED

See `plan/sprint-10c-deferred.md`. Trigger criterion requires GPU render
of Sprint 7 output; autonomous-mode session had no rendering environment.

#### Sprint 11 — PPG path guiding: COMPLETE (vitrum-side, scaffold only — dispatch deferred)

`@vitrum/walkaround-hybrid` ships:
- `src/ppg/types.ts` — `PPGDirectionalBin`, `PPGQuadTreeNode`, `PPGSpatialCell`,
  `PPGBuffers`, `PPG_MAX_SPATIAL_CELLS` (10,000), `PPG_DIRECTIONS` (16), byte-stride
  constants.
- `src/ppg/wgsl/ppgSample.wgsl.ts` — `PPG_SAMPLE_WGSL` WGSL fragment for guided
  direction sampling, @group(2) bindings.
- `src/ppg/wgsl/ppgUpdate.wgsl.ts` — `PPG_UPDATE_WGSL` compute kernel with atomic
  fixed-point radiance accumulation.
- `createPPGBuffers` / `destroyPPGBuffers` in `resourceManager.ts`.
- `HybridEngineOptions.ppgEnabled`, `HybridEngine.setPPGEnabled()`,
  `HybridEngine.ppgEnabled` getter.

Dense linear array kd-tree with brute-force O(N) nearest-cell lookup (Decision: see
`plan/sprint-11-ppg-integration.md`, kd-tree section). Dispatch wiring deferred —
see `plan/sprint-11-ppg-integration.md`. Tests: 81 (PPG suite). Commit range:
`0cebaf9`.

#### Sprint 12 — Hero-wavelength spectral: DEFERRED

See `plan/sprint-12-deferred.md`. Trigger condition not met (no spectral-correctness-
required materials in scope). Sprint 8b's Jakob+Hanika rider may make this unnecessary
for the bevel use case (Decision 10).

#### Sprint 13 — Custom WebGPU neural denoiser: DEFERRED

See `plan/sprint-13-deferred.md`. Three-AND trigger criterion; cannot evaluate any
of the three legs without GPU verification + WebNN timeline tracking.

---

## What's pending (host-side or fork-side)

### Fork patches needed

Each of these is documented and ready to apply. Apply in sprint order (2 before 3,
etc.). Visual A/B verification required after each; reference renders saved to
`tools/reference-renders/`.

| Sprint | Patch doc | Fork files | DoD gate |
|---|---|---|---|
| 2 | `plan/sprint-2-pt-fork-patch.md` | `lights_struct.glsl.js`, `LightsInfoUniformStruct.js` | Pixel-diff identical pre/post (passive field) |
| 3 | `plan/sprint-3-pt-fork-patch.md` | `direct_lighting.glsl.js`, new `light_tree.glsl.js`, `sampling.glsl.js` | ≥3× floor-pixel stddev reduction at 192 samples |
| 4 | `plan/sprint-4-pt-fork-patch.md` | `bsdf_functions.glsl.js`, `surface_record_struct.glsl.js`, `get_surface_record_function.glsl.js` | ≥40% ms/sample reduction on glass-and-came scene |
| 5 | `plan/sprint-5-pt-fork-patch.md` | `trace_scene_function.glsl.js`, `shape_intersection_functions.glsl.js`, `PhysicalPathTracingMaterial.js` | ≥30% BVH node-visit reduction; MRT targets allocated |
| 6 | `plan/sprint-6-pt-fork-patch.md` | `bsdf_functions.glsl.js` (rough refraction lobe) | Visual A/B: hammered glass shows volumetric scatter blur |
| 7 | `plan/sprint-7-pt-fork-patch.md` | `path_tracer.glsl.js`, new `volume_march.glsl.js`, `bsdf.glsl.js`, `materials_data_function.glsl.js` | God-ray shafts visible at ≥0.5 density; opal shows milky glow |
| 8 | `plan/sprint-8-pt-fork-patch.md` | `bsdf_functions.glsl.js` (per-channel IOR, Jakob+Hanika rider) | Bevel rainbow shows smooth spectrum; no chromatic aberration on non-bevels |
| 10a (PT preview) | `plan/sprint-10a-pt-fork-patch.md` | `PhysicalPathTracingMaterial.js` (gMotion MRT channel, prevViewProjMatrix uniform) | SVGF integration in PTSVGFDenoiser.tsx; motion vectors populated |

### Host-side changes needed

- **Sprint 1**: `plan/sprint-1-host-checklist.md` — H1 (HDRI 404), H2 (DPR halving),
  H3 (EffectComposer skip on samples ≤ 8), H4 (OrbitControls dampingFactor = 0.15).
- **Sprint 5 host**: Came segment data construction from `EdgeLines.tsx` equivalent;
  call `cameUniformPacker` to populate UBO; allocate `WebGLMultipleRenderTargets`
  for the 3-channel MRT per `plan/sprint-5-mrt-gbuffer-spec.md`.
- **Sprint 6 host**: Wire `PTSpatialDenoiser.tsx` (37-tap hexagonal filter) as first
  `postprocessing` Effect in EffectComposer chain; auto-disable above 24 samples.
- **Sprint 7 host**: `PhotorealismControls.tsx` UI additions — atmospheric haze section
  (enable, density 0–2.0, color, anisotropy) and glass SSS section (per-type sigma_t
  override). Sign-off question open: hardcoded sigma_t lookup table vs. per-type
  sliders (§7, item 1 of roadmap).
- **Sprint 8 host**: `bevels.ts` baker — remove fake noise-split; import
  `dispersionStrength` from `glassMaterialProfiles.ts` and pass to shader.
- **Sprint 9 host**: adaptive sampling integration per
  `plan/sprint-9-walkaround-integration.md` (pipeline compilation, BGL, texture
  allocation, dispatch order update).
- **Sprint 10a host (walkaround)**: SVGF wiring per
  `plan/sprint-10a-walkaround-integration.md` (replace à-trous dispatch loop).
- **Sprint 10b host**: `plan/sprint-10b-host-checklist.md` — install
  `onnxruntime-web`, bundle OIDN ONNX model, "Denoise" button, float32 readback,
  `preloadOIDNModel` pre-warm, denoised PNG save option, `clearOIDNCache` on exit.
- **Sprint 11 host**: PPG dispatch wiring per `plan/sprint-11-ppg-integration.md`
  (pipeline compilation, BGL, shade pass changes, frame-parity gating).

### Integration deferred (vitrum-side, requires GPU verification)

- **Sprint 9 adaptive sampling dispatch path**: `sampleBudgetKernel` and
  `resolveKernel` shaders authored; `varianceBuffer` allocated; pipeline
  compilation, BGL construction, and dispatch-order integration deferred. See
  `plan/sprint-9-walkaround-integration.md`.
- **Sprint 11 PPG dispatch wiring**: shaders authored, buffers allocated (opt-in
  via `ppgEnabled`); pipeline compilation and shade-pass changes deferred. See
  `plan/sprint-11-ppg-integration.md`.

---

## Stats

- **Vitrum library packages**: 8 (core, three-bindings, shared-bvh, shared-samplers,
  shared-denoisers, pt-webgl, pt-webgpu stub, walkaround-hybrid)
- **Total tests**: 346 passing (16 test files across 6 packages)
  - pt-webgl: 18
  - shared-bvh: 11
  - shared-denoisers: 69
  - shared-samplers: 54
  - three-bindings: 1
  - walkaround-hybrid: 193
- **TypeScript strict**: clean across workspace (`npm run typecheck` passes)
- **LOC vitrum library code**: 19,746 total (TypeScript source files in `packages/`,
  excluding `.d.ts` and `node_modules`)
- **Phase 6 sprint commits**: 6 (7bda9c4 through 0cebaf9)
- **Total commits on main**: 28

---

## Sprint artifact index

| Document | Summary |
|---|---|
| `plan/sprint-0-api-contract.md` | Sprint 0 API contract — `@vitrum/core` public types, backend stubs, workspace tsc |
| `plan/sprint-1-host-checklist.md` | Sprint 1 host items — HDRI 404, DPR halving, EffectComposer skip, OrbitControls damping |
| `plan/sprint-2-pt-fork-patch.md` | Sprint 2 fork patch — `float power` field on Light struct; `s5.a` packing in LightsInfoUniformStruct |
| `plan/sprint-3-pt-fork-patch.md` | Sprint 3 fork patch — mixture PDF + light tree CDF + back-face NEE resample in fork shaders |
| `plan/sprint-4-pt-fork-patch.md` | Sprint 4 fork patch — lobeMask bitfield + lite BSDF + material LOD in fork shaders |
| `plan/sprint-5-pt-fork-patch.md` | Sprint 5 fork patch — analytic H-channel came intersection + `traceScene` hybrid BVH/analytic |
| `plan/sprint-5-mrt-gbuffer-spec.md` | Sprint 5 MRT G-buffer layout — gColor/gNormalDepth/gAlbedo format + channel details, locked |
| `plan/sprint-6-pt-fork-patch.md` | Sprint 6 fork patch — rough refraction lobe additive on BSDF refracted direction |
| `plan/sprint-7-pt-fork-patch.md` | Sprint 7 fork patch — volume march loop restructure + SSS TRANSLUCENT flag + HG phase + equi-angular PDF |
| `plan/sprint-8-pt-fork-patch.md` | Sprint 8 fork patch — Cauchy per-channel IOR + Jakob+Hanika polynomial spectrum rider |
| `plan/sprint-9-walkaround-integration.md` | Sprint 9 integration spec — Welford variance buffer, sample-budget + resolve shader wiring |
| `plan/sprint-10a-walkaround-integration.md` | Sprint 10a walkaround integration — SVGF two-pass dispatch replacing à-trous |
| `plan/sprint-10a-pt-fork-patch.md` | Sprint 10a PT preview integration — motion vector MRT channel, PTSVGFDenoiser replacement |
| `plan/sprint-10b-host-checklist.md` | Sprint 10b host checklist — ORT-Web install, OIDN model bundle, "Denoise" button, float32 readback |
| `plan/sprint-11-ppg-integration.md` | Sprint 11 PPG integration spec — pipeline compilation, BGL, shade-pass changes, frame-parity |
| `plan/sprint-10c-deferred.md` | Sprint 10c (BDPT) deferred — trigger criterion and un-defer instructions |
| `plan/sprint-12-deferred.md` | Sprint 12 (hero spectral) deferred — trigger material list, un-defer criteria |
| `plan/sprint-13-deferred.md` | Sprint 13 (neural denoiser) deferred — 3-AND trigger criteria, bail-out criterion |

---

## Known issues / residual risks

- **RC subsystem TSL→raw conversion** (RD-12): structurally tested (39 binding tests
  in `__tests__/rc-bindings.test.ts`) but not GPU-verified. See `packages/walkaround-hybrid/`
  README Known Issues section.
- **Welford struct duplication**: Decision 13 pinned the layout in
  `walkaround-hybrid/src/shaders/common.wgsl.ts @version 1`. Sprint 10a SVGF's
  `svgf.wgsl.ts` in `shared-denoisers` carries a local copy of the same struct
  (cross-package WGSL imports are not a mechanism in the vitrum build). Layouts are
  byte-identical (`struct WelfordVariance { mean: f32, m2: f32 }`); the SVGF binding
  tests assert compatibility. If the canonical layout is ever bumped, the SVGF copy
  must be updated to match.
- **Jakob+Hanika placeholder**: Sprint 8 ships analytic helpers (Cauchy + 3-coefficient
  polynomial). The full 24 MB precomputed coefficient table from the 2019 paper
  (rgl.epfl.ch/publications/Jakob2019Spectral) is a drop-in upgrade to the analytic
  fallback — not required for correctness, but would improve precision for materials
  with unusual spectra.
- **PPG kd-tree O(N) brute-force scan**: Sprint 11 uses a dense linear array with
  O(N) nearest-cell lookup (up to 10K comparisons per indirect-bounce invocation at
  full cell capacity). A proper kd-tree binary descent would reduce this to O(log N) ≈
  13 comparisons. Explicitly noted as a post-Sprint-11 optimization in
  `plan/sprint-11-ppg-integration.md`.
- **Sprint 9 variance write path**: `varianceBuffer` is allocated but never written.
  The write path is Sprint 10a's responsibility (`varianceUpdateKernel` or inline in
  SVGF temporal pass). Until Sprint 10a's walkaround integration lands, adaptive
  sampling has no input signal to tier pixels on.
- **`core` and `pt-webgpu` have no tests**: `@vitrum/core` is pure types (no runtime
  behavior to test). `@vitrum/pt-webgpu` is a stub that throws `Not implemented` on
  all methods. Neither omission is a risk at this stage but should be revisited when
  either package gains real implementation.

---

## Resumption checklist

When returning to active development on vitrum Phase 6:

1. **Acquire GPU verification capacity** — a browser with WebGL2 (for PT mode)
   and WebGPU (for walkaround mode) is required for all remaining work. Headless
   Chrome with `--enable-webgpu` or a local dev server with DevTools is sufficient.

2. **Apply fork patches in sprint order** — for each sprint from 2 through 8 and
   the Sprint 10a PT-preview patch:
   - Read the corresponding `plan/sprint-N-pt-fork-patch.md`
   - Apply changes to `~/projects/three-gpu-pathtracer/` on branch
     `phase4-normalmap-shadow-rays`
   - Run `npm run build` in the fork directory
   - In `packages/pt-webgl/`: `npm install file:../../../three-gpu-pathtracer`
   - Capture before/after reference renders in `tools/reference-renders/`
   - Verify the sprint's DoD checklist
   - Write the benchmark file (`plan/sprint-N-benchmark.md` per roadmap §9 template)

3. **Wire deferred integrations** (after fork patches are applied and verified):
   - Sprint 9 adaptive sampling: `plan/sprint-9-walkaround-integration.md`
   - Sprint 10a SVGF walkaround: `plan/sprint-10a-walkaround-integration.md`
   - Sprint 11 PPG: `plan/sprint-11-ppg-integration.md`

4. **Apply host-side changes** in the host application (not the vitrum library):
   - Sprint 1 checklist, Sprint 5 came UBO + MRT wiring, Sprint 6 spatial denoiser
     EffectComposer mount, Sprint 7 PhotorealismControls UI, Sprint 8 bevels.ts
     cleanup, Sprint 10b OIDN button + float32 readback, Sprint 11 PPG dispatch

5. **Re-evaluate deferred sprint triggers**:
   - Sprint 10c: run Sprint 7 DoD hero render; measure floor-caustic noise SD;
     threshold decision is user-defined at re-eval time
   - Sprint 12: confirm whether spectral-correctness-required materials are being
     added to the project
   - Sprint 13: evaluate all three trigger criteria in order (SVGF gap visible, WebNN
     production timeline, PPG noise gap); close permanently if any criterion fails

6. **Close Sprint 7 open question** (§7, item 1 of roadmap) before implementing
   Sprint 7 host-side UI: per-glass-type sigma_t — hardcoded lookup table or
   per-type sliders? This affects the material-profile schema.
