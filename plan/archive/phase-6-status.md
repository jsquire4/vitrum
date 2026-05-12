# Phase 6 Status — vitrum library-side completion

**Date**: 2026-05-09
**Branch**: main
**Commits**: 7bda9c4..4c44923 (Sprints 1–13 + audit remediation + selective merge; 33 total on main)

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
  **Dispatch integration WIRED** — `sampleBudgetKernel` and `resolveKernel` are
  dispatched unconditionally in `WalkaroundGPUPipeline.renderFrame` (Pass 0 and the
  resolve pass before composite). `varianceBuffer` is written by the Welford temporal
  pass (Sprint 10a). Full integration spec: `plan/sprint-9-walkaround-integration.md`.
  Commit range: `f5bfde0`. Tests: 63 (Welford suite). Mode scope: walkaround.

### Frontier sprints

#### Sprint 10a — SVGF spatiotemporal denoising: COMPLETE (vitrum-side)

`@vitrum/shared-denoisers` ships `svgf.wgsl.ts` with two entry points
(`svgfVarianceMain`, `svgfAtrousMain`), `svgfBindings.ts` with TypeScript
descriptor types (`SVGFVarianceBindGroupLayout`, `SVGFAtrousBindGroupLayout`,
`SVGF_DEFAULT_UNIFORMS`), and packing helpers (`packSVGFVarianceUniforms`,
`packSVGFUniforms`). Tests: included in shared-denoisers suite (69 total).
**Walkaround wiring COMPLETE** — SVGF two-pass dispatch (`svgfVarianceMain` +
`svgfAtrousMain` × 5) is wired into `WalkaroundGPUPipeline.renderFrame` as the
`denoiserMode === 'svgf'` path; Welford temporal replaces the à-trous path when
SVGF is selected. See `plan/sprint-10a-walkaround-integration.md` for spec.
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

#### Sprint 10c — BDPT for caustics: COMPLETE (vitrum-side scaffold)

`@vitrum/shared-samplers` ships:
- `src/bdptVertex.ts` — `BDPTVertex` interface, kind constants (0–3),
  `BDPT_VERTEX_FLOATS` (12), `BDPT_VERTEX_BYTES` (48),
  `BDPT_MAX_LIGHT_BOUNCES` (3), `BDPT_MAX_EYE_BOUNCES` (12),
  `packBDPTVertex`, `unpackBDPTVertex`. Float-by-float offset map documented
  in module header (CPU↔GLSL contract).
- `src/bdptMIS.ts` — `bdptConnectionMIS` (Veach 1997 power heuristic, β=2
  default) + `buildBDPTStrategyPDFs` (per-strategy PDF table, length s+t+1).
- Tests: 35 new tests in `__tests__/bdpt.test.ts` (pack/unpack round-trips,
  kind constants, strategy PDF length, MIS weight sum-to-1, β=1 balance
  heuristic equivalence, edge cases).

Fork patch **PENDING** — see `plan/sprint-10c-pt-fork-patch.md` for the full
specification (includes trigger criterion in Appendix A).

**Pending fork files**: `light_subpath_kernel.glsl.js` (NEW), `eye_subpath_kernel.glsl.js`
(NEW), `connection.glsl.js` (NEW), `path_tracer.glsl.js` (modify), `PhysicalPathTracingMaterial.js`
(modify — add `uLightPathBuffer`, `uBDPTMaxLightBounces`, `uBDPTEnabled`).

**Vertex storage approach**: texture ping-pong (3 draw calls, one per light-subpath
bounce), NOT MRT-all-at-once. Rationale: MRT would require 9 color attachments
for N=3 bounces × 3 texel groups, exceeding `MAX_DRAW_BUFFERS = 8`.

**BDPT opt-in by default** (`uBDPTEnabled = false`) — zero regression risk until
the fork patch is applied and verified.

#### Sprint 11 — PPG path guiding: COMPLETE (vitrum-side + dispatch wired)

`@vitrum/walkaround-hybrid` ships:
- `src/ppg/types.ts` — `PPGDirectionalBin`, `PPGQuadTreeNode`, `PPGSpatialCell`,
  `PPGBuffers`, `PPG_MAX_SPATIAL_CELLS` (10,000), byte-stride constants.
- `src/ppg/wgsl/ppgSample.wgsl.ts` — `PPG_SAMPLE_WGSL` WGSL fragment for guided
  direction sampling, @group(2) bindings.
- `src/ppg/wgsl/ppgUpdate.wgsl.ts` — `PPG_UPDATE_WGSL` compute kernel with atomic
  fixed-point radiance accumulation.
- `createPPGBuffers` / `destroyPPGBuffers` in `resourceManager.ts`.
- `HybridEngineOptions.ppgEnabled` / `ppgMaxSpatialCells`, `HybridEngine.setPPGEnabled()`,
  `HybridEngine.ppgEnabled` getter.

**Dispatch WIRED** — PPG update kernel dispatched in `WalkaroundGPUPipeline.renderFrame`
when `ppgEnabled === true` (after shade, before denoiser). `buildPpgKdTreeGpuBytes` + kd-tree
upload wired in `HybridEngine._initPipeline`. See `plan/sprint-11-ppg-integration.md` for spec.
Dense linear-array kd-tree with O(N) nearest-cell lookup. Tests: 81 (PPG suite). Commit range:
`0cebaf9`.

#### Sprint 12 — Hero-wavelength spectral: COMPLETE (vitrum-side spectral utilities)

`@vitrum/shared-samplers` ships:
- `src/cieCmf.ts` — CIE 1931 2° standard observer CMF tables (81 entries each,
  380–780 nm at 5 nm steps), CIE D65 illuminant, `sampleCMF(λ)` linear-interpolating
  lookup, `xyzToLinearSRGB` using Bradford-adapted D65 matrix (IEC 61966-2-1).
- `src/wavelengthSampling.ts` — `sampleHeroWavelength(u)` importance-samples the Y CMF
  via piecewise-linear CDF inversion; `wavelengthToRGB(λ, throughput, pdf)` reconstructs
  RGB from a hero-wavelength path result; `Y_CMF_INTEGRAL` constant (~106.857 nm).
- `src/cauchyIor.ts` — `cauchyIOR(λ_nm, A, B, C)` Cauchy dispersion formula (λ in µm
  internally); `abbeNumber(A, B, C)`; `CAUCHY_CROWN_GLASS` (Abbe ≈ 64),
  `CAUCHY_FLINT_GLASS` (Abbe ≈ 36), `CAUCHY_LEAD_CRYSTAL` (Abbe ≈ 32 — Sprint 8 default
  bevel material). B coefficients calibrated to produce correct Abbe numbers.
- `__tests__/spectral.test.ts` — 50 tests covering table dimensions, CMF values,
  interpolation, XYZ→sRGB white point, hero sampling distribution, `wavelengthToRGB`
  channel dominance at 450/550/700 nm, Cauchy IOR monotonicity, and Abbe numbers.

Fork-side kernel rewrite (ray payload vec3→scalar, BSDF wavelength-aware, spectral
accumulator) is documented in `plan/sprint-12-pt-fork-patch.md` and remains gated on
trigger confirmation (uranium glass / dichroic film / gemstone materials). See §7 of
that document for the re-surface decision point. Tests: 50 (spectral suite).

#### Sprint 13 — Custom WebGPU neural denoiser: COMPLETE (vitrum-side scaffold)

`@vitrum/walkaround-hybrid` ships:

**WGSL inference kernels** (`src/neural/wgsl/`):
- `conv2d.wgsl.ts` — `CONV2D_WGSL`: 2D convolution with SAME padding, stride, dilation.
  Entry point: `conv2dKernel`. Workgroup 8×8×1. Bindings @group(0) @binding(0–4).
- `transposedConv2d.wgsl.ts` — `TRANSPOSED_CONV2D_WGSL`: transposed (deconvolutional)
  2D conv for decoder upsampling. Entry point: `transposedConv2dKernel`. Gather formulation.
- `relu.wgsl.ts` — `RELU_WGSL`: elementwise ReLU. Entry point: `reluKernel`. Workgroup 256×1×1.
- `skipConnection.wgsl.ts` — `SKIP_CONNECTION_WGSL`: elementwise add for UNet skip connections.
  Entry point: `skipConnectionKernel`.
- `bilinearUpsample.wgsl.ts` — `BILINEAR_UPSAMPLE_WGSL`: bilinear 2× upsampling with
  center-aligned coordinates. Entry point: `bilinearUpsampleKernel`.

**Inference orchestrator** (`src/neural/InferenceGraph.ts`):
- `InferenceGraph` class: wires kernels into a DAG, allocates GPU buffers, dispatches
  per-layer compute passes. `initialize(device)` / `run(device, encoder, inputs, outputs)` / `dispose()`.
- Types: `InferenceGraphSpec`, `ModelWeights`, `InferenceLayer`, `InferenceLayerKind`.

**UNet architecture spec** (`src/neural/unetArchitecture.ts`):
- `WALKAROUND_DENOISER_UNET_SPEC`: canonical 9→24→48→96→192→96→48→24→3 UNet.
  426,075 parameters (~1.63 MB f32, within 1–3 MB DoD target).
  Input: 9ch (noisy RGB + albedo + normals). Output: 3ch denoised RGB.
- Constants: `UNET_TOTAL_PARAMETERS`, `UNET_WEIGHT_BYTES`, channel-width arrays,
  tensor name arrays.

**Training pipeline scaffolding** (`tools/neural-denoiser-training/`):
- `README.md`: setup, prerequisites, workflow.
- `dataset_spec.md`: noisy/clean pair spec (10K target, .npz format, preprocessing).
- `train.py.md`: PyTorch pseudocode matching `WALKAROUND_DENOISER_UNET_SPEC` exactly.
- `export_weights.md`: binary format spec (`VITRUMW1` header) + JS loader pseudocode.

Integration into `HybridEngine.renderFrame` deferred — GPU-verified wiring not possible
without live WebGPU device. Full integration spec: `plan/sprint-13-walkaround-integration.md`.
Trigger criteria (Decision 14) still apply; library-side is ready for when they are met.
Tests: 101 (Sprint 13 neural suite in `__tests__/sprint13-neural.test.ts`). Commit: `main`.

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
- **Sprint 9 / 10a / 11 host**: Library-side dispatch wiring is complete.
  Remaining host work: expose `denoiserMode`, `ppgEnabled`, and adaptive
  sampling thresholds in the UI (render-mode panel / settings panel). GPU
  verification (frame capture, timing gate) required before enabling by default.
- **Sprint 10b host**: `plan/sprint-10b-host-checklist.md` — install
  `onnxruntime-web`, bundle OIDN ONNX model, "Denoise" button, float32 readback,
  `preloadOIDNModel` pre-warm, denoised PNG save option, `clearOIDNCache` on exit.

### Integration deferred (vitrum-side, requires GPU verification)

- **Sprint 9 adaptive sampling + Sprint 10a SVGF + Sprint 11 PPG dispatch**:
  All three are now wired into `WalkaroundGPUPipeline.renderFrame` and
  `HybridEngine._initPipeline`. GPU verification (frame capture, A/B compare)
  still pending. See the respective integration spec docs for test procedures.
- **Sprint 13 neural denoiser HybridEngine wiring**: `InferenceGraph` authored and
  tested; wiring into `HybridEngine.renderFrame` (G-buffer pack pass, dispatch chain,
  composite read path, `setNeuralDenoiserEnabled` toggle) deferred. See
  `plan/sprint-13-walkaround-integration.md`.

---

## Stats

- **Vitrum library packages**: 8 (core, three-bindings, shared-bvh, shared-samplers,
  shared-denoisers, pt-webgl, pt-webgpu prototype, walkaround-hybrid)
- **Total tests**: 542 passing (22 test files across 6 packages)
  - pt-webgl: 18
  - shared-bvh: 11
  - shared-denoisers: 75 (69 + 6 new HWC↔NCHW round-trip tests from audit remediation M-3)
  - shared-samplers: 143 (54 existing + 35 BDPT tests + 50 new Sprint 12 spectral tests + 4 from audit remediation)
  - three-bindings: 1
  - walkaround-hybrid: 294 (193 existing + 101 new Sprint 13 neural denoiser tests)
- **TypeScript strict**: clean across workspace (`npm run typecheck` passes)
- **LOC vitrum library code**: ~21,800 total (Sprint 13 adds ~2,100 lines across 7 new files)
- **Phase 6 sprint commits**: complete through Sprint 13 + audit remediation + selective merge
- **Total commits on main**: 33

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
| `plan/sprint-10c-pt-fork-patch.md` | Sprint 10c (BDPT) active patch spec — light/eye subpath kernels, ping-pong vertex storage, MIS weight GLSL; Appendix A preserves original trigger criterion |
| `plan/sprint-12-pt-fork-patch.md` | Sprint 12 active patch spec — CIE CMF tables, hero-wavelength sampling, Cauchy IOR, fork-side kernel rewrite spec (gated) |
| `plan/sprint-13-walkaround-integration.md` | Sprint 13 integration spec — InferenceGraph wiring into renderFrame, bind groups, gating, memory, dispatch sizing |

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
- **`core` remains type-only; `pt-webgpu` has moved to prototype status**:
  `@vitrum/core` is pure types (no runtime behavior to test). `@vitrum/pt-webgpu`
  now includes a pre-alpha implementation (progressive accumulation, CPU-built BVH,
  GPU traversal, directional/point direct lighting) and package-level tests; GPU
  visual verification and fidelity parity are still pending.

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
