# Phase 6 Roadmap — PT Production Frontier

**Status**: planning, Sprint 0 in progress
**Created**: 2026-05-09 (revised same day after library-extraction reframe)
**Branch**: `main` (Phase 1–5 of photorealism plan shipped 2026-05-09; sweep findings closed)

## Strategic frame: this is library extraction, not app feature work

The stainedGlass app is the proving-ground host for a SOTA browser-based renderer being extracted into a separate library: **`vitrum`** at `~/projects/vitrum/`. Per `plan/library-architecture.md` in that repo, every Phase 6 sprint deliverable lands in a `@vitrum/*` package, not in stainedGlass app code. The stainedGlass app re-imports from `@vitrum/*` to demonstrate.

**Why this matters for sprint planning**: the cells-go-grey / Canvas-key-remount bug class isn't a sprint; it's resolved by the library API contract (`vitrum/packages/core/src/engine.ts`) which decouples engine lifetime from host React mount lifetime. Sprint 0 (the new prerequisite) draws that contract.

## Sprint 0 — Library API contract (NEW, prerequisite, in progress)

See `~/projects/vitrum/plan/sprint-0-api-contract.md` for the full breakdown. Summary: 2–3 days, lands the public types + lifecycle contract in `@vitrum/core`, stubs the backends, scaffolds the stainedGlass migration shim. After Sprint 0, every Phase 6 sprint below has a clear vitrum-package destination.

---

## 1. Goal

Take our renderer from "produces a great-looking image" (Phase 5 baseline) to "produces an image that competes with offline production renderers (Arnold, Cycles, Renderman) for stained-glass scenes specifically." The non-negotiable target: **a hero render that's indistinguishable from a photograph of a real stained-glass window in a sunlit room**, plus an interactive PT preview that converges fast enough to dial-in lighting and material edits in seconds-not-minutes.

This roadmap is the synthesis of:
- Audits of *Ray Tracing in One Weekend / The Next Week / The Rest of Your Life* and erichlof's `THREE.js-PathTracing-Renderer`
- Five domain-specialist planning passes (light transport, convergence, preview perf, BSDF/traversal, scene fidelity)
- Bleeding-edge frontier techniques flagged as not-in-any-of-the-audits (SVGF, OIDN, PPG, hero-wavelength spectral, neural denoising)

---

## 2. Anti-pattern: avoid the walkaround/PT crossover trap

The walkaround mode (custom WGSL/WebGPU, runs DDGI + Radiance Cascades + ReSTIR DI) and the PT mode (forked three-gpu-pathtracer, GLSL on WebGL2) have different sampling strategies, different denoisers, and different convergence semantics. A feature that lands "in walkaround" is NOT automatically in PT, and vice versa. The render-mode invariant is non-negotiable per `project_render_modes` memory:

> raster = editor (default)
> PT preview = verifier (opt-in)
> walkaround = WebGPU GI explorer (opt-in, room-mode)
> PT final = output

When a feature is needed in BOTH modes, that's two implementations, not one. Each mode's implementation must hit its own "definition of done" criterion before the feature is marked complete. The walkaround caustic-boost work earlier in 2026-05 hit this exact trap — caustics looked great in walkaround, were absent in PT preview, and only re-emerged in PT after a separate set of fixes (camera auto-frame, transmission floor at 0.95, area-light sun, outdoor HDRIs).

**Every Phase 6 sprint below specifies `Mode scope` explicitly. Anything marked `Both` requires two implementations.**

---

## 3. Mode-scope matrix

| Sprint | Technique | Walkaround | PT preview | PT final | Shared infra |
|---|---|---|---|---|---|
| 1 | HDRI 404 fix | — | ✓ | ✓ | — |
| 1 | Preview bounces 5→3 | — | ✓ | — | — |
| 1 | Resolution factor 0.5 | — | ✓ | — | — |
| 1 | Skip post-process during accum | — | ✓ | — | — |
| 1 | OrbitControls damping tuning | ✓ | ✓ | — | shared |
| 2 | Per-cell luminance precompute | ✓ | ✓ | ✓ | shared (BVH build) |
| 3 | Mixture PDF sampling | — | ✓ | ✓ | fork |
| 3 | Light tree | — | ✓ | ✓ | fork |
| 3 | Back-face NEE resample | — | ✓ | ✓ | fork |
| 4 | lobeMask bitfield | — | ✓ | ✓ | fork |
| 4 | Lite BSDF for indirect | — | ✓ | ✓ | fork |
| 4 | Material LOD by depth | — | ✓ | ✓ | fork |
| 5 | Analytic came (CSG) | — | ✓ | ✓ | fork + JS uploader |
| 6 | Rough refraction lobe | — | ✓ | ✓ | fork |
| 6 | Edge-stopping spatial filter | — | ✓ | — | post-pipeline |
| 7 | SSS (opalescent/glueChip/ringMottled) | — | ✓ | ✓ | fork |
| 7 | Volume + equi-angular | — | ✓ | ✓ | fork |
| 8 | RGB-as-3λ spectral | — | ✓ | ✓ | fork |
| 9 | Adaptive sampling | ✓ | — | — | walkaround only |
| 9 | Checkerboard upsampling | ✓ | — | — | walkaround only |
| 10a | SVGF spatiotemporal denoising | ✓ | ✓ | — | both (separate impls) |
| 10b | OIDN final-pass via ONNX | — | — | ✓ | post-pipeline |
| 10c | BDPT for caustics | — | — | ✓ | fork (gated re-eval) |
| 11 | PPG path guiding | ✓ | — | — | walkaround only |
| 12 | Hero-wavelength spectral | — | ✓ | ✓ | fork (kernel rewrite) |
| 13 | Custom WebGPU neural denoiser | ✓ | — | — | walkaround only |

Legend: ✓ = in scope; — = explicitly out of scope; **fork** = patch to `github:jsquire4/three-gpu-pathtracer`.

---

## 4. Sprint plan

Each sprint specifies: **goal**, **mode scope**, **files**, **definition of done**, **dependencies**, **effort**, **risk**.

### Sprint 1 — PT preview speed wins (3.5 hrs)

**Goal**: PT preview runs at ~25–40 fps interactive on desktop, ~10–20 fps on retina. Drop the four explicit HDRI presets from broken (silently 404) to working.

**Mode scope**: PT preview (T1–T4), shared OrbitControls (T2 — applies in raster too).

**Files**: `outdoorScenePresets.ts`, `pathtracerConstants.ts`, `PathTracingLayer.tsx`, `PTPostProcessing.tsx`, `StageOrbitControls.tsx`.

**Definition of done**:
- Outdoor HDRI presets load successfully (browser network panel shows 200, not 404)
- PT_PREVIEW.bounces = 3 in `pathtracerConstants.ts`
- PT_PREVIEW renders at 0.5× DPR (verify via `pathtracer._pathTracer.target.width` halving)
- EffectComposer skipped for first 8 samples after each reset (verify via frame-time profile)
- OrbitControls damping factor = 0.15

**Dependencies**: none.

**Effort**: 3.5 hours.

**Risk**: bilinear upscale softens caustic edges in preview. Acceptable; PT_FINAL stays at 1.0× DPR.

---

### Sprint 2 — Per-cell luminance precompute (1 day)

**Goal**: Foundation infrastructure for Sprint 3 light tree. Walks all emitters once at BVH build time, writes `cellPower[i] = Le[i] × area[i]` to a uniform buffer.

**Mode scope**: Both — walkaround `bvhCompute.ts` writes to its bind group; PT three-gpu-pathtracer's lights uniform gets a `power` field added per entry. Two implementations with shared semantic ("emitter total radiant flux").

**Files**: `src/rendering/scene/walkaround/engines/restir/bvhCompute.ts` + fork's `light_sampling_functions.glsl.js` Material struct extension.

**Definition of done**:
- `cellPower` uniform populated and visible in `__WG__` debug bridge (walkaround)
- `light.power` field populated in fork's lights texture (PT)
- Round-trip test: setting `Le[i]=2×` doubles `cellPower[i]` in both modes
- No visible visual change yet (this is foundation only)

**Dependencies**: none.

**Effort**: 1 day.

**Risk**: surface-area calc must match the actual triangle area used in BVH; verify against `MeshBVHHelper`.

---

### Sprint 3 — Sampling theory upgrade (5.5 days, **3–5× variance reduction in interior PT shots**)

**Goal**: replace uniform light selection + binary env-vs-light coin flip with proper mixture PDF + power-weighted light tree.

**Mode scope**: PT only (preview + final). Walkaround already has its own ReSTIR-based light sampling; not affected.

**Files**: fork's `direct_lighting.glsl.js`, new `light_tree.glsl.js`, `sampling.glsl.js`.

**Definition of done**:
- Mixture PDF (BSDF + env + light) replaces binary branch — verify via shader inspection
- Light tree CDF built CPU-side on scene change; binary-search lookup in GLSL
- Back-face NEE samples redraw up to 4× before contributing zero
- Variance benchmark: render reference scene at 192 samples, measure floor-pixel stddev. Target: ≥3× reduction vs. baseline. Captured in `plan/sprint-3-benchmark.md`.

**Dependencies**: Sprint 2 (per-cell luminance precompute).

**Effort**: 5.5 days (mixture 2d → light tree 3d → back-face resample 0.5d, in order).

**Risk**: light tree CDF rebuild on every scene mutation. Wire to existing scene-dirty signal (PathtracerSceneSync's 50ms debounce); verify rebuild stays <1ms.

---

### Sprint 4 — BSDF cost reduction (3.5 days, **~50–60% BSDF math reduction**)

**Goal**: skip dead Disney lobes and simplify BSDF for indirect bounces.

**Mode scope**: PT only (preview + final). Walkaround custom WGSL is unaffected.

**Files**: fork's `bsdf_functions.glsl.js`, `surface_record_struct.glsl.js`, `get_surface_record_function.glsl.js`.

**Definition of done**:
- `lobeMask` bitfield computed from material at `getSurfaceRecord` time
- Sheen / clearcoat / iridescence branches gated on bit
- `liteMode` flag set for `state.depth > 1`; reduces `bsdfEval` to Lambertian + GGX-only
- Material LOD: `materialLodDepth=2` uniform; texture samples skipped when `state.depth > 2`
- Profile baseline: ms/sample reduction ≥40% on a glass-and-came scene

**Dependencies**: none — fully fork-internal, builds on Phase 4 fork base.

**Effort**: 3.5 days (P1 lobeMask 1d → P2 lite BSDF 1.5d → P3 material LOD 1d, in order).

**Risk**: lite BSDF for indirect bounces is a perceptual call — visually verify on opal panels which depend on subtle shading. If degradation is visible, gate per-material via `forceFullBSDF` flag.

---

### Sprint 5 — Analytic came geometry + MRT G-buffer scaffold (5 days + 1 day rider, **~30–50% BVH traversal reduction**)

**Goal**: replace 1500+ tube/sphere triangle meshes for came/solder with closed-form ray-primitive intersection, hybrid with BVH for glass. Also lay down MRT G-buffer infrastructure while `trace_scene_function.glsl.js` is already open — needed by Sprints 6, 10a, and 10b.

**Mode scope**: PT only. Walkaround's BVH is structurally different (different attribute packing) and is fast enough already.

**Files**: fork's `trace_scene_function.glsl.js`, `shape_intersection_functions.glsl.js`, `PhysicalPathTracingMaterial.js` (uniforms); new `cameUniformUploader.ts` in our project; `WebGLMultipleRenderTargets` allocation in PT pipeline layer.

**MRT G-buffer rider scope** (Decision 12):
- Allocate `WebGLMultipleRenderTargets` with 3 channels: `gColor` (location 0), `gNormalDepth` (location 1, encoded normal + linear depth), `gAlbedo` (location 2, base-color unlit)
- Add fragment outputs to `PhysicalPathTracingMaterial.js` populating each channel from the primary-hit surface record
- Layout locked here; downstream sprints just read from the existing buffers

**Definition of done**:
- 500-segment came UBO populated from `EdgeLines.tsx`-equivalent code path
- `intersectCameSegment` + `intersectCameNode` analytic functions in shader
- `traceScene` runs both BVH and analytic-came intersection, picks closest hit
- Synthetic SurfaceHit fill returns correct material, normal, UV (if applicable) for came hits
- Profile baseline: BVH-walk node-visits per ray ≥30% reduction
- Device-tier fallback: low-end GPUs (`maxFragmentUniformVectors < 256`) auto-disable analytic-came path
- Mesh-came geometry remains in BVH as fallback (no regression risk if uniform path fails)

**Dependencies**: none — fully independent of Sprint 4.

**Effort**: 5 days.

**Risk**: highest of any sprint. UBO packing, normal reconstruction, and synthetic SurfaceHit fill each have edge cases. Mitigation: mesh fallback always available; analytic path disabled by default initially, opt-in via `viewport.analyticCameEnabled`.

---

### Sprint 6 — Visible-quality wins (5 days)

**Goal**: composable visual upgrades that ship after the perf foundation.

**Mode scope**:
- Rough refraction lobe-on-refracted-ray: PT preview + final (fork patch)
- Edge-stopping spatial filter: PT preview only (it's a low-spp clean-up; PT_FINAL has high enough spp that the filter would over-blur)

**Files**: fork's `bsdf_functions.glsl.js` (rough refraction); new `src/rendering/scene/PTSpatialDenoiser.tsx` (37-tap hexagonal kernel as `postprocessing` Effect subclass).

**Definition of done**:
- Rough refraction perturbs refracted ray direction by GGX-distributed roughness lobe (NOT surface normal — that's our existing Phase 4 patch); composes additively
- Spatial denoiser mounted FIRST in EffectComposer chain (before Bloom)
- Spatial denoiser auto-disables when `pathtracer.samples > 24` (temporal accumulation has converged)
- Visual A/B: hammered glass cell shows volumetric scatter blur in PT_FINAL; preview at 4 samples looks visually clean instead of speckle-noisy

**Dependencies**: Sprint 4 ideally lands first so the spatial-filter benchmarks against the new fast baseline.

**Effort**: 5 days (rough refraction 2d + spatial denoiser 3d).

**Risk**: spatial denoiser is a `postprocessing` Effect subclass — moderate boilerplate. Custom shader needs depth + normal G-buffers from PT accumulation; verify availability before committing.

---

### Sprint 7 — Volumetric & SSS (7 days, **the cathedral-with-god-rays look**)

**Goal**: ship the single largest perceptual upgrade — atmospheric haze with god-ray shafts through panels, plus correct internal scattering for opalescent / glueChip / ringMottled glass.

**Mode scope**: PT only. Volume scattering in walkaround would require a third major lighting system addition; defer.

**Files**: fork's `path_tracer.glsl.js` (main loop restructure for volume scatter events), new `volume_march.glsl.js`, `bsdf.glsl.js` (HG phase function shared with SSS), `materials_data_function.glsl.js` (SSS material flag), `PhotorealismControls.tsx` (haze + scatter UI).

**Definition of done**:
- Single-scatter SSS: `TRANSLUCENT` material flag; opalescent/glueChip/ringMottled cells map to it; scatter distance sampled exponentially with HG anisotropy
- Volume: global homogeneous medium with `density`, `scatterAlbedo`, `anisotropy g` uniforms
- Equi-angular PDF (Szécsi/Kulla-Conty): volume-scatter NEE samples distance-along-ray weighted toward closest point to primary light
- UI: "Atmospheric haze" section in PhotorealismControls (enable, density 0–2.0, color, anisotropy), "Glass SSS" section with per-type sigma_t override
- Visual A/B: backlit panel + sun produces visible god-ray shafts at ≥0.5 density; opalescent panel shows milky internal glow

**Dependencies**: Sprint 3 (light tree — equi-angular sampling needs to know which emitter to equi-angularly sample); Sprint 6 (rough refraction — shares HG phase function utility, extract to common.glsl.js).

**Effort**: 7 days (SSS 3d + volume + equi-angular 4–5d).

**Risk**: medium-high. Volume per-sample cost +20%; if PT_FINAL budget is exceeded for hero scenes, reduce default density or expose `volumeQuality` slider. Homogeneous-medium assumption breaks if non-uniform density is requested — explicitly out of scope for this sprint.

**Decision point** (sign-off needed before Sprint 7): room-wide uniform volume only, or per-region density volumes? Default: uniform only.

---

### Sprint 8 — RGB-as-3-wavelengths spectral + Jakob+Hanika upsampling (5 days + 1.5 day rider)

**Goal**: bevel glass renders true rainbow dispersion (R refracts less than B per Cauchy formula), instead of the current fake noise-split in `bevels.ts`. Sprint 8b rider adds Jakob+Hanika spectral upsampling to smooth the 3-color rainbow into a continuous spectrum.

**Mode scope**: PT only.

**Files**: fork's dielectric BSDF in `bsdf.glsl.js`; `glassMaterialProfiles.ts` (add `dispersionStrength?: number`); `createBakedGlassMaterial.ts` (`onBeforeCompile` shader injection); `bevels.ts` baker (remove fake noise-split).

**Definition of done**:
- Per-channel IOR computed via Cauchy formula: `iorRGB = ior + B/λ² + C/λ⁴` at λ ∈ {700, 550, 450} nm
- `dispersionStrength` slider exposed for bevel cells (default 0.018 from Abbe ~32 for lead crystal; 0 for non-bevel)
- Jakob+Hanika spectral upsampling (Decision 10): each RGB channel's IOR derived from a 3-coefficient polynomial spectrum (6 GLSL instructions per channel), not hardcoded Cauchy values. Rainbow is smooth across the visible spectrum, not 3 discrete bands.
- Visual A/B: bevel close-up shows red/green/blue spreading smoothly through prismatic edges; crown-glass non-bevels show no chromatic aberration

**Dependencies**: none.

**Effort**: 6.5 days (3d fork patch + 1d material wiring + 1d validation + 1.5d Jakob+Hanika rider).

**Risk**: this is the FIRST and ONLY single-wavelength path-tracing approximation we ship. If user later adds uranium glass / dichroic / multi-order iridescence, hero-wavelength spectral (Sprint 12) becomes necessary — this is its scaffold. Jakob+Hanika upsampling **may make Sprint 12 unnecessary entirely** for the bevel use case.

---

### Sprint 9 — Convergence efficiency + Welford struct rider (5 days + 2 hr rider, walkaround only)

**Goal**: reduce samples-to-clean for walkaround mode via adaptive sampling and checkerboard temporal upsampling.

**Mode scope**: walkaround only. PT mode receives no per-pixel adaptive feedback (Plan 2's PT-side workaround was rejected as too coarse for the 5-day budget).

**Files**: walkaround's `accumulate.wgsl`, new `sample_budget.wgsl`, new `resolve.wgsl` (checkerboard); `HybridRenderer.ts` orchestration; `common.wgsl` (Welford struct).

**Definition of done**:
- Welford variance buffer (RG32Float texture) populated each frame
- `WelfordVariance` struct extracted to `common.wgsl` as a versioned named type (Decision 13). Layout pinned to prevent Sprints 10a/11/13 from independently re-declaring incompatible variants.
- Per-pixel sample-count tier (1/2/4 rays/frame) driven by variance threshold
- Checkerboard pattern: shaded pixels alternate per frame; gap pixels reproject from prev-frame G-buffer
- Visual A/B at 16 samples: caustic edges show ≥30% lower noise than baseline
- No regression on motion: ghosting acceptable per the existing variance-clamped AABB

**Dependencies**: none, but Sprint 2's per-cell luminance can compose with adaptive's variance feedback (high-variance + high-luminance pixels prioritized).

**Effort**: 5 days + 2 hr rider (adaptive 3d + checkerboard 2.5d + Welford struct extraction 2hr).

**Risk**: ghosting under fast camera motion (no velocity vectors). Mitigated by existing variance-clamped accumulator in walkaround. If user reports ghosting after kickoff, add a 1-day velocity-buffer follow-up.

---

## 5. Sprint 10+ — bleeding-edge / situational

These are gated behind explicit **trigger criteria**, not automatically scheduled. Each has a "ship-now-anyway" cost and a "defer-until-X" criterion.

### Sprint 10a — SVGF spatiotemporal denoising (1.5–2 weeks)

**Goal**: bridge the gap between hand-tuned à-trous (current walkaround denoiser) and fully neural denoising. SVGF uses motion vectors + variance estimates + spatial filtering, no learned weights — closes 60–70% of the perceptual gap to OIDN-class denoising.

**Mode scope**: walkaround (replaces our current à-trous implementation) + PT preview (new spatial denoiser, replaces the simpler Sprint 6 hexagonal filter).

**Trigger criterion**: ship after Sprint 9. SVGF + adaptive sampling are complementary.

**Definition of done**:
- Walkaround à-trous replaced with SVGF: variance-guided, temporally-stable
- PT preview spatial filter replaced with SVGF variant
- Visual A/B at 8 samples: indistinguishable from 64-sample reference for diffuse surfaces

**Risk**: re-tuning the existing à-trous edge-stop coefficients (chromaticity, normal, depth) for the SVGF variance feedback. Budget 3 days for tuning alone.

---

### Sprint 10b — OIDN final-pass via ONNX + WebNN execution provider (3–4 days)

**Goal**: instant "click-to-denoise" on hero renders. Loads pre-trained OIDN ONNX model into the browser, runs inference once on a converged PT_FINAL output.

**Mode scope**: PT final only — explicitly NOT real-time.

**Trigger criterion**: ship anytime; orthogonal to other sprints.

**Definition of done**:
- ONNX Runtime Web loads the OIDN albedo/normal/color UNet (5–10 MB bundle)
- "Denoise" button in PT_FINAL UI runs inference on the accumulated buffer + auxiliary G-buffers (G-buffers come from Sprint 5 MRT scaffold)
- Output saves alongside the raw render
- Inference time <2 seconds for a 2K render on a typical GPU
- ONNX Runtime Web `executionProviders` configured as `['webnn', 'webgpu', 'wasm']` (Decision 11) — picks WebNN if browser supports it (free near-native acceleration on Edge/Chrome with WebNN behind-flag), falls back to WebGPU then WASM

**Risk**: bundle-size hit (5–20 MB). Lazy-load only when "Denoise" button is clicked.

**Effort**: 3–4 days (originally 2–3 days for OIDN-only + 1 day WebNN EP rider).

---

### Sprint 10c — BDPT for caustics (8–10 days)

**Goal**: bidirectional path tracing for true caustic convergence. Light-side rays + connection-PMF MIS.

**Mode scope**: PT final only.

**Trigger criterion**: ship ONLY IF Sprint 7 (volume + equi-angular) doesn't visually close the caustic-convergence gap. Re-evaluate after Sprint 7 with a hero-render side-by-side comparison.

**Definition of done**:
- Light subpath kernel: traces from each emitter up to N=3 bounces, stores vertices in MRT/SSBO
- Eye-subpath connection routine: at each eye vertex, attempt connections to every stored light vertex, evaluate joint PDF + MIS weight
- Visual A/B: floor caustic from sun-through-panel converges at ~256 samples in PT_FINAL vs. ~1024+ samples for pure NEE

**Risk**: highest of any frontier item. WebGL2 lacks compute shaders → vertex storage requires creative MRT or texture ping-pong. Budget 2 weeks minimum. **Defer until Sprint 7 ships and gap is visually re-assessed.**

---

### Sprint 11 — PPG path guiding (3–4 weeks)

**Goal**: online learning of useful directions per spatial cell. ~2–3× sample efficiency on indirect-lit scenes (rooms with sun coming through stained glass — exactly our scene type).

**Mode scope**: walkaround only. WebGL2 PT can't easily do the kd-tree updates (no compute shaders); WebGPU walkaround can.

**Trigger criterion**: ship after Sprint 9 + 10a. PPG composes with adaptive sampling and SVGF — adaptive prioritizes high-variance pixels, PPG cuts that variance via guided directions, SVGF cleans up what's left.

**Definition of done**:
- PPG kd-tree allocated as WebGPU storage buffer (sparse, capped at ~10K spatial cells)
- Each cell holds a quad-tree of directional bins (16-direction discretization initially)
- Per-frame: collect path-completion samples into the structure; ping-pong update on alternate frames
- Path-tracing dispatch reads the cell at each indirect bounce, samples direction per the learned PDF
- Visual A/B: indirect-only-lit scene converges at 30 samples vs. 90 samples baseline

**Risk**: large surface area. Budget 3 weeks for PPG core + 1 week for tuning. Bail-out criterion: if walkaround framerate drops below 30 fps on default scene, disable until performance re-tuned.

---

### Sprint 12 — Hero-wavelength spectral (4–5 weeks)

**Goal**: replace Sprint 8's RGB-as-3λ approximation with full spectral path tracing. Sample one wavelength per path; reconstruct RGB at display via CIE color-matching functions.

**Mode scope**: PT preview + final.

**Trigger criterion**: ship ONLY IF user adds materials that need spectral correctness — uranium glass (fluorescence), dichroic film (multi-order interference), gemstones (absorption bands), or wants smooth-spectrum bevel rainbows beyond 3-color fans.

**Definition of done**:
- Ray payload changes from `vec3 throughput` to `float wavelength + float throughput`
- Every BSDF evaluation site updated to wavelength-aware variant
- Spectral accumulator + CIE CMF reconstruction at display
- Visual A/B: bevel rainbow shows smooth spectrum (8+ visible colors) vs. Sprint 8's 3-color fan

**Risk**: kernel rewrite of the fork. ~5 weeks. Major fork divergence from upstream — every future `git pull` from gkjohnson is a multi-day merge.

**Decision point** (sign-off needed before Sprint 12): is the visible improvement over RGB-as-3λ worth the kernel rewrite + ongoing fork maintenance burden? **User noted "more realistic" preference for RGB-as-3λ**; revisit only if specific spectral materials are added to the project.

---

### Sprint 13 — Custom WebGPU neural denoiser (6–8 weeks)

**Goal**: production-grade real-time neural denoising. Custom UNet-style CNN, trained on offline reference data, running entirely on WebGPU compute shaders.

**Mode scope**: walkaround only (real-time path); PT final uses Sprint 10b's ONNX-based OIDN final pass.

**Trigger criterion** (Decision 14): ship ONLY IF
1. SVGF (Sprint 10a) leaves a visible quality gap AND
2. WebNN API is still 12+ months from production-ready AND
3. PPG (Sprint 11) has shipped and didn't close the noise gap on its own

**Definition of done**:
- Training pipeline (Python/PyTorch) generates reference noisy/clean pairs from offline path-tracer + walkaround output
- Model architecture: UNet, ~1–3 MB weights, trained on ~10K image pairs
- Inference graph: WebGPU compute shaders for conv2d, transposed conv2d, ReLU, skip connections
- ~10–50 ms inference time per frame on typical GPU
- Visual A/B: real-time-denoised output indistinguishable from offline OIDN at 4 samples-per-pixel

**Risk**: research project, not feature work. Budget 8 weeks minimum. Bail-out criterion: if month 1 inference benchmarks don't hit <50 ms, abort and wait for WebNN.

---

## 6. Decision log

Decisions locked during planning, surfaced for transparency:

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Spectral approach | RGB-as-3λ (Sprint 8) + Jakob+Hanika upsampling (Sprint 8b rider) | More realistic to ship; captures 80% of visible effect; Sprint 12 hero-wavelength gated on actual need |
| 2 | Adaptive sampling on PT mode | Skip — walkaround only | PT-side workaround is too coarse to be worth 5-day budget |
| 3 | BDPT | Sprint 10c, **vanilla BDPT** (NOT ReSTIR BDPT), gated re-eval | See decision 8 below |
| 4 | No tier overrides | Confirmed | All audit-derived sprints stay in baseline plan |
| 5 | Walkaround/PT crossover handling | Two implementations per "Both" feature | Per render-mode invariant |
| 6 | Mesh-came retained as PT fallback | Yes | Sprint 5 analytic-came opts in via flag; mesh stays in BVH |
| 7 | Sprint 7 volume scope | Uniform medium only | Per-region density deferred to a future sprint if needed |
| 8 | Sprint 10c BDPT path | Stick with vanilla BDPT, NOT ReSTIR BDPT (Hedstrom 2025) | Verified 2026-05-09: ReSTIR BDPT is hardwired to DirectX 12 + DXR + Falcor + RTX hardware. Slang→WGSL exists but uses DXR ray queries which need the WebGPU ray query extension (Chrome-flags-only, no Firefox/Safari timeline). Honest port = 3–5 months. Citations: github.com/Shmaug/ReSTIR-BDPT. |
| 9 | Sprint 11 path-guiding path | Stick with PPG (Müller 2017), NOT NRC (Müller 2021) | Verified 2026-05-09: original NRC paper never released open source; production NVIDIA-RTX/NRC is closed binaries tied to RTXGI 2.0 + DXR; SIGGRAPH Asia 2025 mobile NRC paper uses HLSL/Vulkan on Samsung Xclipse with tensor-core MLP, not portable to WebGPU. Online training requires backward-pass autograd in compute shaders, which has no production WebGPU framework. PPG has LOWER implementation risk for our stack. Citations: github.com/NVIDIA-RTX/NRC, dl.acm.org/doi/10.1145/3757376.3771399. |
| 10 | Sprint 8b spectral upsampling rider | Add Jakob+Hanika 2019 spectral upsampling (1–2 days) | 2019 paper, 6 GLSL instructions per channel, layers cleanly on top of Sprint 8 RGB-as-3λ. Captures most of Sprint 12 visual payoff with no kernel rewrite. Source: rgl.epfl.ch/publications/Jakob2019Spectral. |
| 11 | Sprint 10b WebNN execution provider | Add `webnn` execution provider to ONNX Runtime Web config (1-day rider) | Few-line change in ORT-Web, near-zero risk, gives free near-native acceleration on Edge/Chrome with WebNN behind-flag. Falls back to WebGPU/WASM. |
| 12 | Sprint 5 MRT G-buffer scaffold | Add as Sprint 5 rider (+1 day) | While `trace_scene_function.glsl.js` is already open. Saves ~2 days of rework across Sprints 6/10a/10b that all need normal+depth+albedo MRT. |
| 13 | Sprint 9 Welford struct in common.wgsl | Add as Sprint 9 rider (+2 hrs) | Versioned named struct prevents Sprints 10a/11/13 from independently re-declaring incompatible variance struct layouts. |
| 14 | Sprint 13 trigger update | Add NRC-failed criterion | Trigger now: SVGF gap visible AND WebNN still behind flag AND PPG (Sprint 11) didn't close noise gap. |

### Items rejected after verification

These were proposed in a prior research pass (alternative-technique audit) and rejected after a follow-up implementation-feasibility audit on 2026-05-09:

- **Replace SVGF with BMFR (D3)** — Koskela 2019, claimed 1.8× faster than SVGF. Status: **needs availability verification before any Sprint 10a A/B**. The original 2019 paper is C++ research code; no WebGPU port confirmed. If pursuing, run a focused 1-day spike to find a public implementation BEFORE committing.
- **Conditional ReSTIR pre-conditioning (D6)** — Kettunen SIGGRAPH Asia 2023. Status: **needs availability verification**. Likely Falcor/DXR-bound like ReSTIR BDPT. Must verify before scheduling.
- **ReSTIR PT Enhanced (D8)** — Lin et al. 2026. Status: **needs availability verification**. 2026 means newest = highest SOTA-availability risk. Must verify before adding to roadmap.

### Verification methodology (lessons learned)

The 2026-05-09 alt-techniques audit produced exciting recommendations to replace BDPT and PPG with SOTA successors. The follow-up implementation-feasibility audit found both targets were hardwired to DirectX 12 + RTX hardware + Falcor framework, with porting estimates of 3–5 months rather than the originally-claimed 2 weeks. **Whenever a SOTA paper is proposed, verify three things before scheduling:**

1. **Where's the public source code?** No code = re-implementation from paper, multiply effort estimate by 4×.
2. **What's the implementation language and runtime?** CUDA / OptiX / DXR / Falcor / HLSL+Vulkan-RT all = effectively unportable to web platform. Slang has a WGSL backend but only for non-DXR-specific dialects.
3. **What hardware does the cited performance number depend on?** RTX tensor cores, DXR ray query hardware, RT cores — none have WebGPU equivalents. Software fallbacks add 5–10× cost.

A claim of "real-time on mobile GPU" needs to be checked: which mobile GPU? Apple Silicon (Metal-only) vs Adreno (Vulkan-portable) vs Mali vs Samsung Xclipse vary widely on which APIs they expose. WebGPU is more constrained than any of these.

---

## 7. Open questions / situational triggers

These need user sign-off **before** the relevant sprint kicks off, not now:

1. **Sprint 7**: per-glass-type sigma_t for SSS — hardcoded lookup table or per-type sliders? (Affects UI + material-profile schema.)
2. **Sprint 10c BDPT**: re-evaluation criterion — what does "caustic gap remains visible" mean operationally? Suggest: hero render of a panel-floor caustic at 256 samples PT_FINAL; if floor caustic noise SD > X, BDPT triggers. Define X at re-eval time.
3. **Sprint 11 PPG**: bail-out at <30 fps — acceptable, or do we want a higher walkaround fps floor?
4. **Sprint 12 hero-wavelength**: trigger material list — uranium glass + dichroic film + gemstones? Confirm before scheduling.
5. **Sprint 13 custom neural denoiser**: bail-out month-1 criterion — &gt;50 ms inference is the abort threshold. Do we tighten?

---

## 8. Sequence summary

```
Sprint 1  (3.5 hrs)   PT preview perf  ─── ships immediate interactivity
Sprint 2  (1 day)     Foundation       ─── unblocks Sprint 3
Sprint 3  (5.5 days)  Sampling theory  ─── 3-5x variance reduction
Sprint 4  (3.5 days)  BSDF cost        ─┬── parallel possible
Sprint 5  (5 days)    Analytic came    ─┘    (file-independent)
Sprint 6  (5 days)    Quality wins     ─── after Sprints 4+5 baseline
Sprint 7  (7 days)    Volume + SSS     ─── needs Sprint 3 light tree
Sprint 8  (5 days)    RGB-3λ spectral  ─── independent
Sprint 9  (5 days)    Convergence      ─── walkaround only

──── Phase 6 baseline complete: 37.5 days ────

Sprint 10a (10 days)  SVGF             ─── after Sprint 9
Sprint 10b (3 days)   OIDN final pass  ─── orthogonal, ship anytime
Sprint 10c (10 days)  BDPT             ─── conditional on Sprint 7
Sprint 11  (15-20d)   PPG path guide   ─── after Sprint 9+10a
Sprint 12  (20-25d)   Hero spectral    ─── conditional on materials
Sprint 13  (30-40d)   Neural denoise   ─── conditional on SVGF gap

──── Phase 6 complete with all triggers: ~3-5 months ────
```

---

## 9. Verification protocol

To prevent the half-implementation trap, **every sprint has a verification step** that exercises the feature in its mode-scoped target. Verifications captured in `plan/sprint-<N>-benchmark.md` per sprint. The benchmark file template:

```
# Sprint N benchmark

## Baseline (pre-sprint)
- ms/sample: <measured>
- floor-pixel stddev at 192 samples: <measured>
- visible artifact: <description>

## Post-sprint
- ms/sample: <measured>
- floor-pixel stddev at 192 samples: <measured>
- visible artifact: <description or "absent">

## Definition-of-done check
- [ ] Item 1 from sprint DoD
- [ ] Item 2 from sprint DoD
- ...

## Sign-off
Verified by: <name>, <date>
```

A sprint isn't "done" until its benchmark file exists and all DoD items are checked.

---

## 10. Memory + tracking

This roadmap is the source of truth. When a sprint kicks off:
1. TaskCreate matching the sprint
2. Update `MEMORY.md` index entry pointing here
3. On completion: write the benchmark file, mark TaskUpdate completed, update `MEMORY.md`

---

**Awaiting kickoff approval. Sprint 1 (3.5 hours, immediately interactive PT preview + HDRI bug fix) is the recommended starting point.**
