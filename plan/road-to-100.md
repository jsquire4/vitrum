# Road to 100% — vitrum

> Authored 2026-06-09 from a full code-truth re-audit (15 packages read line-by-line by
> deep-reader agents; every load-bearing claim re-verified by lead code-read/grep).
> Scope per the user's request: **frontier-feature completion, fidelity, provisioning,
> hygiene.** Deliberately EXCLUDES public-distribution posture, release governance, and
> cross-host GPU-validation evidence (tracked separately in `HARDWARE-VALIDATION-NEEDS.md`).

## Where we actually are

- **Foundations + default render paths: ~90% (release-candidate).** The `@vitrum/core`
  contract, each backend's default integrator, shared-bvh/samplers/denoisers, and
  three-bindings are real, correct, type-clean (typecheck green across 22 pkgs), and
  test-backed (~3,000 assertions / 321 files).
- **The P0 default-path correctness gate is essentially CLEAR.** The three findings the
  2026-06-06 audit left open are resolved in current code (verified this session): ReSTIR-DI
  p̂ uses a consistent smooth normal across RIS/temporal/spatial; DDGI π double-count is gone
  after the octahedral→L2-SH migration + `8aa444a`; PPG directional flux IS accumulated.
- **The real distance to "fully implemented + professional" is three buckets:**
  **A** frontier features that are wired-but-inert/partial, **B** deliberate fidelity
  ceilings in the default path, **C** provisioning, **D** hygiene. That's what this doc lists.

**Legend:** `✅` = verified by lead code-read/grep this session · `◻` = reported by a
deep-reader agent with file:line, not independently re-verified (verify before acting).
**Effort:** S = hours · M = 1–3 days · L = ~1 week · XL = multi-week / research-grade.

---

## Bucket A — Frontier feature completion (largest gap to "fully implemented")

These are the features the README/feature-list advertises that a *default* engine does not
actually deliver. Per the project north-star (fidelity-paramount, implement-don't-remove),
the target is to make each one real and consumed — not to demote it.

| ID | Item | Where | State now | Done = | Effort |
|----|------|-------|-----------|--------|--------|
| **A1** | ✅ **pt-webgpu ReSTIR-PT → composite into beauty + add spatial reuse** | `pt-webgpu/src/index.ts:788,930-939`; `wgsl/pathTrace/restirPt*` | Producer→temporal→resolve run when `restirPtReuse:true` but write to a **separate `rpt_result` debug buffer**; "the beauty image is the unmodified megakernel result." No spatial-reuse pass exists. | Resolve output MIS-composited into the accumulator; spatial reuse pass added; variance-reduction A/B vs megakernel at equal spp. | L |
| **A2** | ✅ **PPG spatial sTree adaptivity** | `walkaround-hybrid/src/ppg/sTree.ts:132`, `PPGCoordinator.ts` | Directional dTree trains (flux atomicAdd is live), but `sTreeAccumulate` has **zero callers** → spatial tree never splits → one global cell for the whole scene. | Feed per-cell sample counts (from the GPU update pass) into `splitOverflowLeaves`; verify the tree subdivides on a multi-region scene and guiding localizes. | M |
| **A3** | ◻ **pt-webgpu true spectral transport** | `wgsl/pathTrace/kernel.wgsl.ts:709`, `material.wgsl.ts` | "Spectral" = hero-wavelength **tint over an RGB path**: throughput is RGB, collapsed to scalar luminance, re-expanded through the hero λ's CMF. Only IOR dispersion + Beer-Lambert μ are truly λ-resolved. | BRDF/albedo evaluated as spectral reflectance at the hero λ (Jakob-Hanika upsampling already exists in shared-samplers); throughput carried spectrally; A/B vs RGB on a dispersive scene. | XL |
| **A4** | ✅/◻ **Photon-map caustics: finish or consolidate** | `pt-webgpu/src/wgsl/pathTrace/caustic.wgsl.ts:914,1019` | Self-documented ~21% of true energy, fires on ~1% of pixels, hardcoded `0.25/1.25` brightness fudge + `0.35` gather radius. MNEE modes are the validated path. | Either a real progressive photon map (density estimation, no fudge) **or** an explicit decision to make MNEE the sole caustic path and retire photon-map. Decision needed (see Questions). | L–XL |
| **A5** | ◻ **pt-webgl BDPT on production drivers** | `pt-webgl/src/ptEngineWebGL2.ts:581,730` | `FEATURE_BDPT` compile is gated off whenever renderer string matches `/angle/i` — i.e. **all of Chrome/Edge/Firefox on Windows**. Inert on every prod desktop driver; only runs on native Linux GL. | Diagnose the ANGLE shader-compile failure and make BDPT compile there, OR formally scope BDPT to pt-webgpu and stop advertising `bdpt-approximate` on pt-webgl. | M |
| **A6** | ✅ **NRC: from opt-in/biased to a validated consumable** | `walkaround-hybrid/src/neural/nrc/*` | Online training (forward+backprop+Adam) is genuinely live on GPU, inference is consumed — but OFF by default, acknowledged-biased, and the zero-padded training tail trains on zero-target samples. | Fix tail-padding bias; converge/quality A/B vs reference; decide default-on tier; document the bias bound. | L |
| **A7** | ◻ **Radiance Cascades as a first-class contributor** | `walkaround-rc/*`, `shade.wgsl.ts:397-418`, `HybridEngine.ts:843` | Pyramid built + merged + MIS-combined into shade when `rcEnabled` — but default OFF, `_rcWeight=0`, first-bounce-only. ~40% of the package (a TSL path) appears unconsumed. | Decide RC's role vs DDGI/ReSTIR-GI; if kept, enable per-tier + validate; delete the unconsumed TSL path if dead. | M |
| **A8** | ✅ **Unbiased ReSTIR-GI reuse by default** | `temporalGi.wgsl.ts`, `spatialGi.wgsl.ts`, gated by `restirPtReuse` | Default temporal/spatial GI reuse is **biased** (clamped-Jacobian reconnection, no MIS). The unbiased GRIS path (full generalized-balance MIS) exists but is off-by-default. | Make GRIS the default (or quantify+document the bias of the default); converged A/B GRIS-on vs off. | M |
| **A9** | ◻ **pt-webgpu BDPT to production quality** | `wgsl/bdpt/*`, `index.ts:819` | Complete + correct MIS, but OFF-default, light subpath is **Lambertian-only**, capped ≤3 light bounces, dispatched as a **single workgroup (serial)**. | Glossy/specular light-path BSDF; parallelize the subpath build; raise bounce cap; on-by-default for the `full` tier with an A/B. | L |
| **A10** | ◻ **Neural denoiser ships no weights** | `walkaround-hybrid/src/neural/.../weights.ts:46-48` | InferenceGraph U-Net dispatch is real, but no checkpoint ships; without host weights it runs He-random output ("will NOT be meaningful"). | Train + vendor a `vi-neural-weights` checkpoint (or wire a known-good model); validate denoise quality; otherwise mark the `neural` denoiser experimental in capabilities. | XL |

---

## Bucket B — Fidelity ceilings in the default path

Deliberate approximations a discerning user hits immediately. Closing these is what makes
the *default* render "hero-fidelity" rather than "good real-time."

| ID | Item | Where | State now | Done = | Effort |
|----|------|-------|-----------|--------|--------|
| **B1** | ✅ **Glossy/metal GI in walkaround** | `risGi.wgsl.ts:164,261` | Glass/metal surfaces get an **empty GI reservoir** (indirect punted to the PT fork); GI target is **Lambertian-only** (`luminance(Lo)·cos·INV_PI`, no GGX/albedo). | Specular/glossy indirect via ReSTIR-GI (full-BRDF p̂) or specular probes; metals receive real indirect. | L–XL |
| **B2** | ◻ **DDGI diffuse-only bounce; metals as diffuse** | `ddgi/wgsl/probeUpdateRays.wgsl.ts:550-552` | Probe bounce treats metals as diffuse reflectors; diffuse-only multi-bounce model. | Glossy/metal-aware probe bounce, or an honest documented ceiling + specular complement. | M–L |
| **B3** | ◻ **walkaround directional IBL** | `resolveHybridEnvironment.ts:201-271`, `HybridEngine.ts:1601` | HDRI reduced to a single solid-angle-averaged **scalar tint**; no directional distribution, no rotation. | Importance-sampled environment (reuse the equirect CDF machinery from pt-webgl2/fork); rotation support. | L |
| **B4** | ✅ **pt-webgl2 mesh-area NEE** | `scene/lightsTexture.ts:95`, `foldEmissiveEmitters.ts` | `mesh-area` emitters filtered out of the analytic light list; lit only by random emissive-surface hits (unbiased but noisy). | Add mesh-area emitters to the lights texture (or a tri-light list) for NEE/MIS; variance A/B on Cornell. | M |
| **B5** | ◻ **Beer-Lambert glass in DDGI probe rays** | `probeUpdateRays.wgsl.ts:288` | Linear `attenuationColor * transmission` tint (comment admits "does NOT reduce to Beer-Lambert"); `thickness`/`attenuationDistance` already carried in the material. | Real Beer-Lambert absorption over path length. | S–M |
| **B6** | ◻ **GTAO view-space reconstruction** | `shaders/gtao.wgsl.ts:146,183` | Fixed view axis `(0,0,-1)` + screen-space pixel offsets; "central-pixel approximation for perspective." | Reconstruct view-space positions; correct at wide FOV / frame edges. | M |
| **B7** | ✅/◻ **BVH planar-SAH + binning** | `shared-bvh/src/buildArrayBvh.ts:132,57,54` | `surfaceArea` returns 0 for any flat box → planar geometry (floors/walls) splits with no SAH discrimination. Fixed K=16 bins, max-leaf 4, no SBVH, recursive builder (JS-stack-bounded). | Thin-AABB epsilon / half-perimeter so planar nodes rank splits; optional SBVH for overlapping geometry; iterative build. | M (planar) / L (SBVH) |
| **B8** | ◻ **Light-tree orientation cone** | `shared-samplers/src/lightTree.ts:48,387` | Spatial-only Conty-Estévez (power/dist² importance); no emission-cone axis/angle term. | Add the orientation-cone term; A/B on directional-emitter scenes. | M |
| **B9** | ◻ **GGX multiscatter energy compensation** | fork `ggx_functions.glsl.js`; pt-webgpu `material.wgsl.ts`; walkaround `ggxBrdf.wgsl.ts` | Single-scatter VNDF only; no multiscatter compensation → energy loss on rough metals. | Kulla-Conty (or similar) multiscatter term in all three BSDF impls. | M |
| **B10** | ◻ **pt-webgpu refraction throughput** | `wgsl/pathTrace/bsdf.wgsl.ts:291` | Phenomenological tint `mix(vec3(1), baseColor, 0.15)`; arbitrary `0.15`. | Physical transmittance (spectral / Fresnel-consistent). | S–M |
| **B11** | ◻ **pt-webgl disc-area + procedural-sky** | `three-bindings/vitrumSceneToThree.ts:503,673` | disc-area → area-preserving rect approximation (warns); procedural-sky env unwired → dark background fallback (engine works around it by pre-baking sky→HDRI). | Native disc-area emitter; real procedural-sky → IBL path. | M |
| **B12** | ◻ **pt-webgpu lite-tier fidelity cliff** | `wgsl/pathTrace/connectLite.wgsl.ts:32,52` | On capped (≤10 storage buffer) adapters: uniform-sphere env sampling, **no area-light BSDF MIS**. | Restore HDRI importance + area-light MIS within the lite binding budget, or document the cliff in capabilities. | M |

---

## Bucket C — Provisioning (turnkey usability)

The code is done; what's missing are shipped assets / managed deps so a consumer gets a
working feature out of the box.

| ID | Item | Where | State now | Done = | Effort |
|----|------|-------|-----------|--------|--------|
| **C1** | ◻ **OIDN is not turnkey** | `shared-denoisers/src/oidnBridge.ts:25-28,305` | Real ONNX-Runtime inference, but requires a host-supplied `.onnx` model **and** an `onnxruntime-web` peer dep; with neither, every `denoiseFinal` throws. | Vendor/host a canonical OIDN model + make the peer dep managed (or surface "asset required" in capabilities so it's not a silent throw). | M |
| **C2** | — **Neural denoiser checkpoint** | (see A10) | Overlaps A10. | — | XL |
| **C3** | ◻ **pt-webgl OIDN color-only** | `pt-webgl/.../oidnFinalDispatcher.ts:46-58`; fork RT not MRT | The fork's primary target isn't MRT, so albedo/normal aux can't be captured → only the color-only OIDN model is usable. | MRT fork render target → enable the `hdr_alb_nrm` model variant. | M |

---

## Bucket D — Hygiene / maintainability (the "obvious gaps, just close them")

Mostly S-effort. These don't change rendering but they mislead readers, ship dead weight,
or silently drop user data — exactly the rot that has made the maturity picture hard to read.

> **Status 2026-06-09:** D1 (dead code), D2 (silent drops), D4 (memory accounting), D5
> (stale comments), D9 (traceTier dedup) are DONE on branch `road-to-100/hygiene-provisioning`
> — behavior-preserving, typecheck clean, tests green. D7 verified mostly-stale. D3 + C1 +
> D8-approach remain (decision-gated — see below). D6 (bind-group churn) is a perf item, deferred.

- **D1 — Dead code removal** ◻: pt-webgl2 `frameParamsPacker.ts` std140 UBO (~280 LOC) + `glResources` `uploadFrameParams`/`#paramsUbo`/`#bindParamsUbo` (no callers, `FRAME_PARAMS_SIZE=256` placeholder); pt-webgl2 `'additive'` accumulation regime + `blend.ts:37` case (unreachable); pt-webgl `debounceMsForEditRate` + `PT_DEBOUNCE_MS_*` (zero consumers); `PPGCoordinator.resetTrainingAccumulators` (no callers); stained-glass `packCameUBO` (no runtime consumer, no shader reads came data); `sTreeAccumulate` (dead — resolve with **A2**, don't just delete); audit P2 dead surface to verify+remove (`heroStrategy` UBO slot, `probesPerFrame` UBO field, `GPU_SKIN_BVH_WGSL`, `expandIndicesToStride4`, `RESTIR_PT_HYBRID_SHIFT` harness, `ownsEnvSampler`, `cleanupAfterSubmit` hook, walkaround-rc TSL path). **Effort: M total.**
- **D2 — Silent data drops** ◻: `three-bindings/src/index.ts:270` drops `THREE.Points/Line/LineSegments/Sprite/BatchedMesh/LOD` with **no warning** (lights get `warnOnce`, geometry types get nothing); pt-webgpu `uploadSceneBuffers.ts:1086,1088` discards `MaterialTextureArray.warnings`; heterogeneous texture sizes copy into a max-size layer with **wrong UVs** and no warning. **Fix: warn-or-handle. Effort: S–M.**
- **D3 — Contract material gaps** ✅ DONE (contract + ingestion) / ◻ consumption tracked: added `specularIntensity`/`specularColor` (+ their maps), `bumpMap`/`bumpScale`, `displacementMap`/`displacementScale`/`displacementBias`, `lightMap`/`lightMapIntensity`, `envMapIntensity` as first-class optional fields on core `MaterialSpec` (+ `MaterialMapFields` slice); `three-bindings.convertMaterial` now extracts them so the THREE→core data loss is closed (+4 tests). **REMAINING — per-backend BSDF consumption (these require golden-breaking material-layout changes, so they're real B-bucket fidelity work, not ingestion): specular F0 modulation + envMapIntensity scale in the 4 BSDFs; bump-map normal perturbation; displacement-map geometry; lightMap additive in shade — plus the reverse `vitrumSceneToThree` round-trip for the pt-webgl path.** **Effort: ingestion M (done); consumption S–L per field.**
- **D4 — Memory accounting** ◻: `debug.estimatedGpuMemoryBytes()` omits all scene buffers (BVH/materials/indices) — `pt-webgpu/src/index.ts:455`. Under-reports materially. **Effort: S.**
- **D5 — Stale comments contradicting code** ◻: GRIS reservoir "written-but-unread until Phase 1/2" (`createRestirGIFrameResources.ts`, `resourceManager.ts:159`) — the GRIS shaders DO read those fields; RC light-model comment `shade.wgsl.ts:404` (RC now does rect-area emitter NEE); `svgfVarianceMain`/`svgfAtrousMain` names in `atrousVariance.wgsl.ts` (NOT SVGF); `cascadeDispatch.ts:18` "not verified" header (since GPU-exercised). **Effort: S.**
- **D6 — Per-frame bind-group churn** ◻: ~20–30 `createBindGroup`/frame in walkaround, zero memoization. Perf hygiene. **Effort: M.**
- **D7 — SVGF texture allocation** ✅ VERIFIED mostly-stale: `createSvgfFrameResources.ts:63-64` collapses the heavy SVGF textures to 1×1 when the denoiser isn't `svgf-real` (`const w = svgfEnabled ? width : 1`) — the audit's "~80MB always allocated" is wrong. Residual: two object-id textures (`svgfCurrentObjectIdTexture`/`svgfPreviousObjectIdTexture`) ARE full-res unconditionally (~28MB@1440p); small follow-up to confirm they're consumed off the svgf path before gating. **Effort: S (residual only).**
- **D8 — fork lint red (corrected diagnosis 2026-06-09)** ✅ diagnosed: NOT the stale-SSS shader gate the audit claimed — `shader-smoke` AND `tsc --noEmit` both PASS. The red is an ESLint plugin crash: the fork pins `eslint@^8.56` but resolves `@typescript-eslint/eslint-plugin@8.59` (typescript-eslint v8 requires ESLint 9), so `no-unused-expressions` throws reading `allowShortCircuit`. Fork-local only (NOT in the root typecheck/test/GPU-smoke CI path). **Decision needed:** bump the fork to ESLint 9 (+ migrate `.eslintrc.json` → flat config, or set `ESLINT_USE_FLAT_CONFIG=false`), OR pin `@typescript-eslint` to a v7 line compatible with ESLint 8. **Effort: S once approach chosen.**
- **D9 — traceTier dedup** ◻: pt-webgl2 `WebGl2TraceTier` union duplicated (`traceTier.ts:18` vs `options.ts:3`); `lite` tier only disables the g-buffer, doesn't shrink textures/bounces as its doc claims. **Effort: S.**

---

## Suggested sequencing

1. **Hygiene + provisioning first (Bucket D + C1).** Mostly S-effort, removes the noise that
   makes maturity hard to read, stops silent data loss, and makes denoising turnkey. A few
   days, high signal-to-effort.
2. **Finish the cheap frontier wins (A2 PPG sTree, A8 GRIS-default, A1 ReSTIR-PT composite).**
   These convert "wired-but-inert" into "actually delivers."
3. **Close the default-path fidelity ceilings users see first (B4 mesh-area NEE, B3 IBL,
   B1 glossy GI, B9 multiscatter).** This is the biggest perceived-quality lift.
4. **Then the research-grade items (A3 true spectral, A10/C2 neural weights, A9 BDPT-prod,
   A4 caustic decision, B7 SBVH).** Multi-week each; schedule deliberately.

## Open decisions (need a call before building)

- **A4:** real progressive photon map, or retire photon-map and make MNEE the sole caustic path?
- **A7:** is Radiance Cascades a default contributor, an opt-in, or retired in favor of DDGI⊕ReSTIR-GI?
- **A6/A10:** ship NRC/neural-denoiser on-by-default (needs trained assets) or keep opt-in+experimental?
- **B1:** glossy GI via ReSTIR-GI full-BRDF p̂, or specular probes, or keep punting glossy to the PT backend?
