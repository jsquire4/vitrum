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
| **A1** | ✅ **DONE (Campaign 2 Wave A) — V28 radiometric pending** | `pt-webgpu/src/index.ts:844-918,1026-1036`; `wgsl/pathTrace/restirPt{Spatial,Compose,Resolve,Temporal,Producer}` | **Implemented:** producer→temporal→**spatial (5-neighbour GRIS, no /p_src)**→resolve, then a COMPOSITE megakernel folds the reconnection-indirect into the BEAUTY accumulator via an E0-direct / indirect estimator split (producer-dropped specular pixels fall through to the full path). OFF-path byte-identical. `restirPtReuse:true`+full-tier only. | Equal-spp variance-reduction A/B vs megakernel on real GPU (V28-B). | done (impl) |
| **A2** | ✅ **DONE (Wave A) — sTree splits + runaway fixed** | `walkaround-hybrid/src/ppg/sTree.ts:179`, `pipeline/PPGCoordinator.ts:157-455` | **Implemented:** per-cell atomic sample counters (reused UBO pad slot, 16 B unchanged) → GPU readback → `splitOverflowLeaves`; children seeded via `cloneDTree`; growth-aware GPU clears (a 2nd latent bug found+fixed). **Runaway fixed:** root-caused (decay=1 diverges linearly) → `PPG_FLUX_DECAY=0.5` Müller per-window decay, analytic steady-state F/(1−d) pinned by `ppgSpatialSplitAndRunaway.test.ts`. | Multi-region guiding-localization A/B on real GPU (V28-B). | done (impl) |
| **A3** | ✅ **DONE (Wave B) — true spectral reflectance** | `wgsl/pathTrace/shadePrologue.wgsl.ts:127-147`, `material.wgsl.ts:732-769` | **Implemented:** in spectral mode the RGB albedo is replaced by a SCALAR spectral reflectance S(λ) at the hero λ via Jakob-Hanika per-material coeffs (solved at pack time), broadcast to all channels so throughput·brdf·NEE·MIS carries a genuine single-wavelength quantity. Material stride 26→27 (stale adjoint-stride latent bug fixed). Flat-spectrum invariant harness pins RGB-mode byte-identity. **Documented approximation:** emitter/env chroma is reconstructed as a D65-relative tristimulus SPD (not authored spectra); materials lacking packed coeffs fall back to RGB-luminance. | Per-material Jakob-Hanika lane for pt-webgl2; dispersive-scene A/B vs RGB (V28-B). | done (impl) |
| **A4** | ✅/◻ **Photon-map caustics: finish or consolidate** | `pt-webgpu/src/wgsl/pathTrace/caustic.wgsl.ts:914,1019` | Self-documented ~21% of true energy, fires on ~1% of pixels, hardcoded `0.25/1.25` brightness fudge + `0.35` gather radius. MNEE modes are the validated path. | Either a real progressive photon map (density estimation, no fudge) **or** an explicit decision to make MNEE the sole caustic path and retire photon-map. Decision needed (see Questions). | L–XL |
| **A5** | ✅ **DONE (Wave A) — pt-webgl2 BDPT host driver** | `pt-webgl2/src/glsl/composeTraceGlsl.ts:275-383`, `gl/glResources.ts`, `index.ts:37-39` | **Implemented:** 3-column ping-pong light-path texture + per-column subpath dispatch loop (`uBdptLightSubpathPass`/`uBdptVertexCol`/`uBdptMaxLightBounces` driven; blit-preserve); the inert-warn is removed. `bdptDriver.test.ts` pins it. | A/B vs pt-webgpu BDPT on a glass Cornell (V28-B). | done (impl) |
| **A6** | ✅ **NRC: from opt-in/biased to a validated consumable** | `walkaround-hybrid/src/neural/nrc/*` | Online training (forward+backprop+Adam) is genuinely live on GPU, inference is consumed — but OFF by default, acknowledged-biased, and the zero-padded training tail trains on zero-target samples. | Fix tail-padding bias; converge/quality A/B vs reference; decide default-on tier; document the bias bound. | L |
| **A7** | ◻ **Radiance Cascades as a first-class contributor** | `walkaround-rc/*`, `shade.wgsl.ts:397-418`, `HybridEngine.ts:843` | Pyramid built + merged + MIS-combined into shade when `rcEnabled` — but default OFF, `_rcWeight=0`, first-bounce-only. ~40% of the package (a TSL path) appears unconsumed. | Decide RC's role vs DDGI/ReSTIR-GI; if kept, enable per-tier + validate; delete the unconsumed TSL path if dead. | M |
| **A8** | ✅ **Unbiased ReSTIR-GI reuse by default** | `temporalGi.wgsl.ts`, `spatialGi.wgsl.ts`, gated by `restirPtReuse` | Default temporal/spatial GI reuse is **biased** (clamped-Jacobian reconnection, no MIS). The unbiased GRIS path (full generalized-balance MIS) exists but is off-by-default. | Make GRIS the default (or quantify+document the bias of the default); converged A/B GRIS-on vs off. | M |
| **A9** | ✅ **MOSTLY DONE (Wave A) — serial build retained** | `wgsl/bdpt/bdptLightSubpath.wgsl.ts:382-433`, `index.ts:311` | **Implemented:** light subpath now carries a REAL glossy/specular BSDF (VNDF) at each vertex (was Lambertian), isotropic point-emitter model, bounce cap 3→8, vertex rows 3→4. **Honest remaining:** the subpath build is RETAINED as a single shared-path serial dispatch (one workgroup — documented; the per-column variant was a spec-undefined cross-workgroup race). Still OFF-default. | Parallelize the subpath build; default-on `full` tier after the A/B. | mostly done |
| **A10** | ✅ **Pipeline E2E (Wave A) — weights not shipped** | `tools/neural-denoiser-training/{train.py,capture-dataset.mjs,export_weights.py}` | **Implemented:** capture→train→export→load CLOSES — `train.py --dry-run/--smoke` exports a valid 535,107-param `.vitrum-model` binary (CANONICAL_PARAM_COUNT pinned to the engine loader; the "vi-neural-weights.json" reference was stale — binary is the real format), round-trip test green; `capture-dataset.mjs` CPU smoke. **Remaining:** no real trained checkpoint ships (real dataset + torch training = hardware/provisioning tail); `neural` denoiser stays experimental in capabilities until weights exist. | Vendor a trained checkpoint; quality A/B. | pipeline done / weights remaining |

---

## Bucket B — Fidelity ceilings in the default path

Deliberate approximations a discerning user hits immediately. Closing these is what makes
the *default* render "hero-fidelity" rather than "good real-time."

| ID | Item | Where | State now | Done = | Effort |
|----|------|-------|-----------|--------|--------|
| **B1** | ✅ **DONE (Wave A) — metals/glossy lit; glass GI out of scope** | `shaders/risGi.wgsl.ts:159-168`, `materialDecode.wgsl.ts`, `shade.wgsl.ts` | **Implemented:** a per-tri `bvh_material` r32uint texture (rough[31:24]\|metal[23:16] — a TEXTURE, so it stays off the 16-storage-buffer floor) carries real roughness/metalness; metals now get DI + analytic NEE + specular indirect (`evalGGXSpecularOnly` re-weight of the chosen GI sample, routed un-demodulated). **Honest design boundary:** the GI reservoir p̂ is DELIBERATELY left Lambertian (specular = deterministic re-weight → no p̂/consumption bias, no demod conflict). Diffuse-default byte-identical. **Remaining:** glass refracted GI still gets an empty reservoir (explicitly out of scope this pass). | Glass refracted GI; NRC xsRough wiring (follow-up). | done (impl, glass out of scope) |
| **B2** | ◻ **DDGI diffuse-only bounce; metals as diffuse** | `ddgi/wgsl/probeUpdateRays.wgsl.ts:550-552` | Probe bounce treats metals as diffuse reflectors; diffuse-only multi-bounce model. | Glossy/metal-aware probe bounce, or an honest documented ceiling + specular complement. | M–L |
| **B3** | ✅ **DONE (Wave B) — directional IBL** | `HybridEngine.ts:1575-1578,1800-1852`, `pipeline/WalkaroundGPUPipeline.ts:1340`, `shaders/environmentSample.wgsl.ts`, `environment/equirectDirectional.ts` | **Implemented:** the scene-load env path builds equirect importance-sampling inverse-CDFs (marginal + conditional, sinθ-weighted), pushes radiance map + CDFs to scene-group bindings 15-19; `environmentSample.wgsl` draws directional samples (rotationY-aware) with the scalar-tint fallback intact (hasEnv=0). Ledger hdri grade promoted approximate→native. **Remaining:** the `updateEnvironment` runtime fast-path is still intensity/tint-only (no equirect rebuild) — re-resolve via setScene for full directional sampling on a swap. | updateEnvironment directional fast-path; A/B (V28-B). | done (impl) |
| **B4** | ✅ **DONE (Wave A) — pt-webgl2 mesh-area NEE** | `scene/meshAreaLights.ts`, `glsl/composeTraceGlsl.ts:197-205,896-913`, `scene/foldEmissiveEmitters.ts` | **Implemented:** a dedicated `uMeshLights` triangle-light texture (6 texels/tri) is NEE-sampled with area-proportional selection → triangle-independent pdf → forward-hit MIS from one global `uTotalEmissiveArea`. The emissive-fold is kept as the BSDF strategy (exactly-one-MIS-estimate algebra documented). The analytic `lightsTexture` still excludes `mesh-area` by design — NEE now comes from the separate mesh-light texture. | Variance A/B on Cornell (V28-B). | done (impl) |
| **B5** | ✅ **DONE (Wave A) — Beer-Lambert DDGI probes** | `ddgi/wgsl/probeUpdateRays.wgsl.ts:276-290` | **Implemented:** real `transmission · exp(−attenuationColor · t/attenuationDistance)` over path length, with `t` thickness-clamped (`clamp(distToExit,0,thickness)`); reduces to Beer-Lambert exactly. | — | done |
| **B6** | ✅ **DONE (Wave B) — GTAO per-pixel view axis** | `shaders/gtao.wgsl.ts:120-188` | **Implemented:** per-pixel view axis reconstructed from the inverse perspective projection (was the constant `(0,0,-1)` central-pixel approximation); correct at wide FOV / frame edges. | — | done |
| **B7** | ✅ **DONE (Wave B) — planar-SAH half-area fix** | `shared-bvh/src/buildArrayBvh.ts:127-147` | **Implemented:** `surfaceArea` now returns a nonzero half-perimeter term for planar boxes (one extent 0) so flat geometry ranks splits — a 2000-tri coplanar floor builds depth 45→9. **Remaining (out of scope):** no SBVH; recursive builder retained. | Optional SBVH; iterative build. | done (planar) / SBVH remaining |
| **B8** | ✅ **DONE (Wave B) — light-tree orientation cones** | `shared-samplers/src/lightTree.ts:48-74,387` | **Implemented:** Conty-Estévez orientation cone (axis + thetaO + thetaE) per node, stride 12→16; spot/area producers wired; full-sphere sentinel keeps the cone term ≡1 (byte-identical when unoriented). | A/B on directional-emitter scenes (V28-B). | done (impl) |
| **B9** | ✅ **DONE (Wave B) — GGX multiscatter (all 3 backends)** | pt-webgpu `material.wgsl.ts`; pt-webgl2 `glsl/render/get_surface_record_function.glsl.js`; walkaround `ggxBrdf.wgsl.ts` | **Implemented:** Kulla-Conty multiscatter energy compensation in all three GGX evals (LUT + furnace test on pt-webgpu; furnace-pinned on pt-webgl2, lite-mode skipped). | — | done |
| **B10** | ✅ **DONE (Wave B) — physical refraction transmittance** | `wgsl/pathTrace/bsdf.wgsl.ts` | **Implemented:** physical Fresnel-consistent transmittance replaces the phenomenological `mix(vec3(1),baseColor,0.15)` tint. | — | done |
| **B11** | ◻ **pt-webgl disc-area + procedural-sky** | `three-bindings/vitrumSceneToThree.ts:503,673` | disc-area → area-preserving rect approximation (warns); procedural-sky env unwired → dark background fallback (engine works around it by pre-baking sky→HDRI). | Native disc-area emitter; real procedural-sky → IBL path. | M |
| **B12** | ◻ **DOCUMENTED (Wave B) — lite-tier fidelity cliff** | `webgpuLimits.ts:35-54`, `index.ts:426-439`, `wgsl/pathTrace/kernelLite.wgsl.ts` | **Wave B = binding-budget PROOF only** (the fidelity-cliff arithmetic, PINNED by `wgslLiteContract.test.ts`): on capped (≤10 storage buffer) adapters there is no headroom to add env importance + area-light MIS storage. The cliff is now documented, NOT closed. **Remaining honest gap:** lite texture-packing follow-up to fit HDRI importance + area-light MIS within the budget (or accept the documented cliff). | Texture-pack env CDF + tri-lights into the lite budget. | documented / lite texture-pack follow-up |

---

## Bucket C — Provisioning (turnkey usability)

The code is done; what's missing are shipped assets / managed deps so a consumer gets a
working feature out of the box.

| ID | Item | Where | State now | Done = | Effort |
|----|------|-------|-----------|--------|--------|
| **C1** | ✅ DONE (capabilities/error approach) | `pt-webgpu/index.ts`, `pt-webgl/ptEngineWebGL2.ts`, `oidnBridge.ts` | Real ONNX inference; needs a host-supplied `.onnx` model **and** the `onnxruntime-web` peer dep. | Both backend factories now throw a clear **two-asset** error naming the model URL AND the `onnxruntime-web` peer dep up front (was modelUrl-only + a late first-frame runtime throw); the `oidn` option JSDoc states it is NOT turnkey + lists both assets; the bridge's missing-runtime error already says `npm install onnxruntime-web`. No binary vendored (per "set aside distribution"). | done |
| **C2** | — **Neural denoiser checkpoint** | (see A10) | Overlaps A10. | — | XL |
| **C3** | ◻ **pt-webgl OIDN color-only** | `pt-webgl/.../oidnFinalDispatcher.ts:46-58`; fork RT not MRT | The fork's primary target isn't MRT, so albedo/normal aux can't be captured → only the color-only OIDN model is usable. | MRT fork render target → enable the `hdr_alb_nrm` model variant. | M |

---

## Bucket D — Hygiene / maintainability (the "obvious gaps, just close them")

Mostly S-effort. These don't change rendering but they mislead readers, ship dead weight,
or silently drop user data — exactly the rot that has made the maturity picture hard to read.

> **Status 2026-06-09:** D1 (dead code), D2 (silent drops), D4 (memory accounting), D5
> (stale comments), D9 (traceTier dedup) are DONE on branch `road-to-100/hygiene-provisioning`
> — behavior-preserving, typecheck clean, tests green. D7 verified mostly-stale. D3 (contract +
> ingestion) and C1 (OIDN clear-error/capabilities) now DONE too. Remaining: D8-approach
> (fork eslint dep — decision-gated), D3 per-backend BSDF consumption (B-bucket fidelity), and
> D6 (bind-group churn — perf, deferred).

- **D1 — Dead code removal** ◻: pt-webgl2 `frameParamsPacker.ts` std140 UBO (~280 LOC) + `glResources` `uploadFrameParams`/`#paramsUbo`/`#bindParamsUbo` (no callers, `FRAME_PARAMS_SIZE=256` placeholder); pt-webgl2 `'additive'` accumulation regime + `blend.ts:37` case (unreachable); pt-webgl `debounceMsForEditRate` + `PT_DEBOUNCE_MS_*` (zero consumers); `PPGCoordinator.resetTrainingAccumulators` (no callers); stained-glass `packCameUBO` (no runtime consumer, no shader reads came data); `sTreeAccumulate` (dead — resolve with **A2**, don't just delete); audit P2 dead surface to verify+remove (`heroStrategy` UBO slot, `probesPerFrame` UBO field, `GPU_SKIN_BVH_WGSL`, `expandIndicesToStride4`, `RESTIR_PT_HYBRID_SHIFT` harness, `ownsEnvSampler`, `cleanupAfterSubmit` hook, walkaround-rc TSL path). **Effort: M total.**
- **D2 — Silent data drops** ◻: `three-bindings/src/index.ts:270` drops `THREE.Points/Line/LineSegments/Sprite/BatchedMesh/LOD` with **no warning** (lights get `warnOnce`, geometry types get nothing); pt-webgpu `uploadSceneBuffers.ts:1086,1088` discards `MaterialTextureArray.warnings`; heterogeneous texture sizes copy into a max-size layer with **wrong UVs** and no warning. **Fix: warn-or-handle. Effort: S–M.**
- **D3 — Contract material gaps** ✅ DONE (contract + ingestion) / ◻ consumption tracked: added `specularIntensity`/`specularColor` (+ their maps), `bumpMap`/`bumpScale`, `displacementMap`/`displacementScale`/`displacementBias`, `lightMap`/`lightMapIntensity`, `envMapIntensity` as first-class optional fields on core `MaterialSpec` (+ `MaterialMapFields` slice); `three-bindings.convertMaterial` now extracts them so the THREE→core data loss is closed (+4 tests). **REMAINING — per-backend BSDF consumption (these require golden-breaking material-layout changes, so they're real B-bucket fidelity work, not ingestion): specular F0 modulation + envMapIntensity scale in the 4 BSDFs; bump-map normal perturbation; displacement-map geometry; lightMap additive in shade — plus the reverse `vitrumSceneToThree` round-trip for the pt-webgl path.** **Effort: ingestion M (done); consumption S–L per field.**
- **D4 — Memory accounting** ◻: `debug.estimatedGpuMemoryBytes()` omits all scene buffers (BVH/materials/indices) — `pt-webgpu/src/index.ts:455`. Under-reports materially. **Effort: S.**
- **D5 — Stale comments contradicting code** ◻: GRIS reservoir "written-but-unread until Phase 1/2" (`createRestirGIFrameResources.ts`, `resourceManager.ts:159`) — the GRIS shaders DO read those fields; RC light-model comment `shade.wgsl.ts:404` (RC now does rect-area emitter NEE); `svgfVarianceMain`/`svgfAtrousMain` names in `atrousVariance.wgsl.ts` (NOT SVGF); `cascadeDispatch.ts:18` "not verified" header (since GPU-exercised). **Effort: S.**
- **D6 — Per-frame bind-group churn** ◻: ~20–30 `createBindGroup`/frame in walkaround, zero memoization. Perf hygiene. **Effort: M.**
- **D7 — SVGF texture allocation** ✅ VERIFIED mostly-stale: `createSvgfFrameResources.ts:63-64` collapses the heavy SVGF textures to 1×1 when the denoiser isn't `svgf-real` (`const w = svgfEnabled ? width : 1`) — the audit's "~80MB always allocated" is wrong. Residual: two object-id textures (`svgfCurrentObjectIdTexture`/`svgfPreviousObjectIdTexture`) ARE full-res unconditionally (~28MB@1440p); small follow-up to confirm they're consumed off the svgf path before gating. **Effort: S (residual only).**
- **D8 — fork lint red** ✅ FIXED (bumped to ESLint 9): the red was an `eslint@8` vs `@typescript-eslint@8` plugin crash (`no-unused-expressions` reading `allowShortCircuit`) — NOT the audit's stale-SSS gate (`tsc` + `shader-smoke` always passed). Fix: bumped the fork to `eslint@^9.39.4` (deduped to root; had to prune an orphan nested `eslint@8.57.1` the lockfile kept reinstalling — uninstall→reinstall on the workspace cleared it), kept `.eslintrc.json` via `ESLINT_USE_FLAT_CONFIG=false` in the lint script, and made its `extends` hoist-proof (`"mdcs"` shareable name instead of a relative `./node_modules/...` path that broke when mdcs hoisted to root). `npm run lint` is now green (0 errors, 1 pre-existing `no-unused-vars` warning in `example/`; tsc + shader-smoke pass). **Future:** eslintrc is deprecated in eslint v10 → a flat-config migration when the repo moves to v10.
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

## Addendum — 2026-06-09 second-wave deep-read (11 agents, lead-verified)

A full line-by-line re-read of all 12 packages found that several bucket entries UNDERSTATE
the gap, plus new fidelity items. **Bugs/broken-surface findings went to `items_to_fix.md`
Section H** (H1–H38, ✅/◻ legend there); this addendum only adjusts THIS doc's picture:

- **"Foundations + default render paths ~90%" needs a caveat: pt-webgl2 is NOT at RC level.**
  ✅ Its entire analytic-light system is inert (`lights.count` never uploaded — items H1),
  `spectral` renders black (H2), directly-visible env never accumulates (H3), and `bdpt` is
  never driven (H5). As shipped it is an emissive-geometry-lit RGB tracer (~60-70%); the
  fork-vs-native A/Bs passed because they exercised exactly the paths that work
  (emissive-fold Cornell, glass, IBL-indirect, textures). Treat pt-webgl2's
  release-candidate label as suspended until H1–H5 land.
- **B1 is WORSE than written:** metals are excluded from DIRECT light too, not just GI
  (`shade.wgsl.ts:230,309` — lo_direct/lo_indirect both return 0 for isGlass||isMetal), so
  metals are effectively unlit except emitter glow; and the comment's escape hatch ("the
  path-traced fork") was deleted 2026-06-09. Also the material payload carries only
  RGB888 + 4-bit transmission + 1-bit metal + 3-bit texId — authored roughness/metalness
  never reach the BRDF (two hardcoded roughnesses, `shade.wgsl.ts:519-521`). Closing B1
  properly requires widening the packed material lane, not just a shader change.
- **A2 (PPG) is two defects, not one:** beyond the sTree never splitting, ◻ the dTree's
  interior nodes never carry flux → descent is uniform above the leaf level AND
  `ppgEvalPdf` mismatches the actual sampling distribution → biased RIS when enabled
  (items H25). The CPU oracle shares the flaw (byte-identity green while both wrong).
- **A6 (NRC) has a structural ceiling:** ◻ the spread-termination predicate is
  constant-true at the default `spreadC` and the training target is the DDGI estimate the
  cache replaces — distillation, no upside (items H26/H27). A6's "validated consumable"
  goal needs the Müller a0 semantics (camera-pdf footprint) + a path-traced training
  target, not just tail-padding fixes.
- **A10 (neural denoiser) is blocked before weights:** ✅(static) the in-place ReLU layers
  bind one buffer as read + read_write in a single bind group — likely hard validation
  failure on any real adapter (items H28). Repro on real GPU before training anything.
- **NEW B13 — walkaround texture sampling is broken at the seam** ✅: UVs zeroed at both
  `restir/bvhCore.ts` build sites (items H15). The G3 texture work (63a6dab) wired UVs
  through shared-bvh, but walkaround discards them. S-effort, render-changing.
- **NEW B14 — DDGI emitter blindness** ◻: probe rays see sun+point/spot+sky only; emissive
  surfaces and area emitters contribute zero DDGI indirect (items H18). The RC emitter NEE
  fix (1e893fa) has no DDGI counterpart. M–L effort.
- **NEW B15 — ✅ DONE (Wave B) scene-scale-aware radiometric clamp defaults**:
  `HybridEngineScaleAwareClamps.ts` + `HybridEngine.ts:719-1145` derive the clamp DEFAULTS
  from the scene diagonal at setScene (1/s² law on the radiometric knobs: irradiance/GI-W/
  firefly clamps), so the Cornell-tuned absolutes no longer cap GI energy in larger scenes;
  hosts that set a clamp explicitly keep their absolute value (override flags captured).
  Root-caused the size-200 estimator instability to this bimodal clipping (the 1/dist²
  suspect was REFUTED — the base estimator is scale-invariant). V28 clamp-sweep scenario
  specced. **Remaining:** real-GPU clamp-sweep A/B (V28-B).
- **NEW B16 — ✅ DONE (Wave B) DI BRDF candidate**: `ris.wgsl.ts:83,255-279` now SAMPLES
  the `M_BRDF=1` GGX-VNDF candidate (measure-converted solid-angle→same RIS measure),
  contributing to glossy DI. Render-changing (NOT byte-identity-preserving). **Remaining:**
  glossy-DI A/B (V28-B).
- **C-bucket correction:** C1's "clear error" fix covers the FACTORY; the runtime
  dispatcher (`oidnDispatcherCore.ts:338-340`) still converts every OIDN failure into one
  console.warn → silent un-denoised frames (items H35). The host-visible failure surface
  (denoiser `state()`) has zero consumers. S effort, high consumer value.

**Second-wave claims-surface audit (same session, items H39–H59) added three structural
buckets that the A–D framing was missing:**

- **NEW C4 — zero examples** ✅: the THREE cutover deleted `examples/` and nothing replaced
  it; every public entry point (`createEngine`, `attachVitrum`, `VitrumCanvas`,
  `createProgressiveEngine`, both PT factories, the hybrid factory) has no runnable example
  (items H57). For "a professional library others can use," this is a provisioning gap on
  par with the OIDN assets. M effort (one core-Scene Cornell example per entry point).
- **NEW C5 — contract-truth reconciliation** ✅: `promiseLedger` rows contradict shipped
  runtime capabilities (pt-webgl2 analytic/mutations/aux); the fidelity matrix's `pt-webgl`
  column describes a deleted package and omits pt-webgl2; CHANGELOG `[Unreleased]` has no
  Removed entry for e14000c; ~6 tool READMEs document dead workflows; 2 packages have no
  README (items H39–H45, H59). S–M effort, zero rendering risk, large honesty payoff.
- **NEW D10 — test-infrastructure gates** ✅/◻: the suite is structurally blind to the
  H1-class (mock GL accepts every uniform; stub GPUDevices validate no sizes; no vitest run
  compiles any shader; the only behavioral gates live outside `npm test`). The top-3 payoff
  tests: GL uniform-upload completeness via recording mock, in-repo naga parse gate, and a
  size-validating GPU stub (items H53–H56 list ten, prioritized). M effort total; this is
  what stops the next H1 from shipping green.
- **MaterialSpec consumption matrix** (items H46–H52): the contract advertises ~60 material
  fields; walkaround's default path consumes ~8 (with roughness/metallic/ior/UVs among the
  casualties — see B1/B13), and a dozen fields have zero consumers in ANY backend
  (anisotropy*, envMapIntensity, aoMap, bumpMap, displacementMap, lightMap,
  angularDiameter, castShadow/receiveShadow, tangents). Extends D3's "consumption tracked"
  list with the verified full set; `denoiser` is additionally a silent no-op on both PT
  backends for any value but `'oidn-final'` (H48).

## Open decisions (need a call before building)

- **A4:** real progressive photon map, or retire photon-map and make MNEE the sole caustic path?
- **A7:** is Radiance Cascades a default contributor, an opt-in, or retired in favor of DDGI⊕ReSTIR-GI?
- **A6/A10:** ship NRC/neural-denoiser on-by-default (needs trained assets) or keep opt-in+experimental?
- **B1:** glossy GI via ReSTIR-GI full-BRDF p̂, or specular probes, or keep punting glossy to the PT backend?
