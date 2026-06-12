# Road to 100% — vitrum

> Authored 2026-06-09 from a full code-truth re-audit (15 packages read line-by-line by
> deep-reader agents; every load-bearing claim re-verified by lead code-read/grep).
> Scope per the user's request: **frontier-feature completion, fidelity, provisioning,
> hygiene.** Deliberately EXCLUDES public-distribution posture, release governance, and
> cross-host GPU-validation evidence (tracked separately in `HARDWARE-VALIDATION-NEEDS.md`).

## Where we actually are

> **Updated 2026-06-10 at the END of trust-remediation rounds R7a–R7d (commits a1b85b1, ba1429d, 3f3aa6d, 9c3e6ba, 1a8ab08).**
> Per `plan/v1-closure-plan-2026-06-10.md §0`, "100%" = everything fully implemented.
> **R7a–R7d campaign additions:** behavioral gate (26/26 pass, permanent CI); anisotropic
> GGX (A-item closed — `materialAnisotropy` now renders); engine error surface (`onError` —
> silent-GPU-error class dead); `@vitrum/gltf-adapter` new package (glTF 2.0 → core Scene);
> `captureFrame` pixel-readback API + `pickPrimitive` real on all 3 backends;
> `CameraLike`/`QualityTier`/presets public; examples/ + debugging runbook docs; IES dead
> chain removed; spectral×photon-map gather spectralized; giState v4 (PPG warm restore);
> SPPM streaming-window corrected (non-progressive — see A4-progressive below).
> **A4-progressive DONE (2026-06-10):** true Hachisuka per-pixel SPPM — `SppmPixelStats`
> binding(9), `sppmGatherProgressive` update rule (N′=N+α·M; R′²=R²·ratio; τ′=(τ+Φ_M)·ratio),
> buffer reset on camera/scene/reset, Cesàro accumulator argument, 36 TS-mirror tests.
> **Implementation distance remaining:** A6 NRC semantics; A8 GRIS-default gate;
> sun-NEE-default gate; B2 DDGI glossy bounce; glass refracted GI (B1 tail);
> production-quality neural weights (starter only); `TextureRef.texCoord` on pt-webgl2
> (documented unkept promise); H-residue (H5/H21/H24-cluster/H32/H34/H35).
> **Big validation tail: V28-B** — GPU A/B recapture for every render-changing landing
> (improvement confirmations, not regression suspects).

- **Foundations + default render paths: solid, advancing toward 100%.** The `@vitrum/core`
  contract, each backend's default integrator, shared-bvh/samplers/denoisers are real,
  correct, type-clean (typecheck green across 12 packages), and test-backed (~3,300+
  assertions). CI rewritten; in-repo shader compile gate (48 WGSL shaders, naga-validated).
  The P0 default-path correctness issues from prior audits are resolved.
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
| **A2** | ✅ **DONE (Wave A + v1-closure Wave 4) — sTree splits + runaway fixed; atomics saturate** | `walkaround-hybrid/src/ppg/sTree.ts:179`, `pipeline/PPGCoordinator.ts:157-455`, `wgsl/ppgUpdate.wgsl.ts:241` | **Implemented:** per-cell atomic sample counters → GPU readback → `splitOverflowLeaves`; children seeded via `cloneDTree`; PPG_FLUX_DECAY=0.5 Müller per-window decay. **Wave 4 fix (06910e2):** flux atomics now saturate instead of u32-wrapping (clamp before atomic add); stale "single global cell" JSDoc deleted (code-verified: grep returns 0 hits). | Multi-region guiding-localization A/B on real GPU (V28-B). | done (impl) |
| **A3** | ✅ **DONE (Wave B) — true spectral reflectance** | `wgsl/pathTrace/shadePrologue.wgsl.ts:127-147`, `material.wgsl.ts:732-769` | **Implemented:** in spectral mode the RGB albedo is replaced by a SCALAR spectral reflectance S(λ) at the hero λ via Jakob-Hanika per-material coeffs (solved at pack time), broadcast to all channels so throughput·brdf·NEE·MIS carries a genuine single-wavelength quantity. Material stride 26→27 (stale adjoint-stride latent bug fixed). Flat-spectrum invariant harness pins RGB-mode byte-identity. **Documented approximation:** emitter/env chroma is reconstructed as a D65-relative tristimulus SPD (not authored spectra); materials lacking packed coeffs fall back to RGB-luminance. | Per-material Jakob-Hanika lane for pt-webgl2; dispersive-scene A/B vs RGB (V28-B). | done (impl) |
| **A4** | ✅ **DONE (A4-progressive: 2026-06-10) — true Hachisuka progressive SPPM** | `pt-webgpu/src/wgsl/pathTrace/sppmBindings.wgsl.ts`, `caustic.wgsl.ts`, `kernel.wgsl.ts`, `gpuResources.ts`, `index.ts` | **Implemented (A4-progressive):** per-pixel `SppmPixelStats` buffer (tau.rgb, radius2, N, _pad×3 = 32 bytes/pixel, group-3 binding 9, full-tier only, 64-byte placeholder when off). Hachisuka & Jensen 2009 §4 / Knaus-Zwicker update rule runs in `sppmGatherProgressive`: N′=N+α·M; ratio=N′/(N+M) [guarded M=0]; R′²=R²·ratio; τ′=(τ+Φ_M)·ratio; L=τ′/(Ne·π·R′²) where Ne=frameAccumulated×photonCount. Buffer GPU-cleared on `reset()`/`setScene()`/camera move (static-eye-point invariant). Accumulator interaction is a Cesàro mean (L_caustic(k)→L_true ⟹ running mean converges). TS-mirror recurrence tests: N(k)=k·α·M closed form, R² monotone-shrink, M=0 stability, first-frame seeding (36 tests). Supersedes the R7a streaming-window form; off-path byte-identical. | Radiometric A/B vs forward-traced oracle (V28-B) — caustic photon-map convergence test. | done |
| **A5** | ✅ **DONE (v1-closure Wave 3, 1d31f0b) — BDPT estimator coherence both backends** | `pt-webgl2/src/glsl/composeTraceGlsl.ts:275-383`, `pt-webgpu/src/wgsl/bdpt/bdptLightSubpath.wgsl.ts:351-431` | **Implemented:** (A) pt-webgl2 light-subpath RNG re-seeds row-independently (`vec2(gl_FragCoord.x, 0)`) so the three rows of one vertex trace the SAME path (were three independent paths = garbage vertices); confined to FEATURE_BDPT subpath branch — off-path byte-identical. (B) pt-webgpu BDPT estimator coherence: light-subpath extension now samples ONE real-BSDF direction at the previous vertex used for BOTH the trace and the stored throughput/pdfFwd (was a sampled-then-discarded direction); pdfRev patched per PBRT §16.3 reciprocal convention. Goldens re-pinned. | A/B vs forward-traced Cornell (V28-B). | done (impl) |
| **A6** | ✅ **NRC: from opt-in/biased to a validated consumable** | `walkaround-hybrid/src/neural/nrc/*` | Online training (forward+backprop+Adam) is genuinely live on GPU and inference is consumed. H26/H27 structural defects are closed: camera-pdf footprint, spread seeding, post-loop `r.Lo` training target, zero-radiance-vs-empty-slot semantics, and atomic slot claims are all wired and tested. Still OFF by default and acknowledged biased. | Converge/quality A/B vs reference; decide default-on tier; keep the documented bias bound honest. | L |
| **A7** | ✅ **DONE (v1-closure Wave 5, caab499) — RC finished (user decision: keep+finish)** | `walkaround-rc/src/`, `shade.wgsl.ts`, `HybridEngineRC.ts` | **Implemented:** RC receiver replaced with correct MC irradiance estimator `E=(4π/N)·ΣL·cos` (was `Le/Wsum·N·0.5` — ray-count-dependent; N=16/N=64 now agree ≈π in tests); real env map bound into the last cascade (was permanently 1×1 black); point/spot lights added to the RC light model (binding 15, DDGI conventions, fingerprint-gated upload); chromatic sun from scene's directional emitter (was achromatic); scene-scale shadow bias. | Real-GPU cascade A/B at N=16/N=64 (V28-B). | done (impl) |
| **A8** | ✅ **DECIDED (2026-06-10) — biased default retained for realtime; unbiased GRIS documented as first-class opt-in** | `HybridEngineOptions.restirPtReuse`, `temporalGi.wgsl.ts`, `spatialGi.wgsl.ts`, `jacobianShift.wgsl.ts`, `restirPHat.wgsl.ts`, `README.md §"Bias & the unbiased GRIS variant"` | **Architecture decision:** The default (`restirPtReuse: false`) retains the pre-GRIS Sprint-17 clamped-Jacobian reuse for the realtime frame budget (the unbiased path adds one visibility ray + full-GBH O(K²) MIS cross-evaluation per accepted neighbour — the dominant cost in the GI reuse passes). **Four bias sources quantified and documented** in the option JSDoc + README table: B1 Jacobian clamp [0.1,10] (`jacobianShift.wgsl.ts`), B2 no reconnection-visibility ray (OFF variants of `spatialGi`/`temporalGi`), B3 no full GBH MIS (OFF combine weights), B4 centroid p̂ in `restirPHat.wgsl.ts` (shared ON/OFF — not fixed by GRIS). The unbiased GRIS path (`restirPtReuse: true`) is first-class, compile-time gated, fully functional (Phase-1 shift + Phase-2 full-GBH spatial, pairwise-MIS temporal), and the JSDoc specifies exactly when to enable it. A compile-time variant-selection pin test added (`__tests__/grisVariantPin.test.ts`). | GPU A/B converged-unbiasedness validation (V19 in `HARDWARE-VALIDATION-NEEDS.md`) — confirms the ON path converges to an unbiased mean and the OFF path's bias matches the documented characterization. | done (decision) |
| **A9** | ✅ **MOSTLY DONE (Wave A + v1-closure Wave 3) — serial build retained; estimator coherence fixed** | `wgsl/bdpt/bdptLightSubpath.wgsl.ts:351-431`, `index.ts:311` | **Implemented:** light subpath carries a REAL glossy/specular BSDF (VNDF) at each vertex, bounce cap 3→8, vertex rows 3→4. **Wave 3 fix (1d31f0b):** estimator coherence — extension now samples ONE real-BSDF direction used for BOTH the trace and stored throughput/pdfFwd (was a sampled-then-discarded direction — biased MIS densities); pdfRev patched per PBRT §16.3. **Honest remaining:** serial dispatch (one workgroup; the per-column variant was a spec-undefined cross-workgroup race — documented). Still OFF-default. | Parallelize the subpath build; default-on `full` tier after the A/B. | mostly done |
| **A10** | ✅ **Pipeline E2E (Wave A) — weights not shipped** | `tools/neural-denoiser-training/{train.py,capture-dataset.mjs,export_weights.py}` | **Implemented:** capture→train→export→load CLOSES — `train.py --dry-run/--smoke` exports a valid 535,107-param `.vitrum-model` binary (CANONICAL_PARAM_COUNT pinned to the engine loader; the "vi-neural-weights.json" reference was stale — binary is the real format), round-trip test green; `capture-dataset.mjs` CPU smoke. **Remaining:** no real trained checkpoint ships (real dataset + torch training = hardware/provisioning tail); `neural` denoiser stays experimental in capabilities until weights exist. | Vendor a trained checkpoint; quality A/B. | pipeline done / weights remaining |

---

## Bucket B — Fidelity ceilings in the default path

Deliberate approximations a discerning user hits immediately. Closing these is what makes
the *default* render "hero-fidelity" rather than "good real-time."

| ID | Item | Where | State now | Done = | Effort |
|----|------|-------|-----------|--------|--------|
| **B1** | ✅ **DONE (Wave A + R8-B tail + B1-ior-per-tri 2026-06-10)** | `shaders/risGi.wgsl.ts`, `materialDecode.wgsl.ts`, `shade.wgsl.ts`, `restir/packingHelpers.ts` | **Wave A:** per-tri `bvh_material` r32uint texture; metals get DI + analytic NEE + specular indirect. **R8-B tail (2026-06-10):** glass primaries get a refracted GI reservoir via 1-interface Snell walk. **B1-ior-per-tri (2026-06-10):** `bvh_material` bits[15:8] now carry IOR quantized [1.0, 3.0] → step ≈ 0.0078; risGi glass walk uses `decodeIor()` per-tri (no more fixed 1.5 constant); shade `lo_transmittedGI` derives Schlick F0 from per-tri IOR via `((ior−1)/(ior+1))²`; rough-glass GI: for roughness > 0.1 the Snell refracted direction is perturbed by a GGX-distributed offset (one sample), giving frosted glass blurred GI. Default IOR=1.5 glass: byte 64, decodes to 1.502, F0=0.04004 (error < 0.003). Mutation path `packBVHRoughMetalFromCore` / `repackBVHMaterialRange` updated. Structural pins + IOR round-trip tests in `roughMetalPacking.test.ts`, `b1GlossyMetalGi.test.ts`. | NRC xsRough wiring. | done |
| **B2** | ✅ **DONE (R8-B, 2026-06-10) — DDGI glossy-aware probe bounce; specular complement** | `ddgi/wgsl/probeUpdateRays.wgsl.ts` | **Implemented:** specular complement via reflected previous-frame SH atlas lookup. `specularWeight = metalness·(1−roughness²)` blends Lambertian indirect toward specular indirect (reflected-direction atlas sample, metal baseColor tint). Blend-not-add — energy-conserving lerp; no double-counting. Gated on `indirectFeedback != 0u` (direct-only probes stay Lambertian). MaterialEntry already carries `roughness` (slot 3) and `metalness` (slot 7) — no new buffer threading needed. Approximation documented in-code: atlas stores cosine-weighted irradiance, not GGX-filtered radiance — honest one-bounce specular complement. **Cite:** Karis (2013) UE4 §4.4; McGuire et al. (2017) probe specular. | V28-B A/B on metallic-sphere scene (R8-C recapture). | done (impl) |
| **B3** | ✅ **DONE (Wave B + v1-closure Wave 4/5, caab499) — env pillar COMPLETE; hdri → native** | `walkaround-hybrid/src/shaders/ris.wgsl.ts:354-376`, `shaders/risGiNrc.wgsl.ts:300-388`, `ddgi/wgsl/probeUpdateRays.wgsl.ts`, `WalkaroundGPUPipeline.ts:1425` | **Implemented:** (Wave B) equirect CDFs built at scene-load (bindings 15-19), directional samples + scalar-tint fallback. (Wave 4/5) `envImportanceSample` is now a live DI NEE candidate in the RIS loop (M_ENV=1 sentinel, measure-consistent source pdf, phat_xi spatial reuse); `risGiNrc` GI-escape reads `envRadiance` (NRC no longer downgrades IBL); DDGI probe misses sample the real HDRI (group-2 bindings 6/7, rotationY identical convention, procedural fallback intact); `updateEnvironment` rebuilds directional CDFs at runtime. Walkaround `hdri` ledger grade promoted to `'native'` — code-verified `promiseLedger.ts:254`. **Remaining:** RC env binding gated pending RC A/B; `updateEnvironment` full fast-path for env-swap still goes through setScene for equirect rebuild. | updateEnvironment fast-path; V28-B A/B. | done (impl) |
| **B4** | ✅ **DONE (Wave A) — pt-webgl2 mesh-area NEE** | `scene/meshAreaLights.ts`, `glsl/composeTraceGlsl.ts:197-205,896-913`, `scene/foldEmissiveEmitters.ts` | **Implemented:** a dedicated `uMeshLights` triangle-light texture (6 texels/tri) is NEE-sampled with area-proportional selection → triangle-independent pdf → forward-hit MIS from one global `uTotalEmissiveArea`. The emissive-fold is kept as the BSDF strategy (exactly-one-MIS-estimate algebra documented). The analytic `lightsTexture` still excludes `mesh-area` by design — NEE now comes from the separate mesh-light texture. | Variance A/B on Cornell (V28-B). | done (impl) |
| **B5** | ✅ **DONE (Wave A) — Beer-Lambert DDGI probes** | `ddgi/wgsl/probeUpdateRays.wgsl.ts:276-290` | **Implemented:** real `transmission · exp(−attenuationColor · t/attenuationDistance)` over path length, with `t` thickness-clamped (`clamp(distToExit,0,thickness)`); reduces to Beer-Lambert exactly. | — | done |
| **B6** | ✅ **DONE (Wave B) — GTAO per-pixel view axis** | `shaders/gtao.wgsl.ts:120-188` | **Implemented:** per-pixel view axis reconstructed from the inverse perspective projection (was the constant `(0,0,-1)` central-pixel approximation); correct at wide FOV / frame edges. | — | done |
| **B7** | ✅ **DONE (Wave B) — planar-SAH half-area fix** | `shared-bvh/src/buildArrayBvh.ts:127-147` | **Implemented:** `surfaceArea` now returns a nonzero half-perimeter term for planar boxes (one extent 0) so flat geometry ranks splits — a 2000-tri coplanar floor builds depth 45→9. **Remaining (out of scope):** no SBVH; recursive builder retained. | Optional SBVH; iterative build. | done (planar) / SBVH remaining |
| **B8** | ✅ **DONE (Wave B) — light-tree orientation cones** | `shared-samplers/src/lightTree.ts:48-74,387` | **Implemented:** Conty-Estévez orientation cone (axis + thetaO + thetaE) per node, stride 12→16; spot/area producers wired; full-sphere sentinel keeps the cone term ≡1 (byte-identical when unoriented). | A/B on directional-emitter scenes (V28-B). | done (impl) |
| **B9** | ✅ **DONE (Wave B) — GGX multiscatter (all 3 backends)** | pt-webgpu `material.wgsl.ts`; pt-webgl2 `glsl/render/get_surface_record_function.glsl.js`; walkaround `ggxBrdf.wgsl.ts` | **Implemented:** Kulla-Conty multiscatter energy compensation in all three GGX evals (LUT + furnace test on pt-webgpu; furnace-pinned on pt-webgl2, lite-mode skipped). | — | done |
| **B10** | ✅ **DONE (Wave B) — physical refraction transmittance** | `wgsl/pathTrace/bsdf.wgsl.ts` | **Implemented:** physical Fresnel-consistent transmittance replaces the phenomenological `mix(vec3(1),baseColor,0.15)` tint. | — | done |
| **B11** | ✅ **pt-webgl2 disc-area NATIVE; pt-webgpu = 32-triangle fan (approximate)** | `pt-webgl2/src/scene/lightsTexture.ts`, `pt-webgpu/src/scene/emitterPacking.ts:128-197` | **pt-webgl2** packs `disc-area` emitters as `CIRC_AREA_LIGHT = 1` with concentric-disc sampling and `intersectsCircle` — geometrically exact. **pt-webgpu** lowers `disc-area` to an area-compensated 32-triangle fan (`discAreaPackedAsTriangles`; fan radius scaled so total area = π·r²) — geometrically approximate; ledger `disc-area` grade for pt-webgpu corrected to `'approximate'` (`promiseLedger.ts`). `procedural-sky` is `'unsupported'` on both PT backends (code-verified). | pt-webgpu native disc-area (32-fan → analytic disc NEE). | M (pt-webgpu only) |
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
- **B1 is DONE (Wave A):** the pre-Wave-A claim that metals were excluded from direct
  light is stale. Current `shade.wgsl.ts:252` (`lo_analyticNEE`) gates only on `isGlass`;
  `shade.wgsl.ts:333` (`lo_direct`) gates only on `isGlass` — metals receive both analytic
  NEE and ReSTIR-DI direct light. `lo_indirectSpecular` (line 580) reflects the GI
  reservoir via the GGX specular lobe for metals/glossy. DDGI's diffuse `lo_indirect`
  (line 429) still exits early for `isGlass || isMetal`, but that is intentional — metals
  consume `lo_indirectSpecular` instead (the DDGI cache is Lambertian-targeted; applying it
  directly to a mirror-like metal would be physically wrong). Glass is the only material
  that remains unlit by direct+indirect (refracted GI is out of scope — tracked in B11
  and road-to-100).
- **A2/H25 (PPG) is now reflected in the closed P0 table:** sTree splitting/runaway
  and dTree interior-flux propagation are implemented; `PPGCoordinator` propagates
  subtree sums bottom-up before refinement, `ppgPdf.wgsl.ts` samples child flux
  proportionally, and `dTreeInteriorFlux.test.ts` pins CPU/GPU sampling/pdf parity.
- **A6/H26-H27 (NRC) no longer carries the filed structural defects:** the GI-RIS
  NRC variant seeds spread accumulation from `0.0`, derives the primary footprint
  from `nrcCfg.cameraPixelPdf`, tracks the first fired candidate, writes one
  post-loop training record with `r.Lo`, and uses atomic slot claims. The remaining
  A6 work is consumability/quality validation and default-tier policy, not those
  stale spread/target/torn-record bugs.
- **A10/H28 (neural denoiser) is no longer blocked on ReLU bind aliasing:** in-place
  ReLU layers allocate distinct output buffers and remap downstream tensor reads;
  `reluPingPong.test.ts` pins the no read/read_write alias invariant. The remaining
  A10 gap is trained weights plus quality A/B.
- **NEW B13 — walkaround texture sampling is broken at the seam** ✅: UVs zeroed at both
  `restir/bvhCore.ts` build sites (items H15). The G3 texture work (63a6dab) wired UVs
  through shared-bvh, but walkaround discards them. S-effort, render-changing.
- **NEW B14 — ✅ DONE (v1-closure Wave 1/2, 0dbaff5) — DDGI emitter NEE complete**: rect/disc-area fixture point-proxy REMOVED (was double-counted against H18 NEE triangles — `coreEmittersToDDGILights.ts:155` map deleted; code-verified 0 hits for fixture-rect pattern); mesh-area emitter triangles now expand into the probe NEE list (`HybridEngineLifecycle.ts:545`, `collectMeshAreaEmitterTrisFromCore` + `setEmitterTris`); emissive-mesh scenes get nonzero DDGI indirect. Code-verified both fix sites.
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
  casualties — see B1/B13), and a dozen fields have zero consumers in ANY backend. **R7b
  update (ba1429d):** `anisotropy`/`anisotropyRotation` now consumed by pt-webgpu (Heitz
  anisotropic GGX). Remaining zero-consumer fields: `envMapIntensity` (pt-webgpu unified
  R7b; walkaround still scalar-tint only), `aoMap`, `bumpMap`, `displacementMap`, `lightMap`
  (pt-webgpu camera-visible only R7b), `angularDiameter`, `castShadow`/`receiveShadow`,
  `tangents`. `TextureRef.texCoord` on pt-webgl2 still zero consumption (documented unkept
  promise, R7c). `denoiser` is a silent no-op on both PT backends for any value but
  `'oidn-final'` (H48).

## Open decisions (need a call before building)

- **A4 DECIDED (2026-06-10):** build real SPPM — DONE (v1-closure Wave 4, 06910e2).
- **A7 DECIDED (2026-06-10):** keep RC + finish — DONE (v1-closure Wave 5, caab499). See B3 update.
- **A6/A10:** ship NRC/neural-denoiser on-by-default (needs trained assets) or keep opt-in+experimental?
- **A8:** make GRIS the default (or quantify+document the bias of the biased default)?
- **sun-NEE default: DECIDED + DONE (2026-06-10, item 4 R8-A).** `lo_sunNEE` wired in `shade.wgsl.ts` — deterministic shadow ray + evalGGX BRDF, default-ON for opaque surfaces, no flag required. `lo_sg_caustic` (stainedGlass flag) unchanged — tinted-glass transmittance path. No double-count vs DDGI indirect: DDGI stores sun→wall radiance at the PROBE bounce surface; lo_sunNEE is sun→receiver DIRECT, disjoint paths. Behavioral gate: `wh/directional-sun` (LUM_THRESHOLD 0.005, intensity 3.0). Render-changing for directional-lit scenes → V28-B recapture in R8-C.
- **B1:** glossy GI via ReSTIR-GI full-BRDF p̂, or specular probes, or keep punting glossy to the PT backend?
- **B1 tail + B1-ior-per-tri: DONE (2026-06-10).** R8-B tail: 1-interface refraction walk in `risGi.wgsl.ts`; shade `lo_transmittedGI` weights by Fresnel-T × Beer tint; `wh/glass-gi` behavioral gate PASS. **B1-ior-per-tri (2026-06-10 follow-up, DONE):** `bvh_material` bits[15:8] carry IOR quantized over [1.0, 3.0] (step ≈ 0.008); `materialDecode.wgsl` exposes `decodeIor()`; risGi glass walk decodes per-tri IOR via `decodeIor(glassPrimaryPacked)` (no more hardcoded 1.5); shade `lo_transmittedGI` computes Schlick F0 from per-tri IOR (`((ior−1)/(ior+1))²`); rough-glass GI: roughness > 0.1 perturbs the Snell refracted direction by a GGX-distributed offset (one sample, per-tri roughness) so frosted glass receives blurred GI; smooth glass stays exact-Snell-byte-identical. Default IOR=1.5 → byte 64 → decodes to 1.502 (error < 0.003, within glass dispersion); `packBVHRoughMetalFromCore` / structural packer / `repackBVHMaterialRange` all updated. 20 new tests in `roughMetalPacking.test.ts` + `b1GlossyMetalGi.test.ts`. Shader-gate 51/51, behavioral-gate 28/28, vitest 1490/1490. Render-changing for non-1.5-IOR glass → V28-B recapture in R8-C.

---

## Addendum — 2026-06-11 condensed implementation spec (glTF + walkaround + arbitrary glTF)

> Authored 2026-06-11 from a code-truth audit of `@vitrum/gltf-adapter` (working tree),
> `promiseLedger.ts`, engine entry points, walkaround consumption, and pt-backend material
> gaps. **No timelines or effort estimates.** Supersedes the conversational plan from the
> same session for execution ordering; does not retract the A–D bucket items above — this
> addendum is the **closure checklist** for the three user-facing 100% targets.
>
> **Baseline:** `main` @ `309fdebe` (oracle math fixes). ~2,148 LOC uncommitted across
> gltf-adapter, engine bridge, unlit all backends, H30 canvas sizing, progressive handoff
> fallback, ledger fixes. **Land Phase 0 first** before treating any claim below as shipped.
>
> **Companion docs:** `plan/road-to-100-gap-ledger-2026-06-11.md`, `items_to_fix.md` §H.
>
> **REVIEW RECONCILIATION (2026-06-12, lead-verified against `main@309fdebe`):**
> 1. **Several P0 rows below were already landed and pushed before this addendum
>    was written** — they are now marked `✅ LANDED` in place. Do NOT re-execute
>    them; verify-on-read if in doubt. Affected: Phase 2A (PTWG-01..05, WEBGL2-01,
>    H49), Phase 3A (W-HYB-01/02/03), Phase 1C (GLTF-01, CORE-01).
> 2. **H49 was wrong, not just stale**: pt-webgl2 ALREADY packs
>    `specularColor`/`specularIntensity` (+ both maps) at `materialsTexture.ts:168-232`;
>    the ledger correctly grades them native. Row struck.
> 3. **H33's fix target was the wrong file**: `worldSpaceMerge.ts` `materialSig`
>    already includes attenuation fields. The real drift is
>    `sceneBvh.ts` `materialSetHashFloats` (~:173-191), which carries
>    `attenuationColor` only and omits `attenuationDistance`/`thickness` —
>    mutating only those fields skips the DDGI rebuild. Row corrected in 3G.
> 4. **V28-B baseline recapture was dropped from this addendum** — restored as
>    Phase 0.3. `309fdebe` is heavily render-changing (env-DI fix alone should
>    visibly brighten HDRI scenes; ALL pre-309fdebe baselines embed the ~66× bias).
> 5. **DECISION LOCKED (user, 2026-06-12): walkaround gets the FULL texture atlas**
>    — Phases 3C/3D/3E as written are committed scope, not optional. The apparent
>    conflict with the acceptance table (ledger-truth is already satisfied by
>    `unsupported` grades post-CAP-01) is resolved in favor of implementation:
>    walkaround material rows are expected to PROMOTE to native/approximate via
>    the atlas, not rest at honest-unsupported.
> 6. **COMPLETENESS PASS (2026-06-12):** three verified holes INSIDE the plan's own
>    acceptance criteria were added (texture wrap modes + mipmaps in Phase 1A;
>    `environment:'none'` phantom skylight + grayscale-directional shortcut class
>    widened into 2C), and **Phase 6** now carries the gap-ledger residue outside
>    the three targets (onError unification, lite single-BLAS, RC footguns,
>    sampled fingerprint, core contract additions; the pt-webgl2 NEE,
>    pt-webgpu trace-lite shader-gate mismatch, attachVitrum recreate scene-loss,
>    and morph-normal skip rows are closed by Wave 1). With Phases 0–6
>    executed, this plan IS the categorical close.
> 7. **FOLLOW-UP CROSSOUT PASS (2026-06-12):** verified and struck/narrowed the
>    rows that landed after this addendum: glTF required `KHR_materials_unlit` /
>    `KHR_materials_pbrSpecularGlossiness`, scalar spec-gloss compatibility
>    scoring, morph-tangent/point-line unsupported policy tests, bridge hook
>    pass-through, walkaround + PT unlit branches (`shadingModel` remains
>    intentionally `approximate`), and the pt-webgpu local BRDF-helper propagation
>    wave. Remaining payload/sampler/schema gaps stay open in place.

### What 100% means (acceptance, not aspiration)

| Target | Done when |
|--------|-----------|
| **glTF** | `loadGltfAsset` → `loadGltfForEngine` handles URL/GLB/JSON+external resources; `analyzeGltfAsset` + `rankGltfBackends` are complete; `GltfSceneController` drives skin/morph/TRS; every extension in `REQUIRED_EXTENSION_SUPPORT` (`featureReport.ts:124-145`) has import + compatibility + test; zero silent `console.warn` in adapter (warnings in return value only). |
| **Walkaround** | Every `MaterialSpec` key graded `native`/`approximate` in `WALKAROUND_MATERIALS` is consumed in GPU shaders; `CONSUMED_MATERIAL_FIELDS` (`consumedMaterialFields.ts`) matches ledger exactly; emitter/environment/shadow grades match runtime; P0 walkaround bugs (W-HYB-01..03, H25-H29) closed. |
| **Arbitrary glTF** | For any asset in Khronos sample set + internal hero fixtures: `loadGltfAsset` succeeds or throws structurally; `evaluateGltfBackendCompatibility(selectedBackend).unsupportedCount === 0` for used features OR `compatibilityMode` rejects before render; rendered output passes material-furnace + reference gate on **recommended** backend; `prefer:'auto'` uses feature report, not triangle count alone. |

**Explicit non-goals** (otherwise "100%" is undefined): point/line primitives, displacement tessellation, production neural weights, true Hachisuka SPPM (A4-progressive shipped — separate from this campaign), cross-host GPU certification.

### Why plans stall at ~85% (gaps this spec closes)

| Trap | Symptom | This spec addresses |
|------|---------|---------------------|
| glTF API without **texture decode bridge** | Scene loads, everything flat gray | Phase 1 § Texture pipeline |
| Compatibility planner without **engine wiring** | Correct backend known, wrong one created | Phase 4 § `createEngine` + `VitrumCanvas` |
| Walkaround **scalar-only** + ledger drift | Textured glTF "works" with warnings | Phase 3 § Texture atlas (non-optional for WA 100%) |
| **PTWG-MAT-01** only on megakernel | glTF clearcoat wrong in BDPT/SPPM/ReSTIR | Phase 2 § Integrator audit matrix |
| **Lite tier** used for arbitrary glTF | Missing maps/TLAS silently | Phase 4 § Lite policy + Phase 2 |
| Animation **without** skin+morph+material sync | Controller patches positions, normals wrong | Phase 1 § Controller + Phase 4 |
| **Tangent** bugs on pt-webgl2 | Normal maps wrong on glTF | Phase 2 § WEBGL2-01 |
| Oracle fixes **without** reference renders | Tests green, images wrong | Phase 5 |
| Uncommitted work | "Plan done", `main` unchanged | Phase 0 |
| `pickBackend` triangle budget (`createEngineScale.ts:46-47`) | Textured asset → walkaround | Phase 4 |
| PPG/NRC/neural **enabled but broken** | Optional features crash bind groups | Phase 3 § Subsystems |
| **items_to_fix §H** residue | Silent wrong behavior | Woven into phases below |

---

### Phase 0 — Land baseline + mechanical gates

#### 0.1 Commit working tree

**Files:** 55 modified + 7 new under `packages/gltf-adapter/`, `packages/engine/`, `packages/core/`, `packages/walkaround-hybrid/`, `packages/pt-webgpu/`.

**Footgun:** ~~`textures.ts` header still says "adapter does not fetch external image URIs" while `assetLoader.ts` does~~ ✅ fixed — the file header now describes the pluggable image-byte handoff. The low-level `getImageBytes` warning remains intentional for callers that bypass `loadGltfAsset`.

#### 0.2 Gate infrastructure (blocks false 100%)

| Gate | Plug-in | Footgun |
|------|---------|---------|
| **GATE-01** | Extend `ledgerVsCapabilities.test.ts` — runtime `buildCapabilities()` must match `BACKEND_PROMISE_LEDGER` for pt-webgl2 lite/full aux buffers | Ledger said `supportsAuxBuffers:false` while full tier had MRT — already bit you (H39) |
| **GATE-02** | Per `native` material row: one test that packs + shader string pin OR readback oracle | Byte-identity SHA tests can be green while both sides share a bug |
| **GATE-06** | `npm run shader-gate` in CI for every `PASS_ORDER` variant including walkaround texture bind layout | WGSL string tests don't compile shaders |
| **GATE-GLTF** | `gltfKhronosSweep.test.ts` — `analyzeGltfAsset` only, no network in CI (fixtures vendored) | Live URL tests flake in CI |

#### 0.3 V28-B baseline recapture (restored 2026-06-12 — was dropped from this addendum)

`309fdebe` (oracle math fixes) is render-changing on: walkaround DI (env ~66×
brighter where the bias applied, selected-xi −30% correction), DDGI visibility
in open scenes, pt-webgpu BDPT connections (~5×), lite rect NEE. **Every
pre-309fdebe reference render and benchmark baseline embeds the old biases**
(same class as the within-leaf-hit precedent: improvement confirmations, not
regression suspects). Recapture via `~/projects/wsl-gpu` before any later
render-changing wave lands, or A/B attribution becomes impossible.

---

### Phase 1 — glTF pipeline 100%

#### 1A — Asset loading (`assetLoader.ts`)

**Already in working tree:** `loadGltfAsset`, external buffer/image fetch, `extensionsRequired` enforcement, `imageBytes` map.

**Still required:**

| Task | Code | Plug-in | Footgun |
|------|------|---------|---------|
| Typed errors | `packages/gltf-adapter/src/errors.ts` | Throw `GltfFetchFailed`, `GltfResourceNotFound` with `{ url, kind }` | Generic `Error` breaks `compatibilityMode` UX |
| Cache hooks | `LoadGltfAssetOptions.cache` | Wrap `fetchArrayBuffer` in `assetLoader.ts` | Cache key must include `baseUri` + URL |
| `loadGltfAndDecodeTextures()` helper | New `texturePipeline.ts` | After `gltfToScene`, walk all `TextureRef`s, call host `decodeImage` → replace handles with backend-ready pixels | **#1 arbitrary-glTF blocker:** Scene has maps but handles are `RawImageHandle` — pt-webgl2 `texturesArray.ts:4-10` needs `{width,height,data}` float RGBA linear |
| sRGB → linear | In decode helper | glTF baseColor textures are sRGB (`KHR_materials_unlit` too) | Double-linear if backend also decodes sRGB |
| NPOT / max dim | In atlas builders (PT + WA) | Nearest resize to `max(dim)` like `texturesArray.ts:12-13` | `sampler2DArray` requires uniform layer size |
| Basis/WebP/DDS | `compression.ts` pattern: host hook + `requires-hook` in report | `featureReport.ts` `EXTENSIONS_REQUIRING_HOST_HOOK` | Failing to pass hook must throw in `strict` mode, not warn-skip |
| **Texture wrap modes (ADDED 2026-06-12 — verified hole)** | `TextureRef.wrapS/wrapT` in core (`repeat`/`clamp`/`mirror`) + adapter reads `gltf.samplers` (parsed today, consumed NOWHERE) + backend honor: pt-webgl2 atlas is hard `CLAMP_TO_EDGE` (`texturesArray.ts:236-237`) → shader-side `fract()` per wrap mode or per-layer sampler strategy; same audit for pt-webgpu samplers | **Tiled/repeating textures are everywhere in real glTF — this alone fails the Khronos-sweep acceptance row.** KHR_texture_transform scale>1 currently edge-smears |
| **Mipmaps (ADDED 2026-06-12 — verified hole)** | pt-webgpu `materialTextureArray.ts` creates `mipLevelCount:1` while sampling with `mipmapFilter:'linear'` — generate mip chain at upload (compute blit or CPU) | Minification aliasing on every real textured asset; also fix the RGBA8-only raw-data `bytesPerRow` assumption in the same file |

#### 1B — Feature report & planner (`featureReport.ts`)

**Already:** `analyzeGltfAsset`, `evaluateGltfBackendCompatibility`, `rankGltfBackends`, per-field ledger crosswalk.

**Still required:**

| Task | Code | Footgun |
|------|------|---------|
| ~~Source paths on every issue~~ ✅ DONE | `featureReport.ts`, `gltfAssetApi.test.ts` | `GltfCompatibilityIssue.path` is now required, analyzer source-path maps cover extensions/primitives/materials/scene cameras, and the test asserts every issue has a non-empty path including `materials[0].normalTexture`-style material paths. |
| ~~Scalar `KHR_materials_pbrSpecularGlossiness` scoring~~ ✅ DONE | `specularGlossinessMaterialCount` + compatibility issue `KHR_materials_pbrSpecularGlossiness` | Texture-alpha glossiness remains the next row |
| Spec-gloss glossiness-alpha | `materials.ts` + issue row `glossinessAlpha: approximate` | RGB imported as `specularColorMap` but roughness not baked from alpha — test in `gltfExtensionPolicy.test.ts` documents gap |
| ~~Morph `TANGENT` unsupported policy~~ ✅ DONE | `hasMorphTargetTangents` + unsupported compatibility issue + warn-skip import path | Core `morphTargetTangents` can remain a future contract expansion, not a blocker |
| ~~Cameras~~ ✅ DONE | `featureReport.ts`, `gltfAssetApi.test.ts` | `sceneGraph.cameraPaths` records `cameras[n]`; compatibility emits `scene:cameras=unsupported` so strict/reject modes can block assets that expect imported cameras. |
| ~~Double-sided~~ ✅ DONE | `featureReport.ts`, `gltfAssetApi.test.ts` | Compatibility emits `material:doubleSided=approximate` at `materials[n].doubleSided`; raw data remains preserved in material extensions, but planner now surfaces the lack of first-class double-sided/backface-normal semantics. |

#### 1C — Import (`gltfToScene.ts`, `materials.ts`, `accessors.ts`)

**Closed:** strip/fan triangulation, morph POSITION/NORMAL, animations, skins, punctual lights, KHR material extensions, `resolveTextureRef` UV/transform.

**Still required:**

| Task | Code | Footgun |
|------|------|---------|
| ~~**GLTF-01** bind matrices~~ ✅ LANDED | `gltfToScene.ts:402-419` emits `bindMatrix`/`bindMatrixInverse` (warn fallback when uncomputable) | — |
| ~~**CORE-01** CUBICSPLINE quats~~ ✅ LANDED | `sampleAnimationClip` normalizes LINEAR/STEP/clamped/CUBICSPLINE rotations | — |
| Generate tangents when missing | After unpack POSITION/NORMAL/uvs, call `generateTangents()` (new in `gltf-adapter` or reuse logic from pt-webgl2) | glTF often omits TANGENT; normal maps break without (WEBGL2-01) |
| `COLOR_0` vertex colors | Already unpacked — add to `GltfFeatureReport` + multiply in backends or `baseColor` bake at import | glTF vertex color × baseColor |
| Sparse accessors | More fixtures in `accessors.ts` tests | Production glTF uses sparse heavily |
| ~~Point/line modes~~ ✅ DONE | Product decision: keep unsupported; `gltfPointLinePrimitivePolicy.test.ts` pins structured compatibility issues + warn-skip import | Don't "half support" — either add `ScenePrimitive` kind later or keep rejecting |

#### 1D — Runtime controller (`sceneController.ts`)

**Already:** `seek`, `advance`, skin bones via `solveSkin`, morph weights, `updatePrimitive` with `setScene` fallback.

**Still required:**

| Task | Code | Plug-in | Footgun |
|------|------|---------|---------|
| Multi-clip blend | `GltfSceneController.blend(clips, weights)` | Sample each clip, accumulate TRS/morph | Order matters: morph before `solveSkin` (same as static import) |
| `KHR_materials_variants` at runtime | `controller.setVariant(name)` → re-run `convertMaterial` for affected primitives → `materialPatch` or `setScene` | Variant switch must invalidate material fast-path caches on walkaround |
| ~~Engine attach API~~ ✅ DONE | `controller.attachEngine(engine, { setScene })` exists and `loadGltfForEngine` attaches after load | Use `attachScene:false` / `setScene:false` when the host already set the scene |
| Patch routing per backend | Use `patchPrimitiveInScene` then `updatePrimitive` — on throw, `setScene` | `ProgressiveHandoffCoordinator` pattern (`progressiveHandoff.ts`) | Partial patch on one engine desyncs handoff pair |

#### 1E — Engine bridge (`engineBridge.ts`)

**Already:** `loadGltfForEngine`, `compatibilityMode`, factory injection.

**Still required:**

| Task | Code | Footgun |
|------|------|---------|
| `@vitrum/engine/gltf` re-export | `packages/engine/src/gltf.ts` wraps `loadGltfForEngine` + `createEngine` | Keeps adapter independent but one-import DX |
| ~~Pass `decodeImage` + `dracoDecode` + `meshoptDecode` through bridge~~ ✅ DONE | `LoadGltfForEngineOptions` extends `LoadGltfAssetOptions`; `loadGltfForEngine` passes options through `loadGltfAsset` | Bridge without hooks still fails required compressed assets, but now through the intended hook contract |
| Return `textureDecodeReport` | List maps that still have `RawImageHandle` after decode pass | Host knows before first frame |

---

### Phase 2 — PT backends (pt-webgl2 + pt-webgpu full) — material & integrator 100%

Required for **arbitrary glTF** on fidelity backends. Walkaround is Phase 3.

#### 2A — P0 correctness (do before fidelity promotion)

> **2026-06-12 reconciliation: ALL rows in this table are ✅ LANDED on `main`**
> (waves `00047313`..`f1b1dd79` + verified on disk). Kept for audit history only
> — do not re-execute.

| ID | Status | Evidence |
|----|--------|----------|
| PTWG-01 | ✅ LANDED | `state==='error'` blocks at `pt-webgpu/src/index.ts:783` |
| PTWG-02 | ✅ LANDED | old-OR-new implicit-emitter repack in `sceneMutationRouter.ts` |
| PTWG-03–04 | ✅ LANDED | `lightSelectInvPdf` all source kinds (`sppmBindings.wgsl.ts:394+`); flux oracle PASSES (energy conserved ±3%) |
| PTWG-05 | ✅ LANDED | Abbe/spectralMinMu lanes split |
| WEBGL2-01 | ✅ LANDED | authored tangent XYZW + nonzero fallback handedness (`attributesTextureArray.ts:252`) |
| H49 | ✅ STRUCK — was WRONG, not stale | `specularColor`/`specularIntensity` + maps ALREADY packed at `materialsTexture.ts:168-232`; ledger grades native |

#### 2B — Material packing gaps → ledger `native`

**pt-webgl2** (`materialsTexture.ts` + GLSL): only `anisotropy*`, `displacement*`, `thicknessMap` remain unsupported.

| Field | Work |
|-------|------|
| `thicknessMap` | Sample in volume path or stay unsupported with compatibility reject |
| `anisotropy*` | Port pt-webgpu `bsdf.wgsl.ts` aniso or keep unsupported + planner rejects |

**pt-webgpu** (`materialPacking.ts`, `materialTextures.ts`, `material.wgsl.ts`):

| Field | Work | Footgun |
|-------|------|---------|
| `normalScale` | Multiply in `applyNormalMap` | Ledger says approximate for `normalMap` partly because of this |
| `transmissionMap`, `alphaMap` | Add descriptor slots in `materialTextures.ts` `DESCRIPTOR_FIELDS` | Group 3 binding count — check `maxStorageTexturesPerShaderStage` |
| `clearcoat*Map`, `sheen*Map`, `iridescence*Map`, `specular*Map` | Extend descriptor list (pt-webgl2 has 17 maps; pt-webgpu has 6) | **H52:** pt-webgpu packs scalars but not maps for extensions |
| `specularIntensity`, `specularColor` | New vec4 lane in `materialPacking.ts` | glTF KHR_materials_specular |
| Per-map UV transform | Stop sharing `baseColor` transform in `material.wgsl.ts` "v1 simplification" | glTF `KHR_texture_transform` per texture |
| `thickness` scalar | Use or keep `unsupported` with honest report | Volume walk uses real path lengths — scalar thickness ambiguous |
| ~~`shadingModel` branch verification~~ ✅ DONE | Packed/consumed as terminal unlit branch in pt-webgpu; ledger intentionally remains `approximate`, not `native` | Native would require different semantics (for example emissive-light participation), so do not over-promote |

**Plug-in pattern for new map:**
1. Add to `materialTextures.ts` collector.
2. Add binding index in `material.wgsl.ts` `materialTexDescriptors`.
3. Sample in `shadePrologue.wgsl.ts` / `bsdf.wgsl.ts`.
4. Update `PT_WEBGPU_MATERIALS` row.
5. Add `scenePack.materials.test.ts` + wgslContract field pin.
6. Reject from `incrementalPatch.ts` material fast-path if map handles change (already pattern for texture maps).

#### 2C — PTWG-MAT-01 integrator audit (mandatory for extension lobes)

> **SCOPE WIDENED (2026-06-12):** the audit is NOT extension-lobes-only. It must
> also close the **grayscale single-directional shortcut class** — in-medium NEE
> (`kernel.wgsl.ts` `params.lightDir.w` path), MNEE cone-search
> (`caustic.wgsl.ts:846,883`), SPPM directional emitter, BDPT bounce-0, and
> ReSTIR-PT `rptDirectAtVertex` all still light from the mean-gray mirrored
> directional the megakernel outgrew (chromatic loss + missing light kinds in
> those paths). Also in scope: BDPT's hardcoded 50-unit emitter placement radius
> (`bdptLightSubpath.wgsl.ts` `emitPos = -lightDir * 50.0`) — derive from scene
> bounds; and `environment:'none'` must stop returning the lit `sampleSky`
> gradient (`connect.wgsl.ts:59-63`) — a no-environment scene gets free ambient
> light today, a contract violation.

Audit **every** `evaluateBrdf` / `brdfDirectionalPdf` call site — glTF extension lobes must match across paths:

| Path | File | Status |
|------|------|--------|
| Eye path NEE | `kernel.wgsl.ts` / `kernelLite.wgsl.ts` | ✅ direct-light NEE and BSDF connection helper calls now use `evaluateBrdfFull` / `brdfDirectionalPdfFull`; still open for path-sampling PDFs in `kernel.wgsl.ts` tied to `sampleNextBounceDirection` |
| BSDF connections | `connect.wgsl.ts`, `connectLite.wgsl.ts` | ✅ local helper propagation closed (area/env full-tier; env lite; area-lite remains deliberate zero stub) |
| BDPT | `bdptConnection.wgsl.ts`, `bdptLightSubpath.wgsl.ts` | ✅ eye↔light connection uses full helpers; **open:** light-subpath scatter still uses base helpers / sampler payload |
| SPPM / caustics | `caustic.wgsl.ts`, `sppmBindings.wgsl.ts` | ✅ receiver-side SPPM/caustic BRDF/PDF helper propagation closed |
| ReSTIR-PT | `restirPtProducer.wgsl.ts`, `restirPtCompose.wgsl.ts`, `reservoirPtHero.wgsl.ts`, `restirPtResolve.wgsl.ts` | ✅ producer direct/onward paths use full helpers; **open:** reservoir payload/resolve still stores only base lobes and source PDFs track the current sampler |
| Adjoint | `adjointPass.wgsl.ts`, `pathTraceAdjoint.wgsl.ts` | OPEN — derivatives still target the base BRDF parameterization |
| Present | `present.wgsl.ts` tonemap only — no BSDF | N/A |

**Footgun:** Fixing megakernel only used to leave BDPT/SPPM wrong for glTF clearcoat scenes with `bdpt:true`; that local helper class is now narrowed. The remaining class is sampler/payload coherence, not just missed function calls.

#### 2D — pt-webgl2 scope gaps for arbitrary glTF

| Gap | Code | Footgun |
|-----|------|---------|
| Analytic primitives | `PT_WEBGL2_SUPPORT` empty + `partitionSceneBySupport` drops analytics | glTF doesn't use analytics — OK if planner never picks pt-webgl2 for analytics |
| Procedural sky | No env — host must supply `hdri` separately | glTF has no sky; document |
| Procedural sky on PT | Copy pt-webgpu Preetham bake (`environmentPacking.ts`) | |
| Mutations all `fallback-rebuild` | `capabilities.ts:85-92` overrides ledger mutations | Animation via controller causes full repack every frame — **performance footgun**; add fast paths mirroring `sceneMutationRouter.ts` |
| No `setSize` | Host uses `FrameInput.viewport` | Document in `Engine` JSDoc; optional `setSize` on PT |
| Denoiser | No in-engine path | `compatibilityReport` should note OIDN unavailable on pt-webgl2 |
| Caustics | Heuristic not MNEE (`options.ts`) | Don't grade `manifold-nee` as native in docs |

#### 2E — pt-webgpu lite tier policy

**For arbitrary glTF 100%:** lite is **not** a target. Code required:

| Task | File | Behavior |
|------|------|----------|
| `loadGltfForEngine` rejects lite for `reject-degraded` | `engineBridge.ts` | If `recommendedBackend` is pt-webgpu but device selects lite, throw or fall back to pt-webgl2 |
| `rankGltfBackends` lite row | `featureReport.ts` | Score lite separately or mark `pt-webgpu-lite` pseudo-backend |
| PTWG-07 verify | `sceneMutationRouter.ts`, lite texture refresh | Emitter/env mutation must refresh `liteLightTex` / `liteEnvTex` |

**Footgun:** `connectLite.wgsl.ts` `bsdfAreaLightConnectionContribution` returns zero **by design** — lite uses one-sided area NEE (`kernelLite.wgsl.ts` after PTWG-LITE-01 fix). Don't "implement stub" without fixing estimator.

#### 2F — Analytic + instancing (pt-webgpu full)

Already native. **glTF instancing:** glTF uses multiple nodes, not `instanced-mesh` kind — adapter flattens to separate primitives. **Gap:** add `EXT_mesh_gpu_instancing` to Phase 1 extension matrix or explicitly `unsupported` with test.

---

### Phase 3 — Walkaround-hybrid 100%

> **SCOPE DECISION LOCKED (user, 2026-06-12): the FULL texture atlas is committed
> scope.** Phases 3C (alpha), 3D (atlas: all maps + UV/tangent buffers), and 3E
> (extension lobes) execute as written — they are requirements, not options.
> "Ledger truth via `unsupported` grades" is NOT an acceptable terminal state for
> walkaround material maps; rows are expected to promote to native/approximate
> through the atlas. Permanent-unsupported exceptions remain only the 3F list
> (spectral/displacement/thin-film/layers).

#### 3A — P0 subsystem & pipeline correctness

> **2026-06-12 reconciliation: W-HYB-01/02/03 are ✅ LANDED on `main`**
> (verified on disk: NRC clear wired at `WalkaroundGPUPipeline.ts:1227`;
> atrous per-iteration 256-byte UBO strides; init failures → `onError`).
> H25/H28/H29 are closed by code + tests; H26-H27 closed in R8-B
> (keep oracle coverage).

| ID | Status | File(s) | Fix | Footgun |
|----|--------|---------|-----|---------|
| W-HYB-01 | ✅ LANDED | — | `clearSlotClaims()` wired before NRC GI-RIS | — |
| W-HYB-02 | ✅ LANDED | — | Per-iteration UBO bindings (256-byte strides) | — |
| W-HYB-03 | ✅ LANDED | — | Init failures route to `onError` | — |
| H25 | ✅ CLOSED | `PPGCoordinator.ts`, `ppgPdf.wgsl.ts`, `dTree.ts`, `dTreeInteriorFlux.test.ts` | Bottom-up interior flux propagation is implemented before dTree refinement; CPU/GPU pdf logic now matches leaf flux / solid angle. | Residual promotion risk: no broad real-GPU PPG A/B in package tests. |
| H26-H27 | ✅ CLOSED (R8-B) | `risGiNrc.wgsl.ts` | Spread + training target fixed — keep oracle tests | |
| H28 | ✅ CLOSED | `layerResourceAllocator.ts`, `reluPingPong.test.ts` | In-place ReLU layers now allocate a distinct output buffer and remap downstream tensor reads. | Broader neural weights/quality remain separate from H28. |
| H29 | ✅ CLOSED | `HybridEngineOptions.ts`, `HybridEngineConfig.ts`, `HybridEngineLifecycle.ts`, `WalkaroundGPUPipeline.ts`, `pipelineCompiler.ts`, `PPGCoordinator.ts`, `resourceManager.ts`, `giStateSnapshot.ts`, `HybridEngineGIState.ts`, `ppgUpdate.wgsl.ts` | `ppgMaxDTreeNodesPerCell` now threads from public engine options through derived config, init host, shader compile, PPG resource allocation, coordinator resize/upload/export/import, and GI snapshot v5 metadata. Focused tests cover config preservation, buffer-size math, WGSL defaults, v5 round-trip, and v4 default compatibility. | Residual promotion risk: broad real-GPU PPG A/B remains a hardware-validation queue item, not an implementation gap. |

#### 3B — Emitters, environment, shadows (ledger truth)

| Item | File(s) | Current | Required for native |
|------|---------|---------|---------------------|
| Point/spot DI | `shade.wgsl.ts` `lo_analyticNEE`, `analytic_lights` binding 13 | H41 wired in code | Verify emitter upload populates binding 13 — trace from `coreEmittersToDDGILights` / emitter pack |
| Mesh-area `color`/`intensity` | `restir/bvhSceneHelpers.ts:316-318` | Ignored (H23) | Multiply Le |
| Emitter `castShadow` | DDGI/ReSTIR paths | unsupported | Pack flag; gate in `shadingTerms.wgsl.ts` |
| `primitiveCastShadow` GI-side | DDGI, ReSTIR-GI, RC | approximate | Extend `bvhCastShadowMask` to GI rays (`shared-bvh`, `probeUpdateRays.wgsl.ts`, `risGi.wgsl.ts`) |
| `updateLighting` sun | `HybridEngine.ts:1524+` | DDGI sun not re-synced (items_to_fix) | Call `_ddgi.setLights(orientDdgiSunLights(...))` on `updateLighting` |
| `procedural-sky` | `resolveHybridEnvironment.ts` | Scalar approx | Either bake Preetham to probe rays or keep `approximate` + planner never recommends WA for procedural-sky assets |
| RC sun RGB | `HybridEngineFrameOrchestrator.ts:363-368` | Monochrome (H24) | Pass color |

#### 3C — Alpha & blending (glTF `alphaMode`)

Walkaround has **no** alpha today (`alphaMode/opacity/alphaMap` unsupported).

| Step | Code | Footgun |
|------|------|---------|
| Pack alphaMode + cutoff | `packingHelpers.ts` new bits in `bvhIndex.w` or `bvh_material` | 4-bit transmission lane already crowded |
| Shade discard | `shade.wgsl.ts` | Must happen before ReSTIR writes reservoirs |
| Composite blend | `composite.wgsl.ts` | Swapchain `rgba8unorm` blend state — walkaround writes swapchain via composite |
| `alphaMap` | Requires Phase 3D texture atlas | |

#### 3D — Texture atlas (non-optional for walkaround material 100%)

**Architecture (mirror pt-webgl2):**

```
Scene MaterialSpec.*Map
  → walkaround-hybrid/src/scene/textureAtlas.ts (NEW)
  → GPU texture_2d_array + layerOf map
  → per-tri materialId + uvSet in BVH buffers
  → shade.wgsl.ts / ris.wgsl.ts sample
```

| Component | File(s) | Notes |
|-----------|---------|-------|
| Atlas build | `textureAtlas.ts` | Reuse pixel read logic from `pt-webgl2/texturesArray.ts:79+` |
| UV buffer | `bvhCore.ts`, `shared-bvh/worldSpaceMerge.ts` | **Must** propagate `uvs` stride-2 and `uv1` — merge already can (`worldSpaceMerge` per H33 fix) |
| Tangent buffer | `bvh_normal` exists; add `bvh_tangent` or pack in aux buffer | Normal maps require TBN in `materialDecode.wgsl.ts` |
| Bind group | `bindGroupLayouts.ts`, `WalkaroundGPUPipeline.ts` | New group or extend scene group — watch bind limit |
| Material index per tri | Extend `bvhIndex.w` or parallel `bvh_matId` buffer | Scalar lanes stay for fallback when no map |
| `materialPatch` fast path | `HybridEnginePrimitiveUpdates.ts` | Texture handle change → invalidate atlas slice, not full `setScene` |
| Ledger | `WALKAROUND_MATERIALS`, `CONSUMED_MATERIAL_FIELDS` | One row promotion per map with test |

**Footguns:**
- Sampling baseColor UV for all maps (pt-webgpu v1 bug) — use per-map `TextureRef.texCoord` + `transform` from glTF.
- `materialPatch` with maps currently may not rebuild atlas — test in `mutationMatrix.test.ts`.
- ReSTIR primary hit uses different UV than shade — must share `materialDecode` helpers.
- Atlas rebuild on every animation frame if UVs deform — morph targets need UV-aware or full atlas refresh.

#### 3E — Extension lobes on walkaround (clearcoat, sheen, iridescence, specular, anisotropy)

After 3D: extend `ggxBrdf.wgsl.ts` + `materialDecode.wgsl.ts` + packing lanes.

**Footgun:** Walkaround is not a path tracer — clearcoat/sheen are approximations. Grade `approximate` unless energy conservation verified; planner must surface this.

#### 3F — Fields intentionally permanent `unsupported` on walkaround

Document in ledger + planner: `displacement*`, `spectralAttenuation`, `dispersionAbbeNumber`, `thinFilmStack`, `scattering*`, `frontLayer`/`backLayer` (unless stained-glass scope). **Arbitrary glTF 100%** routes assets using these to pt-webgpu via `rankGltfBackends` — walkaround 100% ≠ all fields native.

#### 3G — Structural debt (items_to_fix §H)

| Item | File | Action |
|------|------|--------|
| H32 glass TLAS shadow | `shared-bvh/wgsl/tlasTraversal.wgsl.ts`; `sceneTraversal.wgsl.ts` | ✅ CODE CLOSED: `traceTlasAny` now forwards `skipGlass` into a single closest-hit path and walkaround forwards the flag. Add a behavioral TLAS glass-shadow oracle before deleting all residual audit notes. |
| H33 materialSig Beer-Lambert | `shared-bvh/src/sceneBvh.ts`; `shared-bvh/src/__tests__/sceneBvhVersionTag.test.ts` | ✅ CLOSED (Wave 2): `materialSetHashFloats` now includes packed `attenuationDistance` (with the canonical no-attenuation sentinel) and `thickness`, so no-tag `SceneBvh.updateFromCore()` rebuilds when only Beer-Lambert distance/depth changes. Regression tests pin attenuationDistance-only and thickness-only edits. |
| H34 BVH degenerates | `buildArrayBvh.ts`, `tlas.ts` | Filter NaN tris |
| Phantom emitter H22 | `emitterList.ts:395-405` | Remove or gate |
| GRIS dead alloc H24 | `resourceManager.ts` | Gate on `regir.enabled` |
| DDGI error swallow | `DDGI.ts:303-346` | Propagate to `onError` |

---

### Phase 4 — Arbitrary glTF orchestration (cross-backend)

#### 4A — Single host path

```
loadGltfAsset(url, { fetch, dracoDecode, meshoptDecode, decodeImage })
  → textureDecodePass()
  → rankGltfBackends(report, policy)
  → createEngine({ prefer, scene, gltfAsset: result })  // NEW: optional gltfAsset
  → controller.attachEngine(engine)
  → loop: controller.advance(dt); engine.renderFrame(...)
```

| Task | File | Footgun |
|------|------|---------|
| `createEngine` accepts `gltfAsset?: GltfAssetResult` | `createEngine.ts`, `createEngineInternals.ts` | When present, `pickBackend` defers to `recommendedBackend.backend` |
| Replace triangle-only auto | `createEngineScale.ts` `pickBackend` | 500k tri budget ignores material richness |
| `VitrumCanvas` `gltf` prop | `VitrumCanvas.tsx` | Load on mount, recreate engine on url change |
| `ProgressiveHandoffCoordinator` + glTF | `progressiveHandoff.ts` | Already has scene fallback; add `controller` reference for animated handoff |
| Shared-device handoff | `createProgressiveEngine.ts` | Textures must be `GPUTexture` compatible — decode to GPU on WebGPU path, not CPU-only handles |
| Examples | `examples/gltf-viewer/` (NEW) | Examples exist (`examples/attach-vitrum/`) but **no glTF example** — gap for DX 100% |

#### 4B — Compatibility enforcement

| Mode | When to throw |
|------|----------------|
| `best-effort` | Never; warnings in `GltfAssetResult.warnings` + `Engine.onWarning` |
| `reject-unsupported` | Any used field `unsupported` on selected backend |
| `reject-degraded` | Any non-`native` issue including `approximate`, `requires-hook` without hook |

**Plug-in:** `engineBridge.ts` `enforceCompatibility` — extend to check `report.primitives.unsupportedModes.length` for used modes.

#### 4C — Texture handle contract (all backends)

| Backend | Expects `TextureRef.handle` | Decoder output |
|---------|------------------------------|----------------|
| pt-webgl2 | `{width,height,data:Float32Array}` RGBA linear or DataTexture-shaped | `texturesArray.ts:79` |
| pt-webgpu | Opaque; uploaded via `webGpuTextureUpload` path in scene pack | GPU texture handle after upload |
| walkaround (Phase 3D) | Same as pt-webgl2 for atlas build | CPU pixels → atlas |

**New shared package or `gltf-adapter/decodeTextures.ts`:** `decodeSceneTextures(scene, { decodeImage, target: 'cpu-linear' | 'webgpu' })` — single entry.

**Footgun:** `createImageBitmap` in browser returns sRGB — convert to linear before atlas.

#### 4D — Animation + temporal GI

| Concern | Code | Footgun |
|---------|------|---------|
| ReSTIR temporal reset | `HybridEngine.ts` `reset()` on topology change | Controller morph/topology must call `reset` or reservoirs ghost |
| DDGI probe invalidation | `updatePrimitive` material vs transform | Transform refit must invalidate probe cache (`HybridEngineGiPropagation.ts`) |
| pt-webgpu accum | `renderFrame` motion vectors | Animated scenes need correct `prevView`/`prevProj` in `attachVitrum` |
| Skinning GPU path | `GpuSkinningSubsystem` vs CPU `solveSkin` | Controller uses CPU `solveSkin` — OK; GPU skinning path must receive bone patches too |

#### 4E — Engine integration residue (H31)

| Item | File |
|------|------|
| `backendId` on attach handle | `vanilla.ts` |
| `createProgressiveEngine` `onError` on canvas configure | `createProgressiveEngine.ts:307` |
| `analyticPrimitiveToMesh` UVs | `packages/core/src/analyticToMesh.ts` |
| `idempotentDispose` errors | `idempotentDispose.ts` → `onError` |

#### 4F — Extensions not yet in spec (gap fill for true arbitrary glTF)

| Extension | Status | Action |
|-----------|--------|--------|
| `EXT_mesh_gpu_instancing` | Not imported | Implement → `instanced-mesh` primitive OR `unsupported` + test |
| `KHR_texture_basisu` | Hook only | Default browser transcoder path + docs |
| `EXT_meshopt_compression` fallback buffer | Implemented | Verify with real samples |
| Multiple UV sets | `TEXCOORD_1` imported | pt-webgpu uv-set bitmask; walkaround needs uv1 buffer |
| `KHR_materials_emissive_strength` | Imported | Verify × on all backends |
| Draco `extensionsRequired` without hook | Throws | Good — keep |

---

### Phase 5 — Closure: prove 100% (not 85%)

#### 5A — Material furnace + glTF sweep

**New:** `tools/gltf-material-sweep/`

For each fixture in `tools/reference-assets/gltf/`:
1. `loadGltfAsset` + `decodeSceneTextures`
2. `evaluateGltfBackendCompatibility` for each backend
3. Render 64spp on **recommended** backend
4. Assert `meanLum > ε`, no GPU validation errors
5. Compare hash to golden PNG (tolerance for MC noise on PT)

**Footgun:** Testing only `analyzeGltfAsset` without render proved glTF API "done" but left textures black.

#### 5B — Oracle suite (keep green)

| Oracle | File | Regression guard |
|--------|------|------------------|
| PTWG-BDPT-01 | `oracle.bdptConnectionCosine.test.ts` | BDPT glTF area lights |
| HYB-GI-01/02 | `oracle.restirDiEstimator.test.ts` | Env + area DI |
| HYB-DDGI-01 | `oracle.ddgiVisibilityMoments.test.ts` | Probe visibility |
| PTWG-LITE-01 | `oracle.liteRectMis.test.ts` | Lite policy |

#### 5C — Mutation matrix GPU observability

Extend `walkaround-hybrid/src/__tests__/mutationMatrix.test.ts` + pt-webgpu mutation tests:
- After `updatePrimitive`/`updateEmitter`/`updateEnvironment`, assert bind group recreation flags, buffer generation counters, or mock `writeBuffer` call counts.

#### 5D — Documentation sync (part of 100% — prevents false claims)

| Artifact | Action |
|----------|--------|
| `BACKEND_PROMISE_LEDGER` | Sole truth; READMEs cite ledger not prose |
| `plan/renderer-fidelity-matrix.md` | Remove deleted `pt-webgl` column; add pt-webgl2 |
| `items_to_fix.md` §H | Close items as fixed or strike |
| ~~H30~~ ✅ CLOSED | Canvas backing store sizing is now applied before engine construction; `attachVitrumLoop.test.ts` pins CSS×DPR sizing |
| H57 | Strike "no examples" — add `gltf-viewer` instead |

#### 5E — Behavioral gate expansion

Add glTF fixtures to behavioral gate configs (currently 29/29): at minimum unlit, textured PBR, transmission glass, skinned animated, Draco (with mock decoder).

---

### Master checklist: 65 material fields × walkaround path to ledger truth

| Category | Fields | Walkaround work |
|----------|--------|-----------------|
| Scalars consumed | baseColor, roughness, metallic, emissive*, transmission, ior, attenuation*, thickness, shadingModel, extensions | `shadingModel` verified `approximate`; fix H23 emissive |
| Alpha | alphaMode, alphaCutoff, opacity, alphaMap | 3C + 3D |
| Maps (17+) | all `*Map` | 3D atlas + decode pipeline |
| Disney scalars | sheen*, clearcoat*, iridescence*, specular*, anisotropy* | 3E |
| Volume/spectral | spectral*, scattering*, thinFilm, front/back layer | Permanent unsupported + planner routes to PT |
| Displacement | displacement* | Permanent unsupported all backends |

**pt-webgl2:** 9 unsupported → 0–3 unsupported (anisotropy, displacement, thicknessMap decision).

**pt-webgpu:** 22 unsupported → 0–5 (thickness, displacement, some maps if bind limits force tier split).

---

### Phase 6 — Ledger residue outside the three targets (ADDED 2026-06-12)

> The three-target addendum does not retract the gap ledger's categorical close
> condition. These verified-open items are NOT covered by Phases 0–5 and must be
> implemented or explicitly downgraded before "100%" signoff:

| Item | File(s) | Fix or downgrade |
|------|---------|------------------|
| pt-webgl2 NEE 3-way selection bias | `packages/pt-webgl2/src/glsl/render/direct_light_contribution_function.glsl.js`; `packages/pt-webgl2/src/glsl/composeTraceGlsl.test.ts` | ✅ DONE (Wave 1): analytic/mesh/env NEE now use one shared strategy variate (`neeStrategyU`) with cumulative cutoffs, so slot probabilities match the PDFs. Focused source/probability tests pin the single-draw selector and the old `1/3,4/9,2/9` regression. |
| Engine `onError` shape unification | `createEngine` / `Engine.onError` / `attachVitrum.onEngineError` / `createProgressiveEngine.onError` | Four shapes, three names; progressive drops the phase/backend event. One `EngineError`-based shape + deprecation aliases |
| `attachVitrum` auto-recreate scene loss | `packages/engine/src/lifecycle/vanilla.ts`; `packages/engine/src/__tests__/attachVitrumAutoRecreate.test.ts` | ✅ DONE (Wave 1): lifecycle now tracks the latest scene submitted through the exposed engine handle and recreates with that scene after device/context loss. Regression test simulates fatal `device-lost` and verifies the second `createEngine` call receives the updated scene. |
| pt-webgpu trace-lite shader-gate mismatch | `packages/pt-webgpu/src/wgsl/pathTrace/causticLite.wgsl.ts`; `kernelLite.wgsl.ts`; `wgslContract.test.ts`; `wgslLiteContract.test.ts` | ✅ DONE (Wave 1): lite MNEE stub signature now matches the lite kernel material-extension call shape, and lite BSDF-environment reconnection receives the scalar clearcoat/sheen/iridescence fields it already evaluates. `npm run shader-gate` compiles `pt-webgpu/trace-lite`; contract tests pin stub/caller parity and the updated lite SHA/length. |
| Lite tier single-BLAS | `uploadSceneBuffers.ts` lite path | Now honestly labeled, but `mergeWorldSpaceFromCore` (already consumed by 2 backends) would make multi-primitive lite real — implement (preferred) or keep ledgered |
| RC exported-surface footguns | `cascadeDispatch.ts:298,317-320,728`; `HybridEngineRC.ts` | ✅ light-buffer lifecycle now invalidates bindings on nonzero→zero transitions; remaining: validate `cascadeDims` (2× ray-grid invariant), throw on violation, and cover bounds-change/stale-merge-uniform cases |
| shared-bvh sampled fingerprint in correctness path | `bufferFingerprint.ts` + `sceneBvh.ts:131` | Sampled hash gates a REBUILD SKIP (stale BVH on miss), not just re-upload as documented — full-hash the geometry arrays or add a cheap length/sum guard |
| `solveSkin` morph-normal silent skip | `packages/core/src/skinSolver.ts:242`; `packages/core/src/__tests__/skinSolver.test.ts` | ✅ DONE (Wave 1): active morphs now throw when `morphTargetNormals.length !== morphTargets.length`, and malformed normal-delta entry lengths remain throw-on-read. Focused test pins both cases. |
| Core contract additions from Wave 3 | `material.ts`, `primitives.ts` | `morphTargetTangents`, `thicknessMap` (+ glTF adapter wiring + `doubleSided` decision from 1B) |

### Suggested commit sequence (no dates)

1. Land glTF API + engine bridge + controller + unlit all backends
2. Texture decode helper + pt-webgl2/pt-webgpu upload integration
3. P0 walkaround + pt-webgpu correctness (W-HYB, PTWG, H25-H29)
4. WEBGL2-01 + H49 + GLTF-01 + CORE-01
5. PTWG-MAT-01 integrator audit + material descriptor expansion
6. Walkaround texture atlas + UV/tangent buffers
7. Walkaround alpha + shadow GI parity
8. `createEngine` + `pickBackend` glTF-aware + `examples/gltf-viewer`
9. glTF material sweep + behavioral gate fixtures
10. Ledger/README/fidelity matrix reconciliation

### Execution dependency

```
P0 land commit
  → P0 correctness (W-HYB, PTWG, PPG, NRC, neural binds)
    → PT material parity (2B, 2C) + WEBGL2-01
      → glTF API harden (1A-1E) + engine/gltf wrapper (1E)
        → pickBackend + compatibility enforcement (4A, 4B)
          → Walkaround alpha (3C) → Walkaround atlas (3D) → Walkaround lobes (3E)
            → Material furnace (5A) + gates (Phase 0.2, Phase 5)
              → 100% signoff
```

Walkaround atlas (3D) and PT material parity (2B) can run in parallel after P0.

### Summary

- **Condensed to 5 phases:** land gates → glTF → PT → walkaround → orchestration → proof.
- **Specificity:** file-level plug-in points, decoder contracts, bind-group footguns, integrator audit matrix, texture atlas architecture.
- **Gap fill vs 85%:** texture decode bridge, EXT_mesh_gpu_instancing decision, animation×temporal GI, lite-tier rejection for fidelity, PTWG-MAT all paths, walkaround alpha/blending, examples/gltf-viewer, render-based glTF sweep (not analyze-only), `pickBackend` fix, double-sided/vertex-color, tangent generation at import, engine `gltfAsset` passthrough, documentation sync as part of done.

Walkaround **100%** and arbitrary glTF **100%** are not the same: arbitrary glTF routes rich assets to PT backends via the planner; walkaround 100% still means permanent `unsupported` for spectral/displacement with explicit rejection, not silent gray materials.

---

## Forward-looking — the post-100% SOTA wave (ADDED 2026-06-12, NOT in campaign scope)

> Phases 0–6 above deliver **contract-complete**. This section is the separate
> axis: convergence/throughput engineering where vitrum is below current SOTA
> practice even after the campaign closes. Tracked here per roadmap §0.5
> (frontier: tracked but deprioritized behind the fidelity grind). Ordered by
> value-per-effort. None of these block 100% signoff.
>
> Context: post-campaign vitrum is already at-or-beyond published in-browser
> SOTA on *breadth* (no public browser engine ships spectral + BDPT + ReSTIR-PT
> + MNEE + progressive SPPM + inverse rendering + a gated realtime GI track).
> The items below are where the *engineering* axis lags the field.

### F1 — Low-discrepancy sampling (biggest convergence win per effort)

Both converged backends run PCG only; pt-webgl2's Sobol/stratified branches are
pinned dead (1×1 dummy textures, `featureTypes.ts`), shared-samplers has only
Hammersley/PCG. SOTA is Owen-scrambled Sobol or PMJ02 + blue-noise screen-space
distribution — typically a 2–4× effective-convergence multiplier.
**Work:** real table generation in `shared-samplers` (Owen-Sobol or PMJ02,
CPU-baked, uploaded as textures/buffers), per-dimension assignment audit
(bounce/lobe/light dims), blue-noise rank-1 screen scramble; revive or replace
the dead pt-webgl2 RANDOM_TYPE branches; pt-webgpu equivalent in
`kernel.wgsl.ts` RNG plumbing. Validate via equal-time RMSE A/B on the
reference scenes (self-validating: error curves, not eyeballs).

### F2 — Compressed wide BVH traversal (biggest throughput win)

Binary SAH + stack traversal is solid but compute-shader SOTA is 8-wide
compressed BVH (CWBVH-style): ~2× traversal throughput, smaller memory
footprint. Light tree is median-split, not full adaptive Estévez-Kulla (already
documented in `lightTree.ts:33-35`).
**Work:** CWBVH build + traversal kernels in `shared-bvh` behind the existing
single-sourced stride/WGSL contract pattern; CPU brute-force oracles like the
existing T1 set; per-backend opt-in until parity proven. Becomes decisive if/when
a WebGPU ray-tracing extension ships (whole-field handicap today: no RT cores
in the browser for anyone).

### F3 — Shipped denoiser weights (out-of-the-box UX)

OIDN arrives via host-supplied ONNX with no weights shipped (A10 production
neural weights = declared non-goal of the campaign). SOTA UX is denoised by
default. **Work:** license-vetted OIDN weight distribution (or train the
in-repo UNet to production quality), wasm/webgpu execution path that needs no
host wiring, `denoiser:'auto'` default that engages when weights resolve.

### F4 — Wavefront path tracing (largest rearchitecture — only if profiling demands)

Megakernel with a hard 8-bounce structural cap (`kernel.wgsl.ts` bounceLimit).
Wavefront scheduling (per-bounce queue compaction) is how current GPU PTs kill
warp divergence at depth. **Work:** queue/compaction infrastructure, kernel
split (generate/extend/shade/connect), persistent state buffers. Big; gate the
decision on divergence profiling, not fashion. The 8-bounce cap lift falls out.

### F5 — Heterogeneous volumes

Homogeneous media + SSS random walk only; pt-webgl2's fog-volume GLSL is dead
code (uniforms never uploaded). SOTA is null-collision delta/ratio tracking over
grids (NanoVDB-class). **Work:** core contract for volume primitives first
(extension point exists: `AnalyticShape`/`Material.extensions`), then
delta-tracking integrator on pt-webgpu. Only if the product wants smoke/clouds —
stained-glass design center may never need it.

### F-BRIDGE — Experimental no-hardware-RT bridge (ADDED 2026-06-12)

> Strategy: compute shaders can't out-muscle RT cores at traversal, so don't
> compete — **trace fewer rays (reuse/upscale), cheaper rays (proxies/LOD), or
> none (caches)**. Stacked estimate ~25–65× effective vs the 10–50× hardware-RT
> promise (amortization-shaped: fast-moving scenes benefit less). All items pass
> the feasibility bar: public source, web-portable, not RTX-locked.
>
> | Lever | What | Gain | Feasibility |
> |-------|------|------|-------------|
> | Subgroups + f16 | Wave intrinsics for traversal compaction; f16 BVH/material bandwidth | 1.3–2× | Shipping in Chrome today; plumbing only |
> | Blue-noise error diffusion | Heitz–Belcour screen-space scrambling — same error, far cleaner look | perceptual | Paper + ref code public; tiny |
> | FSR2-class temporal upscaling | Render 50–60% res, reconstruct; motion vectors + depth already produced | 3–4× | FSR2 MIT HLSL → WGSL port; no browser PT has done it |
> | SHaRC radiance cache | World-space hash-grid cache, early path termination; non-neural cousin of NRC, can share its termination seam | ~2× path length | NVIDIA open source, plain compute |
> | SDF/proxy secondary rays | Software-Lumen trick: diffuse GI traces SDF/voxel proxy, only primary/specular touch triangles | 5–10× on secondary | Well documented (UE source readable); walkaround proves the dual-representation pattern |
> | Stochastic bounce LOD | Bounces ≥2 trace a decimated BVH (geometry version of the dead `materialLodDepth` idea) | scene-dep | Simple, low risk |
> | Shadow-map NEE assist | Rasterized shadow map answers dominant-light shadow rays; trace only the penumbra band | large (shadow rays dominate) | Walkaround raster primary shows the seam exists |
> | Server-side RT-baked GI | Converge GI state on real RT hardware off-device, ship via GI-state v4 export; browser refines | n/a (offload) | Works today; productizes app-idea #3 |
> | Persistent threads + ray sorting | Morton-binned wavefront (Aila/Laine lineage) — F4 done aggressively | 1.5–2× | Classic public literature; higher effort |
> | Subgroup matrix (experimental) | WMMA-class MLP inference for NRC/neural denoiser | NRC-specific | Chrome experimental flag; wait for stabilization |
>
> Unlock target: T2-4 (multiplayer GI editing), T2-5 (sensor twins), and
> 30–60fps path tracing of moderate scenes **without** the WebGPU RT extension.

### F6 — Unbiased realtime GI default (GRIS default-on)

Default ReSTIR-GI reuse is deliberately biased (clamped-Jacobian; A8 decision
keeps exact GRIS off-default). The unbiased-reuse literature is the published
SOTA. **Work:** GPU-validate the existing GRIS path at production settings
(perf + flicker), then flip the default; the oracle infrastructure from this
campaign (`oracle.restirDiEstimator`) extends to pin the GI estimator the same
way.
