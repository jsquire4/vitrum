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
| **B12** | ✅ **DONE (source-verified 2026-06-12) — lite-tier texture packing shipped** | `webgpuLimits.ts:45-73`, `wgsl/pathTrace/material.wgsl.ts:112-142`, `wgsl/pathTrace/kernelLite.wgsl.ts:143-304`, `scene/litePackedTextures.ts`, `gpuResources.ts:711-728,1686-1755`, `liteTierCapabilities.test.ts`, `webgpuLimits.test.ts` | **Implemented:** the lite tier packs HDRI radiance/pdf + CDF into sampled textures (`liteEnvTex`, `liteEnvCdfTex`) and point/spot/rect-area light data into `liteLightTex`, avoiding the storage-buffer cliff while staying inside the baseline sampled-texture budget. Capabilities/tests now advertise point/spot/rect-area + HDRI support on lite; disc-area/mesh-area remain explicitly unsupported. | GPU A/B promotion evidence still belongs to the validation matrix, not this binding-budget gap. | done |

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

> **Status 2026-06-12:** D2 (silent drops), D4 (memory accounting), D5 (stale comments),
> D7 (SVGF allocation), D8 (fork lint), D9 (traceTier dedup), D3 (contract + ingestion),
> and C1 (OIDN clear-error/capabilities) are closed in HEAD. D1 is no longer a single
> "delete all names" task: source verification found stale names, public/test helpers, and
> one real pt-webgl2 residue. Remaining: D6 (bind-group churn — perf, deferred), D3
> per-backend BSDF consumption (B-bucket fidelity), and the still-open D1 policy cleanups
> called out below.

- **D1 — Dead code removal / source reconciliation** ◑ VERIFIED + partial cleanup 2026-06-12: pt-webgl2 `frameParamsPacker.ts`/`uploadFrameParams`/`#paramsUbo`/`#bindParamsUbo` are stale Road references (no active implementation remains; the misleading `glResources` FrameParams-UBO prose was removed); pt-webgl2 `'additive'` accumulation is now fully removed from the compile contract and emitted GLSL (`FEATURE_ADDITIVE_ACCUM` blocks pruned; `composeTraceGlsl.test.ts` gates it); retired `pt-webgl` debounce symbols are absent from active packages (legacy staging/docs only); `PPGCoordinator.resetTrainingAccumulators`, `heroStrategy`, old `GPU_SKIN_BVH_WGSL`, `ownsEnvSampler`, and `cleanupAfterSubmit` are stale names/renames rather than live deletion targets. **Still deliberate/not blindly deleted:** `packCameUBO` is an exported stained-glass host utility with tests/README but no in-repo runtime shader consumer; `sTreeAccumulate`/`resetAccumulators` are test fixtures/helpers while production PPG uses GPU readback counts; `probesPerFrame` remains ABI layout/packing even if shaders do not branch on it; `expandIndicesToStride4` is an exported shared-bvh convenience with tests; `RESTIR_PT_HYBRID_SHIFT_HARNESS_WGSL` is a test/GPU-oracle harness; walkaround-rc TSL references are historical mapping docs/comments with raw-WebGPU boundary tests. **Open action:** decide whether to deprecate public/test helpers or keep them with explicit public/test-only wording; do not remove as "dead runtime code" without API intent. **Effort: S–M policy cleanup.**
- **D2 — Silent data drops** ✅ DONE / reconciled in HEAD: the old `three-bindings/src/index.ts` drop claim is stale (no `packages/three-bindings`, `@vitrum/three-bindings`, `sceneFromThree`, or `vitrumSceneToThree` remains after the glTF-adapter path replaced it); pt-webgpu `MaterialTextureArray.warnings` route through `UploadedSceneBuffers.warnings` / the engine structured warning path; heterogeneous texture-array layers now expose per-layer UV-fit scales (`MaterialTextureArray.layerUvScales`) and the full-tier material samplers address the copied source rect per map instead of sampling padded black texels. Residual exact mip/border policy belongs to the broader glTF texture-decode/sampler campaign, not this silent-drop row. **Effort: closed.**
- **D3 — Contract material gaps** ✅ DONE (contract + ingestion) / ◻ consumption tracked: added `specularIntensity`/`specularColor` (+ their maps), `bumpMap`/`bumpScale`, `displacementMap`/`displacementScale`/`displacementBias`, `lightMap`/`lightMapIntensity`, `envMapIntensity` as first-class optional fields on core `MaterialSpec` (+ `MaterialMapFields` slice); `three-bindings.convertMaterial` now extracts them so the THREE→core data loss is closed (+4 tests). **REMAINING — per-backend BSDF consumption (these require golden-breaking material-layout changes, so they're real B-bucket fidelity work, not ingestion): specular F0 modulation + envMapIntensity scale in the 4 BSDFs; bump-map normal perturbation; displacement-map geometry; lightMap additive in shade — plus the reverse `vitrumSceneToThree` round-trip for the pt-webgl path.** **Effort: ingestion M (done); consumption S–L per field.**
- **D4 — Memory accounting** ✅ VERIFIED closed in HEAD: `UploadedSceneBuffers.gpuMemoryBytes()` sums live scene buffers + material texture arrays (`packages/pt-webgpu/src/scene/uploadSceneBuffers.ts:1408`), and `debug.estimatedGpuMemoryBytes()` includes those bytes under `byCategory.scene`, `byTextureFormat`, and `byBufferUsage.storage` before telemetry emits the scalar total (`packages/pt-webgpu/src/index.ts:787`, `:950`). **Effort: closed.**
- **D5 — Stale comments contradicting code** ✅ CLOSED 2026-06-12: source-read verified the RC light-model and current `packages/walkaround-rc/src/cascadeDispatch.ts` verification-status citations were already corrected; `createRestirGIFrameResources.ts` now says the GRIS reconnection cache is read by the reuse variants today, and `atrousVariance.wgsl.ts` now calls `svgfVarianceMain`/`svgfAtrousMain` legacy entry-point names rather than evidence that the module is Schied SVGF. Focused stale-comment gates pin those statements. **Effort: closed.**
- **D6 — Per-frame bind-group churn** ◻: ~20–30 `createBindGroup`/frame in walkaround, zero memoization. Perf hygiene. **Effort: M.**
- **D7 — SVGF texture allocation** ✅ CLOSED 2026-06-12: `createSvgfFrameResources.ts` now collapses the heavy SVGF history textures and the current/previous object-id textures to 1×1 whenever the active denoiser is not `svgf-real`; shade writes object IDs through a dimension-guarded helper so the inactive 1×1 storage texture remains a legal frame-layout placeholder. `gpuMemoryEstimate.test.ts`, `svgfObjectIdResources.test.ts`, and `svgfObjectId.test.ts` pin the inactive placeholder sizes, active full-res sizes, and guarded shader store. **Effort: closed.**
- **D8 — fork lint red** ✅ FIXED (bumped to ESLint 9): the red was an `eslint@8` vs `@typescript-eslint@8` plugin crash (`no-unused-expressions` reading `allowShortCircuit`) — NOT the audit's stale-SSS gate (`tsc` + `shader-smoke` always passed). Fix: bumped the fork to `eslint@^9.39.4` (deduped to root; had to prune an orphan nested `eslint@8.57.1` the lockfile kept reinstalling — uninstall→reinstall on the workspace cleared it), kept `.eslintrc.json` via `ESLINT_USE_FLAT_CONFIG=false` in the lint script, and made its `extends` hoist-proof (`"mdcs"` shareable name instead of a relative `./node_modules/...` path that broke when mdcs hoisted to root). `npm run lint` is now green (0 errors, 1 pre-existing `no-unused-vars` warning in `example/`; tsc + shader-smoke pass). **Future:** eslintrc is deprecated in eslint v10 → a flat-config migration when the repo moves to v10.
- **D9 — traceTier dedup** ✅ CLOSED 2026-06-12: source-read verified `WebGl2TraceTier` is owned by `packages/pt-webgl2/src/traceTier.ts` and re-exported from `options.ts` instead of being duplicated; `traceTier.ts` and the package README now describe `lite` as aux-buffer-only degradation with the trace kernel unchanged rather than promising hidden bounce/texture caps. `traceTier.test.ts` pins both the single-source type surface and the lite-tier policy wording. **Effort: closed.**

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

- **NEW C4 — examples / DX surface** ✅ SOURCE-RECONCILED 2026-06-12:
  the old "zero examples" claim is stale. The repo now has runnable Vite examples for
  `createEngine`, `attachVitrum`, `VitrumCanvas`, `createProgressiveEngine`, both direct
  PT factories, and **`loadGltfWithEngine()`** (`examples/gltf-viewer`). H57 is closed
  as a provisioning gap; future examples are product polish, not a Road blocker.
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
| ~~Typed errors~~ ✅ DONE | `packages/gltf-adapter/src/errors.ts`, `assetLoader.ts`, `gltfAssetApi.test.ts` | `GltfFetchFailed` / `GltfResourceNotFound` now carry `{ url, kind }` plus HTTP status when applicable; loader tests assert typed rejection. | Generic fetch/resource errors are gone from the high-level loader path. |
| ~~Cache hooks~~ ✅ DONE | `LoadGltfAssetOptions.cache`, `assetLoader.ts`, `gltfAssetApi.test.ts` | `fetchArrayBuffer` now checks/sets a host cache for resolved asset/buffer/image URLs, keyed by final URL + resource kind. | Relative assets with different `baseUri`s resolve to different absolute keys. |
| ~~`loadGltfAndDecodeTextures()` helper + decode report~~ ✅ API DONE | `assetLoader.ts`, `texturePipeline.ts`, `engineBridge.ts`, `index.ts`, `gltfAssetApi.test.ts` | High-level loading now returns `textureDecodeReport` and exports `loadGltfAndDecodeTextures()` as the explicit adapter-owned decode/report entry. The report walks the converted `Scene` and classifies each material `TextureRef` by field/path/UV/transform/handle-kind plus backend readiness (`pt-webgl2`, `pt-webgpu`, `walkaround-hybrid`). | This closes the adapter diagnostics/API surface; backend atlas/upload completion remains in 2B/3D. |
| sRGB → linear | In decode helper | glTF baseColor textures are sRGB (`KHR_materials_unlit` too) | Double-linear if backend also decodes sRGB |
| NPOT / max dim | In atlas builders (PT + WA) | Nearest resize to `max(dim)` like `texturesArray.ts:12-13` | `sampler2DArray` requires uniform layer size |
| Basis/WebP/DDS source policy | ✅ CODE CLOSED (`featureReport.ts`, `textures.ts`, `gltfExtensionPolicy.test.ts`, `gltfAssetApi.test.ts`): required/no-base-fallback texture-source extensions report `requires-hook`; optional alternates with a base `texture.source` fallback are compatibility-clean until the host opts into `textureSourceExtensions`. | `reject-degraded` no longer rejects deterministic PNG/JPEG fallback assets just because they also advertise KTX2/WebP/DDS alternates. | Default browser transcoder/decode path remains a future 4F polish item. |
| **Texture wrap modes (ADDED 2026-06-12 — verified hole)** | ~~`TextureRef.wrapS/wrapT` in core (`repeat`/`clamp`/`mirror`) + adapter reads `gltf.samplers`~~ ✅ API DONE (`material.ts`, `textures.ts`, `gltfTextureSweep.test.ts`, `gltfAssetApi.test.ts`, `texturePipeline.ts`). ✅ pt-webgpu full-tier material textures now pack per-map wrap pairs (`materialTextures.ts`) and WGSL applies repeat/clamp/mirrored-repeat before UV-fit scaling (`material.wgsl.ts`). ✅ pt-webgl2 now packs per-map wrap pairs at material texels 95..104 and routes every material texture sample in surface + attenuation shaders through manual repeat/clamp/mirror wrapping (`materialsTexture.ts`, `material_struct.glsl.js`, `get_surface_record_function.glsl.js`, `attenuate_hit_function.glsl.js`). Still open: walkaround sampler/wrap via the Phase 3D atlas. | API no longer drops sampler semantics and `textureDecodeReport` exposes wrap modes. Runtime parity for clamp/mirror/repeat is closed on pt-webgpu and pt-webgl2; walkaround remains blocked on its material atlas. |
| **Mipmaps (ADDED 2026-06-12 — verified hole)** | pt-webgpu raw-data upload now normalizes 1/2/3/4-channel 8-bit `{data,width,height}` payloads to explicit RGBA8 rows and warns/leaves the layer black for unsupported typed-array layouts (`materialTextureArray.ts`, `materialTextureArray.test.ts`). Still open: true mip generation/selection. The sampler object has `mipmapFilter:'linear'`, but the WGSL material path currently uses `textureSampleLevel(..., 0.0)`, so just generating mip chains would not fix minification aliasing; this needs a real LOD policy (ray-cone/differential heuristic or other documented approximation) plus mip uploads. | Minification aliasing remains on distant/high-frequency textures until mip generation and non-zero LOD selection land. |

#### 1B — Feature report & planner (`featureReport.ts`)

**Already:** `analyzeGltfAsset`, `evaluateGltfBackendCompatibility`, `rankGltfBackends`, per-field ledger crosswalk.

**Still required:**

| Task | Code | Footgun |
|------|------|---------|
| ~~Source paths on every issue~~ ✅ DONE | `featureReport.ts`, `gltfAssetApi.test.ts` | `GltfCompatibilityIssue.path` is now required, analyzer source-path maps cover extensions/primitives/materials/scene cameras, and the test asserts every issue has a non-empty path including `materials[0].normalTexture`-style material paths. |
| ~~Scalar `KHR_materials_pbrSpecularGlossiness` scoring~~ ✅ DONE | `specularGlossinessMaterialCount` + compatibility issue `KHR_materials_pbrSpecularGlossiness` | Texture-alpha glossiness remains the next row |
| ~~Spec-gloss glossiness-alpha~~ ✅ DONE (explicit approximate downgrade) | `materials.ts` warns that `specularGlossinessTexture` RGB is imported as `specularColorMap` while glossiness-in-alpha is not baked into `roughnessMap`; `featureReport.ts` emits `KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha=approximate` with source path. Test: `gltfExtensionPolicy.test.ts`. | No silent roughness-map lie: scalar `glossinessFactor` drives roughness unless a future texture bake/channel-swizzle contract is added. |
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
| ~~Generate tangents when missing~~ ✅ DONE | `gltfToScene.ts`, `tangents.ts`, `gltfAdapter.test.ts` | Normal/bump/clearcoat-normal mapped primitives now synthesize xyzw tangents from POSITION/NORMAL/TEXCOORD_0 when authored TANGENT is absent, and preserve authored tangents unchanged. |
| ~~`COLOR_0` vertex colors~~ ✅ DONE for adapter + planner + pt-webgl2 | `gltfToScene.ts` imports `COLOR_0`; `featureReport.ts` records source paths and compatibility; pt-webgl2 now threads merged vertex colors into `attributesArray` layer 3 and enables the GLSL `material.vertexColors` multiply for affected material slots. pt-webgpu + walkaround-hybrid are structured `unsupported` compatibility issues until their attribute/material tiers consume colors. | glTF vertex color × baseColor is native on pt-webgl2; unsupported paths reject/degrade explicitly |
| ~~Sparse accessors~~ ✅ DONE | `accessors.ts`, `accessors.test.ts`, `gltfAdapter.test.ts`, `gltfModesMorphsAnimations.test.ts` | Sparse patches now have focused coverage for base+pure-sparse accessors, unsigned-byte sparse indices, byte offsets, strided base data, normalized values, integer index accessors, and out-of-range/invalid sparse-index diagnostics. |
| ~~Point/line modes~~ ✅ DONE | Product decision: keep unsupported; `gltfPointLinePrimitivePolicy.test.ts` pins structured compatibility issues + warn-skip import | Don't "half support" — either add `ScenePrimitive` kind later or keep rejecting |

#### 1D — Runtime controller (`sceneController.ts`)

**Already:** `seek`, `advance`, skin bones via `solveSkin`, morph weights, `updatePrimitive` with `setScene` fallback.

**Still required:**

| Task | Code | Plug-in | Footgun |
|------|------|---------|---------|
| ~~Multi-clip blend~~ ✅ DONE | `GltfSceneController.blend(clips, weights, time, { times?, loop?, engine? })` samples each clip, normalizes positive weights, blends channels per node/path (including hemisphere-corrected rotation nlerp), then runs the same transform/morph/skin patch path as single-clip animation. Tests: `sceneController.test.ts` transform blend + morph-before-skin-solve blend. | Morph blending now happens before `solveSkin`; sparse clips blend per authored channel so unrelated channels are not damped. |
| ~~`KHR_materials_variants` at runtime~~ ✅ DONE | `GltfSceneController.setVariant(name/index/undefined)` uses importer-emitted `materialVariantBindings` + `convertedMaterials` to patch only affected primitive materials; `loadGltfForEngine()` now forwards that metadata into bridge-created controllers; `updatePrimitive` fast path falls back to `setScene(nextScene)` on rejection; `resetPose()` preserves the active material variant. Tests: `sceneController.test.ts` runtime switch + fallback, `gltfAssetApi.test.ts` bridge-created controller switch, `gltfExtensionPolicy.test.ts` import-time selection. | Variant switch now invalidates material fast-path caches by issuing material patches through the same engine patch channel as animation updates. |
| ~~Engine attach API~~ ✅ DONE | `controller.attachEngine(engine, { setScene })` exists and `loadGltfForEngine` attaches after load | Use `attachScene:false` / `setScene:false` when the host already set the scene |
| ~~Patch routing per backend~~ ✅ DONE | `sceneController.ts` applies `patchPrimitiveInScene` first, tries `updatePrimitive`, and falls back to one `setScene(nextScene)` with a controller warning if an incremental patch throws; `sceneController.test.ts` pins the throw→fallback path. | — | Partial patch desync is closed for the glTF controller path |

#### 1E — Engine bridge (`engineBridge.ts`)

**Already:** `loadGltfForEngine`, `compatibilityMode`, factory injection.

**Still required:**

| Task | Code | Footgun |
|------|------|---------|
| ~~`@vitrum/engine/gltf` re-export~~ ✅ DONE | `packages/engine/src/gltf.ts` exports the adapter bridge and adds `loadGltfWithEngine()`, which injects `createEngine` and maps the glTF planner's selected backend to the matching engine preference. `packages/engine/package.json` exposes `./gltf` and declares the adapter dependency. Test: `gltfSubpathExport.test.ts`. | Adapter still owns loading/planning/controller logic; the engine subpath is a thin one-import DX wrapper. |
| ~~Pass `decodeImage` + `dracoDecode` + `meshoptDecode` through bridge~~ ✅ DONE | `LoadGltfForEngineOptions` extends `LoadGltfAssetOptions`; `loadGltfForEngine` passes options through `loadGltfAsset` | Bridge without hooks still fails required compressed assets, but now through the intended hook contract |
| ~~Return `textureDecodeReport`~~ ✅ DONE | `GltfAssetResult` + `GltfForEngineResult` expose the scene-level report; tests pin raw-image fallback entries. | Host knows before first frame which maps are raw/opaque/CPU-readable/ignored. |

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
| ~~`normalScale`~~ ✅ DONE | Packed in the group-3 material texture descriptor and applied in `applyNormalMap` by scaling tangent-space xy before normalization; ledger row promoted to `native`. Tests: `materialTextures.test.ts`, `engineContract.test.ts`. | `normalMap` itself remains `approximate` because pt-webgpu still derives hit tangents instead of consuming authored tangent.xyzw / handedness |
| ~~`transmissionMap`, `alphaMap`~~ ✅ DONE | `materialTextures.ts` packs both maps into the linear texture array; `material.wgsl.ts` samples alpha coverage in alphaMode paths and transmission R in the full-tier prologue. Tests: `materialTextures.test.ts`, `wgslContract.test.ts`, `wgslLiteContract.test.ts`. | Texture-map handle changes remain full-rebuild material patches; that is mutation performance, not a missing render path |
| ~~`clearcoatMap`, `clearcoatRoughnessMap`, `sheen*Map`, `iridescence*Map`, `specular*Map`~~ ✅ DONE/approx | `materialTextures.ts` extends the full-tier descriptor stride to 63 vec4s, packs these extension-lobe maps into the correct sRGB/linear arrays, records per-map texCoord / KHR_texture_transform / wrap / UV-fit data, and `material.wgsl.ts` samples the glTF channels before `shadePrologue.wgsl.ts` modulates decoded lobe parameters. Tests: `materialTextures.test.ts`, `wgslContract.test.ts`, `liteTierCapabilities.test.ts`; `shader-gate --self-test` and WSL `validate:gpu:smoke` pass. | Still `approximate` in the promise ledger because `clearcoatNormalMap` / `clearcoatNormalScale` remain unsupported, BDPT light-subpath texture-map payloads are still scalar-only, inverse/adjoint gradients target the base parameterization, and material-lobe reference A/B is still pending. |
| ~~`specularIntensity`, `specularColor`~~ ✅ DONE/approx | Packed in material vec4 #27 and consumed by ordinary PT BRDF/PDF paths, MNEE/SPPM receiver paths, BDPT light-subpath surface scattering, and ReSTIR-PT visible-domain reservoir/resolve payloads. Tests: `scenePack.materials.test.ts`, `wgslContract.test.ts`, `wgslLiteContract.test.ts`, `bdptGlossyLightSubpath.test.ts`; `shader-gate --self-test` compiles production shaders. | Still `approximate` in `BACKEND_PROMISE_LEDGER` until inverse/adjoint gradients and remaining specialty texture-map payload schemas have the same scalar/material-lobe coherence |
| ~~Per-map UV transform~~ ✅ DONE for pt-webgpu consumed maps | `materialTextures.ts` now packs two UV metadata vec4s per consumed map (`texCoord`, offset, scale, rotation), and `material.wgsl.ts` passes map-specific slots for baseColor/emissive/normal/ORM/AO/light/bump/anisotropy/alpha/transmission plus the extension-lobe maps above. Tests: `materialTextures.test.ts`, `wgslContract.test.ts`; `shader-gate` compiles the full-tier trace. | This is full-tier megakernel coverage only; ReSTIR-PT visible-vertex map sampling still needs its own payload/schema pass. |
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
> bounds. ✅ **environment:'none' phantom skylight CLOSED (2026-06-12):**
> full + lite no-map env lookups now return black radiance and zero env pdf
> (`connect.wgsl.ts`, `connectLite.wgsl.ts`); procedural sky remains lit through
> the CPU-baked HDRI path, and `environmentPacking.test.ts` pins kind:none.

Audit **every** `evaluateBrdf` / `brdfDirectionalPdf` call site — glTF extension lobes must match across paths:

| Path | File | Status |
|------|------|--------|
| Eye path NEE | `kernel.wgsl.ts` / `kernelLite.wgsl.ts` | ✅ direct-light NEE, BSDF connection helpers, `sampleNextBounceDirection`, and BDPT eye-stack forward/reverse PDFs now use a normalized sampled-density helper for base/clearcoat/sheen. Transmissive dielectric materials deliberately stay on the existing delta reflection/refraction density until a layered-glass sampler exists. |
| BSDF connections | `connect.wgsl.ts`, `connectLite.wgsl.ts` | ✅ local helper propagation closed (area/env full-tier; env lite; area-lite remains deliberate zero stub) |
| BDPT | `bdptConnection.wgsl.ts`, `bdptLightSubpath.wgsl.ts` | ✅ eye↔light connection uses full helpers; scalar light-subpath scatter now reuses `sampleNextBounceDirection` and records `brdfDirectionalPdfFullSampled` forward/reverse densities. **Open:** light-subpath texture-map payloads and independent radiometric oracle coverage. |
| SPPM / caustics | `caustic.wgsl.ts`, `sppmBindings.wgsl.ts` | ✅ receiver-side SPPM/caustic BRDF/PDF helper propagation closed; MNEE cone-vs-BSDF MIS now uses the sampled-density helper for the BRDF alternative |
| ReSTIR-PT | `restirPtProducer.wgsl.ts`, `restirPtCompose.wgsl.ts`, `reservoirPtHero.wgsl.ts`, `restirPtResolve.wgsl.ts` | ✅ producer direct/onward paths use full helpers; ✅ scalar-lobe reservoir payload/target/resolve now uses full visible-domain helpers, including `KHR_materials_specular` scalar colour/intensity (`ReservoirPTHero` 52-word layout); ✅ visible-vertex payload now mirrors the main shade prologue for alpha pass-through, baseColor/AO/ORM/normal/bump/transmission/extension maps, layer tint/roughness, thin-film, and spectral albedo; ✅ suffix/reconnection vertices now alpha-skip and decode the same hit-local material-map/layer/thin-film/spectral domain before Lo evaluation, using the mapped suffix normal for the reservoir geometry; ✅ producer source sampler now samples a normalized base/clearcoat/sheen lobe mixture and stores the matching `pdfSrc` |
| Adjoint | `adjointPass.wgsl.ts`, `pathTraceAdjoint.wgsl.ts` | OPEN — derivatives still target the base BRDF parameterization |
| Present | `present.wgsl.ts` tonemap only — no BSDF | N/A |

**Footgun:** Fixing megakernel only used to leave BDPT/SPPM wrong for glTF clearcoat scenes with `bdpt:true`; that local helper class is now narrowed. The remaining class is specialty texture payloads, inverse/adjoint derivatives, and proof/A-B coverage, not just missed function calls.

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

Already native. **glTF instancing:** glTF uses multiple nodes, not `instanced-mesh` kind — adapter flattens to separate primitives. **`EXT_mesh_gpu_instancing` decision DONE (2026-06-12):** explicitly unsupported + test-covered; optional use emits an adapter warning and imports the base mesh once, required use remains a hard required-extension rejection, and compatibility reports the extension as unsupported with the node source path.

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
| Point/spot DI | `shade.wgsl.ts` `lo_analyticNEE`, `analytic_lights` binding 13; `restir/emitterHelpers.ts`; `pipeline/BvhBufferHost.ts`; `HybridEngineDdgiSync.ts` | ✅ CODE CLOSED: point/spot emitters pack into the binding-13 analytic texture payload as `Le = color × intensity` with spot direction/cone metadata; `syncDdgiFromCoreScene()` refreshes `pipeline.updateAnalyticLights(scene)` on init and emitter updates before the scene bind group is consumed. | Regressions: `bvhBufferHost.test.ts` pins point/spot upload payload, and `HybridEngineDdgiSync.test.ts` pins the lifecycle/helper bridge that refreshes binding 13. |
| Mesh-area `color`/`intensity` | `restir/bvhCore.ts`; `restir/__tests__/directLightEmitterCore.test.ts` | ✅ CODE CLOSED: mesh-area emitters now override the referenced primitive material slot with `Le = emitter.color × emitter.intensity` before ReSTIR emitter-list packing, and the same override feeds the TLAS/merged emissive-Le glow buffer. | Keep the existing H23 regression: `color=[10,10,10]`, `intensity=10` produces `[100,100,100]` and total emissive power follows the override. |
| Emitter `castShadow` | DDGI/ReSTIR/RC direct-light paths | ✅ CODE CLOSED / ledger native: analytic point/spot payload packs binding-13 lane `[13]`; shared `EmitterTri` packs lane `[19]`; ReSTIR-DI candidate visibility + shade visibility gate on `e.castShadowDisabled`; DDGI/RC area-emitter NEE skip the emitter shadow ray; DDGI and RC fixture/sun lights carry castShadow-disabled flags; main direct sun gates its visibility ray from the scene directional emitter flag. | Regressions: `directLightEmitterCore`, `probeUpdateLights`, `rcLightsLayoutPin`, `emitterCastShadowWgsl`, `rcLightEvalWgsl`, `hybridEngineFrameOrchestrator`, and `engineContract`. |
| `primitiveCastShadow` GI-side | DDGI, ReSTIR-GI, RC | ✅ CODE CLOSED / ledger native | Shared material flag bit 1 + predicate-backed shared-BVH traversal now skip `castShadow:false` geometry in DDGI probe direct-light visibility and RC probe direct-light visibility; ReSTIR-GI + GRIS visibility use `traceSceneAnyCastMask` through `bvh_material` bit 0. Tests: `materialEntry`, `bvhCastShadowMask`, `ddgiMaterialPack`, `emitterCastShadowWgsl`, `rcLightEvalWgsl`, `engineContract`. |
| `updateLighting` sun | `HybridEngine.ts`; `HybridEngineDdgiSync.ts` | ✅ SOURCE-VERIFIED STALE: `updateLighting({ primaryLightDir })` re-syncs DDGI sun lights through `_syncDdgiLightsFromCoreScene()` or `orientDdgiSunLights(...)`; `primaryLightIntensity` also updates the DDGI sun multiplier with scene-directional single-count handling. | Keep mutation-matrix coverage |
| `procedural-sky` | `resolveHybridEnvironment.ts` | Scalar approx | Either bake Preetham to probe rays or keep `approximate` + planner never recommends WA for procedural-sky assets |
| RC sun RGB | `HybridEngineFrameOrchestrator.ts`; `hybridEngineFrameOrchestrator.test.ts` | ✅ CODE CLOSED: RC uses the scene directional emitter's RGB × intensity when present, with the legacy grey `primaryLightIntensity` fallback only when no scene directional exists. | Keep regression test with RC enabled |

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
| H34 BVH degenerates | `buildArrayBvh.ts`, `tlas.ts`; `nanTriangleFilter.test.ts`, `oobIndexFilter.test.ts`, `tlas.test.ts` | ✅ CODE CLOSED: BLAS already filters non-finite and out-of-range-index triangles with warnings and returns the empty-BVH shape when every triangle is invalid; this wave added TLAS parity by filtering non-finite build instances (AABB or `worldToLocal`) with warnings, throwing clearly when no finite instance remains, and rejecting non-finite/inverted refit AABBs before mutating node bounds. |
| Phantom emitter H22 | `walkaround-hybrid/src/restir/emitterList.ts:297-369`; `walkaround-hybrid/__tests__/hRemediationItems.test.ts` | ✅ CODE CLOSED: the zero-real-emitter placeholder remains only as the required non-empty WebGPU storage binding, but it is gated inert (`Le=[0,0,0]`, `intensity=0`, `power=0`, `totalEmissivePower=0`). The CDF fallback writes finite `[1]` instead of NaN, and the regression test pins both the inert payload and finite-CDF path. |
| H24 SceneBvh equal-length edits | `shared-bvh/src/sceneBvh.ts`; `sceneBvhVersionTag.test.ts` | ✅ CODE CLOSED: untagged `SceneBvh.updateFromCore()` uses exact buffer fingerprints, and the test pins equal-length vertex edits plus large unsampled-byte edits. |
| H24 material resolver | `restir/bvhCore.ts`; `restir/__tests__/bvhCoreMaterialResolver.test.ts` | ✅ CODE CLOSED: duplicate mesh-like primitive ids now throw before TLAS packing can reuse the first material slot, and unknown resolver calls throw instead of falling back to material 0. |
| H24 GPU skinning host-resource skip | `skin/GpuSkinningSubsystem.ts`; `gpuSkinningBindRouting.test.ts` | ✅ CODE CLOSED: missing GPU skinning position/normal buffers or mesh ranges now CPU-skin every skinned mesh instead of silently skipping. Count-only-cache risk is source-verified stale because cached bind groups key on live shared buffer identity. |
| H24 NRC training diagnostics | `pipeline/WalkaroundGPUPipeline.ts`; `pipeline/__tests__/nrcTrainingDiagnostics.test.ts` | ✅ CODE CLOSED: `trainFromRecords()` rejections emit deduped non-fatal `EngineError`s through the engine error channel while live, and remain suppressed after dispose. |
| H24 denoiser state consumers | `WalkaroundGPUPipeline.getActiveDenoiserState`; `HybridEngineFrameOrchestrator.ts`; `frameStatsDenoiserState.test.ts` | ✅ SOURCE-VERIFIED STALE: active denoiser state, including OIDN failed/retryable states, is emitted through `FrameStats.denoiserState`. |
| H24 RC sun RGB | `HybridEngineFrameOrchestrator.ts`; `hybridEngineFrameOrchestrator.test.ts` | ✅ CODE CLOSED: RC sun dispatch now preserves scene directional chroma and intensity; scalar grey is only the no-scene-directional fallback. |
| ReGIR dead alloc H24 | `pipeline/ReGIRCoordinator.ts`; `pipeline/BvhBufferHost.ts`; `pipeline/resourceManager.ts`; `pipeline/pipelineCompiler.ts`; `regirWiring.test.ts` | ✅ CODE CLOSED: ReGIR is fully opt-in. `gridRegionBytes()` returns `0` when `regirConfig.enabled` is false, `BvhBufferHost` pads the combined light-tree buffer only by that byte count, `compilePipelines()` only builds `regir-build` when `regirEnabled` is true, and the pass still gates on `coord.live`. Regression tests pin the disabled path as an unpadded light-tree upload and the enabled path as exactly `cells × survivors × REGIR_FLOATS_PER_SURVIVOR × 4` bytes of appended grid storage. |
| DDGI error swallow | `DDGI.ts:303-346` | ✅ CODE CLOSED: DDGI init/BVH/probe-frame failures now emit non-fatal `EngineError` diagnostics through `HybridEngine.onError`; failed probe frames do not advance the grid to `ready`. Focused tests pin direct DDGI reporting plus HybridEngine forwarding. |

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
| ~~`createEngine` accepts `gltfAsset?: GltfAssetResult`~~ ✅ DONE | `createEngineInternals.ts` exposes a structural `gltfAsset` hint; `createEngine.ts` passes `gltfAsset.recommendedBackend.backend` into `pickBackend`; `@vitrum/engine/gltf` supplies the loaded asset automatically. | Structural type avoids importing adapter runtime code into the core create path |
| ~~Replace triangle-only auto~~ ✅ DONE for glTF assets | `createEngineScale.ts` `pickBackend` uses the glTF recommendation when `prefer:'auto'`; explicit `prefer` still wins, and WebGL-only hosts fall back to `pt-webgl2` for WebGPU recommendations. Test: `createEngineScale.test.ts`. | Non-glTF scenes still use the 500k-triangle heuristic |
| ~~`VitrumCanvas` `gltf` prop~~ ✅ DONE | `VitrumCanvas.tsx` now accepts `gltf` + `gltfOptions`, loads via `loadGltfAsset`, forwards the imported scene to `attachVitrum`, passes the `gltfAsset` recommendation through the lifecycle into `createEngine`, and recreates on `gltf` / `gltfOptions` identity changes. Tests: `vitrumCanvasMount.test.tsx`, `attachVitrumLoop.test.ts`. | Direct `scene` remains supported; `gltf` is the creation-time alternative. |
| ~~`ProgressiveHandoffCoordinator` + glTF~~ ✅ DONE | `progressiveHandoff.ts`, `createProgressiveEngine.ts`, `progressiveHandoff.test.ts` | Structural `controller` option advances once per `frame()` (default 1/60s or host delta callback) and receives a synthetic patch target that forwards `setScene` / `updatePrimitive` to both engines through the coordinator's existing scene-authority/reset path. Empty-animation glTF controllers are skipped safely; `createProgressiveEngine` forwards the controller options. |
| Shared-device handoff | `createProgressiveEngine.ts` | Textures must be `GPUTexture` compatible — decode to GPU on WebGPU path, not CPU-only handles |
| ~~Examples~~ ✅ DONE | `examples/gltf-viewer/` | Self-contained Vite app now exercises `loadGltfWithEngine()`, backend recommendation, `textureDecodeReport`, controller attachment, and the capture protocol. |

#### 4B — Compatibility enforcement

| Mode | When to throw |
|------|----------------|
| `best-effort` | Never; warnings in `GltfAssetResult.warnings` + `Engine.onWarning` |
| `reject-unsupported` | Any used field `unsupported` on selected backend |
| `reject-degraded` | Any non-`native` issue including `approximate`, `requires-hook` without hook |

✅ **DONE (2026-06-12):** `engineBridge.ts` strict modes now reject selected-backend unsupported primitive modes through the same compatibility issue stream as material/extension rows, and the thrown diagnostic preserves the source path (`primitive:mode:1=unsupported at meshes[0].primitives[0].mode`). Test: `gltfAssetApi.test.ts` rejects a point/line-mode asset before invoking the injected engine factory.

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
| ~~`EXT_mesh_gpu_instancing`~~ ✅ policy closed | Not imported by design today; optional node-level use warns and imports the base mesh once; compatibility reports unsupported with node source path; required use throws via `extensionsRequired` | Native import to `instanced-mesh` remains future work, but arbitrary glTF now gets explicit rejection/degradation instead of silent loss |
| `KHR_texture_basisu` | Host hook required when selected, required, or no base `texture.source` fallback; optional fallback assets load without degraded rejection | Default browser transcoder path + docs |
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
| ~~H57~~ ✅ CLOSED | `examples/gltf-viewer/` added; `examples/README.md` lists the glTF path and debug capture fields. |

#### 5E — Behavioral gate expansion

Add glTF fixtures to behavioral gate configs (currently 29/29): at minimum unlit, textured PBR, transmission glass, skinned animated, Draco (with mock decoder).

---

### Master checklist: 65 material fields × walkaround path to ledger truth

| Category | Fields | Walkaround work |
|----------|--------|-----------------|
| Scalars consumed | baseColor, roughness, metallic, emissive*, transmission, ior, attenuation*, thickness, shadingModel, extensions | `shadingModel` verified `approximate`; mesh-area Le override and DDGI material-emissive direct probe hits closed; remaining scalar work belongs to atlas/lobe parity rows |
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
| Engine `onError` shape unification | `createEngine` / `Engine.onError` / `attachVitrum.onEngineError` / `createProgressiveEngine.onError` | ✅ DONE (Wave 7, construction-event half): `createProgressiveEngine.onError` now mirrors `CreateEngineOptions.onError(error, CreateEngineErrorEvent)` instead of discarding the phase/backend/recoverability event, and its own final canvas-configure failure reports `{ phase:'canvas-configure', backend:'walkaround-hybrid', recoverable:true }`. Runtime GPU errors remain intentionally on the core `Engine.onError(EngineError)` channel; `attachVitrum.onEngineError` is the lifecycle alias for that runtime channel. Focused progressive facade tests pin both event-forwarding paths. |
| `attachVitrum` auto-recreate scene loss | `packages/engine/src/lifecycle/vanilla.ts`; `packages/engine/src/__tests__/attachVitrumAutoRecreate.test.ts` | ✅ DONE (Wave 1): lifecycle now tracks the latest scene submitted through the exposed engine handle and recreates with that scene after device/context loss. Regression test simulates fatal `device-lost` and verifies the second `createEngine` call receives the updated scene. |
| pt-webgpu trace-lite shader-gate mismatch | `packages/pt-webgpu/src/wgsl/pathTrace/causticLite.wgsl.ts`; `kernelLite.wgsl.ts`; `wgslContract.test.ts`; `wgslLiteContract.test.ts` | ✅ DONE (Wave 1): lite MNEE stub signature now matches the lite kernel material-extension call shape, and lite BSDF-environment reconnection receives the scalar clearcoat/sheen/iridescence fields it already evaluates. `npm run shader-gate` compiles `pt-webgpu/trace-lite`; contract tests pin stub/caller parity and the updated lite SHA/length. |
| Lite tier single-BLAS | `uploadSceneBuffers.ts` lite path | Now honestly labeled, but `mergeWorldSpaceFromCore` (already consumed by 2 backends) would make multi-primitive lite real — implement (preferred) or keep ledgered |
| RC exported-surface footguns | `cascadePyramid.ts`; `cascadeDispatch.ts`; `HybridEngineRC.ts`; `cascadeDimsOverride.test.ts`; `cascadeDispatchInvalidation.test.ts`; `rcMergedRefit.test.ts` | ✅ DONE (Waves 4-5): `validateCascadeDims()` rejects empty/malformed overrides, non-positive probes, non-square ray counts, broken 2× ray-grid steps, and invalid intervals before allocation/dispatch. `RCSubsystem.refitCascadeBounds()` invalidates dispatcher bindings when probe bounds change, so merge uniforms rebuild with fresh `probeOriginWorld`/`roomSize`; merged-instance refit test pins invalidation without dispatcher recreation. Raw `RCDispatcher.dispatchFrameRaw()` now self-invalidates cached bind groups when direct callers change `bvhMode`, buffer sets, env bindings, device, cascade output buffers, or cascade bounds; focused tests pin stable-frame reuse plus TLAS/bounds rebuilds. README/package-boundary docs are reconciled around the no-`/three` package surface. |
| RC direct-light glass visibility | `packages/walkaround-rc/src/wgsl/rcLightEval.wgsl.ts`; `packages/walkaround-rc/__tests__/rcLightEvalWgsl.test.ts`; `packages/walkaround-hybrid/src/HybridEngineRC.ts`; `packages/walkaround-hybrid/__tests__/rcMergedRefit.test.ts` | ✅ DONE (Wave 6): rect-area emitter NEE and point/spot fixture direct-light shadow rays now use `rcTraceAny(..., skipGlass=true)` instead of closest-hit occlusion, so transmissive geometry no longer fully blocks coarse RC direct light. Merged-mode RC now uploads the same canonical `bvhIndex.w` payload as the ReSTIR/TLAS path, so `trans4` glass filtering works outside TLAS mode too. Tests pin both the WGSL call sites and the merged-mode glass payload. |
| shared-bvh sampled fingerprint in correctness path | `bufferFingerprint.ts` + `sceneBvh.ts:131`; `bufferFingerprint.test.ts`; `sceneBvhVersionTag.test.ts` | ✅ DONE (Wave 3): sampled `fingerprintBuffer(s)` remains available for versioning/upload heuristics, but `SceneBvh.updateFromCore()` now uses exact `fingerprintBuffersExact()` for the rebuild-skip gate. Regression tests pin an unsampled interior-byte miss in the sampled helper and prove the large-scene no-tag path rebuilds instead of keeping stale buffers. |
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

### Active performance track

These are real SOTA/performance gaps, but they do not block the contract claim
that one predictable API can ingest and route arbitrary glTF assets. Promote
them to hard 100% blockers only if the Road definition is widened from
contract-complete to contract-complete plus SOTA throughput/convergence.

1. Low-discrepancy sampling (`LD-SAMPLING-01`): shared Owen-scrambled Sobol or
   PMJ02 tables, per-dimension assignment audit, blue-noise screen scrambling,
   pt-webgpu integration, and pt-webgl2 RANDOM_TYPE revival/replacement.
2. Compressed wide BVH traversal (`WBVH-01`): opt-in CWBVH-style builder,
   packed node layout, WGSL traversal, CPU brute-force oracle, backend
   capability flag, and binary-BVH fallback until parity/perf are proven.

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
LD-SAMPLING-01 can also run in parallel as a performance-quality track. WBVH-01
should wait until the current BVH/material mutation contracts are stable enough
that the wide layout can be proven against the binary-BVH oracle without moving
targets.

### Summary

- **Condensed to 5 phases:** land gates → glTF → PT → walkaround → orchestration → proof.
- **Specificity:** file-level plug-in points, decoder contracts, bind-group footguns, integrator audit matrix, texture atlas architecture.
- **Gap fill vs 85%:** texture decode bridge, EXT_mesh_gpu_instancing decision, animation×temporal GI, lite-tier rejection for fidelity, PTWG-MAT all paths, walkaround alpha/blending, examples/gltf-viewer, render-based glTF sweep (not analyze-only), `pickBackend` fix, double-sided/vertex-color, tangent generation at import, engine `gltfAsset` passthrough, documentation sync as part of done.
- **Performance work preserved:** low-discrepancy sampling and compressed wide
  BVH traversal are tracked as active SOTA/performance work, but not as
  arbitrary-glTF API contract blockers. Shipped denoiser weights remain
  post-100/provisioning work.

Walkaround **100%** and arbitrary glTF **100%** are not the same: arbitrary glTF routes rich assets to PT backends via the planner; walkaround 100% still means permanent `unsupported` for spectral/displacement with explicit rejection, not silent gray materials.

---

## Forward-looking — the post-100% SOTA wave (ADDED 2026-06-12)

> Phases 0–6 above deliver **contract-complete**. This section is the separate
> axis: convergence/throughput engineering where vitrum is below current SOTA
> practice even after the campaign closes. Tracked here per roadmap §0.5.
> **2026-06-12 scope update:** F1 low-discrepancy sampling and F2 compressed
> wide BVH traversal are tracked as an active performance track above, but they
> remain separate from the arbitrary-glTF API contract. F3 shipped denoiser
> weights stays post-100/provisioning work for now.
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
