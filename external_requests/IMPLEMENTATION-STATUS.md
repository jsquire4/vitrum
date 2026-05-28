# external_requests/ Implementation Status — 2026-05-10

## Scope note

This file covers RFEs 06–14 (fork shader patches, 2026-05-10). For RFEs 01–05
(@vitrum/core contract additions), see
[../plan/archive/external-requests-status.md](../plan/archive/external-requests-status.md).

## 2026-05-10 closure update

- Fork Sprint 12 Gap §5 (spectral attenuation Beer-Lambert) **uses packed per-material spectral μ(λ)** from `MaterialsTexture` texels 20–27 inside `transmissionAttenuationHero` (hero-wavelength exp(−μ·d)); RGB fallback remains when no spectral curve is packed.
- Fork **direct-light / `bsdfResult`** now threads **`state.wavelength`** into `bsdfEval` (removed hard-coded 550 nm).
- Fork **`transmissionEval`** PDF now follows the **Walter et al. EGSR07** microfacet BTDF Jacobian form (`ggxPDF(wo,wh) / (η wi·wh + wo·wh)²`), replacing the incorrect Fresnel-only stub.
- pt-webgpu **material packing stride** is now **22 vec4s/material** with **thin-film layer triplets** `(ior, thicknessNm, extinctionCoefficient)`, stack **`incidentIor`** + **`angleDependent`**, and **bounded multi-light storage buffers** (`point` / `spot` / `rect-area` / `mesh-area`) bound at `@binding(20–23)` with counts in `FrameParams`.
- Vitrum **`npm run fork-shader-smoke`** invokes the fork `scripts/shader-smoke-check.js`; Playwright capture adapter appends **`vitrumScenario` / `vitrumSeed` / …** query params for host pages.
- Final GPU render/perf validation remains pending and tracked in `plan/archive/gap-closure-verification-2026-05-10.md`.

## Sprint 4 BSDF Cost Reduction: APPLIED (2026-05-12)

- Fork commit: `2640b75`
- Date: 2026-05-12
- Files changed in fork:
  - `src/shader/structs/surface_record_struct.glsl.js` — added `uint lobeMask` (P1) and `bool liteMode` (P2) fields to `SurfaceRecord`
  - `src/materials/pathtracing/glsl/get_surface_record_function.glsl.js` — added `uniform int materialLodDepth` (P3); `int pathDepth` parameter; `bool useTextures` guard on all 16 texture fetches (including TBN normal map path); lobeMask bitfield computed from material thresholds; `liteMode = (pathDepth > 1)`; fog particle fast-path sets `lobeMask=1/liteMode=false`
  - `src/shader/bsdf/bsdf_functions.glsl.js` — sheen and clearcoat lobes in `bsdfEval` gated on `lobeMask` bits 2/3 and `!surf.liteMode` (P1+P2); iridescence `evalIridescence` call in `specularEval` gated on lobeMask bit 4 and `!surf.liteMode` (P1+P2)
  - `src/materials/pathtracing/PhysicalPathTracingMaterial.js` — `materialLodDepth: { value: 2 }` uniform added; `getSurfaceRecord` call site passes `int(state.depth)` as `pathDepth`
- Adaptations from spec:
  - Spec referenced `forceFullBSDF` material flag — deferred. The `liteMode` mechanism is in place; the per-material override can be added via a future Material struct bit when GPU profiling reveals an opal/glueChip scene degradation.
  - The `materialLodDepth` uniform is declared in the GLSL function file (not as a separate fragment-shader `uniform` line) since `get_surface_record_function` is assembled into the shader before `main()`.
- Gaps:
  - GPU profiling of ms/sample reduction pending (DoD target: ≥40% on glass-and-came scene). Runtime verification required.
  - `forceFullBSDF` per-material override not yet implemented. Add when opal scene A/B shows degradation.

## Sprint 6 Rough Refraction GGX Lobe: APPLIED (2026-05-12)

- Fork commit: `e9c7516`
- Date: 2026-05-12
- Files changed in fork:
  - `src/shader/bsdf/bsdf_functions.glsl.js` — added `perturbDirectionByGGX(refractDir, roughness, u1, u2)` helper (Heitz 2014 visible NDF); called after `refract()` in both `transmissionDirection()` and `dispersionTransmissionDirection()` via `rand2(47)`. Roughness < 1e-4 fast-paths to specular (no perturbation). Composition with Phase 4: Phase 4 modifies N before `refract`; Sprint 6 perturbs refDir after — correct order.
- Adaptations from spec:
  - Spec showed a call site as `refDir = perturbDirectionByGGX(refDir, rough, rand(), rand())` using scalar `rand()` calls. The fork uses the established `rand2(seed)` pattern for paired uniform samples — `rand2(47)` (seed 47 is unused by any prior sprint). This is functionally equivalent and consistent with fork conventions.
  - `dispersionTransmissionDirection` was not mentioned in the spec call site but logically requires the same perturbation (it is the hero-wavelength dispersion path that calls into the same transmission branch). Perturbation added there too.
- Gaps:
  - GPU visual A/B on hammered/seedy glass pending (no GPU access in-session). Shader smoke: pass.
  - Firefly risk at roughness > 0.8 noted in spec risk section — existing `filteredGlossyFactor` clamp should cover it; runtime verification needed.

## Sprint 5 MRT G-buffer: APPLIED (2026-05-12)

- Fork commit: `49081a3`
- Date: 2026-05-12
- Files changed in fork:
  - `src/materials/pathtracing/PhysicalPathTracingMaterial.js` — MRT layout declarations (`layout(location=0..2) out vec4 gColor/gNormalDepth/gAlbedo`); renamed all `gl_FragColor` usages to `gColor`; G-buffer accumulator variables (`gbufWritten`, `gbufNormalEnc`, `gbufLinearDepth`, `gbufAlbedo`); primary-hit capture block at `state.firstRay` using camera world matrix + `surf.normal`/`surf.color`; G-buffer write at end of `main()` with sky fallbacks.
  - `scripts/shader-smoke-check.js` — radiance-clamp pattern updated to accept `gColor` in place of `gl_FragColor`.
  - `build/index.module.js`, `build/index.module.js.map`, `build/index.umd.cjs`, `build/index.umd.cjs.map` — rebuilt bundles.
- MRT layout applied (per spec Decision 12):
  - location 0: `gColor` RGBA16F — accumulated radiance (unchanged from prior single output)
  - location 1: `gNormalDepth` RGBA16F — world normal encoded [0,1] in `.rgb`; linear depth in `.w`
  - location 2: `gAlbedo` RGBA8 — demodulated base color (surf.color), no lighting
- Adaptations from spec:
  - Spec referenced host-side `WebGLMultipleRenderTargets` allocation — that is vitrum-side (`@vitrum/pt-webgl`), not fork-side; not changed here per scope.
  - Host-side format defaults (RGBA16F for gColor/gNormalDepth, RGBA8 for gAlbedo) documented in spec; fragment shader outputs are format-agnostic.
- Gaps / H6.3 hit list (earlier-sprint paths affected by `gl_FragColor → gColor` rename):
  - Sprint 7 volume scatter: writes `gColor.rgb` accumulations — structurally unchanged, compiles correctly.
  - Sprint 8 dispersion: routes through `gColor.rgb` — structurally unchanged, compiles correctly.
  - Sprint 12 spectral: `wavelengthToRGB`/`throughputRgb` pipe into `gColor.rgb` — unchanged.
  - Sprint 14 layered BSDF: `activeLayerWeight` call in main loop affects `gColor.rgb` indirectly — unchanged.
  - None of the above sprints write to gNormalDepth/gAlbedo, so no regressions in their output paths.
  - GPU visual A/B (no-MRT host vs MRT host gColor agreement) pending as part of H6.3 verification.

## Sprint 10c BDPT Fork Dispatch: APPLIED (2026-05-12)

- **Fork commit**: `98f4446` — feat(sprint-10c): BDPT integrator — light-subpath ping-pong + eye↔light connections + Veach §10.3 MIS
- **Vitrum commit**: `398dfce` — feat(pt-webgl): Sprint 10c vitrum-side BDPT bridge
- **Blocker resolved**: Sprints 4 (`2640b75`), 5 (`49081a3`), 6 (`e9c7516`) are all applied; Sprint 5 MRT prerequisite met.

### Fork files changed

- `src/materials/pathtracing/glsl/bdpt_light_subpath.glsl.js` (NEW)
  `writeLightSubpathVertex()` — traces up to 3 bounces from emitter surface using
  cosine-weighted hemisphere scatter. Writes position/normal/pdfFwd/throughput/pdfRev
  into 3×3 RGBA32F ping-pong texture (one column per bounce). Geometry term
  `G(x↔y) = |cosθ_x · cosθ_y| / ‖x−y‖²` (Veach §8.3.2). Invalid vertices flagged
  kind=3 (BDPT_KIND_INVALID). Specular surfaces (transmission>0.5 + roughness<0.05)
  skipped (Veach §10.3.5 zero-weight rule).

- `src/materials/pathtracing/glsl/bdpt_connection.glsl.js` (NEW)
  `evaluateBdptConnection()` — shadow ray via existing `attenuateHit()`, BSDF eval
  at eye vertex (`bsdfResult`), Lambertian approximation at light vertex, geometric
  term, 2-strategy power-heuristic MIS weight (w = p_s² / (p_s² + p_alt²), β=2 —
  GLSL port of `bdptConnectionMIS_full` from `@vitrum/shared-samplers`). NaN/Inf
  guard + firefly clamp at 100 per component.

- `src/materials/pathtracing/glsl/index.js`
  Exports `bdpt_connection` + `bdpt_light_subpath` (alphabetical).

- `src/materials/pathtracing/PhysicalPathTracingMaterial.js`
  `FEATURE_BDPT` define (default 0; synced from `uBdptEnabled` in `onBeforeRender`).
  New uniforms: `uBdptEnabled` (bool), `uBdptMaxLightBounces` (int=3),
  `uBdptLightPathTex` (sampler2D). GLSL includes compiled under `#if FEATURE_BDPT`.
  Main loop connection call at each indirect bounce (depth>0, not firstRay).

### Vitrum files changed

- `packages/pt-webgl/src/forkUniformBridge.ts`
  `ForkBridgeBdptOptions` type; `driveForkMaterialUniforms()` gains bdptOptions arg.
  Safety guard: null lightPathTex forces enabled=false.

- `packages/pt-webgl/src/ptEngineWebGL2.ts`
  `#bdpt` and `#bdptMaxLightBounces` fields from extensions
  `'vitrum.ptWebgl.bdpt'` + `'vitrum.ptWebgl.bdptMaxLightBounces'`.

- `packages/pt-webgl/src/index.ts`
  Exports `ForkBridgeBdptOptions`, `ForkBridgeCausticOptions`, `driveForkMaterialUniforms`.

- `packages/pt-webgl/src/__tests__/forkUniformBridge.test.ts`
  3 new BDPT tests (40 total, all pass).

### Vertex storage approach

Texture ping-pong: RGBA32F, width=BDPT_MAX_LIGHT_BOUNCES=3, height=3 rows.
Host issues 3 draw calls (one per bounce) advancing `uBdptVertexCol` 0→2,
then unbinds framebuffer. Eye-ray accumulation pass reads via `texelFetch()`.
Stays within `MAX_DRAW_BUFFERS=8` limit (3 MRT attachments per draw call).

### MIS port faithfulness vs. H4 reference

- GLSL `bdptMISWeight2(p_s, p_alt)` = `p_s² / (p_s² + p_alt²)` — identical
  math to `bdptConnectionMIS_full` power heuristic with β=2 and 2 strategies.
- Full Veach §10.3 recursive ratio sweep (all k+1 strategies) deferred: requires
  pdfRev from full BSDF evaluation in the light kernel. Documented known gap.
- Specular zero-weight rule implemented (Veach §10.3.5).

### Known gaps / documented approximations

1. **pdfRev approximation**: light-subpath kernel uses cosine-hemisphere pdfRev
   (Lambertian). Full Veach requires re-evaluating after path completion. Visually
   acceptable for caustic convergence verification but biases variance estimate.
   Track: `plan/archive/phase-6-status.md`.
2. **Full strategy enumeration**: 2-strategy MIS approximation (connection vs.
   unidirectional) replaces k+1-strategy recursive sweep. Adequate for caustic
   acceleration; full sweep deferred to a future patch.
3. **GPU visual A/B** (DoD): caustic convergence 3–5× speedup requires GPU run
   with BDPT enabled vs. pure NEE. No GPU access in-session.
4. **Reference renders** (DoD): `tools/reference-renders/sprint-10c-before.png`
   and `sprint-10c-after.png` pending GPU run.

### Build + smoke status

`npm run build` clean. `npm run shader-smoke` passes. Vitrum `npm test` 664 pass,
3 skipped. `npm run typecheck` clean across workspace.

## RFE-07 Sprint 7 Volume Scattering: APPLIED

- Fork commit hash: `260c432`
- Files changed in fork:
  - `src/shader/bsdf/volume_march.glsl.js` (NEW) — `sampleExponential`, `equiAngularPdf`, `hg_phase`, `sampleHG_glsl`, `volumeMarch`
  - `src/shader/bsdf/bsdf_functions.glsl.js` — `TRANSLUCENT_BIT` constant; `sssSample()` SSS function
  - `src/shader/bsdf/index.js` — exports `volume_march`
  - `src/materials/pathtracing/PhysicalPathTracingMaterial.js` — Sprint 7 uniforms (`u_volumeDensity`, `u_scatterAlbedo`, `u_anisotropyG`, `u_sssSigmaT`, `u_sssAlbedo`, `u_sssAnisotropyG`); main loop volume scatter event block
- Gaps:
  - No known code gaps in the fork patch set for this RFE. Runtime validation is still pending (no GPU verification in-session).

## RFE-06 Sprint 8 Spectral Dispersion: APPLIED

- Fork commit hash: `7ffd15d`
- Files changed in fork:
  - `src/shader/bsdf/bsdf_functions.glsl.js` — `evalSpectrum(coeffs, lambda)`; `dispersionTransmissionDirection(wo, surf, channelMask)`; `bsdfSample` transmission branch gated on `u_dispersionStrength > 1e-4`
  - `src/materials/pathtracing/PhysicalPathTracingMaterial.js` — Sprint 8 uniforms (`u_ior0`, `u_dispersionStrength`, `u_jakobCoeffs`) committed in Sprint 7 commit
- Gaps: none material. The `u_dispersionStrength == 0` fast path is in place — non-bevel glass takes the existing `transmissionDirection` path unchanged.

## RFE-08 Sprint 12 Spectral Accumulator: COMPLETE (H6.6 / SG.D — 2026-05-12)

- Fork commit hash: `8917492` (kernel); per-material upload path was already wired in the
  RFE-09 / MaterialsTexture packing pass (no new fork commit needed).
- Files changed in fork (kernel + consumer, all prior commits):
  - `src/shader/bsdf/spectral_accumulator.glsl.js` (NEW) — `sampleCmfX/Y/Z`, `sampleHeroWavelength` (Y-CMF CDF binary search), `wavelengthToRGB` (XYZ→linear sRGB, Bradford D65 matrix)
  - `src/shader/bsdf/bsdf_functions.glsl.js` — `cauchyIORatLambda(lambdaNm, A, B, C)`; `evalSpectrumAtHero(lambdaNm)`
  - `src/shader/bsdf/index.js` — exports `spectral_accumulator`
  - `src/materials/pathtracing/PhysicalPathTracingMaterial.js` — Sprint 12 uniforms (`uCmfX[81]`, `uCmfY[81]`, `uCmfZ[81]`, `uYCmfCdf[82]`, `uYCmfIntegral`, `iorCauchyA/B/C`); `spectral_accumulator` GLSL block in fragment shader
  - `src/uniforms/MaterialsTexture.js` — per-material spectral grid at samples 20–27 (32 floats, 380–780 nm uniform grid); `hasSpectralAttenuation` feature flag at sample 17.a bit 0; thin-film stack (35 layers) at samples 28–54; all read from `userData.vitrumSpectralAttenuation` / `userData.vitrumThinFilmStack`.
  - `src/shader/common/util_functions.glsl.js` — `readSpectralAttenuationMu(materialsTex, materialIndex, spectralIdx)`; `spectralAttenuationMuHero(materialsTex, materialIndex, heroWavelength)`; `transmissionAttenuationHero(materialsTex, dist, attColor, attDist, hasSpectral, materialIndex, heroWavelength)` — routes to spectral μ(λ) exp(−μ·d) when hasSpectral, otherwise RGB Beer-Lambert + heroScalarFromRgb fallback.
  - `src/materials/pathtracing/glsl/attenuate_hit_function.glsl.js` — calls `transmissionAttenuationHero` with `material.hasSpectralAttenuation` + `materialIndex` at volume exit hit.
  - `src/shader/structs/material_struct.glsl.js` — `hasSpectralAttenuation` field decoded from featureFlags bit 0.
- Vitrum files changed (H6.6 completion):
  - `packages/pt-webgl/src/__tests__/materialsTextureSpectral.test.ts` (NEW) — 9 tests covering wire-format layout self-check, no-spectral fallback, constant-μ curve, ramp-curve linearity, multi-material stride correctness, scatteringCoefficient sample-15 packing, thinFilmStack layer-count + incidentIor + angleDependent, hasFrontLayer/hasBackLayer feature flags, and PBR field non-corruption invariant.
- All prior gaps fully resolved:
  - Ray payload restructure — APPLIED
  - Main loop spectral accumulation — APPLIED
  - BSDF hero-wavelength Cauchy IOR — APPLIED
  - Thin-film TMM — APPLIED
  - Per-material spectral Beer-Lambert (the H6.6 / SG.D payoff) — COMPLETE
- Remaining: GPU visual A/B (cobalt/iron/SeCd/goldRuby render with spectral vs. RGB Beer-Lambert). No GPU in-session; verified via unit tests + shader smoke.

## RFE-09 pt-webgl Material -> Fork Uniform Bridge: APPLIED (runtime-unverified)
<!-- NOTE: This "RFE-09" is the uniform-bridge sprint deliverable (not a standalone external_requests/ file).
     The file external_requests/09-runtime-lighting-updates.md is a separate, later-filed RFE (Status: Proposed)
     that proposes a runtime updatePrimaryLight/updateSkyDome/updateLights API. It is NOT implemented here. -->

- Files changed in vitrum + fork:
  - `packages/pt-webgl/src/forkUniformBridge.ts` (NEW) — scene material scan + uniform driving
    for CMF/CDF + spectral bridge tables.
  - `packages/pt-webgl/src/index.ts` — bridge call integrated on `setScene`.
  - `packages/pt-webgl/src/__tests__/forkUniformBridge.test.ts` (NEW) — bridge behavior coverage.
  - `packages/pt-webgl/package.json` — `@vitrum/shared-samplers` dependency added.
  - Fork `MaterialsTexture.js` now packs per-material scalar drives from `userData.vitrum*`
    (`scatteringCoefficient`, `scatteringAnisotropy`, `scatteringCoefficientRGB`,
    `dispersionAbbeNumber` -> derived dispersion strength) and shader BSDF paths consume them.
- Gaps:
  - GPU runtime visual validation pending.

## RFE-10 three-bindings userData Propagation: APPLIED

- Implementation: `packages/three-bindings/src/material.ts` (lines 93–157)
- Pre-existed at the time of RFE-10 filing — committed in `1036a8c`
  ("fix: wire vitrum.Material ↔ THREE userData.vitrum* round-trip").
- Reads and type-checks: `vitrumDispersionAbbeNumber`, `vitrumScatteringCoefficient`,
  `vitrumScatteringCoefficientRGB`, `vitrumScatteringAnisotropy`,
  `vitrumSpectralAttenuation` (SpectralCurve object + Float32Array fallback),
  `vitrumThinFilmStack`, `vitrumFrontLayer`, `vitrumBackLayer`.
- Round-trip tests: `packages/three-bindings/src/__tests__/material-vitrum-roundtrip.test.ts` (14 tests).
- Residual open items:
  - `frontLayer`/`backLayer` BSDF evaluation — deferred on RFE-12.
  - `vitrum.Material → fork uniforms` direction — completed in RFE-09.

## RFE-12 / RFE-03 Layered BSDF Fork Patch: APPLIED (2026-05-11)

- Fork commit: `ee379dc` — feat(layered-bsdf): Sprint 14 — wire activeLayerWeight into bsdfEval color path
- Prerequisite fork commit: `d6b88b3` — fix(adaptive): count proportional tile work for adaptive repeat sampling
- Files changed in fork:
  - `src/shader/bsdf/bsdf_functions.glsl.js` — added `color *= activeLayerWeight(surf, heroWavelength)` in `bsdfEval` after all lobe evaluations. The helper function and all supporting fields were already present from the earlier phase4-normalmap-shadow-rays merge.
  - `build/index.module.js`, `build/index.module.js.map`, `build/index.umd.cjs`, `build/index.umd.cjs.map` — rebuilt bundles.
- Infrastructure already on main (from phase4-normalmap-shadow-rays merge `49f1e4b`):
  - `src/shader/structs/material_struct.glsl.js` — `frontLayerTransmission`, `frontLayerRoughness`, `hasFrontLayer`, `backLayerTransmission`, `backLayerRoughness`, `hasBackLayer`; featureFlags unpacking at bits 2 and 4.
  - `src/uniforms/MaterialsTexture.js` — packs `userData.vitrumFrontLayer` / `userData.vitrumBackLayer` into material texels 18–19.
  - `src/shader/structs/surface_record_struct.glsl.js` — `activeLayerTransmission`, `activeLayerRoughness`, `hasActiveLayer`.
  - `src/materials/pathtracing/glsl/get_surface_record_function.glsl.js` — per-face layer selection by `frontFaceHit`; roughness override applied to `surf.filteredRoughness` before lobe evaluation.
  - `src/shader/bsdf/bsdf_functions.glsl.js` — `activeLayerWeight()` helper using `heroScalarFromRgb` for spectral path.
- Gaps:
  - GPU A/B visual verification pending (per spec gate condition).
  - `pt-webgl` vitrum-side bridge (`sceneToThree.ts`) should populate `userData.vitrumFrontLayer` / `userData.vitrumBackLayer` from `Material.frontLayer` / `Material.backLayer` — this is a separate vitrum-side follow-up (see note below).

## RFE-14 Thin-Film TMM Evaluator: APPLIED (runtime-unverified)

- Files changed in fork:
  - `src/shader/bsdf/thin_film_tmm.glsl.js` (NEW) — fixed-bound 35-layer TMM TE evaluator
    reading per-material stacks from `MaterialsTexture`.
  - `src/shader/bsdf/index.js` — exports `thin_film_tmm`.
  - `src/materials/pathtracing/PhysicalPathTracingMaterial.js` — thin-film uniforms + shader include.
  - `src/shader/bsdf/bsdf_functions.glsl.js` — thin-film modulation in specular/transmission eval.
  - `src/uniforms/MaterialsTexture.js` — per-material 35-layer thin-film payload packing.
  - `src/materials/pathtracing/glsl/get_surface_record_function.glsl.js` and
    `src/shader/structs/{material,surface_record}_struct.glsl.js` — material-index and layer-count wiring.
- Gaps:
  - GPU visual/perf verification pending.

## H6.3 — Sprint 5 MRT compatibility verification: COMPLETE (2026-05-12)

Fork commit: `ff1dace`

Verified that the H6.2 MRT rename (`gl_FragColor` → `gColor`) did not break Sprint
7/8/12/14 output paths. No forward-porting was required.

- **Sprint 7** (volume scatter + SSS, fork 260c432): `volumeMarch()` and `sssSample()`
  in `src/shader/bsdf/volume_march.glsl.js` / `bsdf_functions.glsl.js` return scalars
  or `ScatterRecord` — they never wrote to `gl_FragColor`. The scatter event updates
  `state.throughput` and the main loop accumulates via `gColor.rgb`. Clean.

- **Sprint 8** (Cauchy chromatic dispersion, fork 7ffd15d): `evalSpectrum()`,
  `cauchyIORatLambda()`, and `dispersionTransmissionDirection()` in
  `bsdf_functions.glsl.js` return scalars or direction vectors. Dispersion path
  exits via `ScatterRecord`; the main loop accumulates into `gColor.rgb`. Clean.

- **Sprint 12** (hero-wavelength spectral kernel, fork 8917492): `sampleHeroWavelength()`
  and `wavelengthToRGB()` in `src/shader/bsdf/spectral_accumulator.glsl.js` return
  float/vec3. Main loop calls `wavelengthToRGB(state.wavelength, state.throughput,
  state.wavelengthPdf)` → `throughputRgb` → `gColor.rgb`. Clean.

- **Sprint 14** (layered BSDF, fork ee379dc): `activeLayerWeight()` in
  `bsdf_functions.glsl.js` returns a float. `bsdfEval()` multiplies local `color`
  (vec3) by that scalar and returns; the caller accumulates into `gColor.rgb` through
  the standard path. Clean.

`npm run shader-smoke` passes at ff1dace. No `gl_FragColor` found in any sprint-owned
source file (`src/shader/bsdf/`, `src/shader/common/`, or
`src/materials/pathtracing/PhysicalPathTracingMaterial.js`).

## Residual risks

GPU verification was not available; shader correctness is unverified beyond syntactic
compile (rollup bundles JS strings, not compiled GLSL). Hosts running the fork should
A/B verify against pre-patch reference renders before shipping.

Specific risks:
- `TRANSLUCENT_BIT` packing and shader read-path are now wired, but per-material SSS
  behavior still requires scene-level visual verification in a mixed-material test scene.
- Sprint 12 `sampleHeroWavelength` GLSL uses a fixed-iteration binary search (7
  iterations, covers 128 > 82 entries). This is correct but if WebGL rejects the
  loop with a non-constant bound, the loop bound `7` may need to be a `#define`.
- Sprint 12 payload conversion code path is landed: `RenderState` and primary contribution
  paths now use scalar throughput + hero wavelength through BSDF sampling/eval paths, but
  final validation still requires GPU visual/perf A/B on representative scenes.
- Float32Array uniform upload for `uCmfX[81]` etc.: Three.js MaterialBase handles
  array uniforms via `setValues`; verify the uniform binding actually sets all 81
  entries in the target WebGL implementation.

## Vitrum library impact

Vitrum-side bridge and `pt-webgpu` changes are now included in addition to fork patches.
Tests: workspace tests pass via `npm test --workspaces --if-present`.
TypeScript: workspace typecheck is currently clean across touched packages (including
`@vitrum/pt-webgl` and `@vitrum/pt-webgpu`) in this execution wave.
