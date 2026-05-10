# external_requests/ Implementation Status — 2026-05-10

## 2026-05-10 closure update

- Sprint 12 Gap §5 (spectral attenuation Beer-Lambert) is now APPLIED in fork codepaths.
- RFE-03 layered BSDF implementation is landed in fork material packing + BSDF selection path.
- pt-webgpu now includes HDRI CPU-payload importance sampling and reciprocal environment MIS.
- pt-webgpu `causticStrategy` now maps to active `'manifold-nee'` and `'photon-map'` approximation paths (runtime-unverified).
- Final GPU render/perf validation remains pending and tracked in `plan/gap-closure-verification-2026-05-10.md`.

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

## RFE-08 Sprint 12 Spectral Accumulator: PARTIAL

- Fork commit hash: `8917492`
- Files changed in fork:
  - `src/shader/bsdf/spectral_accumulator.glsl.js` (NEW) — `sampleCmfX/Y/Z`, `sampleHeroWavelength` (Y-CMF CDF binary search), `wavelengthToRGB` (XYZ→linear sRGB, Bradford D65 matrix)
  - `src/shader/bsdf/bsdf_functions.glsl.js` — `cauchyIORatLambda(lambdaNm, A, B, C)`; `evalSpectrumAtHero(lambdaNm)` (both ready to connect once payload restructure lands)
  - `src/shader/bsdf/index.js` — exports `spectral_accumulator`
  - `src/materials/pathtracing/PhysicalPathTracingMaterial.js` — Sprint 12 uniforms (`uCmfX[81]`, `uCmfY[81]`, `uCmfZ[81]`, `uYCmfCdf[82]`, `uYCmfIntegral`, `iorCauchyA/B/C`); `spectral_accumulator` GLSL block in fragment shader
  - `SPRINT_12_GAPS.md` (NEW) — full gap documentation
- Gaps (see `SPRINT_12_GAPS.md`):
  - Ray payload restructure (`vec3 throughput` → `float wavelength + float throughput`) — APPLIED in shader payload + BSDF transport paths (runtime verification pending)
  - Main loop spectral accumulation — APPLIED (`sampleHeroWavelength` seeds payload; contribution sites derive `throughputRgb` via `wavelengthToRGB`)
  - BSDF hero-wavelength IOR switchover from 3-channel to continuous Cauchy — APPLIED for transmission branch (`dispersionTransmissionDirection` now uses `cauchyIORatLambda(heroWavelength, A, B, C)`)
  - Thin-film stack TMM evaluation (35-layer TiO₂/SiO₂) — APPLIED in fork; runtime visual/perf verification pending
  - Spectral attenuation Beer-Lambert RFE-01 — APPLIED (runtime visual/perf verification pending)

## RFE-09 pt-webgl Material -> Fork Uniform Bridge: APPLIED (runtime-unverified)

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
- Round-trip tests: `packages/three-bindings/src/__tests__/material-vitrum-roundtrip.test.ts` (17 tests).
- Residual open items:
  - `frontLayer`/`backLayer` BSDF evaluation — deferred on RFE-12.
  - `vitrum.Material → fork uniforms` direction — completed in RFE-09.

## RFE-12 Layered BSDF Fork-Patch Plan: APPLIED

- Files changed:
  - `plan/sprint-14-layered-bsdf-fork-patch.md` (NEW) — implementation plan for RFE-03.
- Gaps:
  - None for this meta-RFE (plan-only deliverable).

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
TypeScript: workspace typecheck has known pre-existing `@vitrum/pt-webgl` type mismatches;
`@vitrum/pt-webgpu` and other touched packages typecheck clean.
