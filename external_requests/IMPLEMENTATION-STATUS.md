# external_requests/ Implementation Status — 2026-05-09

## RFE-07 Sprint 7 Volume Scattering: APPLIED

- Fork commit hash: `260c432`
- Files changed in fork:
  - `src/shader/bsdf/volume_march.glsl.js` (NEW) — `sampleExponential`, `equiAngularPdf`, `hg_phase`, `sampleHG_glsl`, `volumeMarch`
  - `src/shader/bsdf/bsdf_functions.glsl.js` — `TRANSLUCENT_BIT` constant; `sssSample()` SSS function
  - `src/shader/bsdf/index.js` — exports `volume_march`
  - `src/materials/pathtracing/PhysicalPathTracingMaterial.js` — Sprint 7 uniforms (`u_volumeDensity`, `u_scatterAlbedo`, `u_anisotropyG`, `u_sssSigmaT`, `u_sssAlbedo`, `u_sssAnisotropyG`); main loop volume scatter event block
- Gaps:
  - `TRANSLUCENT_BIT` is defined but not wired into `MaterialsTexture.js` packing (the material struct doesn't yet expose a flags uint per glass type). TODO in `bsdf_functions.glsl.js` at the constant definition. Glass-type flag mapping (`opalescent`, `glueChip`, `ringMottled`) requires a follow-up `MaterialsTexture.js` extension.

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
  - Ray payload restructure (`vec3 throughput` → `float wavelength + float throughput`) — pervasive, ~3 days, deferred
  - Main loop spectral accumulation (depends on payload restructure)
  - BSDF hero-wavelength IOR switchover from 3-channel to continuous Cauchy (depends on payload restructure)
  - Thin-film stack TMM evaluation (35-layer TiO₂/SiO₂, not started — too complex for session without GPU verification)
  - Spectral attenuation Beer-Lambert RFE-01 (not started — depends on payload restructure)

## Residual risks

GPU verification was not available; shader correctness is unverified beyond syntactic
compile (rollup bundles JS strings, not compiled GLSL). Hosts running the fork should
A/B verify against pre-patch reference renders before shipping.

Specific risks:
- `TRANSLUCENT_BIT` (Sprint 7) is not yet wired into material packing. SSS will not
  activate on opalescent/glueChip/ringMottled until `MaterialsTexture.js` is extended
  to pack the bit into sample 14. The `u_sssSigmaT` uniform path works for any material
  where the host sets that uniform directly.
- Sprint 12 `sampleHeroWavelength` GLSL uses a fixed-iteration binary search (7
  iterations, covers 128 > 82 entries). This is correct but if WebGL rejects the
  loop with a non-constant bound, the loop bound `7` may need to be a `#define`.
- Float32Array uniform upload for `uCmfX[81]` etc.: Three.js MaterialBase handles
  array uniforms via `setValues`; verify the uniform binding actually sets all 81
  entries in the target WebGL implementation.

## Vitrum library impact

None — fork patches landed in fork only. Vitrum library types and code are unchanged.
Tests: 542 pass (all workspaces, `npm test --workspaces --if-present`).
TypeScript: `npx tsc --noEmit -p .` — clean, no errors.
