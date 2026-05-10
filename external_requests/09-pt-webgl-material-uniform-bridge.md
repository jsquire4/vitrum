# RFE-09 — pt-webgl Material → Fork Uniform Bridge

**Date:** 2026-05-09
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Status:** APPLIED (runtime-unverified). The bridge/fork split is now explicit:
- `userData.vitrum*` stamps are packed per material in fork `MaterialsTexture`.
- BSDF/SSS/dispersion/thin-film/layered paths read per-material payload values.
- `@vitrum/pt-webgl` host bridge uploads global CMF/CDF spectral tables on scene set and does not override per-material scalar uniforms.

---

## What this request is for

`packages/pt-webgl/src/sceneToThree.ts` already stamps `userData.vitrum*` fields on each
`THREE.MeshPhysicalMaterial` it builds from a `vitrum.Material` (RFE-06/07/08 data carried as
`vitrumDispersionAbbeNumber`, `vitrumScatteringCoefficient`, `vitrumScatteringAnisotropy`,
`vitrumSpectralAttenuation`, `vitrumThinFilmStack`, `vitrumFrontLayer`, `vitrumBackLayer`).

The fork's `PhysicalPathTracingMaterial` carries matching uniforms for Sprints 7, 8, and 12
(`u_volumeDensity`, `u_anisotropyG`, `u_sssSigmaT`, `u_ior0`, `u_dispersionStrength`,
`u_jakobCoeffs`, `iorCauchyA/B/C`, `uCmfX/Y/Z`). These uniforms are never driven from the
`vitrum.Material` fields. The data flow has a gap between the `userData` stamp and the
uniform upload.

Without this bridge, even the APPLIED Sprint 7 and Sprint 8 fork patches produce no visual
difference: the uniforms remain at their zero-default values regardless of what the host
material specifies. End-to-end usage of RFE-06, RFE-07, and (once RFE-13 lands) RFE-08
is blocked until this bridge is in place.

---

## Affected files

- `packages/pt-webgl/src/sceneToThree.ts` — stamps `userData.vitrum*` on THREE materials but
  does not upload corresponding fork uniforms. The `vitrumMaterialToThree()` function at
  line 65 is the stamping site.
- `packages/pt-webgl/src/index.ts` — the `PTEngineWebGL2` engine drives `WebGLPathTracer` but
  performs no per-material uniform extraction from the constructed THREE scene.
- Fork: `src/materials/pathtracing/PhysicalPathTracingMaterial.js` — uniforms
  `u_volumeDensity`, `u_scatterAlbedo`, `u_anisotropyG`, `u_sssSigmaT`, `u_sssAlbedo`,
  `u_sssAnisotropyG`, `u_ior0`, `u_dispersionStrength`, `u_jakobCoeffs`,
  `iorCauchyA/B/C`, `uCmfX[81]`, `uCmfY[81]`, `uCmfZ[81]`, `uYCmfCdf[82]`,
  `uYCmfIntegral` are defined and wired in the shader but receive no host-driven values.

---

## What the bridge must do

For each mesh in the THREE scene, after `vitrumMaterialToThree()` has created the material:

1. **RFE-07 (volume scattering):** read `ud.vitrumScatteringCoefficient` → set
   `pathTracer.material.u_volumeDensity`. Read `ud.vitrumScatteringAnisotropy` → set
   `u_anisotropyG`. Read `ud.vitrumScatteringCoefficientRGB` (optional) → set
   `u_scatterAlbedo`. Read `ud.vitrumScatteringCoefficient` as σ_t → set `u_sssSigmaT`.

2. **RFE-06 (dispersion):** read `ud.vitrumDispersionAbbeNumber` V_d → convert to
   Cauchy B coefficient → set `u_dispersionStrength = (n_D − 1) / V_d × K` (see
   `plan/sprint-8-pt-fork-patch.md §1` for the K scaling factor). Set `u_ior0` from
   material IOR. Compute `jakobCoeffs` via `@vitrum/shared-samplers/jakobHanika.ts` →
   set `u_jakobCoeffs`.

3. **RFE-08 / Sprint 12 (spectral):** on engine init, upload CIE CMF tables from
   `@vitrum/shared-samplers/cieCmf.ts` → set `uCmfX[81]`, `uCmfY[81]`, `uCmfZ[81]`,
   `uYCmfCdf[82]`, `uYCmfIntegral`. Per-material: read `ud.vitrumSpectralAttenuation` →
   prepare Beer-Lambert upload when the payload restructure (RFE-13) lands.
   Read `iorCauchyA/B/C` from `@vitrum/shared-samplers/cauchyIor.ts` coefficients
   corresponding to material glass type.

The exact upload mechanism (per-sample vs. per-material uniform rebind) is vitrum's authors'
call. The acceptance criteria below are outcome-focused, not API-prescriptive.

---

## Acceptance criteria

- [ ] For a baked opalescent material with `scatteringCoefficient=2.5`, a rendered scene
  shows the milky SSS glow (visual A/B vs. the current flat output confirms RFE-07 is
  end-to-end active, not just shader-compiled).
- [ ] For a bevel material with `dispersionAbbeNumber=30`, a rendered bevel shows
  perceptible chromatic splitting (3-colour prismatic fan) — Sprint 8 dispersion is
  end-to-end active.
- [ ] Non-scattering, non-dispersive glass types (cathedral, antique) are visually
  unchanged — the bridge applies zero values where `vitrum.Material` fields are absent.
- [ ] CMF uniform arrays are uploaded on engine init without a host-side initialisation
  call — the bridge is not caller-opt-in.
- [ ] `npm test --workspaces --if-present` passes without regression.

---

## References

- `external_requests/06-sprint8-spectral-dispersion-fork-patch.md` — Sprint 8 fork patch
  (APPLIED as of fork commit `7ffd15d`); this RFE is the host-side driver for it.
- `external_requests/07-sprint7-volume-scattering-fork-patch.md` — Sprint 7 fork patch
  (APPLIED as of fork commit `260c432`); this RFE is the host-side driver for it.
- `external_requests/08-sprint12-spectral-accumulator-fork-patch.md` — Sprint 12 (PARTIAL).
- `external_requests/13-fork-sprint12-ray-payload-restructure.md` — prerequisite for full
  spectral uniform upload (Beer-Lambert per-λ).
- `IMPLEMENTATION-STATUS.md` §Vitrum library impact: "None — fork patches landed in fork
  only. Vitrum library types and code are unchanged."

---

## Out-of-scope notes

- This RFE does NOT cover the fork-side shader changes (those are RFE-06/07/08/13/14).
- It does NOT cover the `three-bindings` userData-reading path (that is RFE-10 and is
  already implemented for the `three-bindings → vitrum.Material` direction).
- It does NOT cover per-face layered BSDF upload (that is RFE-03 / RFE-12).
- The Cauchy coefficient conversion math is specified in `plan/sprint-8-pt-fork-patch.md §1`
  and in `@vitrum/shared-samplers/src/cauchyIor.ts`; this RFE does not re-specify it.

---

## Consumer-side state

The stainedGlass app bakes all RFE fields into `MeshPhysicalMaterial.userData` via
`packages/stained-glass-physics/src/baking/` (Phase 2b Tier A, commit `34a54e6`). The adapter
`packages/stained-glass-physics/src/baking/vitrumMaterialAdapter.ts` converts baked THREE
materials to a `LocalMaterial` with all fields populated. The data is ready; only the
uniform upload path is missing.
