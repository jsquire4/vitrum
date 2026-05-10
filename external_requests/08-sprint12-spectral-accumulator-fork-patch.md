# RFE-08 — Sprint 12 Hero-Wavelength Spectral Accumulator Fork Patch

**Date:** 2026-05-09
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Status:** APPLIED IN LOCAL FORK (runtime verification pending).

---

## What this request is for

Apply the Sprint 12 fork-side kernel rewrite from `plan/sprint-12-pt-fork-patch.md` to
`~/projects/three-gpu-pathtracer/`. This replaces the RGB-as-3λ approximation (Sprint 8)
with full hero-wavelength spectral path tracing and CIE CMF-based spectral accumulation.

---

## Why it is needed (and why it is gated)

The stainedGlass app stamps `userData.vitrumThinFilmStack` on dichroic glass materials
(35-layer TiO₂/SiO₂ quarter-wave stack, `thinFilmStacks.ts`). TMM evaluation of a
multi-layer thin-film stack produces a wavelength-dependent R(λ)/T(λ) spectrum. The RGB
Beer-Lambert path has no mechanism to represent this — it requires a per-wavelength
accumulator.

Additionally, RFE-01 spectral attenuation curves (81 samples, 380–780 nm) stamped via
`userData.vitrumSpectralAttenuation` are unusable without the spectral kernel: the fork
today can only consume scalar or RGB attenuation, not a per-wavelength μ(λ) curve.

**Decision gate (from `plan/sprint-12-pt-fork-patch.md §7`):** before beginning this
~24-working-day fork rewrite, confirm one or more of the trigger conditions with the user:
- Uranium glass (fluorescence emission by wavelength)
- Dichroic film (multi-order thin-film interference — 3-colour approximation shows aliasing)
- Bevel rainbows where Sprint 8's 3-colour fan is visibly discrete in hero renders
- Gemstones with absorption bands (e.g. alexandrite colour-shift at ~680 nm)

Do NOT begin the kernel rewrite without trigger confirmation. See `plan/sprint-12-pt-fork-patch.md §7`.

---

## What data the app already provides (ready to consume)

| Source | Field | Coverage |
|--------|-------|----------|
| `material.userData.vitrumSpectralAttenuation` | `LocalSpectralCurve` (81 samples) | 7 colorant families |
| `material.userData.vitrumThinFilmStack` | `LocalThinFilmStack` (35 layers) | dichroic only |
| `LocalMaterial.spectralAttenuation` | forwarded | same |
| `LocalMaterial.thinFilmStack` | forwarded | same |

Host-side utilities already shipped in `@vitrum/shared-samplers`:
- `sampleHeroWavelength`, `wavelengthToRGB`, `Y_CMF_INTEGRAL` in `src/wavelengthSampling.ts`
- CIE 1931 2° CMF tables (81 entries), `sampleCMF`, `xyzToLinearSRGB` in `src/cieCmf.ts`
- `cauchyIOR`, `CAUCHY_CROWN_GLASS`, `CAUCHY_FLINT_GLASS`, `CAUCHY_LEAD_CRYSTAL` in `src/cauchyIor.ts`

---

## Effort estimate

Per `plan/sprint-12-pt-fork-patch.md §5`: ~24 working days total. This is a pervasive
kernel change (ray payload restructure touches every shader function). Merging upstream
three-gpu-pathtracer updates after this patch requires multi-day conflict resolution.

---

## Exact fork work required

See `plan/sprint-12-pt-fork-patch.md §2` for the full specification. Summary:

1. **Ray payload** (`path_tracer.glsl.js` or equivalent):
   - Change from `vec3 throughput` to `float wavelength + float throughput`.

2. **Wavelength sampling** (path initialisation):
   - `sampleHeroWavelength(u, out float pdf)` GLSL function.
   - Upload 82-entry Y-CMF CDF array + 81-entry Y table + `Y_CMF_INTEGRAL` as uniforms.

3. **BSDF wavelength-awareness** (`bsdf_functions.glsl.js`):
   - All IOR lookups use `cauchyIOR(wavelength, A, B, C)` (three Cauchy coefficients).
   - New uniforms: `iorCauchyA`, `iorCauchyB`, `iorCauchyC` (replaces `ior0` + `dispersionStrength`).

4. **New file: `spectral_accumulator.glsl.js`**:
   - `sampleCmfX/Y/Z(lambda)` — linear interpolation on uniform CMF arrays.
   - `vec3 wavelengthToRGB(lambda, throughput, pdfLambda)` — XYZ → linear sRGB.

5. **Framebuffer accumulation**:
   - Use `wavelengthToRGB(...)` output per path instead of direct RGB throughput.

6. **`PhysicalPathTracingMaterial.js`** (or equivalent):
   - New uniforms: `iorCauchyA/B/C`, `yCmfCdf[82]`, `yCmfY[81]`, `yCmfIntegral`, `cmfX/Y/Z[81]`.

---

## Relationship to Sprint 8 (RFE-06)

Sprint 12 supersedes Sprint 8's Cauchy IOR uniforms (`ior0`, `dispersionStrength`) with
the three-coefficient form (`iorCauchyA/B/C` in µm). See migration mapping in
`plan/sprint-12-pt-fork-patch.md §3`. Sprint 8 can land first as an intermediate step;
Sprint 12 migrates the Cauchy coefficients to µm scale without changing call sites.

---

## Verification criteria

- [ ] Ray payload uses scalar `wavelength + throughput` (not `vec3 throughput`)
- [ ] `sampleHeroWavelength` GLSL implemented with Y-CMF CDF binary search
- [ ] All BSDF sites use `iorCauchyA/B/C` + hero `wavelength` for IOR
- [ ] `spectral_accumulator.glsl.js` present with CMF sampling + XYZ → sRGB
- [ ] `PhysicalPathTracingMaterial.js` carries all new uniforms
- [ ] Visual A/B: bevel rainbow shows smooth spectrum (8+ visible colours) vs Sprint 8's 3-colour fan
- [ ] GPU throughput regression < 30% vs Sprint 8 baseline on 1080p scene
