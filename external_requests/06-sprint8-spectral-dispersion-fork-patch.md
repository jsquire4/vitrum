# RFE-06 — Sprint 8 Spectral Dispersion Fork Patch

**Date:** 2026-05-09
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Status:** NOT APPLIED. All Sprint 7/8/12 patch markers absent from the fork.

---

## What this request is for

Apply the Sprint 8 fork-side shader patches from `plan/sprint-8-pt-fork-patch.md` to
`~/projects/three-gpu-pathtracer/`. These patches enable physically-based chromatic
dispersion (per-channel IOR via the Cauchy formula) and Jakob+Hanika spectral upsampling.

---

## Why it is needed

The stainedGlass app stamps `userData.vitrumDispersionAbbeNumber` on every baked
`MeshPhysicalMaterial` for glass types with a documented Abbe number (e.g. bevels:
V_d = 30, lead crystal). The consumer-side adapter (`vitrumMaterialAdapter.ts`,
Phase 2b Tier B) forwards this value to `LocalMaterial.dispersionAbbeNumber`.

The rendering pipeline (`@vitrum/pt-webgl`) has no shader code to consume this value.
Without the fork patches, bevel cells render with a single-IOR refraction path and no
chromatic splitting. The `THREE.MeshPhysicalMaterial.dispersion` field (set to `30/V_d`)
is the only current fallback, and it uses three.js's built-in prismatic approximation —
a first-order simulation without the stochastic wavelength selection the Sprint 8 patch
introduces.

---

## What data the app already provides (ready to consume)

| Source | Field | Value for bevels |
|--------|-------|-----------------|
| `material.userData.vitrumDispersionAbbeNumber` | Abbe V_d | 30 |
| `LocalMaterial.dispersionAbbeNumber` | forwarded by `toVitrumMaterial` | 30 |
| `material.dispersion` (THREE.js fallback) | 30/30 = 1.0 | already set |

The Jakob+Hanika host-side computation (`rgbToSpectralCoefficients`) is already available
in `@vitrum/shared-samplers/src/jakobHanika.ts`. The Cauchy IOR utilities are in
`@vitrum/shared-samplers/src/cauchyIor.ts` (`cauchyIOR`, `CAUCHY_LEAD_CRYSTAL`, etc.).

---

## Exact fork work required

See `plan/sprint-8-pt-fork-patch.md` §1–§2 for the full specification. Summary:

1. **`src/shader/shaders/pathtracing/bsdf_functions.glsl.js`** (or equivalent location in the fork):
   - Add per-channel IOR from Cauchy formula at `{700, 550, 450} nm`:
     ```glsl
     float iorR = u_ior0 + u_dispersionStrength / (700.0 * 700.0);
     float iorG = u_ior0 + u_dispersionStrength / (550.0 * 550.0);
     float iorB = u_ior0 + u_dispersionStrength / (450.0 * 450.0);
     ```
   - Stochastic wavelength selection (1/3 probability each channel, 3× throughput weight).
   - `evalSpectrum(vec3 coeffs, float lambda)` — 6-instruction sigmoid polynomial.
   - Gate: only activate when `u_dispersionStrength > 1e-4`.

2. **`src/materials/pathtracing/PhysicalPathTracingMaterial.js`** (or equivalent):
   - Add uniforms: `ior0`, `dispersionStrength`, `jakobCoeffs` (vec3).

---

## Relationship to Sprint 12

Sprint 8 uses RGB-as-3λ (three discrete channels). Sprint 12 (`plan/sprint-12-pt-fork-patch.md`)
replaces this with hero-wavelength spectral PT (full payload restructure, ~24 working days).
Sprint 8 is the lower-cost stepping stone for bevel rainbow quality; Sprint 12 is the full
spectral engine (see `plan/sprint-12-pt-fork-patch.md §7` decision point criteria).

---

## Verification criteria

- [ ] Cauchy IOR formula present in fork BSDF shader at {700, 550, 450} nm
- [ ] Stochastic wavelength selection gate (`u_dispersionStrength > 1e-4`)
- [ ] `PhysicalPathTracingMaterial.js` carries `ior0`, `dispersionStrength`, `jakobCoeffs`
- [ ] Visual A/B: bevel cell shows 3-colour prismatic fan at normal-incidence refraction
- [ ] Non-bevel glass (cathedral, opalescent) unaffected (dispersionStrength = 0 fast path)
