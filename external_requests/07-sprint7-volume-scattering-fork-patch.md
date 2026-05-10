# RFE-07 — Sprint 7 Volume Scattering Fork Patch

**Date:** 2026-05-09
**Requester:** stainedGlass app (`~/projects/stainedGlass`)
**Status:** NOT APPLIED. No Sprint 7 patch markers found in the fork.

---

## What this request is for

Apply the Sprint 7 fork-side shader patches from `plan/sprint-7-pt-fork-patch.md` to
`~/projects/three-gpu-pathtracer/`. These patches add:
- Homogeneous volume scatter in the main path-tracing loop (god-ray haze).
- Per-material SSS single scatter via HG phase function (opalescent / glueChip / ringMottled).

---

## Why it is needed

The stainedGlass app stamps `userData.vitrumScatteringCoefficient` and
`userData.vitrumScatteringAnisotropy` on baked materials for scattering glass types
(opalescent σ_s=2.5 mm⁻¹ g=0.75; wispy σ_s=1.2 g=0.60; ringMottled σ_s=2.5 g=0.75;
dalleDeVerre σ_s=0.5 g=0.50). The consumer-side adapter (`vitrumMaterialAdapter.ts`)
forwards these to `LocalMaterial.scatteringCoefficient` / `scatteringAnisotropy`.

Without the fork patches, opalescent glass renders as a flat semi-transparent slab.
The milky internal glow and forward-scattered internal light characteristic of real
opalescent glass is absent.

Note: the fork already contains a `volumeParticle` flag and fog-volume infrastructure
(see `src/shader/bvh/inside_fog_volume_function.glsl.js`). This is pre-existing and
unrelated to the Sprint 7 per-material SSS. Sprint 7 adds the homogeneous-medium
scatter as a separate code path inside the main path-tracing loop.

---

## What data the app already provides (ready to consume)

| Source | Field | Example (opalescent) |
|--------|-------|----------------------|
| `material.userData.vitrumScatteringCoefficient` | σ_s mm⁻¹ | 2.5 |
| `material.userData.vitrumScatteringAnisotropy` | HG g | 0.75 |
| `LocalMaterial.scatteringCoefficient` | forwarded | 2.5 |
| `LocalMaterial.scatteringAnisotropy` | forwarded | 0.75 |

Host-side JS utilities already shipped in `@vitrum/shared-samplers`:
- `evaluateHG / sampleHG / pdfHG` in `src/hgPhase.ts`
- `sampleEquiAngular` in `src/equiAngular.ts`

---

## Exact fork work required

See `plan/sprint-7-pt-fork-patch.md` §1–§6 for the full specification. Summary:

1. **New file: `volume_march.glsl.js`** (or under `src/shader/bsdf/`):
   - `sampleExponential(u, density, maxT)` — exponential scatter distance
   - `equiAngularPdf(t, tC, D, thetaRange)`
   - `hg_phase(cosTheta, g)` — HG phase function
   - `volumeMarch(ro, rd, tSurface, u)` — returns scatter distance

2. **`PhysicalPathTracingMaterial.js`** (or main loop file):
   - New uniforms: `volumeDensity`, `scatterAlbedo` (vec3), `anisotropyG`, `sssSigmaT`, `sssAlbedo` (vec3), `sssAnisotropyG`

3. **Main path-tracing loop** (`PhysicalPathTracingMaterial.js` shader or equivalent):
   - After surface hit test, check `u_volumeDensity > 0`, sample scatter distance via `volumeMarch`.
   - If `tScatter < tSurface`: evaluate HG phase, sample new direction, perform equi-angular NEE.

4. **`bsdf_functions.glsl.js`**:
   - SSS single-scatter branch gated on `TRANSLUCENT_BIT` material flag.
   - Uses `sampleExponential` + HG phase for interior scatter.

5. **Material data function** (wherever glass material flags are packed):
   - `TRANSLUCENT_BIT = 0x10u` (bit 4).
   - Glass type mappings: `opalescent`, `glueChip`, `ringMottled` → `TRANSLUCENT_BIT`.

---

## Verification criteria

- [ ] `volume_march.glsl.js` (or equivalent) present with `sampleExponential`, `equiAngularPdf`, `hg_phase`, `volumeMarch`
- [ ] Main loop has volume scatter event before surface shading
- [ ] `TRANSLUCENT_BIT` defined and mapped to opalescent / glueChip / ringMottled
- [ ] Visual A/B: backlit opalescent panel shows milky internal glow (σ_s=2.5, g=0.75)
- [ ] Visual A/B: sun-lit scene with `volumeDensity > 0` shows god-ray shafts
- [ ] Non-scattering profiles (cathedral, bevels) unaffected (`volumeDensity=0` fast path)
