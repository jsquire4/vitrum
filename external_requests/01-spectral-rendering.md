# Spectral Material Attenuation and Dispersion

**Status:** Implemented (contract + fork patches applied; GPU A/B pending)
**Originating need:** Consumer materials whose transmission spectrum varies significantly within a single RGB channel, causing hue drift with geometry thickness that RGB rendering cannot correctly reproduce.
**Priority:** High

## 1. Motivation

RGB path tracing treats light as three independent monochromatic wavelengths. Beer-Lambert attenuation is applied per-channel:

```
T_R = exp(−μ_R · d),  T_G = exp(−μ_G · d),  T_B = exp(−μ_B · d)
```

This is physically correct only when the spectral absorption coefficient μ(λ) is flat within each channel's bandpass. Real absorbing dielectrics violate this assumption. The result is **hue drift with thickness**: two objects with the same RGB attenuation color but different thicknesses will render with systematically wrong hues. The error compounds when the same material is reused at multiple thicknesses in a single scene.

Two illustrative cases span the problem space:

- **Colored gemstones.** A ruby at 1 mm vs. 10 mm has dramatically different hue in reality (due to the chromium Cr³⁺ absorption triad). RGB rendering produces the same hue at both thicknesses, only scaling brightness — a visible fidelity gap in jewelry visualization.
- **Architectural glazing.** A cobalt-blue glass pane absorbs in three overlapping d-d transition bands (525, 595, 650 nm). As pane thickness increases from 2 mm to 10 mm, the three bands attenuate at different rates, shifting the transmitted hue from saturated blue toward near-black. RGB misses this drift.

Dispersion (wavelength-dependent IOR) is a related problem: a glass prism separates white light into its constituent wavelengths because each wavelength refracts at a slightly different angle. RGB rendering cannot produce prismatic chromatic aberration — the three channel-rays refract identically, so bevel-edge rainbow effects and caustic color separation are impossible without a spectral or hero-wavelength trace.

## 2. Proposed API Surface

The following additions extend the existing `Material` interface in `scene.ts`. All fields are optional; absent fields degrade to current RGB behavior.

```typescript
// Addition to the existing Material interface in scene.ts

interface Material {
  // ... existing fields unchanged ...

  // ── Spectral attenuation ───────────────────────────────────────────────
  /**
   * Spectral attenuation coefficient table, sampled at uniformly spaced
   * wavelengths. Each entry is μ(λ) in inverse scene-length-units (m⁻¹ or
   * mm⁻¹, matching attenuationDistance units).
   *
   * When present, the engine uses per-wavelength Beer-Lambert instead of the
   * RGB approximation. Length must be >= 3; typical use is 8–32 samples
   * spanning 380–700 nm. The engine linearly interpolates between samples.
   *
   * When absent and attenuationColor is present, engine falls back to the
   * existing RGB Beer-Lambert approximation.
   */
  spectralAttenuation?: SpectralCurve;

  // ── Dispersion ─────────────────────────────────────────────────────────
  /**
   * Abbe number V_d = (n_d − 1) / (n_F − n_C).
   * Higher values = lower dispersion. When set, the engine computes
   * wavelength-dependent IOR via the two-term Cauchy approximation:
   *   n(λ) ≈ n_d + (n_d − 1) / (V_d · (λ² − λ_d²) · C)
   * and uses hero-wavelength sampling so each path traces a single wavelength.
   * Range: 20 (dense flint, high dispersion) to 90 (crown glass, low).
   * Default: undefined (no dispersion).
   */
  dispersionAbbeNumber?: number;
}

/**
 * A spectral reflectance or attenuation curve.
 * wavelengthStart / wavelengthEnd: visible range in nanometers (e.g. 380/700).
 * values: per-sample coefficient. Must have >= 3 entries, uniformly spaced.
 */
export interface SpectralCurve {
  readonly wavelengthStart: number;   // nm, e.g. 380
  readonly wavelengthEnd: number;     // nm, e.g. 700
  readonly values: Float32Array;      // μ(λ) in units matching attenuationDistance
}
```

`spectralAttenuation` lives on `Material` (alongside the existing `attenuationColor`/`attenuationDistance` fields). `dispersionAbbeNumber` also lives on `Material`. Neither requires new scene-level or engine-level types.

## 3. Algorithmic Notes

**Hero Wavelength Spectral Sampling (HWSS)** — Wilkie et al., EGSR 2014 [1]:

Each path is assigned one uniformly sampled "hero" wavelength λ_h. Three companion wavelengths are placed at equal spectral intervals (λ_h + k·ΔΛ, k=1,2,3) to tile the visible range. All directional sampling (BSDF, phase function, Russian roulette) uses λ_h; Beer-Lambert and dispersion are evaluated for all four wavelengths simultaneously. The four wavelength samples are accumulated into a per-path spectral estimate and converted to XYZ (then RGB) via CIE color-matching functions at the end.

Cost: roughly 2–3× per-sample vs. RGB PT. Since spectral rendering converges to the correct hue faster for colored scenes (fewer samples needed to pass through the "wrong hue" transient), the practical convergence cost is closer to 1.5–2× for heavily colored scenes.

**Dispersion** uses the wavelength-dependent IOR to compute refraction angle per wavelength at each dielectric interface.

## 4. Backend Implementation Guidance

**`@vitrum/pt-webgl` (forked three-gpu-pathtracer):**
Three-gpu-pathtracer currently traces in RGB. HWSS requires changes to the core path-trace shader: (a) a per-ray wavelength uniform, (b) wavelength-indexed Beer-Lambert texture sampling, (c) a CIE XYZ→RGB accumulation pass. The four-wavelength hero packet can be implemented as four parallel path evaluations per invocation (GPU-parallelism-friendly on wide warps). `dispersionAbbeNumber` maps to per-refraction IOR perturbation.

**`@vitrum/pt-webgpu` (future):**
WebGPU compute shaders are better suited to HWSS because workgroup-shared memory allows the four hero wavelengths to share BVH traversal. A single `PathState` buffer carries a `wavelength: f32` alongside radiance.

**`@vitrum/walkaround-hybrid` (real-time GI):**
Full HWSS is too expensive for real-time. A practical approximation: split the Beer-Lambert per-material into three separate Cauchy-fitted RGB attenuation passes at the scene level (precomputed; not per-path), and approximate dispersion with a scalar chromatic aberration post-process. Report in `EngineCapabilities` extensions that spectral accuracy is approximate.

## 5. Validation

**Reference scene:** A solid glass sphere (r = 50 mm) filled with cobalt-blue medium (μ peaks at 525, 595, 650 nm). Render at two slab thicknesses by scaling the sphere — 5 mm and 30 mm equivalent path. Ground truth: Mitsuba 3 with `spectral` integrator and measured Co²⁺ absorption coefficients.

**Expected outcome:** RGB rendering shows same saturated blue at both thicknesses (only darker). HWSS rendering shows saturated blue at 5 mm, shifting toward a cooler near-neutral blue-gray at 30 mm. The mismatch between RGB and spectral at 30 mm is the validation signal.

**Dispersion test:** A glass prism (V_d = 30, simulating lead crystal). A collimated white-light beam should produce a rainbow fringe at the exit face. Compare fringe spread against the known angular separation Δθ derived from the Abbe number.

## 6. Consumer-Side Integration

A consumer supplying spectral attenuation for a strongly colored medium:

```typescript
import type { Material, SpectralCurve } from '@vitrum/core';

// Cobalt blue absorption: sampled every 40 nm from 380–700 nm
const cobaltBlueAttenuation: SpectralCurve = {
  wavelengthStart: 380,
  wavelengthEnd: 700,
  // μ(λ) in mm⁻¹: low absorption in blue, high in green/red
  values: new Float32Array([0.05, 0.04, 0.10, 0.60, 0.90, 0.85, 0.60, 0.30, 0.10]),
};

const material: Material = {
  baseColor: [0.1, 0.1, 0.8],
  roughness: 0.02,
  metallic: 0.0,
  transmission: 1.0,
  ior: 1.52,
  attenuationDistance: 3.0,           // mm; fallback for non-spectral backends
  attenuationColor: [0.0, 0.05, 0.9], // fallback RGB approximation
  spectralAttenuation: cobaltBlueAttenuation,
};
```

A consumer wanting dispersion for a high-index medium:

```typescript
const lensGlass: Material = {
  baseColor: [1, 1, 1],
  roughness: 0.001,
  metallic: 0.0,
  transmission: 1.0,
  ior: 1.78,
  dispersionAbbeNumber: 25.0,  // dense flint; strong dispersion
};
```

## 7. Open Questions

- CIE color-matching function storage: embed as a small constant lookup in the shader (31 samples, 10 nm spacing, RGB CMFs) or pass as a texture? Texture is more flexible.
- `SpectralCurve` wavelength range: enforce 380–700 nm for now, or allow IR (useful for thermal materials)? Defer to a later spec.

## 8. References

[1] A. Wilkie, S. Nawaz, M. Droske, A. Weidlich, J. Hanika. "Hero Wavelength Spectral Sampling." *Computer Graphics Forum (Proc. EGSR 2014)*, vol. 33, no. 4, pp. 123–131, 2014. DOI: 10.1111/cgf.12419.

[2] M. Rubin. "Optical Properties of Soda-Lime Silica Glasses." *Solar Energy Materials*, vol. 12, no. 4, pp. 275–288, 1985. (Cauchy fit for soda-lime glass; indexed at refractiveindex.info.)

[3] Schott AG. "N-BK7 Optical Glass Data Sheet." Schott Advanced Optics. Available at us.schott.com. (n_d = 1.5168, V_d = 64.17; Sellmeier coefficients.)

[4] B. Burley. "Physically-Based Shading at Disney." SIGGRAPH 2012 Course Notes, 2012. Available at disneyanimation.com/publications/.

[5] Academy Software Foundation. "OpenPBR Surface v1.1.1 Specification." Available at academysoftwarefoundation.github.io/OpenPBR/. (transmission_dispersion_abbe_number field; spectral attenuation model.)
