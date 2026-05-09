# Volume Scattering — Henyey-Greenstein Phase Function and Density Media

**Status:** Proposed
**Originating need:** Consumer materials containing suspended particles that scatter light internally (milky, opalescent, or wispy media) cannot be physically modeled with surface roughness alone; they require volumetric scattering with a configurable phase function.
**Priority:** High

## 1. Motivation

Many physically important media contain distributed scatterers that redirect light as it travels through the volume. The current `Material` interface handles volume *absorption* via Beer-Lambert (`attenuationColor` / `attenuationDistance`) but provides no mechanism for volume *scattering*. The absence creates a hard fidelity gap:

- **Opalescent and milky materials.** Milky glass, jade, wax, biological tissue, and milk all derive their visual character from internal scattering. Modeling them as opaque + rough surface gives an approximation that fails completely at oblique angles and under backlit conditions (the characteristic warm glow of backlit milk or opal is a pure volume-scattering effect).
- **Underwater / atmospheric rendering.** Fog, sea water, and haze require forward-biased volume scattering. Without HG phase function support, all media must be treated as either purely absorbing or fully opaque.
- **Participating media in interior scenes.** Dusty air in a sunlit room, smoke, or translucent candle wax require volumetric path tracing with scattering events.

Surface roughness is not a substitute: it affects only rays arriving at the surface, not paths propagating through the medium's interior. For a 5 mm slab of opalescent glass, the scattering mean free path is shorter than the slab thickness — multiple internal scattering events occur, redistributing ray directions in a way that surface roughness cannot replicate.

## 2. Proposed API Surface

The following optional fields extend `Material`:

```typescript
// Addition to the existing Material interface in scene.ts

interface Material {
  // ... existing fields unchanged ...

  // ── Volume scattering ─────────────────────────────────────────────────
  /**
   * Scattering coefficient σ_s, in inverse scene-length-units (matching
   * attenuationDistance units). The total extinction coefficient is
   * σ_t = σ_a + σ_s, where σ_a is derived from attenuationDistance.
   *
   * When scatteringCoefficient > 0 and transmission > 0, the backend
   * activates volumetric path tracing (delta tracking) for rays passing
   * through this medium. Default: undefined (no volumetric scattering).
   *
   * Units: mm⁻¹ (or m⁻¹, matching attenuationDistance).
   */
  scatteringCoefficient?: number;

  /**
   * Henyey-Greenstein phase function asymmetry parameter g ∈ (−1, 1).
   *   g = 0:    isotropic scatter (large particles, Rayleigh regime).
   *   g > 0:    forward-biased scatter (Mie regime, biological tissue ~0.9).
   *   g < 0:    backward-biased scatter (rare; retroreflective powders).
   * Default: 0 (isotropic) when scatteringCoefficient is set.
   */
  scatteringAnisotropy?: number;

  /**
   * Per-channel (RGB) scattering coefficients. When present, overrides the
   * scalar scatteringCoefficient with wavelength-dependent values, producing
   * chromatic scattering (e.g., sky-blue tint in forward-scattered white light).
   * Units and defaults same as scatteringCoefficient.
   */
  scatteringCoefficientRGB?: Vec3;
}
```

All three fields live on `Material`. No scene-level or engine-level changes are required.

## 3. Algorithmic Notes

**Henyey-Greenstein phase function** — Henyey & Greenstein 1941 [1]:

```
p_HG(cos θ) = (1 / 4π) · (1 − g²) / (1 + g² − 2g cos θ)^(3/2)
```

Sampling is analytic (closed-form inversion exists for cos θ), making it GPU-friendly.

**Delta tracking (null-collision)** — Novák et al. 2018 [2]:

For homogeneous media (constant σ_t), the standard free-flight sampling exp(−σ_t · t) applies directly. For the heterogeneous generalization, a null-collision (fictitious scattering event) is inserted at rate σ_null = σ_t_max − σ_t(x) to homogenize the majorant, enabling unbiased sampling. For thin slabs (2–6 mm) with moderate scattering, the mean free path is long relative to slab thickness and the method converges quickly.

**Typical parameter values for opalescent glass:** σ_s ≈ 1–5 mm⁻¹, g ≈ 0.7–0.8 (Mie regime, forward-biased). Opal white transmitting warm backlit glow needs g close to 0.75 with moderate σ_s.

**Practical cost:** Each volumetric scatter event adds one recursive path bounce. For a slab where the expected number of scattering events is low (< 3), the cost overhead is modest. For highly scattering media (σ_s >> σ_a, many events), Russian roulette absorbs the tail.

## 4. Backend Implementation Guidance

**`@vitrum/pt-webgl` (forked three-gpu-pathtracer):**
Three-gpu-pathtracer's path trace shader currently handles surface-only BSDFs. Volume scattering requires adding a free-flight sampling step inside the ray-march loop: at each transmitted-ray segment, sample a potential scattering distance t from exp(−σ_t · t); if t < d_surface, perform an HG scattering event and continue. The HG sampler is a ~10-line GLSL addition. The existing `attenuationColor`/`attenuationDistance` Beer-Lambert already runs inside the transmitted-ray segment — the volume scatter step inserts alongside it.

**`@vitrum/pt-webgpu` (future):**
WebGPU compute pipelines handle volumetric paths cleanly via path-state structs. The `PathState` buffer gains a `mediumSigmaT: f32` and `mediumG: f32` field. Volume scattering is a first-class path event type alongside surface reflection/refraction.

**`@vitrum/walkaround-hybrid` (real-time GI):**
Real-time volumetric path tracing is feasible for thin slabs at low scattering albedo. For the walkaround engine, a practical compromise is a single-scatter approximation: trace the primary ray through the volume, sample one scattering direction, and estimate indirect contribution via the irradiance cache. This degrades gracefully to Beer-Lambert absorption when σ_s is small. Report as `supportsVolumeScattering: 'single-scatter'` in engine capabilities extensions.

## 5. Validation

**Reference scene 1 (opalescent slab):** A 5 mm plane-parallel slab with σ_s = 2 mm⁻¹, σ_a = 0.1 mm⁻¹, g = 0.75, lit from behind. Expected: warm diffuse glow in transmission; slight blue tint in reflection (short-wavelength preferential backscatter). Validate against Mitsuba 3 `volpath` integrator with matched parameters.

**Reference scene 2 (foggy room):** A 2 m room filled with σ_s = 0.005 m⁻¹ isotropic haze. A point light in the center should produce a visible volumetric glow. Validate angular falloff against the analytic single-scatter solution for a point source in a homogeneous medium.

**Published reference:** Novák et al. 2018 [2] Figure 12 — ratio tracking vs. delta tracking convergence on a heterogeneous participating medium. Use as a convergence sanity check.

## 6. Consumer-Side Integration

A consumer describing an opalescent medium:

```typescript
import type { Material } from '@vitrum/core';

const opalWhite: Material = {
  baseColor: [0.9, 0.88, 0.85],
  roughness: 0.03,         // fire-polished surface, smooth
  metallic: 0.0,
  transmission: 1.0,
  ior: 1.52,
  attenuationColor: [1.0, 0.97, 0.90],
  attenuationDistance: 50.0,           // mm; slight warm absorption
  scatteringCoefficient: 2.5,          // mm⁻¹; strong internal scatter
  scatteringAnisotropy: 0.75,          // forward-biased (Mie regime)
};
```

A consumer describing chromatic scattering (sky/atmosphere-like medium):

```typescript
const hazedAir: Material = {
  baseColor: [1, 1, 1],
  roughness: 0.0,
  metallic: 0.0,
  transmission: 1.0,
  ior: 1.0003,
  scatteringCoefficientRGB: [0.012, 0.008, 0.003],  // blue-biased Rayleigh-like
  scatteringAnisotropy: 0.0,
};
```

## 7. Open Questions

- **Heterogeneous density:** Should `scatteringCoefficient` accept a `TextureRef` (3D density volume)? Deferred — the homogeneous case covers the initial use cases; heterogeneous can be added as a `VolumeMedium` scene-level primitive later.
- **Index-matched media:** When the medium's IOR matches the surrounding medium, there are no surface Fresnel events. The current `ior` + `transmission` fields handle this correctly already.

## 8. References

[1] L.C. Henyey, J.L. Greenstein. "Diffuse Radiation in the Galaxy." *The Astrophysical Journal*, vol. 93, pp. 70–83, 1941.

[2] J. Novák, I. Georgiev, J. Hanika, W. Jarosz. "Monte Carlo Methods for Volumetric Light Transport Simulation." *Computer Graphics Forum*, vol. 37, no. 2, pp. 551–576, 2018. (Delta tracking, ratio tracking, null-scattering framework.) DOI: 10.1111/cgf.13332.

[3] Academy Software Foundation. "OpenPBR Surface v1.1.1 Specification." Available at academysoftwarefoundation.github.io/OpenPBR/. (`transmission_scatter`, `transmission_scatter_anisotropy` fields.)

[4] Autodesk. "Standard Surface Specification." Available at autodesk.github.io/standard-surface/. (`transmission_scatter`, `transmission_scatter_anisotropy`.)
