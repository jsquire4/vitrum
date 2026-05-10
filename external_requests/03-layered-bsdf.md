# Layered BSDF — Two-Sided / Front-Back-Asymmetric Material

**Status:** Applied (runtime-unverified)
**Originating need:** Consumer materials that have physically distinct optical properties on their front and back faces — absorbing surface layers, diffusion coatings, or asymmetric films — cannot be represented by the current symmetric `Material` interface.
**Priority:** Medium (originating use case deferred; primitive is broadly applicable to asymmetric dielectrics)

## 1. Motivation

The current `Material` interface assigns a single BSDF to both sides of any surface. This is adequate for homogeneous symmetric materials (bare glass, metal, diffuse walls) but fails for a wide class of physically important surfaces:

- **Coated architectural glass.** Low-e coatings (silver-based films) on building glass reflect IR differently from the interior vs. exterior face. The coating is on one face only; the bare glass face has a very different reflectance and emission profile.
- **Optical components with anti-reflection on one face.** Lenses and beam-splitter plates often have AR coatings on the entry face and a dielectric film on the exit face. A single symmetric BSDF cannot represent this.
- **Painted or decorated surfaces.** Any surface with a surface-applied absorber on one face (a fired ceramic trace, a lacquer coat, a retroreflective film) and a bare dielectric on the other requires distinct front and back BSDFs.
- **Layered dielectrics in general.** Multi-coat paint systems (automotive clear-coat over colored base over primer) have depth-ordered optical properties that emerge from the direction of incidence.

The primitive is framed as a general front/back BSDF asymmetry, not as anything domain-specific. The implementation mechanism follows Belcour 2018 [1].

## 2. Proposed API Surface

Rather than duplicating the entire `Material` interface, the proposal is a lightweight asymmetry extension. The vast majority of use cases require only a per-face absorption override — the surface geometry (roughness, normal map, IOR) is the same on both faces, only the attenuation layer differs.

```typescript
// New top-level type in scene.ts

/**
 * Describes a thin absorbing layer on one face of a dielectric surface.
 * Models a surface-fused coating or diffusion-tinted region that modulates
 * the transmitted and reflected radiance independently from the bulk material.
 *
 * The layer is treated as an infinitesimally thin absorber applied BEFORE
 * the bulk BSDF is evaluated — i.e., incoming radiance is multiplied by
 * layerTransmission before entering the bulk, and outgoing radiance is
 * multiplied by layerTransmission before exiting. This is the simplified
 * (non-multiple-scattering) form of Belcour 2018's atomic decomposition.
 */
export interface SurfaceAbsorptionLayer {
  /**
   * Per-channel (RGB) transmission of the thin absorbing layer.
   * [1, 1, 1] = no absorption (identity layer).
   * [0, 0, 0] = fully opaque mask.
   * Each component ∈ [0, 1].
   */
  readonly transmission: Vec3;

  /**
   * Optional roughness override for this face's surface.
   * When set, replaces Material.roughness for this face only.
   * Range: [0, 1].
   */
  readonly roughness?: number;

  /**
   * Optional normal map texture for this face only.
   * Useful when the two faces have different surface treatments.
   */
  readonly normalMap?: TextureRef;
  readonly normalScale?: number;
}

// Addition to the existing Material interface in scene.ts

interface Material {
  // ... existing fields unchanged ...

  // ── Per-face BSDF asymmetry ───────────────────────────────────────────
  /**
   * Thin absorbing layer applied to the front (outward-normal) face.
   * "Front" is the face whose normal points in the direction used to
   * define the mesh's vertex normals.
   *
   * When present, the BSDF for rays hitting this face is:
   *   L_front = SurfaceAbsorptionLayer.transmission ⊙ L_bulk
   * where L_bulk is the standard Material BSDF evaluation.
   */
  frontLayer?: SurfaceAbsorptionLayer;

  /**
   * Thin absorbing layer applied to the back (inward-normal) face.
   * Symmetric semantics to frontLayer.
   */
  backLayer?: SurfaceAbsorptionLayer;
}
```

This is the minimal viable interface. Both `frontLayer` and `backLayer` are optional; if neither is present, behavior is identical to the current symmetric BSDF.

## 3. Algorithmic Notes

**Belcour 2018** [1] derives a statistically exact layered BSDF by tracking the first three directional moments (energy, mean, variance) of each interface's BSDF and composing them through an atomic decomposition. For the simplified case of a single thin absorbing layer over a dielectric bulk, the composition reduces to scalar multiplication of the layer transmission against the bulk BSDF output — a tractable approximation that avoids the full moment-tracking machinery.

The full Belcour framework handles: arbitrary number of layers, each with its own BRDF/BTDF, absorption, and scattering operators. For future extension to thicker intermediate layers (e.g., a painted layer with both absorption and scattering), the same infrastructure supports richer compositions.

**Multiple-scattering correction.** When the absorbing layer is optically thick, inter-layer multiple-scattering becomes relevant (some energy trapped between the layer and bulk bounces multiple times before exiting). Heitz et al. 2016 [2] addresses this for single-surface roughness; Belcour 2018 extends it to the layered case. For thin absorption layers (transmission > 0.5), the single-bounce approximation is adequate.

## 4. Backend Implementation Guidance

**`@vitrum/pt-webgl` (forked three-gpu-pathtracer):**
The per-face check requires knowing the ray's side-of-incidence (sign of dot(rayDir, geometricNormal)). Three-gpu-pathtracer already computes this for two-sided materials. The change is: at shade time, select `frontLayer` or `backLayer` based on face orientation, multiply the sampled BSDF value by `layer.transmission`, and optionally substitute `layer.roughness` and `layer.normalMap` for the affected face. This is a shader-level addition of approximately 20–30 lines.

**`@vitrum/pt-webgpu` (future):**
Same logic. The `PathState` struct carries a `faceSide: u32` (0 = front, 1 = back) that is trivially computed at the intersection. Material buffer stores both layer structs adjacently.

**`@vitrum/walkaround-hybrid` (real-time GI):**
The walkaround shading pass already distinguishes face sides for shadow termination. Adding front/back layer selection is a direct extension. Performance impact is one conditional per material sample — negligible.

## 5. Validation

**Reference scene:** A thin glass plane (5 mm, ior 1.52) with a partial absorbing layer (`transmission: [0.1, 0.1, 0.1]`) on the front face only. Illuminated from both sides with equal-intensity directional lights.

**Expected outcome:** The front-illuminated side should show strongly attenuated transmission and reflection through the dark absorber; the back-illuminated side should show normal glass reflectance and transmission. Flip the camera to both sides and compare — a symmetric BSDF would produce identical results from both sides; the layered BSDF must not.

**Mitsuba 3 reference:** Mitsuba's `twosided` and `coating` BSDFs can be composed to produce the ground-truth reference.

## 6. Consumer-Side Integration

A consumer describing a glass pane with a surface absorber on one face:

```typescript
import type { Material, SurfaceAbsorptionLayer } from '@vitrum/core';

// Dark absorbing layer on front face; back face is bare glass
const frontCoatedGlass: Material = {
  baseColor: [0.05, 0.05, 0.05],
  roughness: 0.02,
  metallic: 0.0,
  transmission: 1.0,
  ior: 1.52,
  attenuationColor: [0.2, 0.3, 0.8],
  attenuationDistance: 3.0,

  frontLayer: {
    transmission: [0.08, 0.06, 0.04],   // near-opaque dark absorber
    roughness: 0.01,
  },
  // backLayer absent → bare glass BSDF on back face
};
```

## 7. Open Questions

None that block the MVP interface. The full Belcour moment-tracking machinery is a separate, more complex extension if thicker layers with scattering are needed.

## 8. References

[1] L. Belcour. "Efficient Rendering of Layered Materials using an Atomic Decomposition with Statistical Operators." *ACM Transactions on Graphics (Proc. SIGGRAPH 2018)*, vol. 37, no. 4, article 73, 2018. DOI: 10.1145/3197517.3201289.

[2] E. Heitz, J. Hanika, E. d'Eon, C. Dachsbacher. "Multiple-Scattering Microfacet BSDFs with the Smith Model." *ACM Transactions on Graphics (Proc. SIGGRAPH 2016)*, vol. 35, no. 4, article 58, 2016. DOI: 10.1145/2897824.2925943.

[3] Autodesk. "Standard Surface Specification." Available at autodesk.github.io/standard-surface/. (Coat layer: `coat`, `coat_color`, `coat_IOR`, `coat_roughness` — single-layer clearcoat analogue.)
