# Sprint 6 — PT Fork Patch Plan

**Status: APPLIED 2026-05-12 — fork commit `e9c7516`**

**Sprint goal**: Composable visual upgrades — rough refraction lobe + edge-stopping spatial denoiser.

**Fork repo**: `~/projects/three-gpu-pathtracer/` branch `phase4-normalmap-shadow-rays`.
**Do NOT modify the fork without explicit user instruction.** This document
specifies what must be patched in the fork once Sprint 6 kicks off.

---

## 1. Rough refraction lobe

### 1.1 `src/shader/shaders/pathtracing/bsdf_functions.glsl.js`

**Goal**: perturb refracted ray direction by a GGX-distributed roughness lobe,
additive to the existing Phase 4 normal-map patch (which perturbs surface
normals, not refraction directions).  This is a separate physical effect:
rough glass has scattered transmission lobes, not just scattered normals.

#### Existing state (Phase 4)

Phase 4 added normal-map perturbation:
```glsl
// Phase 4 contribution:
normal = perturbNormalByMap(surfaceRecord.normalMap, normal, tangent, bitangent);
refractedDir = refract(incidentDir, normal, eta);
```

The Phase 4 patch modifies the surface normal before refraction.  The rough
refraction lobe is a DIFFERENT perturbation applied to the refracted direction
AFTER computing the specular refraction:

```glsl
// Sprint 6 contribution (additive, NOT replacing Phase 4):
refractedDir = refract(incidentDir, geometricNormal, eta);  // specular refraction
refractedDir = perturbDirectionByGGX(refractedDir, roughness, u1, u2);  // roughness lobe
```

#### Implementation

```glsl
// Perturb a refracted direction by a GGX lobe.
// This approximates the rough-transmission BSDF evaluated at normal incidence.
//
// refractDir: specular refraction direction (unit vector)
// roughness:  Disney roughness α (0 = specular glass, 1 = frosted glass)
// u1, u2:     uniform random samples for GGX importance sampling
// Returns: perturbed refraction direction (unit vector)
vec3 perturbDirectionByGGX(vec3 refractDir, float roughness, float u1, float u2) {
  if (roughness < 1e-4) return refractDir;  // specular: no perturbation

  float alpha = roughness * roughness;

  // GGX microfacet importance sample in hemisphere (Heitz 2014 visible NDF)
  float phi   = 2.0 * PI * u1;
  float cosTheta = sqrt((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2));
  float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  vec3 localH = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

  // Transform localH to the frame where refractDir is the +Z axis
  vec3 up = abs(refractDir.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
  vec3 tangent   = normalize(cross(up, refractDir));
  vec3 bitangent = cross(refractDir, tangent);
  vec3 perturbed = localH.x * tangent + localH.y * bitangent + localH.z * refractDir;

  return normalize(perturbed);
}
```

**Call site**: in the transmission sampling branch of `sampleBSDF`, after
computing the specular refraction direction:

```glsl
// Existing specular refraction
refDir = refract(-V, N, eta);

// Sprint 6: perturb by roughness lobe
float rough = surfaceRecord.roughness;
refDir = perturbDirectionByGGX(refDir, rough, rand(), rand());
```

**Composition with Phase 4**: Phase 4 modifies `N` before this call;
Sprint 6 modifies `refDir` after.  They compose correctly in order.

#### Definition of done

- Rough refraction perturbs refracted ray direction by GGX lobe.
- Does NOT change specular reflection (reflection branch unchanged).
- Does NOT interact with Phase 4 normal map perturbation (orthogonal).
- Visual A/B: hammered/seedy glass cell shows volumetric scatter blur in
  PT_FINAL (multiple ray fans from a single glass hit point).
- Crown glass and flat clear glass (roughness ~0) show no change.

---

## 2. Edge-stopping spatial filter (host-side, not fork-side)

The 37-tap hexagonal spatial filter WGSL is implemented in
`@vitrum/shared-denoisers/src/wgsl/spatialFilter.wgsl.ts` (vitrum-side).

The host-side integration (`PTSpatialDenoiser` as a `postprocessing` Effect
subclass) is **not** a fork patch — it lives in the host application and
consumes the WGSL constant from the vitrum package.

**Host-side integration spec** (for the host application developer):

1. Create a custom `postprocessing` Effect subclass `PTSpatialDenoiser` that:
   - Wraps `SPATIAL_FILTER_WGSL` (imported from `@vitrum/shared-denoisers`)
   - Binds the G-buffers from `FrameOutput.normalDepth` (for normal + depth)
   - Binds `FrameOutput.primaryRadiance` as `inputColor`
   - Takes σ parameters from props: `sigmaColor`, `sigmaNormal`, `sigmaDepth`

2. Mount `PTSpatialDenoiser` FIRST in the EffectComposer chain (before Bloom).

3. Auto-disable gate: check `pathtracer.samples > 24` before mounting the
   denoiser pass; disable it after that threshold.

```typescript
// Host-side gate (in PathTracingLayer.tsx or equivalent):
const useSpatialFilter = pathtracer.samples <= 24;
if (useSpatialFilter) {
  composer.addPass(new EffectPass(camera, spatialDenoiserEffect));
}
```

---

## 3. Risk notes

- **Rough refraction + Phase 4 ordering**: always verify that Phase 4 runs
  BEFORE Sprint 6 at the call site.  If both patches land in the same
  sampleBSDF block, the order must be:
  1. Phase 4: apply normal map to `N`
  2. Sprint 6: compute `refDir = refract(-V, N, eta)`, then perturb by GGX

- **Firefly risk**: very rough glass (roughness > 0.8) combined with bright
  caustic emitters can produce firefly spikes where the perturbed refraction
  samples a very-bright emitter at low probability.  Mitigation: the existing
  `filteredGlossyFactor` clamp already handles fireflies in direct lighting;
  verify it applies to transmission samples too.

- **Spatial filter σ tuning**: the three σ values (color, normal, depth) require
  scene-specific tuning.  Suggested starting points:
  - sigmaColor  = 0.1 (HDR chromaticity; tight enough to preserve caustic edges)
  - sigmaNormal = 32.0 (as exponent; harsh threshold for geometry edges)
  - sigmaDepth  = 0.01 (world units; tight to prevent depth-discontinuity blur)

  These should be surfaced as props in the host's `PTSpatialDenoiser` component.
