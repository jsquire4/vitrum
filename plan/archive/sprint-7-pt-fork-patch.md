# Sprint 7 — PT Fork Patch Plan

**Sprint goal**: Volumetric scattering + SSS — atmospheric haze with god-ray shafts,
and correct single-scatter SSS for opalescent / glueChip / ringMottled glass types.

**Fork repo**: `~/projects/three-gpu-pathtracer/` branch `phase4-normalmap-shadow-rays`.
**Do NOT modify the fork without explicit user instruction.** This document
specifies what must be patched in the fork once Sprint 7 kicks off.

**Volume scope decision (Decision 7)**: uniform homogeneous medium only.
Per-region density volumes are explicitly deferred. A `volumeQuality` slider
can cap the density to maintain PT_FINAL budget if needed.

---

## 1. Path tracer main loop restructure

### 1.1 `src/shader/shaders/pathtracing/path_tracer.glsl.js`

The main path-tracing loop gains a volume scatter event at each bounce.

**Current loop** (simplified):

```glsl
for (int i = 0; i < bounces; i++) {
  traceScene(ray, hit);
  shading / NEE;
  bounce ray;
}
```

**Sprint 7 loop**:

```glsl
for (int i = 0; i < bounces; i++) {
  traceScene(ray, hit);

  // Volume: march through the homogeneous medium to find a potential
  // scatter event before the surface hit.
  if (u_volumeDensity > 0.0) {
    float tScatter = volumeMarch(ray, hit.t);
    if (tScatter < hit.t) {
      // Scatter event: evaluate HG phase function, sample new direction,
      // perform NEE to the primary light via equi-angular PDF.
      vec3 scatterPos = ray.origin + tScatter * ray.dir;
      float cosTheta_NEE = ...;
      float phaseVal = hg_phase(cosTheta_NEE, u_anisotropyG);
      float thetaPdf = equiAngularPdf(tScatter, tClosest, D, thetaRange);
      throughput *= u_scatterAlbedo * phaseVal / thetaPdf;
      // Continue ray from scatter position
      ray.origin = scatterPos;
      ray.dir = sampleHG(rand(), rand(), u_anisotropyG, ray.dir);
      continue;
    }
  }

  // Surface hit as before
  shading / NEE;
  bounce ray;
}
```

---

## 2. New file: `volume_march.glsl.js`

New shader chunk defining the homogeneous volume march utilities.

```glsl
// volume_march.glsl.js — homogeneous medium march for Sprint 7.
//
// Scope: uniform density, single scatter, equi-angular NEE.
// Per-region density volumes are out of scope (Decision 7).

// Uniforms provided by PhysicalPathTracingMaterial.js:
//   u_volumeDensity:  float  — extinction coefficient σ_t (0 = no medium)
//   u_scatterAlbedo:  vec3   — σ_s / σ_t per channel (0 = absorb only, 1 = scatter only)
//   u_anisotropyG:    float  — HG anisotropy g ∈ (-1, 1)

// Sample scatter distance using exponential distribution.
// Returns sampled t in [0, maxT]; pdf = density * exp(-density * t).
float sampleExponential(float u, float density, float maxT) {
  if (density <= 0.0) return maxT + 1.0; // no medium
  float t = -log(max(1e-10, 1.0 - u)) / density;
  return min(t, maxT);
}

// Equi-angular PDF for a scatter at distance t along the ray,
// given light at perpendicular distance D and closest-approach t_c.
float equiAngularPdf(float t, float tC, float D, float thetaRange) {
  if (D < 1e-6 || thetaRange < 1e-6) return 1.0 / max(1e-6, t);
  float ratio = (t - tC) / D;
  return 1.0 / (D * thetaRange * (1.0 + ratio * ratio));
}

// Henyey-Greenstein phase function (GLSL mirror of hgPhase.ts).
// See @vitrum/shared-samplers/src/hgPhase.ts for the JS implementation.
float hg_phase(float cosTheta, float g) {
  float g2 = g * g;
  float denom = 1.0 + g2 - 2.0 * g * cosTheta;
  return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

// Main volume march: given a ray and surface hit at tSurface, return the
// scatter distance tScatter (or tSurface if no scatter in medium).
float volumeMarch(vec3 ro, vec3 rd, float tSurface, float u) {
  return sampleExponential(u, u_volumeDensity, tSurface);
}
```

---

## 3. HG phase function and equi-angular — GLSL mirror

The JavaScript implementations in `@vitrum/shared-samplers`:

- `evaluateHG / sampleHG / pdfHG` (`hgPhase.ts`)
- `sampleEquiAngular` (`equiAngular.ts`)

The GLSL equivalents (`hg_phase`, `sampleHG_glsl`, `equiAngularPdf`) live in
`volume_march.glsl.js` and in the main `bsdf.glsl.js` for SSS.

These must match the JS implementations numerically (same formula, same
clamping conventions) so that CPU-side tests cover the shader logic.

---

## 4. `bsdf_functions.glsl.js` — HG phase for SSS

Single-scatter SSS uses the same HG phase function as volume scattering.
Add to the transmission branch of the BSDF:

```glsl
// SSS single scatter (when TRANSLUCENT material flag is set):
if (materialFlags & TRANSLUCENT_BIT) {
  float sigma_t = u_sssSigmaT;  // per-material scatter distance
  float tScatter = sampleExponential(rand(), sigma_t, INF);
  vec3 scatterPos = ro + rd * tScatter;
  rd = sampleHG_glsl(rand(), rand(), u_sssAnisotropyG, rd);
  throughput *= u_sssAlbedo * exp(-sigma_t * tScatter);
}
```

---

## 5. `materials_data_function.glsl.js` — TRANSLUCENT flag

Add a `TRANSLUCENT` material flag bit:

```glsl
const uint TRANSLUCENT_BIT = 0x10u;  // bit 4
```

Map these glass types to `TRANSLUCENT`:

- `opalescent` → `TRANSLUCENT_BIT | SCATTER_PARAMS(sigma_t=0.5, g=0.3)`
- `glueChip` → `TRANSLUCENT_BIT | SCATTER_PARAMS(sigma_t=1.0, g=-0.1)`
- `ringMottled`→ `TRANSLUCENT_BIT | SCATTER_PARAMS(sigma_t=0.8, g=0.1)`

These are initial defaults; the host should expose per-type sliders in
PhotorealismControls (Sprint 7 UI, host side — not documented here).

---

## 6. New uniforms in `PhysicalPathTracingMaterial.js`

```javascript
// Volume uniforms (Sprint 7)
volumeDensity:   { value: 0.0 },   // 0 = disabled, >0 = haze density σ_t
scatterAlbedo:   { value: new THREE.Vector3(0.8, 0.85, 0.9) }, // σ_s / σ_t
anisotropyG:     { value: 0.0 },   // HG g for volume

// Per-material SSS uniforms (Sprint 7)
sssSigmaT:       { value: 0.5 },   // scatter distance reciprocal
sssAlbedo:       { value: new THREE.Vector3(0.9, 0.9, 0.9) },
sssAnisotropyG:  { value: 0.0 },
```

---

## 7. Definition of done

**Vitrum-side** (already complete after Sprint 7 implementation):

- [x] `evaluateHG / sampleHG / pdfHG` in `@vitrum/shared-samplers/src/hgPhase.ts`
- [x] `sampleEquiAngular` in `@vitrum/shared-samplers/src/equiAngular.ts`
- [x] All tests pass (HG normalization, equi-angular distribution property)
- [x] Volume scope locked: uniform medium only (Decision 7)

**Fork-side** (to be verified in `~/projects/three-gpu-pathtracer/`):

- [ ] `path_tracer.glsl.js` — main loop with volume scatter event
- [ ] `volume_march.glsl.js` — new file: exponential scatter, equi-angular PDF, HG
- [ ] `bsdf_functions.glsl.js` — SSS single scatter via HG
- [ ] `materials_data_function.glsl.js` — `TRANSLUCENT_BIT` + glass type mappings
- [ ] `PhysicalPathTracingMaterial.js` — volume + SSS uniforms
- [ ] Visual A/B: backlit panel + sun produces visible god-ray shafts at ≥0.5 density
- [ ] Visual A/B: opalescent panel shows milky internal glow

---

## 8. Risk notes

- **Volume per-sample cost**: each volume march adds ~1 RNG call + exponential
  - possibly a NEE sample. Expected +20% per-sample cost. If PT_FINAL budget
    is exceeded for hero scenes, expose a `volumeQuality: 'low' | 'full'` slider
    in the host app to disable equi-angular NEE in the low mode.

- **Equi-angular vs. exponential MIS**: for scenes with the light inside the
  volume, pure equi-angular sampling degrades. The proper solution is MIS
  between equi-angular and exponential distance sampling (Kulla & Conty §4).
  Sprint 7 starts with equi-angular only; MIS upgrade is a follow-up if
  variance is high for interior-light scenes.
