# Sprint 12 — Hero-Wavelength Spectral PT Fork Patch

**Status**: Vitrum-side COMPLETE. Fork-side kernel rewrite is GATED (see §7 Decision Point).
**Supersedes**: `plan/sprint-12-deferred.md` (deleted).
**Created**: 2026-05-09
**Mode scope**: PT preview + PT final.

---

## Overview

Sprint 12 replaces Sprint 8's RGB-as-3λ approximation with full hero-wavelength
spectral path tracing. Each path samples one wavelength stochastically; the
scalar throughput is reconstructed as RGB at the accumulator via CIE CMF lookup.

**Vitrum-side utilities are complete and tested.** The fork-side kernel rewrite is
a separate, large engineering effort (estimated 4–5 weeks). This document
specifies the full patch so it is ready to execute when the trigger condition is
met.

---

## 1. Vitrum-side deliverables (COMPLETE)

The following are shipped and tested in `@vitrum/shared-samplers`:

| File                         | Status                                                                                             |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `src/cieCmf.ts`              | NEW — CIE 1931 2° CMF tables (81 entries), D65 illuminant, `sampleCMF`, `xyzToLinearSRGB`          |
| `src/wavelengthSampling.ts`  | NEW — `sampleHeroWavelength`, `wavelengthToRGB`, `Y_CMF_INTEGRAL`                                  |
| `src/cauchyIor.ts`           | NEW — `cauchyIOR`, `abbeNumber`, `CAUCHY_CROWN_GLASS`, `CAUCHY_FLINT_GLASS`, `CAUCHY_LEAD_CRYSTAL` |
| `src/index.ts`               | UPDATED — all Sprint 12 symbols re-exported                                                        |
| `__tests__/spectral.test.ts` | NEW — 50 tests covering all three modules                                                          |

**Tests**: 50 new tests added. All 431 workspace tests pass.
**TypeScript strict**: clean across `@vitrum/shared-samplers`.

---

## 2. Fork-side kernel rewrite (GATED — do not begin without trigger confirmation)

### 2.1 Ray payload change

**Current** (Sprint 8 RGB-as-3λ):

```glsl
// Payload carries a vec3 throughput (one value per RGB channel)
vec3 throughput = vec3(1.0);
```

**Sprint 12** (hero-wavelength):

```glsl
// Payload carries scalar wavelength + scalar throughput
float wavelength;  // nm, in [380, 780]
float throughput;  // scalar (dimensionless)
```

**Affected files**:

- `src/shader/shaders/pathtracing/path_tracer.glsl.js` — payload struct, main loop
- Every function that passes or reads throughput (6–8 call sites throughout the tracer)

### 2.2 Wavelength sampling — path initialisation

At the start of each path, sample a hero wavelength using the Y-CMF importance
sampler (GLSL mirror of `sampleHeroWavelength`):

```glsl
// CDF inversion of Y_CMF, precomputed as a uniform array.
// The CPU side uploads the 82-entry CDF array once on init.
uniform float uYCdf[82];   // normalised CDF: uYCdf[0]=0, uYCdf[81]=1

float sampleHeroWavelength(float u, out float pdf) {
  // Binary search
  int lo = 0, hi = 80;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (uYCdf[mid + 1] <= u) lo = mid + 1;
    else hi = mid;
  }
  float cdfLo = uYCdf[lo];
  float cdfHi = uYCdf[lo + 1];
  float yLo = yCmfY[lo];     // uYTable: Y CMF values (81 entries)
  float yHi = yCmfY[lo + 1];
  float t = (cdfHi > cdfLo) ? (u - cdfLo) / (cdfHi - cdfLo) : 0.0;
  float lambda = float(380 + lo * 5) + t * 5.0;
  float yAtLambda = mix(yLo, yHi, t);
  pdf = yAtLambda / uYIntegral;  // uYIntegral = ~106.857
  return lambda;
}
```

**Host side**: upload the 82-entry CDF array + 81-entry Y table + Y_CMF_INTEGRAL
once after renderer init. These are static constants from `@vitrum/shared-samplers`.

### 2.3 BSDF wavelength-awareness

Every BSDF evaluation site must use the sampled `wavelength` for IOR lookup:

```glsl
// Before Sprint 12: three discrete channels
float iorR = uIor + uDispersionStrength / (700.0 * 700.0);
float iorG = uIor + uDispersionStrength / (550.0 * 550.0);
float iorB = uIor + uDispersionStrength / (450.0 * 450.0);

// Sprint 12: single wavelength from the payload
float lambdaUm = wavelength * 0.001;  // nm → µm
float iorAtLambda = uIorA + uIorB / (lambdaUm * lambdaUm) + uIorC / pow(lambdaUm, 4.0);
```

**New uniforms on `PhysicalPathTracingMaterial.js`**:

```javascript
// Sprint 12 Cauchy coefficients (replaces scalar ior + dispersionStrength)
iorCauchyA: { value: 1.5 },      // base IOR
iorCauchyB: { value: 0.0 },      // µm² dispersion (0 = no dispersion)
iorCauchyC: { value: 0.0 },      // µm⁴ higher-order term (optional)
yCmfCdf:    { value: new Float32Array(82) },  // normalised Y CMF CDF
yCmfY:      { value: new Float32Array(81) },  // Y CMF values
yCmfIntegral: { value: 106.857 },             // ∫ Y dλ (nm)
```

**Host-side initialization** (once, after renderer init):

```typescript
import { Y_CMF_INTEGRAL, CIE_Y_TABLE } from '@vitrum/shared-samplers';

// Build CDF (mirrors wavelengthSampling.ts module-level IIFE)
const cdf = new Float32Array(82);
// ... trapezoidal integration of CIE_Y_TABLE ...
material.uniforms.yCmfCdf.value = cdf;
material.uniforms.yCmfY.value = new Float32Array(CIE_Y_TABLE);
material.uniforms.yCmfIntegral.value = Y_CMF_INTEGRAL;
```

**Affected fork files**:

- `src/shader/shaders/pathtracing/bsdf_functions.glsl.js` — all BSDF eval sites
- `src/shader/shaders/pathtracing/direct_lighting.glsl.js` — PDF MIS denominator
- `PhysicalPathTracingMaterial.js` — new uniforms

### 2.4 Spectral accumulator

Replace the RGB accumulation target with per-wavelength → RGB reconstruction:

```glsl
// At each path termination, convert (wavelength, throughput, pdf) to RGB:
float x = sampleCmfX(wavelength);  // CIE x̄(λ), interpolated from uCmfX[]
float y = sampleCmfY(wavelength);  // CIE ȳ(λ)
float z = sampleCmfZ(wavelength);  // CIE z̄(λ)

float weight = throughput / pdfLambda;
vec3 xyz = vec3(x, y, z) * weight;

// XYZ → linear sRGB (Bradford-adapted D65 matrix)
vec3 rgb = mat3(
   3.2404542, -0.9692660,  0.0556434,
  -1.5371385,  1.8760108, -0.2040259,
  -0.4985314,  0.0415560,  1.0572252
) * xyz;

// Accumulate into framebuffer (alpha blend with previous samples)
gl_FragColor += vec4(rgb, 1.0);
```

**New fork file**: `src/shader/shaders/pathtracing/spectral_accumulator.glsl.js`

- Contains: `sampleCmfX`, `sampleCmfY`, `sampleCmfZ` (linear interpolation on uniform arrays)
- XYZ → linear sRGB matrix
- `vec3 wavelengthToRGB(float lambda, float throughput, float pdfLambda)`

**Additional uniforms**:

```javascript
cmfX: { value: new Float32Array(81) },  // CIE x̄ table
cmfY: { value: new Float32Array(81) },  // CIE ȳ table (same as yCmfY)
cmfZ: { value: new Float32Array(81) },  // CIE z̄ table
```

### 2.5 Accumulation target format change

The framebuffer accumulates RGB contributions (not XYZ); no format change needed
on the `WebGLRenderTarget`. The XYZ → sRGB conversion happens inside the shader
before accumulation.

---

## 3. Cauchy IOR migration from Sprint 8

Sprint 8 used two shader uniforms:

- `uIor0` — base IOR
- `uDispersionStrength` — scalar Cauchy B coefficient in nm² scale

Sprint 12 replaces these with three Cauchy coefficients (A, B, C) in µm scale,
matching the `cauchyIOR(lambdaNm, A, B, C)` signature in `cauchyIor.ts`.

**Migration mapping** (host-side, in `createBakedGlassMaterial.ts`):

```typescript
import { CAUCHY_LEAD_CRYSTAL, cauchyIOR } from '@vitrum/shared-samplers';

// Old Sprint 8 path:
// material.uniforms.dispersionStrength.value = profile.dispersionStrength ?? 0;

// New Sprint 12 path:
const { A, B, C } = profile.cauchyCoeffs ?? CAUCHY_LEAD_CRYSTAL;
material.uniforms.iorCauchyA.value = A;
material.uniforms.iorCauchyB.value = B;
material.uniforms.iorCauchyC.value = C;
```

---

## 4. Definition of done (fork-side)

- [ ] Ray payload changed from `vec3 throughput` to `float wavelength + float throughput`
- [ ] `sampleHeroWavelength` GLSL function implemented, using `yCmfCdf` + `yCmfY` uniforms
- [ ] All BSDF evaluation sites use `iorCauchyA/B/C` + hero `wavelength` for IOR
- [ ] `spectral_accumulator.glsl.js` implements CMF sampling + XYZ → sRGB
- [ ] Framebuffer accumulation uses the spectral `vec3 wavelengthToRGB(...)` output
- [ ] `PhysicalPathTracingMaterial.js` updated with all new uniforms
- [ ] Visual A/B: bevel rainbow shows smooth spectrum (8+ visible colours) vs Sprint 8's 3-colour fan
- [ ] All Sprint 8 glass material visual tests still pass (non-bevel glass unaffected)
- [ ] GPU profiling: throughput regression < 30% vs Sprint 8 baseline on 1080p scene

---

## 5. Effort and risk

### Effort estimate

| Phase                     | Scope                                           | Effort                          |
| ------------------------- | ----------------------------------------------- | ------------------------------- |
| Ray payload restructure   | path_tracer.glsl.js                             | 3 days                          |
| BSDF wavelength-awareness | bsdf_functions.glsl.js, direct_lighting.glsl.js | 5 days                          |
| Spectral accumulator      | spectral_accumulator.glsl.js                    | 4 days                          |
| Uniform wiring            | PhysicalPathTracingMaterial.js, host-side       | 2 days                          |
| Validation + visual A/B   | Reference renders, DoD checklist                | 5 days                          |
| Buffer                    | Integration friction, edge cases                | 5 days                          |
| **Total**                 |                                                 | **~24 working days (~5 weeks)** |

### Risk callouts

1. **Fork divergence**: every future `git pull` from `gkjohnson/three-gpu-pathtracer`
   after this patch becomes a multi-day merge. The payload restructure is a
   pervasive change that touches every shader function; merge conflicts are
   near-guaranteed on upstream updates.

2. **Performance regression**: hero-wavelength tracing reduces effective variance
   reduction vs. the 3-sample RGB-as-3λ approach for scenes dominated by broad
   spectral features. Expect 10–30% more samples needed for the same noise level
   on non-dispersive glass. For bevel materials with true spectral variation, the
   tradeoff is favorable.

3. **Uniform array upload cost**: 81-entry CMF arrays × 3 channels + 82-entry CDF.
   At 4 bytes per float, total uniform upload is ~1.2 KB per draw call. Negligible
   vs. texture uploads; not a bottleneck.

4. **Edge case: wavelength outside CMF range**: the accumulator must clamp or
   return 0 for λ < 380 or λ > 780 nm. Mirror the `sampleCMF` guard from
   `cieCmf.ts` in GLSL.

5. **Achromatic glass**: for glass with no dispersion (`iorCauchyB = 0`), the
   hero-wavelength path gives the same result as RGB-as-3λ (no channel splitting).
   The GLSL should detect B ≈ 0 and skip the IOR wavelength computation for
   performance parity with Sprint 8.

---

## 6. Jakob+Hanika rider compatibility

Sprint 8b's `jakobHanika.ts` spectral upsampling remains valid for host-side RGB →
spectral-coefficient conversion. In the Sprint 12 context, the polynomial
coefficients can optionally modulate the throughput at each wavelength:

```glsl
// Optional: weight throughput by the spectral reflectance at the hero wavelength
float specReflectance = evalSpectrum(uJakobCoeffs, wavelength);
throughput *= specReflectance;
```

This composes cleanly with the hero-wavelength accumulator: the spectral
reflectance provides a smooth per-wavelength weight across the full visible range
instead of the 3-discrete-band approximation from Sprint 8.

---

## 7. Decision point (re-surface before starting fork work)

> **Is the visible improvement over RGB-as-3λ worth the kernel rewrite + ongoing
> fork maintenance burden?**

From the roadmap Decision 1:

> "Jakob+Hanika upsampling may make Sprint 12 unnecessary entirely for the bevel
> use case."

**Trigger condition**: start fork work ONLY IF the user confirms one or more of:

- Uranium glass (fluorescence emission by wavelength)
- Dichroic film (multi-order thin-film interference — 3-colour approximation shows aliasing)
- Gemstones with visible absorption bands (e.g., alexandrite colour-shift at ~680 nm)
- Bevel rainbows where the 3-colour fan is visibly discrete in hero renders

If the user confirms the trigger, re-surface this question with a side-by-side
hero render comparison before beginning the 5-week fork rewrite.

---

## 8. Sprint artifact index update

See `plan/phase-6-status.md` for the updated status entry.
