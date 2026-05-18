# Sprint 8 — PT Fork Patch Plan

**Sprint goal**: RGB-as-3-wavelengths spectral + Jakob+Hanika upsampling.
Bevel glass renders true rainbow dispersion via per-channel IOR from the Cauchy
formula. The Jakob+Hanika rider smooths the 3-color fan into a continuous
polynomial spectrum.

**Fork repo**: `~/projects/three-gpu-pathtracer/` branch `phase4-normalmap-shadow-rays`.
**Do NOT modify the fork without explicit user instruction.** This document
specifies what must be patched in the fork once Sprint 8 kicks off.

---

## 1. Dielectric BSDF — per-channel IOR via Cauchy formula

### 1.1 `src/shader/shaders/pathtracing/bsdf_functions.glsl.js`

**Goal**: evaluate the dielectric (glass) BSDF three times — once per
representative wavelength {λ_R, λ_G, λ_B} — using IOR computed from the Cauchy
dispersion formula at each wavelength.

#### Cauchy formula

For a given glass with base IOR `n₀` at 589.3 nm (sodium D line) and Abbe
number `V_d` (or dispersion coefficient `B`):

```
ior(λ) = n₀ + B / λ² + C / λ⁴
```

Simplified (first-order Cauchy, sufficient for visual quality):

```
ior(λ) ≈ n₀ + B / λ²
```

The `B` coefficient relates to the Abbe number as:

```
B ≈ (n₀ - 1) / (V_d × (1/λ_F² - 1/λ_C²))
```

For lead crystal (Abbe V_d ≈ 32, n₀ ≈ 1.58):

- B ≈ 0.0085 μm² (λ in μm) → B ≈ 8.5e6 nm² (λ in nm)
- Dispersion strength `u_dispersionStrength = 0.018` (default, per roadmap)

Practical per-channel IOR at representative wavelengths:

```glsl
// u_ior0:              Base IOR at 589.3 nm (sodium D line)
// u_dispersionStrength: B coefficient (nm²) scaled to sensible slider range.
//                       Default 0.018 ≈ 8500 / (589.3²) for lead crystal.

const float LAMBDA_R = 700.0;  // nm — red channel
const float LAMBDA_G = 550.0;  // nm — green channel
const float LAMBDA_B = 450.0;  // nm — blue channel

float iorR = u_ior0 + u_dispersionStrength / (LAMBDA_R * LAMBDA_R);
float iorG = u_ior0 + u_dispersionStrength / (LAMBDA_G * LAMBDA_G);
float iorB = u_ior0 + u_dispersionStrength / (LAMBDA_B * LAMBDA_B);
```

#### Evaluation

For each refraction event in the dielectric branch of `sampleBSDF`, evaluate
the refraction direction three times:

```glsl
vec3 refDirR = refract(-V, N, 1.0 / iorR);
vec3 refDirG = refract(-V, N, 1.0 / iorG);
vec3 refDirB = refract(-V, N, 1.0 / iorB);

// Average the throughput contributions (RGB-as-3λ approximation):
// Each channel contributes 1/3 of the total power at its refracted direction.
// The path continues along one randomly chosen direction for efficiency
// (stochastic wavelength selection), weighted by 3× throughput for that channel.
float choiceRand = rand();
vec3 refDir;
vec3 channelMask;
if (choiceRand < 1.0/3.0) {
  refDir = refDirR;
  channelMask = vec3(3.0, 0.0, 0.0);  // weight the red channel
} else if (choiceRand < 2.0/3.0) {
  refDir = refDirG;
  channelMask = vec3(0.0, 3.0, 0.0);
} else {
  refDir = refDirB;
  channelMask = vec3(0.0, 0.0, 3.0);
}
throughput *= channelMask;
```

**Gate**: only apply spectral dispersion when `u_dispersionStrength > 1e-4`.
When dispersion is off, fall back to the single-IOR path (no performance hit
for non-bevel glass).

---

## 2. Jakob+Hanika spectral upsampling rider

### 2.1 GLSL polynomial evaluation

The 3-coefficient polynomial from `rgbToSpectralCoefficients` (vitrum-side,
`packages/shared-samplers/src/jakobHanika.ts`) maps directly to 6 GLSL
instructions per channel:

```glsl
// Evaluate polynomial spectrum at wavelength lambda (nm).
// coeffs = (c0, c1, c2) from rgbToSpectralCoefficients.
float evalSpectrum(vec3 coeffs, float lambda) {
  float x = coeffs.x + coeffs.y * lambda + coeffs.z * lambda * lambda;
  // sigmoid: 0.5 + x / (2 * sqrt(1 + x*x))
  return 0.5 + x * inversesqrt(1.0 + x * x) * 0.5;
}

// Per-channel IOR weighted by the polynomial spectrum at that wavelength:
float specR = evalSpectrum(u_jakobCoeffs, LAMBDA_R);
float specG = evalSpectrum(u_jakobCoeffs, LAMBDA_G);
float specB = evalSpectrum(u_jakobCoeffs, LAMBDA_B);

// Effective IOR for this glass color at each channel:
float iorR = u_ior0 + u_dispersionStrength * specR / (LAMBDA_R * LAMBDA_R);
float iorG = u_ior0 + u_dispersionStrength * specG / (LAMBDA_G * LAMBDA_G);
float iorB = u_ior0 + u_dispersionStrength * specB / (LAMBDA_B * LAMBDA_B);
```

**Host side**: the host computes `rgbToSpectralCoefficients(r, g, b)` from the
glass color and uploads the result as `u_jakobCoeffs` (vec3 uniform). This
is a per-material uniform, not a per-frame one (the glass color changes only
on material edit, not every frame).

```typescript
// Host-side pseudo-code (in createBakedGlassMaterial.ts or similar):
import { rgbToSpectralCoefficients } from '@vitrum/shared-samplers';

const [c0, c1, c2] = rgbToSpectralCoefficients(glassR, glassG, glassB);
material.uniforms.jakobCoeffs.value.set(c0, c1, c2);
```

### 2.2 `PhysicalPathTracingMaterial.js` — new uniforms

```javascript
// Sprint 8 uniforms
ior0:               { value: 1.5 },
dispersionStrength:  { value: 0.0 },    // 0 = disabled; 0.018 = lead crystal
jakobCoeffs:         { value: new THREE.Vector3(0, 0, 0) }, // (c0, c1, c2)
```

`jakobCoeffs` defaults to (0, 0, 0) which gives `evalSpectrum(_, λ) = sigmoid(0) = 0.5`
— a flat 50% spectrum that, combined with the Cauchy formula, gives equal IOR
perturbation at all wavelengths (no colorimetric modification). Hosts set
this to the real spectral coefficients for chromatic glass.

---

## 3. Glass material profiles (host side)

### 3.1 `glassMaterialProfiles.ts` (host app file, NOT fork)

Add `dispersionStrength?: number` to the glass material profile schema:

```typescript
export interface GlassMaterialProfile {
  ior: number;
  roughness: number;
  // ... existing fields ...
  /** Cauchy B coefficient for chromatic dispersion.
   *  0 = no dispersion (default for most glass types).
   *  0.018 = lead crystal (bevel cells, default).
   *  Range [0, 0.05]; anything above 0.05 is artistically exaggerated. */
  dispersionStrength?: number;
}
```

### 3.2 `createBakedGlassMaterial.ts` (host app file, NOT fork)

Wire `dispersionStrength` and `jakobCoeffs` into the `onBeforeCompile` shader
injection:

```typescript
material.onBeforeCompile = (shader) => {
  shader.uniforms.dispersionStrength = { value: profile.dispersionStrength ?? 0 };
  shader.uniforms.jakobCoeffs = {
    value: new THREE.Vector3(...rgbToSpectralCoefficients(r, g, b)),
  };
  // ... existing Phase 4 injection ...
};
```

---

## 4. `bevels.ts` baker (host side) — remove fake noise-split

**Current (fake) implementation** in `bevels.ts`:

```typescript
// HACK: simulate dispersion by splitting the bevel mesh into
// R/G/B sub-meshes with slightly different normals.  Remove in Sprint 8.
const rMesh = createBevelMesh({ normalNoise: 0.02 });
const gMesh = createBevelMesh({ normalNoise: 0.01 });
const bMesh = createBevelMesh({ normalNoise: 0.0 });
```

**Sprint 8**: remove the sub-mesh split entirely. The bevel cell is a single
mesh with `dispersionStrength = 0.018` in its material profile. The Cauchy
formula in the fork shader handles dispersion physically.

---

## 5. Definition of done

**Vitrum-side** (already complete after Sprint 8 implementation):

- [x] `rgbToSpectralCoefficients` + `evaluateSpectrum` in `@vitrum/shared-samplers/src/jakobHanika.ts`
- [x] Full documentation of placeholder vs. full table in source (see jakobHanika.ts header)
- [x] Tests pass: achromatic flat spectrum, chromatic primaries peak at correct λ

**Fork-side** (to be verified in `~/projects/three-gpu-pathtracer/`):

- [ ] Cauchy IOR formula at {700, 550, 450} nm in `bsdf_functions.glsl.js`
- [ ] Stochastic wavelength selection (1/3 probability per channel)
- [ ] `evalSpectrum` GLSL function (6 instructions per channel)
- [ ] `PhysicalPathTracingMaterial.js` — `ior0`, `dispersionStrength`, `jakobCoeffs` uniforms
- [ ] `dispersionStrength = 0` fast path (no spectral split for non-bevel glass)

**Host-side** (to be verified in the host application):

- [ ] `glassMaterialProfiles.ts` — `dispersionStrength` field added
- [ ] `createBakedGlassMaterial.ts` — `jakobCoeffs` computed + uploaded
- [ ] `bevels.ts` — fake noise-split removed; single mesh with `dispersionStrength = 0.018`

---

## 6. Best-judgment notes on Jakob+Hanika placeholder

**Decision**: use the compact placeholder approximation rather than the full precomputed table.

**Rationale**:

1. The full precomputed table (≥24 MB) is too large for a browser library bundle.
2. The table's redistribution license is not confirmed for open-source use.
3. The placeholder captures the primary visual effect (spectral peak at each
   channel's representative wavelength) which is sufficient for bevel rainbow
   dispersion.
4. Sprint 12 (hero-wavelength spectral) already handles full spectral accuracy
   as a gated future sprint.

**Accuracy gap**: the placeholder may show slight banding on saturated colors
(e.g., a deep ruby red may not have as smooth a spectrum as the full table
would produce). For the primary use case (clear lead-crystal bevel cells with
subtle rainbow splitting), this is visually acceptable.

**Upgrade path**: if the full table becomes available, `rgbToSpectralCoefficients`
can be replaced with a table lookup without changing the call sites or the
coefficient format. The polynomial structure (`c0 + c1·λ + c2·λ²`) is
identical in both the placeholder and the full Jakob+Hanika 2019 table.
