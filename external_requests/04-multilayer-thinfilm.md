# Multi-Layer Thin-Film Interference via Transfer Matrix Method

**Status:** Proposed
**Originating need:** Consumer materials with vapor-deposited multi-layer optical stacks (distributed Bragg reflectors, structural-color coatings) cannot be faithfully reproduced with the current single-layer iridescence model, which approximates multi-layer behavior using a heuristic thickness range.
**Priority:** Medium

## 1. Motivation

The existing `Material.iridescenceThicknessRange` / `iridescenceIor` fields implement a single-layer thin-film interference model (Belcour & Barla 2017 [1]). This is correct for single thin-film coatings (anodized aluminum, iridescent glass with one SnO₂ coat, soap bubbles) but fails for multi-layer stacks that behave as optical band-pass filters.

Two illustrative cases:

- **Archaeological metalwork (Roman/Byzantine gilded glass, Lycurgus Cup).** Ancient glass artifacts with dichroic effects contain multi-layer colloidal or deposited films whose spectral response has sharp reflection bands — a feature impossible to reproduce with a single-layer model. Conservation scientists use TMM to analyze and document these films [4]. A renderer used for museum visualization needs to match the measured spectral response, not approximate it with a heuristic.
- **Structural-color products.** Dichroic optical coatings (30–70 alternating TiO₂/SiO₂ layers, total thickness 760–890 nm) act as distributed Bragg reflectors: they reflect a narrow wavelength band and transmit the complementary color [3]. The single-layer model produces the right angular color-shift behavior but cannot reproduce the simultaneous two-color (reflect + transmit complement) effect, because the single-layer model's transmission is simply 1 − R, not a spectrally selective band-pass.

The gap is observable: a surface described with single-layer iridescence shows a single dominant hue in reflection and its complement in transmission, but as film thickness varies across a surface, both hues drift continuously. A real Bragg reflector shows a stable narrow-band reflection color with a sharply complementary transmission — the bands don't drift; they remain at a fixed spectral position determined by the layer periods.

## 2. Proposed API Surface

The proposal is a new optional field on `Material` that replaces the `iridescence*` trio with a richer stack description. The existing `iridescence*` fields remain unchanged (for the single-layer case); the new field is an opt-in upgrade.

```typescript
// New types in scene.ts

/**
 * A single layer in a thin-film optical stack.
 * The stack is ordered from topmost (air-adjacent) layer to
 * bottom (substrate-adjacent) layer.
 */
export interface ThinFilmLayer {
  /** Refractive index of this layer (real part). */
  readonly ior: number;
  /** Extinction coefficient (imaginary part of complex IOR). 0 for
   *  dielectric layers; > 0 for lossy or metallic layers. */
  readonly extinctionCoefficient?: number;
  /** Physical layer thickness in nanometers. */
  readonly thicknessNm: number;
}

/**
 * A multi-layer thin-film stack for Transfer Matrix Method evaluation.
 * The engine computes per-wavelength reflectance and transmittance by
 * multiplying the 2×2 characteristic matrices of each layer (Abeles formalism).
 * The resulting spectral R(λ) and T(λ) are convolved with CIE CMFs to
 * produce the RGB BSDF weight used at each shade point.
 */
export interface ThinFilmStack {
  /**
   * Ordered array of thin-film layers, topmost first (closest to incident
   * medium, usually air). The substrate IOR is taken from Material.ior.
   * Minimum: 1 layer. No maximum, but backends may cap for performance
   * (e.g., 64 layers).
   */
  readonly layers: ReadonlyArray<ThinFilmLayer>;

  /**
   * IOR of the incident medium (the medium the ray arrives from).
   * Typically 1.0 (air). Default: 1.0.
   */
  readonly incidentIor?: number;

  /**
   * If true, evaluate TMM per viewing angle (cos θ from Snell's law through
   * each layer). This produces the correct viewing-angle-dependent spectral
   * shift. If false, evaluate at normal incidence only (faster, less accurate
   * for grazing angles). Default: true.
   */
  readonly angleDependent?: boolean;
}

// Addition to the existing Material interface in scene.ts

interface Material {
  // ... existing fields unchanged ...

  // ── Multi-layer thin-film (TMM) ───────────────────────────────────────
  /**
   * Multi-layer thin-film stack evaluated via the Transfer Matrix Method.
   * When present, overrides the single-layer iridescence model
   * (iridescence / iridescenceIor / iridescenceThicknessRange).
   * The TMM result is applied as a wavelength-dependent BSDF weight
   * on the specular lobe.
   */
  thinFilmStack?: ThinFilmStack;
}
```

`ThinFilmStack` and `ThinFilmLayer` are new exported types in `scene.ts`. The `thinFilmStack` field on `Material` is the only change to the existing interface.

## 3. Algorithmic Notes

**Transfer Matrix Method (Abeles formalism)** — Born & Wolf 1959 [5], Macleod 2010 [6]:

For each wavelength λ and each layer i with complex IOR N_i = n_i + ik_i and thickness D_i:

```
Phase thickness: δ_i = (2π / λ) · N_i · D_i · cos(θ_i)
where θ_i is the refraction angle in layer i (Snell's law from incident medium).

Characteristic matrix:
  M_i = [ cos(δ_i)         i · sin(δ_i) / p_i ]
        [ i · p_i · sin(δ_i)    cos(δ_i)       ]

where p_i = N_i · cos(θ_i) (s-polarization) or p_i = N_i / cos(θ_i) (p-polarization).

System matrix: M = M_N · M_{N-1} · … · M_1

Reflectance amplitude: r = (M[0][0] · p0 + M[0][1] · p_sub − M[1][0] − M[1][1] · p_sub) / (denominator)
Reflectance: R(λ) = |r|²
Transmittance: T(λ) = 1 − R(λ) (for lossless dielectric stacks; modified for lossy layers)
```

For a 64-layer stack evaluated at 8 spectral samples (380–700 nm, 40 nm spacing), this is 64 × 8 complex 2×2 matrix multiplications per shade point — feasible in a GPU shader as a small loop. The result is a 8-sample spectral R(λ) curve, convolved with CIE CMFs to yield an RGB weight.

**Reference implementation:** The open-source `tmm` Python library (PyPI) [7] implements this exactly and is the canonical validation reference.

## 4. Backend Implementation Guidance

**`@vitrum/pt-webgl` (forked three-gpu-pathtracer):**
The TMM loop is implementable in GLSL as a small function (under 100 lines). The stack layers are passed as a uniform array of `(ior, extinctionCoef, thicknessNm)` structs. At shade time, the function is called once per specular event, returning an RGB weight. Cap at 64 layers for uniform array sizing. For angle-dependent evaluation, the per-layer θ_i is computed via Snell's law given the incident angle — adds 8 `asin` calls per wavelength sample.

**`@vitrum/pt-webgpu` (future):**
WebGPU's storage buffers allow variable-length layer arrays without the uniform-array cap. The TMM function can be a standalone WGSL function in a shared shader module. Spectral evaluation over 16 or 32 wavelength samples (vs. 8) is feasible.

**`@vitrum/walkaround-hybrid` (real-time GI):**
Real-time TMM evaluation at full spectral resolution per shade point is expensive. Recommended approximation: precompute the TMM spectral curve at engine load time (CPU), fit it as a sum of two Gaussians in wavelength space, and store the Gaussian parameters as material uniforms. The real-time shader evaluates the Gaussian fit — accurate enough for the walkaround use case.

## 5. Validation

**Reference scene — Bragg reflector:** A 30-layer TiO₂ (n=2.35, 75 nm) / SiO₂ (n=1.45, 130 nm) stack on a clear glass substrate. At normal incidence the constructive reflection peak is at approximately:

```
λ_peak = 2 · (n₁D₁ + n₂D₂) = 2 · (2.35·75 + 1.45·130) = 2 · (176 + 189) = 2 · 365 = 730 nm
```

(near-IR; adjusting layer thicknesses to center in visible range is straightforward.)

Validate spectral reflectance curve against the `tmm` Python library [7] and against the Texas Tech TiO₂/SiO₂ DBR paper [4] Figure 3 (measured reflectance of 11-layer stack; similar construction).

**Expected visual:** The plane appears strongly colored in reflection (narrow band) and shows the complementary color in transmission — not the gradually drifting rainbow of single-layer iridescence.

## 6. Consumer-Side Integration

A consumer describing a 30-pair TiO₂/SiO₂ Bragg reflector coating on a glass substrate:

```typescript
import type { Material, ThinFilmStack, ThinFilmLayer } from '@vitrum/core';

// Build alternating TiO₂/SiO₂ layer pairs targeting ~550 nm reflection peak
// Quarter-wave condition: n·D = λ/4 → D_TiO2 = 550/(4·2.35) ≈ 58 nm
//                                       D_SiO2 = 550/(4·1.45) ≈ 95 nm
const braggLayers: ThinFilmLayer[] = [];
for (let i = 0; i < 30; i++) {
  braggLayers.push({ ior: 2.35, thicknessNm: 58 }); // TiO₂
  braggLayers.push({ ior: 1.45, thicknessNm: 95 }); // SiO₂
}

const dichroicStack: ThinFilmStack = {
  layers: braggLayers,
  incidentIor: 1.0,
  angleDependent: true,
};

const dichroicGlass: Material = {
  baseColor: [1, 1, 1],
  roughness: 0.005,
  metallic: 0.0,
  transmission: 1.0,
  ior: 1.52,
  thinFilmStack: dichroicStack,
};
```

## 7. Open Questions

- **Polarization:** The TMM distinguishes s and p polarization. Unpolarized rendering (the common case) averages (R_s + R_p) / 2. Backends should compute both and average unless polarized rendering is a separate future feature.
- **Spatial variation:** Stack parameters are currently per-material (uniform over the mesh). Spatially varying film thickness could be supported via a `thicknessMap: TextureRef` on `ThinFilmLayer`, but this is deferred.

## 8. References

[1] L. Belcour, P. Barla. "A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence." _ACM Transactions on Graphics (Proc. SIGGRAPH 2017)_, vol. 36, no. 4, art. 65, 2017. DOI: 10.1145/3072959.3073620.

[2] "Thin-film interference." Wikipedia. https://en.wikipedia.org/wiki/Thin-film_interference.

[3] "Dichroic glass." Wikipedia. https://en.wikipedia.org/wiki/Dichroic_glass. (30–50 layers; total coating ~760–890 nm; Bragg mirror; complementary reflection/transmission.)

[4] A. Gharbi et al. "SiO₂/TiO₂ distributed Bragg reflector near 1.5 µm fabricated by e-beam evaporation." Texas Tech University, Nanophotonics Group, 2013. https://www.depts.ttu.edu/ece/nanophotonics/papers/2013_SiO2_TiO2%20distributed%20Bragg%20reflector%20near%201.5%20%CE%BCm%20fabricated%20by%20e-beam.pdf. (11-layer TiO₂/SiO₂ stack; measured reflectance curve.)

[5] M. Born, E. Wolf. _Principles of Optics: Electromagnetic Theory of Propagation, Interference and Diffraction of Light._ 1st ed. Pergamon Press, 1959. 7th expanded ed. Cambridge University Press, 1999. (TMM derivation in Chapters 1 and 7.)

[6] H.A. Macleod. _Thin-Film Optical Filters._ 4th ed. CRC Press, 2010. (Standard reference for the Abeles TMM in optical coating design.)

[7] `tmm` Python library. https://pypi.org/project/tmm/. (Reference open-source implementation of the TMM for multilayer thin-film stacks.)
