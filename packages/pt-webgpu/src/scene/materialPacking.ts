import type { MaterialSpec } from '@vitrum/core';
import { CIE_LAMBDA_MIN, CIE_LAMBDA_MAX, rgbToSpectralCoefficients } from '@vitrum/shared-samplers';

const THIN_FILM_LAYER_LIMIT = 8;
const SPECTRAL_SAMPLE_COUNT = 32;
/**
 * Matches WGSL `MATERIAL_VEC4_STRIDE`. MUST stay in lockstep with the constant
 * in `wgsl/pathTrace/material.wgsl.ts` and the `matId * MATERIAL_VEC4_STRIDE`
 * indexing in `caustic.wgsl.ts`; changing it here without the WGSL constant
 * silently misaligns every material read.
 *
 * Layout (vec4 index):
 *   0  baseColor.rgb, roughness
 *   1  emissive.rgb, metallic
 *   2  transmission, ior, scatteringCoeff, scatteringAnisotropy
 *   3  scatteringRgb.xyz, hasSpectralAttenuation
 *   4  frontLayerTx.xyz, frontLayerRoughness
 *   5  backLayerTx.xyz, backLayerRoughness
 *   6  thinFilmEnabled, thinFilmLayerCount, incidentIor, angleDependent
 *   7..12  thin-film layers (8 × {ior, thicknessNm, k}) = 24 floats
 *   13..20 spectral attenuation samples (32 floats)
 *   21 spectralAvgMu, dispersionAbbeNumber, spectralMaxMu, spectralSampleCount
 *   22 σ_a.rgb (Beer-Lambert absorption coefficient), hasSigmaA flag  ← WS4
 *   23 clearcoat, clearcoatRoughness, sheen, sheenRoughness              ← H52
 *   24 sheenColor.rgb, iridescence                                       ← H52
 *   25 iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
 *      castShadowDisabled (1.0 ⇔ source primitive set castShadow:false; 0.0 default)  ← H52 / SHADOW-01
 *   26 baseColor Jakob-Hanika sigmoid coeffs c0,c1,c2 (raw-nm), hasSpectralReflectance flag  ← A3
 *
 * A3: vec4 #26 (`MATERIAL_VEC4_STRIDE` bumped 26 → 27) carries the Jakob &
 * Hanika 2019 RGB→spectrum upsampling coefficients for the material's baseColor,
 * solved ONCE at pack time. In spectral mode the GPU evaluates the sigmoid
 * reflectance S(λ) = sigmoid(c0 + c1·λ + c2·λ²) at the hero wavelength to carry a
 * genuine SCALAR spectral reflectance through the path (replacing the RGB→
 * luminance tint). The .w flag is 1 when the coefficients are valid (always, for
 * every material — black collapses to S≈0 via the solver's pure-black shortcut).
 * spectralEnabled=false ignores this vec4 entirely (RGB path byte-identical).
 * Ref: Jakob & Hanika 2019 (shared-samplers/jakobHanika.ts).
 *
 * WS4: vec4 #22 (`MATERIAL_VEC4_STRIDE` bumped 22 → 23) carries the RGB
 * absorption coefficient σ_a derived from `attenuationColor`/`attenuationDistance`
 * so the volumetric random walk has a real per-channel extinction term that does
 * not depend on the (optional) spectral-attenuation curve.
 *
 * H52: vec4s #23–#25 (`MATERIAL_VEC4_STRIDE` bumped 23 → 26) carry the three
 * Disney extension lobes — clearcoat (additive GGX at F0=0.04), sheen (Charlie
 * retro-reflective), and iridescence (thin-film Fresnel modification of F0).
 * All three scalars default to 0, so zero-default scenes are NUMERICALLY
 * IDENTICAL to the pre-H52 path (the lobes multiply by their scalar and are
 * skipped when it is 0).
 * Refs: glTF KHR_materials_clearcoat, KHR_materials_sheen, KHR_materials_iridescence.
 */
const MATERIAL_VEC4_STRIDE = 27;
export const MATERIAL_FLOAT_STRIDE = MATERIAL_VEC4_STRIDE * 4;

// Visible-light wavelength range, canonical from shared-samplers/cieCmf.
// Previously duplicated as SPECTRAL_LAMBDA_MIN_NM/SPECTRAL_LAMBDA_MAX_NM.
const SPECTRAL_LAMBDA_MIN_NM = CIE_LAMBDA_MIN;
const SPECTRAL_LAMBDA_MAX_NM = CIE_LAMBDA_MAX;

function sampleSpectralCurve(curve: MaterialSpec['spectralAttenuation'], lambdaNm: number): number {
  if (curve == null || curve.values.length === 0) return 0;
  const start = curve.wavelengthStart;
  const end = curve.wavelengthEnd;
  const denom = Math.max(end - start, 1e-6);
  const t = Math.min(1, Math.max(0, (lambdaNm - start) / denom));
  const f = t * (curve.values.length - 1);
  const i0 = Math.floor(f);
  const i1 = Math.min(i0 + 1, curve.values.length - 1);
  const a = curve.values[i0] ?? 0;
  const b = curve.values[i1] ?? a;
  return a + (b - a) * (f - i0);
}

/** Optional per-primitive packing context for {@link materialToPackedVec4s}.
 *  pt-webgpu material slots are PER-PRIMITIVE (no dedup), so primitive-level
 *  flags ride the material payload. */
export interface MaterialPackContext {
  /** SHADOW-01 — the source primitive's `castShadow` flag. `false` packs 1.0
   *  into vec4 #25 .w (castShadowDisabled); `true`/undefined packs 0.0, which
   *  is byte-identical to the pre-SHADOW-01 pad. */
  readonly castShadow?: boolean | undefined;
}

export function materialToPackedVec4s(
  material: MaterialSpec,
  context: MaterialPackContext = {},
): number[] {
  const finite = (v: number, fallback = 0): number => (Number.isFinite(v) ? v : fallback);
  const clamp01 = (v: number): number => Math.min(1, Math.max(0, finite(v)));
  const base = material.baseColor;
  const emissive = material.emissive ?? [0, 0, 0];
  const emissiveIntensity = material.emissiveIntensity ?? 1;
  const roughness = material.roughness ?? 0.5;
  const metallic = material.metallic ?? 0;
  const transmission = material.transmission ?? 0;
  const ior = material.ior ?? 1.5;
  const scatteringCoeff = material.scatteringCoefficient ?? 0;
  const scatteringAnisotropy = material.scatteringAnisotropy ?? 0;
  const scatteringRgb = material.scatteringCoefficientRGB ?? [
    scatteringCoeff,
    scatteringCoeff,
    scatteringCoeff,
  ];
  const frontLayerRaw = material.frontLayer?.transmission ?? [1, 1, 1];
  const frontLayerTx: readonly [number, number, number] = [
    clamp01(frontLayerRaw[0] ?? 1),
    clamp01(frontLayerRaw[1] ?? 1),
    clamp01(frontLayerRaw[2] ?? 1),
  ];
  const frontLayerRoughness =
    material.frontLayer?.roughness == null ? -1 : clamp01(material.frontLayer.roughness);
  const backLayerRaw = material.backLayer?.transmission ?? [1, 1, 1];
  const backLayerTx: readonly [number, number, number] = [
    clamp01(backLayerRaw[0] ?? 1),
    clamp01(backLayerRaw[1] ?? 1),
    clamp01(backLayerRaw[2] ?? 1),
  ];
  const backLayerRoughness =
    material.backLayer?.roughness == null ? -1 : clamp01(material.backLayer.roughness);
  const thinFilmLayers = material.thinFilmStack?.layers ?? [];
  const thinFilmLayerCount = Math.min(thinFilmLayers.length, THIN_FILM_LAYER_LIMIT);
  const thinFilmEnabled = thinFilmLayerCount > 0 ? 1 : 0;
  const incidentIor = Math.max(finite(material.thinFilmStack?.incidentIor ?? 1, 1), 1);
  const angleDependentFlag = material.thinFilmStack?.angleDependent === true ? 1 : 0;
  const spectralCurve = material.spectralAttenuation;
  let spectralSampleCount = 0;
  let spectralAvgMu = 0;
  let spectralMaxMu = Number.NEGATIVE_INFINITY;
  const spectralSamples = new Array<number>(SPECTRAL_SAMPLE_COUNT).fill(0);
  if (spectralCurve != null && spectralCurve.values.length > 0) {
    spectralSampleCount = SPECTRAL_SAMPLE_COUNT;
    let sum = 0;
    for (let i = 0; i < SPECTRAL_SAMPLE_COUNT; i += 1) {
      const t = i / Math.max(SPECTRAL_SAMPLE_COUNT - 1, 1);
      const lambda = SPECTRAL_LAMBDA_MIN_NM + t * (SPECTRAL_LAMBDA_MAX_NM - SPECTRAL_LAMBDA_MIN_NM);
      const v = Math.max(sampleSpectralCurve(spectralCurve, lambda), 0);
      spectralSamples[i] = v;
      sum += v;
      spectralMaxMu = Math.max(spectralMaxMu, v);
    }
    spectralAvgMu = sum / SPECTRAL_SAMPLE_COUNT;
    if (!Number.isFinite(spectralMaxMu)) spectralMaxMu = 0;
  } else {
    spectralMaxMu = 0;
  }
  const dispersionAbbe = Math.max(finite(material.dispersionAbbeNumber ?? 0), 0);
  const packed = [
    base[0],
    base[1],
    base[2],
    roughness,
    emissive[0] * emissiveIntensity,
    emissive[1] * emissiveIntensity,
    emissive[2] * emissiveIntensity,
    metallic,
    transmission,
    ior,
    scatteringCoeff,
    scatteringAnisotropy,
    scatteringRgb[0],
    scatteringRgb[1],
    scatteringRgb[2],
    spectralSampleCount > 0 ? 1 : 0,
    frontLayerTx[0],
    frontLayerTx[1],
    frontLayerTx[2],
    frontLayerRoughness,
    backLayerTx[0],
    backLayerTx[1],
    backLayerTx[2],
    backLayerRoughness,
    thinFilmEnabled,
    thinFilmLayerCount,
    incidentIor,
    angleDependentFlag,
  ];
  for (let i = 0; i < THIN_FILM_LAYER_LIMIT; i += 1) {
    if (i < thinFilmLayerCount) {
      const layer = thinFilmLayers[i];
      packed.push(
        Math.max(finite(layer?.ior ?? 1, 1), 1),
        Math.max(finite(layer?.thicknessNm ?? 0), 0),
        Math.max(finite(layer?.extinctionCoefficient ?? 0), 0),
      );
    } else {
      packed.push(0, 0, 0);
    }
  }
  packed.push(...spectralSamples);
  packed.push(spectralAvgMu, dispersionAbbe, spectralMaxMu, spectralSampleCount);

  // WS4 — volumetric absorption coefficient σ_a (vec4 #22).
  // attenuationColor is the transmittance reached after travelling
  // attenuationDistance through the medium (glTF KHR_materials_volume):
  //   T(d) = attenuationColor = exp(-σ_a · attenuationDistance)
  //   ⇒ σ_a = -ln(attenuationColor) / attenuationDistance   (clamped ≥ 0).
  // The hasSigmaA flag (.w) lets the kernel distinguish "no absorption set"
  // from a deliberate σ_a = 0 (a clear medium). Total extinction in the walk
  // is σ_t = σ_a + σ_s, with σ_s from scatteringCoefficient(RGB).
  // Ref: PBR4e §11.1 "Volume Scattering Processes"; glTF KHR_materials_volume.
  const attColor = material.attenuationColor;
  const attDistance = material.attenuationDistance;
  if (
    attColor != null &&
    attDistance != null &&
    Number.isFinite(attDistance) &&
    attDistance > 0
  ) {
    const sigmaAChannel = (c: number): number => {
      const t = Math.min(Math.max(finite(c, 1), 1e-4), 1);
      return Math.max(-Math.log(t) / attDistance, 0);
    };
    packed.push(
      sigmaAChannel(attColor[0]),
      sigmaAChannel(attColor[1]),
      sigmaAChannel(attColor[2]),
      1, // hasSigmaA
    );
  } else {
    packed.push(0, 0, 0, 0);
  }

  // H52 — Disney extension lobes: clearcoat / sheen / iridescence.
  // All scalar fields default to 0; zero-default scenes are numerically
  // identical to the pre-H52 path because the WGSL lobes multiply by their
  // scalar and the kernel short-circuits when it is 0.
  //
  // Refs: glTF KHR_materials_clearcoat (Spec rev 3.0); KHR_materials_sheen;
  //       KHR_materials_iridescence; Belcour & Barla, "A Practical Extension
  //       to Microfacet Theory for the Modeling of Varying Iridescence,"
  //       ACM TOG 36(4) (SIGGRAPH 2017).

  // vec4 #23: clearcoat, clearcoatRoughness, sheen, sheenRoughness
  const clearcoat = clamp01(material.clearcoat ?? 0);
  const clearcoatRoughness = clamp01(material.clearcoatRoughness ?? 0);
  const sheen = clamp01(material.sheen ?? 0);
  const sheenRoughness = clamp01(material.sheenRoughness ?? 0);
  packed.push(clearcoat, clearcoatRoughness, sheen, sheenRoughness);

  // vec4 #24: sheenColor.rgb, iridescence
  const sheenColor = material.sheenColor ?? [0, 0, 0];
  const iridescence = clamp01(material.iridescence ?? 0);
  packed.push(
    clamp01(sheenColor[0] ?? 0),
    clamp01(sheenColor[1] ?? 0),
    clamp01(sheenColor[2] ?? 0),
    iridescence,
  );

  // vec4 #25: iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
  //           castShadowDisabled (SHADOW-01 — formerly a zero pad; default scenes
  //           pack 0.0 here, byte-identical to the pre-SHADOW-01 layout).
  const iridescenceIor = Math.max(finite(material.iridescenceIor ?? 1.3, 1.3), 1.0);
  const iridescenceRange = material.iridescenceThicknessRange ?? [100, 400];
  const iridescenceMin = Math.max(finite(iridescenceRange[0] ?? 100, 100), 0);
  const iridescenceMax = Math.max(finite(iridescenceRange[1] ?? 400, 400), 0);
  packed.push(iridescenceIor, iridescenceMin, iridescenceMax, context.castShadow === false ? 1 : 0);

  // A3 — vec4 #26: Jakob & Hanika 2019 RGB→spectrum upsampling coefficients for
  // the material's baseColor (linear sRGB → 3-coefficient sigmoid polynomial),
  // solved ONCE here at pack time. In spectral mode the GPU evaluates
  //   S(λ) = sigmoid(c0 + c1·λ + c2·λ²)
  // at the hero wavelength to get a genuine scalar spectral reflectance, carried
  // through the path instead of an RGB→luminance tint. .w = 1 (coefficients
  // always valid; pure black resolves to S≈0 via the solver's shortcut). The RGB
  // path never reads this vec4, so spectralEnabled=false stays byte-identical.
  // Ref: Jakob & Hanika 2019, "A Low-Dimensional Function Space for Efficient
  //      Spectral Upsampling" (shared-samplers/jakobHanika.ts).
  const [specC0, specC1, specC2] = rgbToSpectralCoefficients(
    finite(base[0] ?? 0),
    finite(base[1] ?? 0),
    finite(base[2] ?? 0),
  );
  packed.push(finite(specC0), finite(specC1), finite(specC2), 1);

  return packed;
}
