import type { MaterialSpec } from '@vitrum/core';
import {
  rgbToSpectralCoefficients,
  resolveEmissiveIntensity,
  sampleSpectralGrid,
  sigmaAFromAttenuation,
} from '@vitrum/shared-samplers';

import { thinFilmRgb } from '../math/thinFilm.js';
/** pt-webgpu thin-film layer capacity. Declared per-backend in
 *  `@vitrum/core` `BackendSupportDetails.thinFilmLayerLimit` (D1 = Option B);
 *  the WGSL `const THIN_FILM_LAYER_LIMIT = 8u;` in
 *  `wgsl/pathTrace/material.wgsl.ts` MUST stay in lockstep with this value. */
export const THIN_FILM_LAYER_LIMIT = 8;
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
 *   26 baseColor Jakob-Hanika sigmoid coeffs c0,c1,c2 (raw-nm),
 *      materialFlags: bit0 hasSpectralReflectance, bit1 unlit shadingModel,
 *      bit2 doubleSided                                                    ← A3 / GLTF material sides
 *   27 specularColor.rgb, specularIntensity                         ← SPEC-01
 *   28 volumeThickness, hasVolumeThickness, thinFilmRgbLutBaseVec4, _
 *      (the LUT offset is absolute; zero means no sparse RGB LUT)
 *
 * A3: vec4 #26 (`MATERIAL_VEC4_STRIDE` bumped 26 → 27) carries the Jakob &
 * Hanika 2019 RGB→spectrum upsampling coefficients for the material's baseColor,
 * solved ONCE at pack time. In spectral mode the GPU evaluates the sigmoid
 * reflectance S(λ) = sigmoid(c0 + c1·λ + c2·λ²) at the hero wavelength to carry a
 * genuine SCALAR spectral reflectance through the path (replacing the RGB→
 * luminance tint). The .w flag's bit0 is set when the coefficients are valid
 * (always, for every material — black collapses to S≈0 via the solver's
 * pure-black shortcut). bit1 carries `MaterialSpec.shadingModel === 'unlit'`;
 * bit2 carries `MaterialSpec.doubleSided === true` (false when omitted).
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
export const THIN_FILM_RGB_LUT_BINS = 64;
const THIN_FILM_RGB_LUT_FLOATS_PER_BIN = 16;
export const THIN_FILM_RGB_LUT_MAX_ABS_ERROR = 0.02;
const THIN_FILM_RGB_LUT_CERTIFICATION_SAMPLES_PER_INTERVAL = 7;
export const MATERIAL_VEC4_STRIDE = 29;
export const MATERIAL_FLOAT_STRIDE = MATERIAL_VEC4_STRIDE * 4;

// Visible-light 380→780 nm spectral grid sampling now lives in shared-samplers
// `sampleSpectralGrid` (single-sourced with pt-webgl2). The grid bounds are
// canonical there via SPECTRAL_GRID_START_NM/END_NM.

/** Optional per-primitive packing context for {@link materialToPackedVec4s}.
 *  pt-webgpu material slots are PER-PRIMITIVE (no dedup), so primitive-level
 *  flags ride the material payload. */
export interface MaterialPackContext {
  /** SHADOW-01 — the source primitive's `castShadow` flag. `false` packs 1.0
   *  into vec4 #25 .w (castShadowDisabled); `true`/undefined packs 0.0, which
   *  is byte-identical to the pre-SHADOW-01 pad. */
  readonly castShadow?: boolean | undefined;
  /** Absolute vec4 offset into the materials storage buffer for the optional
   * sparse RGB thin-film LUT. Zero is the no-LUT sentinel. */
  readonly thinFilmRgbLutBaseVec4?: number | undefined;
}

export function materialToPackedVec4s(
  material: MaterialSpec,
  context: MaterialPackContext = {},
): number[] {
  const finite = (v: number, fallback = 0): number => (Number.isFinite(v) ? v : fallback);
  const clamp01 = (v: number): number => Math.min(1, Math.max(0, finite(v)));
  const base = material.baseColor;
  const emissive = material.emissive ?? [0, 0, 0];
  const emissiveIntensity = resolveEmissiveIntensity(material.emissiveIntensity);
  const roughness = material.roughness ?? 0.5;
  const metallic = material.metallic ?? 0;
  const transmission = material.transmission ?? 0;
  const hasVolumeThickness =
    material.thickness != null || material.thicknessMap != null;
  const volumeThickness = Math.max(finite(material.thickness ?? 0), 0);
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
  const angleDependentFlag = material.thinFilmStack?.angleDependent !== false ? 1 : 0;
  const spectralCurve = material.spectralAttenuation;
  // 380→780 nm uniform grid μ(λ) sampling + avg/max/count, single-sourced with
  // pt-webgl2 via shared-samplers `sampleSpectralGrid`. pt-webgpu's original
  // sampler accepted a length ≥ 1 curve with raw wavelength bounds → default
  // options ({}) reproduce that byte-for-byte.
  const spectralGrid = sampleSpectralGrid(spectralCurve);
  const spectralSamples = spectralGrid.samples;
  const spectralSampleCount = spectralGrid.sampleCount;
  const spectralAvgMu = spectralGrid.avg;
  const spectralMaxMu = spectralGrid.max;
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
    // σ_a channel math single-sourced with pt-webgl2 via shared-samplers
    // `sigmaAFromAttenuation` (same 1e-4 transmittance clamp + Beer-Lambert log).
    const sigmaA = sigmaAFromAttenuation(
      [attColor[0] ?? 1, attColor[1] ?? 1, attColor[2] ?? 1],
      attDistance,
    );
    packed.push(
      sigmaA[0],
      sigmaA[1],
      sigmaA[2],
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
  // through the path instead of an RGB→luminance tint. .w bit0 = coefficients
  // valid (always; pure black resolves to S≈0 via the solver's shortcut), bit1 =
  // MaterialSpec.shadingModel === 'unlit', bit2 = MaterialSpec.doubleSided.
  // Ref: Jakob & Hanika 2019, "A Low-Dimensional Function Space for Efficient
  //      Spectral Upsampling" (shared-samplers/jakobHanika.ts).
  const [specC0, specC1, specC2] = rgbToSpectralCoefficients(
    finite(base[0] ?? 0),
    finite(base[1] ?? 0),
    finite(base[2] ?? 0),
  );
  const materialFlags =
    1 +
    (material.shadingModel === 'unlit' ? 2 : 0) +
    (material.doubleSided === true ? 4 : 0);
  packed.push(finite(specC0), finite(specC1), finite(specC2), materialFlags);

  // SPEC-01 — KHR_materials_specular scalar factors for dielectric F0. Defaults
  // are specularColor=[1,1,1], specularIntensity=1 so old scenes keep the same
  // 4% dielectric F0. Metallic F0 remains baseColor-driven in WGSL.
  const specularColor = material.specularColor ?? [1, 1, 1];
  const specularIntensity = clamp01(material.specularIntensity ?? 1);
  packed.push(
    clamp01(specularColor[0] ?? 1),
    clamp01(specularColor[1] ?? 1),
    clamp01(specularColor[2] ?? 1),
    specularIntensity,
  );

  // VOL-THICKNESS — KHR_materials_volume thicknessFactor. The optional
  // thicknessMap (full tier only) multiplies this scalar in WGSL. The presence
  // flag preserves glTF's explicit thicknessFactor=0 "no volume" default while
  // letting attenuation paths distinguish "author supplied a slab clamp" from
  // "use geometric segment length".
  packed.push(volumeThickness, hasVolumeThickness ? 1 : 0,
    Math.max(0, finite(context.thinFilmRgbLutBaseVec4 ?? 0)), 0);

  return packed;
}

/** CPU mirror of the shader's direction-aware cosine grid. For high-to-low
 * index transport, bins are split at the critical angle so interpolation never
 * crosses the TIR discontinuity. */
export function thinFilmRgbLutCosTheta(
  bin: number,
  incidentIor: number,
  substrateIor: number,
  reverse: boolean,
): number {
  const clampedBin = Math.min(THIN_FILM_RGB_LUT_BINS - 1, Math.max(0, bin));
  const etaIncident = reverse ? substrateIor : incidentIor;
  const etaTransmitted = reverse ? incidentIor : substrateIor;
  const criticalCos = etaIncident > etaTransmitted
    ? Math.sqrt(Math.max(0, 1 - (etaTransmitted * etaTransmitted) /
      (etaIncident * etaIncident)))
    : 0;
  if (criticalCos > 1e-6) {
    const halfBins = THIN_FILM_RGB_LUT_BINS / 2;
    if (clampedBin < halfBins) {
      const u = clampedBin / (halfBins - 1);
      return criticalCos * u * u;
    }
    const u = (clampedBin - halfBins) / (halfBins - 1);
    return criticalCos + (1 - criticalCos) * u * u;
  }
  const u = clampedBin / (THIN_FILM_RGB_LUT_BINS - 1);
  return u * u;
}

/** Inverse of {@link thinFilmRgbLutCosTheta}; mirrors the WGSL lookup exactly. */
export function thinFilmRgbLutPosition(
  cosTheta: number,
  incidentIor: number,
  substrateIor: number,
  reverse: boolean,
): number {
  const cos = Math.min(1, Math.max(0, cosTheta));
  const etaIncident = reverse ? substrateIor : incidentIor;
  const etaTransmitted = reverse ? incidentIor : substrateIor;
  const criticalCos = etaIncident > etaTransmitted
    ? Math.sqrt(Math.max(0, 1 - (etaTransmitted * etaTransmitted) /
      (etaIncident * etaIncident)))
    : 0;
  if (criticalCos > 1e-6) {
    const halfBins = THIN_FILM_RGB_LUT_BINS / 2;
    if (cos <= criticalCos) {
      return Math.sqrt(Math.min(1, cos / criticalCos)) * (halfBins - 1);
    }
    return halfBins + Math.sqrt(Math.min(1,
      (cos - criticalCos) / Math.max(1 - criticalCos, 1e-8),
    )) * (halfBins - 1);
  }
  return Math.sqrt(cos) * (THIN_FILM_RGB_LUT_BINS - 1);
}

/** Build the optional RGB-mode thin-film payload for one material. The data is
 * appended after the fixed-stride material table; non-film materials return an
 * empty array and therefore pay zero storage-buffer cost. Each bin stores
 * forward then reverse {R.rgb,T.rgb,R_Y,T_Y}. Hero mode bypasses this LUT. */
export function thinFilmRgbLutForMaterial(material: MaterialSpec): number[] {
  const layers = material.thinFilmStack?.layers.slice(0, THIN_FILM_LAYER_LIMIT) ?? [];
  if (layers.length === 0) return [];
  const finite = (v: number, fallback: number): number =>
    Number.isFinite(v) ? v : fallback;
  const incidentIor = Math.max(
    finite(material.thinFilmStack?.incidentIor ?? 1, 1), 1,
  );
  const substrateIor = Math.max(finite(material.ior ?? 1.5, 1.5), 1);
  const rgbInput = {
    layers,
    incidentIor,
    substrateIor,
    angleDependent: material.thinFilmStack?.angleDependent !== false,
  };
  const lut: number[] = [];
  for (let bin = 0; bin < THIN_FILM_RGB_LUT_BINS; bin += 1) {
    const forwardCos = thinFilmRgbLutCosTheta(bin, incidentIor, substrateIor, false);
    const reverseCos = thinFilmRgbLutCosTheta(bin, incidentIor, substrateIor, true);
    const forward = thinFilmRgb({ ...rgbInput, cosTheta: forwardCos, reverse: false });
    const reverse = thinFilmRgb({ ...rgbInput, cosTheta: reverseCos, reverse: true });
    lut.push(
      ...forward.reflectance, ...forward.transmittance,
      forward.reflectanceEnergy, forward.transmittanceEnergy,
      ...reverse.reflectance, ...reverse.transmittance,
      reverse.reflectanceEnergy, reverse.transmittanceEnergy,
    );
  }
  // Per-stack admission certificate. Probe every interpolation interval in
  // both directions, including the independently split TIR regions. This keeps
  // the public domain truthful for unusually thick/high-Q stacks: a stack whose
  // sparse table cannot meet the declared bound is rejected during setScene()
  // instead of silently rendering an inaccurate approximation.
  const packedLut = new Float32Array(lut);
  let maxError = 0;
  for (const reverse of [false, true]) {
    const directionOffset = reverse ? 8 : 0;
    for (let bin = 0; bin < THIN_FILM_RGB_LUT_BINS - 1; bin += 1) {
      const cos0 = thinFilmRgbLutCosTheta(bin, incidentIor, substrateIor, reverse);
      const cos1 = thinFilmRgbLutCosTheta(bin + 1, incidentIor, substrateIor, reverse);
      if (cos1 <= cos0) continue;
      for (
        let probe = 1;
        probe <= THIN_FILM_RGB_LUT_CERTIFICATION_SAMPLES_PER_INTERVAL;
        probe += 1
      ) {
        const alpha = probe /
          (THIN_FILM_RGB_LUT_CERTIFICATION_SAMPLES_PER_INTERVAL + 1);
        const cosTheta = cos0 + (cos1 - cos0) * alpha;
        const reference = thinFilmRgb({ ...rgbInput, cosTheta, reverse });
        const expected = [
          ...reference.reflectance, ...reference.transmittance,
          reference.reflectanceEnergy, reference.transmittanceEnergy,
        ];
        const lutPosition = thinFilmRgbLutPosition(
          cosTheta, incidentIor, substrateIor, reverse);
        const lutAlpha = lutPosition - bin;
        for (let lane = 0; lane < 8; lane += 1) {
          const a = packedLut[bin * 16 + directionOffset + lane]!;
          const b = packedLut[(bin + 1) * 16 + directionOffset + lane]!;
          // Mirror f32 storage plus WGSL mix arithmetic conservatively.
          const actual = Math.fround(
            Math.fround(a) + Math.fround(Math.fround(b - a) * Math.fround(lutAlpha)),
          );
          maxError = Math.max(maxError, Math.abs(actual - expected[lane]!));
        }
      }
    }
  }
  if (maxError > THIN_FILM_RGB_LUT_MAX_ABS_ERROR) {
    throw new Error(
      `pt-webgpu thin-film RGB LUT error ${maxError} exceeds ` +
      `${THIN_FILM_RGB_LUT_MAX_ABS_ERROR}; use spectral mode or simplify the stack.`,
    );
  }
  if (lut.length !== THIN_FILM_RGB_LUT_BINS * THIN_FILM_RGB_LUT_FLOATS_PER_BIN) {
    throw new Error('pt-webgpu thin-film RGB LUT packing invariant failed.');
  }
  return lut;
}
