import type { Material } from '@vitrum/core';
import { CIE_LAMBDA_MIN, CIE_LAMBDA_MAX } from '@vitrum/shared-samplers';

const THIN_FILM_LAYER_LIMIT = 8;
const SPECTRAL_SAMPLE_COUNT = 32;
/** Matches WGSL `MATERIAL_VEC4_STRIDE` (base layers + 8×(ior,thicknessNm,k) + spectral grid + tail). */
const MATERIAL_VEC4_STRIDE = 22;
export const MATERIAL_FLOAT_STRIDE = MATERIAL_VEC4_STRIDE * 4;

// Visible-light wavelength range, canonical from shared-samplers/cieCmf.
// Previously duplicated as SPECTRAL_LAMBDA_MIN_NM/SPECTRAL_LAMBDA_MAX_NM.
const SPECTRAL_LAMBDA_MIN_NM = CIE_LAMBDA_MIN;
const SPECTRAL_LAMBDA_MAX_NM = CIE_LAMBDA_MAX;

function sampleSpectralCurve(curve: Material['spectralAttenuation'], lambdaNm: number): number {
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

export function materialToPackedVec4s(material: Material): number[] {
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
  let spectralMinMu = Number.POSITIVE_INFINITY;
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
      spectralMinMu = Math.min(spectralMinMu, v);
      spectralMaxMu = Math.max(spectralMaxMu, v);
    }
    spectralAvgMu = sum / SPECTRAL_SAMPLE_COUNT;
    if (!Number.isFinite(spectralMinMu)) spectralMinMu = 0;
    if (!Number.isFinite(spectralMaxMu)) spectralMaxMu = 0;
  } else {
    spectralMinMu = 0;
    spectralMaxMu = 0;
  }
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
  packed.push(spectralAvgMu, spectralMinMu, spectralMaxMu, spectralSampleCount);
  return packed;
}
