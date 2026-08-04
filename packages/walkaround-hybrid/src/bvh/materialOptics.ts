import { effectiveMaterialIor, type MaterialSpec } from '@vitrum/core';
import { materialDefinesBulkOpticalMedium } from '@vitrum/shared-bvh';
import {
  CIE_D65_TABLE,
  CIE_LAMBDA_MIN,
  CIE_LAMBDA_STEP,
  CIE_TABLE_LENGTH,
  CIE_X_TABLE,
  CIE_Y_TABLE,
  CIE_Z_TABLE,
  dispersionStrengthFromAbbe,
  sampleSpectralGrid,
  xyzToLinearSRGB,
} from '@vitrum/shared-samplers';

/** Canonical realtime optical reduction shared by shade, ReSTIR, DDGI, and RC. */
export const MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT = 32;
export const MATERIAL_OPTICAL_ANGLE_SAMPLE_COUNT = 8;
export const MATERIAL_OPTICAL_META_TEXELS =
  2 + MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT + MATERIAL_OPTICAL_ANGLE_SAMPLE_COUNT * 4;

export const MATERIAL_OPTICAL_RELATIVE_OFFSETS = {
  HEADER: 0,
  DISPERSION_IOR_RGB: 1,
  SPECTRAL_SAMPLES: 2,
  THIN_FILM_FRONT_REFLECTANCE: 34,
  THIN_FILM_FRONT_TRANSMITTANCE: 42,
  THIN_FILM_BACK_REFLECTANCE: 50,
  THIN_FILM_BACK_TRANSMITTANCE: 58,
} as const;

type Complex = readonly [number, number];
type Rgb = readonly [number, number, number];

const cAdd = (a: Complex, b: Complex): Complex => [a[0] + b[0], a[1] + b[1]];
const cSub = (a: Complex, b: Complex): Complex => [a[0] - b[0], a[1] - b[1]];
const cMul = (a: Complex, b: Complex): Complex => [
  a[0] * b[0] - a[1] * b[1],
  a[0] * b[1] + a[1] * b[0],
];
const cScale = (a: Complex, s: number): Complex => [a[0] * s, a[1] * s];
const cAbs2 = (a: Complex): number => a[0] * a[0] + a[1] * a[1];

function cDiv(a: Complex, b: Complex): Complex {
  const denominator = cAbs2(b);
  if (!(denominator > 1e-30) || !Number.isFinite(denominator)) return [0, 0];
  return [
    (a[0] * b[0] + a[1] * b[1]) / denominator,
    (a[1] * b[0] - a[0] * b[1]) / denominator,
  ];
}

function cSqrt(value: Complex): Complex {
  const radius = Math.hypot(value[0], value[1]);
  const real = Math.sqrt(Math.max(0, 0.5 * (radius + value[0])));
  const imaginary = Math.sqrt(Math.max(0, 0.5 * (radius - value[0])));
  return [real, value[1] < 0 ? -imaginary : imaginary];
}

function cExpI(value: Complex): Complex {
  const amplitude = Math.exp(Math.max(-80, Math.min(0, -value[1])));
  return [amplitude * Math.cos(value[0]), amplitude * Math.sin(value[0])];
}

interface ScatterMatrix {
  readonly rL: Complex;
  readonly tLR: Complex;
  readonly rR: Complex;
  readonly tRL: Complex;
}

function scatterIdentity(): ScatterMatrix {
  return { rL: [0, 0], tLR: [1, 0], rR: [0, 0], tRL: [1, 0] };
}

/** Stable Redheffer star product; avoids the transfer-matrix overflow of thick lossy films. */
function cascadeScatter(a: ScatterMatrix, b: ScatterMatrix): ScatterMatrix {
  const inverse = cDiv([1, 0], cSub([1, 0], cMul(a.rR, b.rL)));
  return {
    rL: cAdd(a.rL, cMul(cMul(cMul(a.tRL, b.rL), inverse), a.tLR)),
    tLR: cMul(cMul(b.tLR, inverse), a.tLR),
    rR: cAdd(b.rR, cMul(cMul(cMul(b.tLR, a.rR), inverse), b.tRL)),
    tRL: cMul(cMul(a.tRL, inverse), b.tRL),
  };
}

function physicalCosine(ior: Complex, transverse: number): Complex {
  const ratio = cDiv([transverse, 0], ior);
  let cosine = cSqrt(cSub([1, 0], cMul(ratio, ratio)));
  const kz = cMul(ior, cosine);
  if (kz[1] < -1e-12 || (Math.abs(kz[1]) <= 1e-12 && kz[0] < 0)) {
    cosine = cScale(cosine, -1);
  }
  return cosine;
}

function materialIorAtWavelength(material: MaterialSpec, wavelengthNm: number): number {
  const baseIor = effectiveMaterialIor(material.ior);
  const abbe = material.dispersionAbbeNumber;
  if (!(abbe != null && Number.isFinite(abbe) && abbe > 0 && baseIor > 1)) return baseIor;
  const bNm2 = dispersionStrengthFromAbbe(baseIor, abbe);
  const a = baseIor - bNm2 / (589.3 * 589.3);
  return Math.max(1, a + bNm2 / (wavelengthNm * wavelengthNm));
}

function polarizedThinFilm(
  material: MaterialSpec,
  wavelengthNm: number,
  cosTheta: number,
  reverse: boolean,
  pPolarized: boolean,
): readonly [reflectance: number, transmittance: number] {
  const stack = material.thinFilmStack;
  if (stack == null) return [0, 1];
  const layers = reverse ? [...stack.layers].reverse() : [...stack.layers];
  const incidentIor = Math.max(1e-6, stack.incidentIor ?? 1);
  const substrateIor = materialIorAtWavelength(material, wavelengthNm);
  const first: Complex = reverse ? [substrateIor, 0] : [incidentIor, 0];
  const last: Complex = reverse ? [incidentIor, 0] : [substrateIor, 0];
  const media: Complex[] = [
    first,
    ...layers.map((layer): Complex => [layer.ior, Math.max(0, layer.extinctionCoefficient ?? 0)]),
    last,
  ];
  const c0 = stack.angleDependent === false ? 1 : Math.max(1e-4, Math.min(1, cosTheta));
  const transverse = first[0] * Math.sqrt(Math.max(0, 1 - c0 * c0));
  const cosines = media.map((ior) => physicalCosine(ior, transverse));
  const admittance = media.map((ior, index) => pPolarized
    ? cDiv(cosines[index]!, ior)
    : cMul(ior, cosines[index]!));
  let network = scatterIdentity();

  for (let index = 0; index < media.length - 1; index += 1) {
    const sum = cAdd(admittance[index]!, admittance[index + 1]!);
    const rL = cDiv(cSub(admittance[index]!, admittance[index + 1]!), sum);
    network = cascadeScatter(network, {
      rL,
      tLR: cDiv(cScale(admittance[index]!, 2), sum),
      rR: cScale(rL, -1),
      tRL: cDiv(cScale(admittance[index + 1]!, 2), sum),
    });
    if (index < layers.length) {
      const phase = cScale(
        cMul(media[index + 1]!, cosines[index + 1]!),
        2 * Math.PI * layers[index]!.thicknessNm / wavelengthNm,
      );
      const propagation = cExpI(phase);
      network = cascadeScatter(network, {
        rL: [0, 0],
        tLR: propagation,
        rR: [0, 0],
        tRL: propagation,
      });
    }
  }

  return [
    cAbs2(network.rL),
    Math.max(admittance.at(-1)![0], 0) /
      Math.max(admittance[0]![0], 1e-15) * cAbs2(network.tLR),
  ];
}

function thinFilmRt(
  material: MaterialSpec,
  wavelengthNm: number,
  cosTheta: number,
  reverse: boolean,
): readonly [reflectance: number, transmittance: number] {
  const [rs, ts] = polarizedThinFilm(material, wavelengthNm, cosTheta, reverse, false);
  const [rp, tp] = polarizedThinFilm(material, wavelengthNm, cosTheta, reverse, true);
  let reflectance = Math.max(0, 0.5 * (rs + rp));
  let transmittance = Math.max(0, 0.5 * (ts + tp));
  if (!Number.isFinite(reflectance) || !Number.isFinite(transmittance)) return [1, 0];
  const energy = reflectance + transmittance;
  if (energy > 1 + 1e-7) return [1, 0];
  if (energy > 1) {
    reflectance /= energy;
    transmittance /= energy;
  }
  return [reflectance, transmittance];
}

let cieNormY = 0;
for (let i = 0; i < CIE_TABLE_LENGTH; i += 1) {
  const endpoint = i === 0 || i === CIE_TABLE_LENGTH - 1 ? 0.5 : 1;
  cieNormY += endpoint * CIE_D65_TABLE[i]! * CIE_Y_TABLE[i]! * CIE_LAMBDA_STEP;
}

function thinFilmRgb(material: MaterialSpec, cosTheta: number, reverse: boolean): {
  readonly reflectance: Rgb;
  readonly transmittance: Rgb;
} {
  let xr = 0; let yr = 0; let zr = 0;
  let xt = 0; let yt = 0; let zt = 0;
  for (let i = 0; i < CIE_TABLE_LENGTH; i += 1) {
    const wavelengthNm = CIE_LAMBDA_MIN + i * CIE_LAMBDA_STEP;
    const [reflectance, transmittance] = thinFilmRt(material, wavelengthNm, cosTheta, reverse);
    const endpoint = i === 0 || i === CIE_TABLE_LENGTH - 1 ? 0.5 : 1;
    const weight = endpoint * CIE_D65_TABLE[i]! * CIE_LAMBDA_STEP / cieNormY;
    xr += reflectance * weight * CIE_X_TABLE[i]!;
    yr += reflectance * weight * CIE_Y_TABLE[i]!;
    zr += reflectance * weight * CIE_Z_TABLE[i]!;
    xt += transmittance * weight * CIE_X_TABLE[i]!;
    yt += transmittance * weight * CIE_Y_TABLE[i]!;
    zt += transmittance * weight * CIE_Z_TABLE[i]!;
  }
  const gamut = (rgb: readonly [number, number, number]): Rgb => [
    Math.max(0, Math.min(1, rgb[0])),
    Math.max(0, Math.min(1, rgb[1])),
    Math.max(0, Math.min(1, rgb[2])),
  ];
  return {
    reflectance: gamut(xyzToLinearSRGB(xr, yr, zr)),
    transmittance: gamut(xyzToLinearSRGB(xt, yt, zt)),
  };
}

function interpolateCieTable(table: Readonly<Float32Array>, wavelengthNm: number): number {
  const f = Math.max(0, Math.min(CIE_TABLE_LENGTH - 1,
    (wavelengthNm - CIE_LAMBDA_MIN) / CIE_LAMBDA_STEP));
  const lo = Math.floor(f);
  const hi = Math.min(CIE_TABLE_LENGTH - 1, lo + 1);
  return (table[lo] ?? 0) + ((table[hi] ?? 0) - (table[lo] ?? 0)) * (f - lo);
}

const spectralRgbWeights: readonly Rgb[] = (() => {
  const raw: Rgb[] = [];
  const sums = [0, 0, 0];
  for (let i = 0; i < MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT; i += 1) {
    const t = i / (MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT - 1);
    const wavelengthNm = 380 + t * 400;
    const endpoint = i === 0 || i === MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT - 1 ? 0.5 : 1;
    const d65 = interpolateCieTable(CIE_D65_TABLE, wavelengthNm);
    const xyz: Rgb = [
      interpolateCieTable(CIE_X_TABLE, wavelengthNm) * d65 * endpoint,
      interpolateCieTable(CIE_Y_TABLE, wavelengthNm) * d65 * endpoint,
      interpolateCieTable(CIE_Z_TABLE, wavelengthNm) * d65 * endpoint,
    ];
    const rgb = xyzToLinearSRGB(xyz[0], xyz[1], xyz[2]);
    raw.push([rgb[0], rgb[1], rgb[2]]);
    sums[0] = (sums[0] ?? 0) + rgb[0];
    sums[1] = (sums[1] ?? 0) + rgb[1];
    sums[2] = (sums[2] ?? 0) + rgb[2];
  }
  return raw.map((rgb): Rgb => [
    rgb[0] / Math.max(Math.abs(sums[0]!), 1e-12),
    rgb[1] / Math.max(Math.abs(sums[1]!), 1e-12),
    rgb[2] / Math.max(Math.abs(sums[2]!), 1e-12),
  ]);
})();

function writeRgb(out: Float32Array, texel: number, rgb: Rgb, alpha = 0): void {
  const base = texel * 4;
  out[base] = rgb[0];
  out[base + 1] = rgb[1];
  out[base + 2] = rgb[2];
  out[base + 3] = alpha;
}

/** True when the core contract activates a participating transmissive volume. */
export function materialHasParticipatingMedium(
  material: MaterialSpec | undefined,
): boolean {
  if (material == null || !(material.transmission != null && material.transmission > 0)) {
    return false;
  }
  if (material.scatteringCoefficientRGB != null) {
    return material.scatteringCoefficientRGB.some((value) => value > 0);
  }
  return (material.scatteringCoefficient ?? 0) > 0;
}

/** Actual positive RGB Beer absorption, not mere presence of a clear payload. */
export function materialHasPositiveRgbAbsorption(
  material: MaterialSpec | undefined,
): boolean {
  const color = material?.attenuationColor;
  const distance = material?.attenuationDistance;
  return color != null && distance != null && Number.isFinite(distance) &&
    distance > 0 && (color[0] < 1 || color[1] < 1 || color[2] < 1);
}

/**
 * Reference length encoded in the realtime optical header.
 *
 * Ordinary volume materials use authored `thickness`, matching the pre-raised
 * Beer tint. A zero-thickness participating medium is still volumetric under
 * the core contract; use its finite attenuation distance as the reference (or
 * one scene unit for identity absorption) so geometric segment transport can
 * retain a positive topology lane without adding another atlas texel.
 */
export function materialOpticalReferenceDistance(
  material: MaterialSpec | undefined,
): number {
  if (material == null) return 0;
  const thickness = Math.max(0, material.thickness ?? 0);
  if (thickness > 0) return thickness;
  if (!materialDefinesBulkOpticalMedium(material)) return 0;
  if (materialHasPositiveRgbAbsorption(material)) {
    return material.attenuationDistance!;
  }
  return 1;
}

/** RGB Beer reference paired with {@link materialOpticalReferenceDistance}. */
export function materialRealtimeBeerReference(
  material: MaterialSpec,
): Rgb | null {
  if (
    !(material.transmission != null && material.transmission > 0) ||
    !materialDefinesBulkOpticalMedium(material) ||
    (material.thickness ?? 0) > 0
  ) {
    return null;
  }
  if (materialHasPositiveRgbAbsorption(material)) return material.attenuationColor!;
  return [1, 1, 1];
}

/**
 * Preintegrate spectral attenuation and thin-film transport into the fixed
 * material-meta ABI. The spectral grid is the same 32-sample 380–780 nm grid
 * used by both converged path tracers; thin-film bins use full 5 nm CIE/D65
 * quadrature and preserve forward/reverse stack order.
 */
export function packMaterialOpticalMeta(material: MaterialSpec | undefined): Float32Array {
  const out = new Float32Array(MATERIAL_OPTICAL_META_TEXELS * 4);
  if (material == null) return out;
  const spectral = sampleSpectralGrid(material.spectralAttenuation, { minValueCount: 3 });
  const hasSpectral = spectral.sampleCount === MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT;
  const hasThinFilm = material.thinFilmStack != null && material.thinFilmStack.layers.length > 0;
  out[0] = hasSpectral ? MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT : 0;
  out[1] = hasThinFilm ? 1 : 0;
  out[2] = material.thinFilmStack?.angleDependent === false ? 0 : 1;
  // The fourth header lane carries the authored homogeneous-medium path
  // length. Thin-film layer count is a CPU validation concern and is not
  // needed by the preintegrated shader lookup.
  const referenceDistance = materialOpticalReferenceDistance(material);
  // Positive = authored thickness cap (thicknessMap may scale it). Negative =
  // synthetic topology/reference distance for a zero-thickness bulk medium;
  // shaders use closed-geometry segment distance and must ignore thicknessMap.
  out[3] = referenceDistance === 0
    ? 0
    : (material.thickness ?? 0) > 0
      ? referenceDistance
      : -referenceDistance;

  writeRgb(out, MATERIAL_OPTICAL_RELATIVE_OFFSETS.DISPERSION_IOR_RGB, [
    materialIorAtWavelength(material, 610),
    materialIorAtWavelength(material, 550),
    materialIorAtWavelength(material, 460),
  ], effectiveMaterialIor(material.ior));

  if (hasSpectral) {
    for (let i = 0; i < MATERIAL_OPTICAL_SPECTRAL_SAMPLE_COUNT; i += 1) {
      const base = (MATERIAL_OPTICAL_RELATIVE_OFFSETS.SPECTRAL_SAMPLES + i) * 4;
      out[base] = Math.max(0, spectral.samples[i] ?? 0);
      out[base + 1] = spectralRgbWeights[i]![0];
      out[base + 2] = spectralRgbWeights[i]![1];
      out[base + 3] = spectralRgbWeights[i]![2];
    }
  }

  if (hasThinFilm) {
    for (let i = 0; i < MATERIAL_OPTICAL_ANGLE_SAMPLE_COUNT; i += 1) {
      const cosTheta = Math.max(1e-4, i / (MATERIAL_OPTICAL_ANGLE_SAMPLE_COUNT - 1));
      const front = thinFilmRgb(material, cosTheta, false);
      const back = thinFilmRgb(material, cosTheta, true);
      writeRgb(out, MATERIAL_OPTICAL_RELATIVE_OFFSETS.THIN_FILM_FRONT_REFLECTANCE + i, front.reflectance);
      writeRgb(out, MATERIAL_OPTICAL_RELATIVE_OFFSETS.THIN_FILM_FRONT_TRANSMITTANCE + i, front.transmittance);
      writeRgb(out, MATERIAL_OPTICAL_RELATIVE_OFFSETS.THIN_FILM_BACK_REFLECTANCE + i, back.reflectance);
      writeRgb(out, MATERIAL_OPTICAL_RELATIVE_OFFSETS.THIN_FILM_BACK_TRANSMITTANCE + i, back.transmittance);
    }
  }
  return out;
}

/** Exposed for deterministic CPU tests of the Cauchy/Abbe reduction. */
export function materialDispersionIorRgb(material: MaterialSpec): Rgb {
  return [
    materialIorAtWavelength(material, 610),
    materialIorAtWavelength(material, 550),
    materialIorAtWavelength(material, 460),
  ];
}

/** Exposed for energy/convergence tests of the preintegrated TMM. */
export function materialThinFilmRgb(
  material: MaterialSpec,
  cosTheta: number,
  reverse = false,
): { readonly reflectance: Rgb; readonly transmittance: Rgb } {
  return thinFilmRgb(material, cosTheta, reverse);
}
