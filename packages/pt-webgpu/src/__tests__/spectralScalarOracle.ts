import {
  CIE_D65_TABLE,
  CIE_LAMBDA_MIN,
  CIE_LAMBDA_STEP,
  CIE_Y_TABLE,
  HERO_LAMBDA_MIN,
  Y_CMF_INTEGRAL,
} from '@vitrum/shared-samplers';

export type SpectralOracleVec3 = readonly [number, number, number];

export function heroLambdaTo01Oracle(lambdaNm: number): number {
  return Math.min(1, Math.max(0, (lambdaNm - HERO_LAMBDA_MIN) / 400));
}

export function spectralRgbFactorAtHeroOracle(
  rgb: SpectralOracleVec3,
  lambdaNm: number,
): number {
  const value: SpectralOracleVec3 = [
    Math.max(rgb[0], 0),
    Math.max(rgb[1], 0),
    Math.max(rgb[2], 0),
  ];
  const t = heroLambdaTo01Oracle(lambdaNm);
  const wB = Math.max(1 - Math.abs(t - 0.15) / 0.35, 0);
  const wG = Math.max(1 - Math.abs(t - 0.5) / 0.35, 0);
  const wR = Math.max(1 - Math.abs(t - 0.85) / 0.35, 0);
  const wSum = Math.max(wR + wG + wB, 1e-6);
  return Math.max((value[0] * wR + value[1] * wG + value[2] * wB) / wSum, 0);
}

export function activeLayerWeightRgbOracle(
  rgb: SpectralOracleVec3,
  lambdaNm: number,
  spectralEnabled: boolean,
): SpectralOracleVec3 {
  if (!spectralEnabled) return rgb;
  const scalar = spectralRgbFactorAtHeroOracle(rgb, lambdaNm);
  return [scalar, scalar, scalar];
}

export function spectralCombinedReflectanceAtHeroOracle(
  combinedRgb: SpectralOracleVec3,
  authoredRgb: SpectralOracleVec3,
  authoredExact: number,
  lambdaNm: number,
): number {
  const combinedApprox = spectralRgbFactorAtHeroOracle(combinedRgb, lambdaNm);
  const authoredApprox = spectralRgbFactorAtHeroOracle(authoredRgb, lambdaNm);
  if (authoredApprox <= 1e-7) {
    return Math.min(1, Math.max(0, combinedApprox));
  }
  return Math.min(1, Math.max(0, authoredExact * combinedApprox / authoredApprox));
}

const D65_Y_INTEGRAL = (() => {
  let integral = 0;
  for (let i = 0; i < CIE_Y_TABLE.length; i += 1) {
    integral += (CIE_D65_TABLE[i] ?? 0) * (CIE_Y_TABLE[i] ?? 0) * CIE_LAMBDA_STEP;
  }
  return integral;
})();

function d65NormalizedAtHero(lambdaNm: number): number {
  if (lambdaNm < CIE_LAMBDA_MIN || lambdaNm > 780) return 0;
  const tableOffset = (lambdaNm - CIE_LAMBDA_MIN) / CIE_LAMBDA_STEP;
  const lo = Math.floor(tableOffset);
  const hi = Math.min(lo + 1, CIE_D65_TABLE.length - 1);
  const t = tableOffset - lo;
  const d65 =
    (CIE_D65_TABLE[lo] ?? 0) +
    t * ((CIE_D65_TABLE[hi] ?? 0) - (CIE_D65_TABLE[lo] ?? 0));
  return d65 * Y_CMF_INTEGRAL / Math.max(D65_Y_INTEGRAL, 1e-9);
}

export function spectralEmissionAtHeroOracle(
  rgb: SpectralOracleVec3,
  lambdaNm: number,
): number {
  const luminance = Math.max(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2], 0);
  if (luminance < 1e-8) return 0;
  return spectralRgbFactorAtHeroOracle(rgb, lambdaNm) * d65NormalizedAtHero(lambdaNm);
}

export function bdptEmitterThroughputOracle(
  emissionRgb: SpectralOracleVec3,
  lambdaNm: number,
  pdfPosition: number,
  spectralEnabled: boolean,
): SpectralOracleVec3 {
  const pdf = Math.max(pdfPosition, 1e-8);
  if (!spectralEnabled) {
    return [emissionRgb[0] / pdf, emissionRgb[1] / pdf, emissionRgb[2] / pdf];
  }
  const scalar = spectralEmissionAtHeroOracle(emissionRgb, lambdaNm) / pdf;
  return [scalar, scalar, scalar];
}
