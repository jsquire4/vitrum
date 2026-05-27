/**
 * Hero-wavelength MIS sampling + CMF→RGB reconstruction (Wilkie et al. 2014 §3.3).
 * Tables: heroWavelengthTables.ts (from @vitrum/shared-samplers wavelengthSampling).
 */
import { HERO_WAVELENGTH_TABLES_WGSL } from './heroWavelengthTables.js';

export const HERO_WAVELENGTH_WGSL = /* wgsl */ `
${HERO_WAVELENGTH_TABLES_WGSL}

fn heroSampleTable(lambdaNm: f32, table: array<f32, 82>) -> f32 {
  if (lambdaNm < HERO_CIE_LAMBDA_MIN || lambdaNm > 780.0) {
    return 0.0;
  }
  let f = (lambdaNm - HERO_CIE_LAMBDA_MIN) / HERO_CIE_LAMBDA_STEP;
  let lo = u32(floor(f));
  let hi = min(lo + 1u, HERO_CIE_TABLE_LENGTH - 1u);
  let t = f - f32(lo);
  return table[lo] + t * (table[hi] - table[lo]);
}

fn heroSampleCmf(lambdaNm: f32) -> vec3f {
  return vec3f(
    heroSampleTable(lambdaNm, HERO_X_CMF),
    heroSampleTable(lambdaNm, HERO_Y_CMF),
    heroSampleTable(lambdaNm, HERO_Z_CMF),
  );
}

fn heroMisMixturePdf(lambdaNm: f32) -> f32 {
  let x = heroSampleTable(lambdaNm, HERO_X_CMF) / HERO_X_CMF_INTEGRAL;
  let y = heroSampleTable(lambdaNm, HERO_Y_CMF) / HERO_Y_CMF_INTEGRAL;
  let z = heroSampleTable(lambdaNm, HERO_Z_CMF) / HERO_Z_CMF_INTEGRAL;
  return (x + y + z) / 3.0;
}

fn heroSampleCmfCdfInverse(
  u: f32,
  cmfTable: array<f32, 82>,
  cdfTable: array<f32, 82>,
  integral: f32,
) -> vec2f {
  let uClamped = clamp(u, 0.0, 1.0 - 1e-7);
  var lo: u32 = 0u;
  var hi: u32 = HERO_CIE_TABLE_LENGTH - 1u;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (cdfTable[mid + 1u] <= uClamped) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  let cdfLo = cdfTable[lo];
  let cdfHi = cdfTable[lo + 1u];
  let vLo = cmfTable[lo];
  let vHi = cmfTable[lo + 1u];
  let t = select(0.0, (uClamped - cdfLo) / (cdfHi - cdfLo), cdfHi > cdfLo);
  let lambdaNm = clamp(
    HERO_CIE_LAMBDA_MIN + (f32(lo) + t) * HERO_CIE_LAMBDA_STEP,
    HERO_CIE_LAMBDA_MIN,
    780.0,
  );
  let vAtLambda = vLo + t * (vHi - vLo);
  let pdf = vAtLambda / integral;
  return vec2f(lambdaNm, pdf);
}

fn sampleHeroWavelengthMIS(uStrategy: f32, uLambda: f32) -> vec3f {
  let s = clamp(uStrategy, 0.0, 1.0 - 1e-7);
  var heroLambda: f32;
  if (s < 1.0 / 3.0) {
    heroLambda = heroSampleCmfCdfInverse(uLambda, HERO_X_CMF, HERO_X_CMF_CDF, HERO_X_CMF_INTEGRAL).x;
  } else if (s < 2.0 / 3.0) {
    heroLambda = heroSampleCmfCdfInverse(uLambda, HERO_Y_CMF, HERO_Y_CMF_CDF, HERO_Y_CMF_INTEGRAL).x;
  } else {
    heroLambda = heroSampleCmfCdfInverse(uLambda, HERO_Z_CMF, HERO_Z_CMF_CDF, HERO_Z_CMF_INTEGRAL).x;
  }
  let pdf = heroMisMixturePdf(heroLambda);
  return vec3f(heroLambda, pdf, 0.0);
}

fn heroWavelengthToRgb(lambdaNm: f32, throughputScalar: f32, pdfLambda: f32) -> vec3f {
  if (pdfLambda <= 0.0) {
    return vec3f(0.0);
  }
  let cmf = heroSampleCmf(lambdaNm);
  let weight = throughputScalar / (pdfLambda * HERO_Y_CMF_INTEGRAL);
  let x = cmf.x * weight;
  let y = cmf.y * weight;
  let z = cmf.z * weight;
  return vec3f(
    3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
    -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
    0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
  );
}

fn heroLambdaTo01(lambdaNm: f32) -> f32 {
  return clamp((lambdaNm - HERO_CIE_LAMBDA_MIN) / 400.0, 0.0, 1.0);
}
`;

export const HERO_WAVELENGTH_MODULE_NAME = 'heroWavelength';
