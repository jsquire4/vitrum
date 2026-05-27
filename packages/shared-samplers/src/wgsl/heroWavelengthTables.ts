/**
 * Embeds CIE CMF CDF tables for WGSL hero-wavelength MIS (mirrors wavelengthSampling.ts).
 */
import { CIE_X_TABLE, CIE_Y_TABLE, CIE_Z_TABLE } from '../cieCmf.js';
import {
  X_CMF_CDF,
  Y_CMF_CDF,
  Z_CMF_CDF,
  X_CMF_INTEGRAL,
  Y_CMF_INTEGRAL,
  Z_CMF_INTEGRAL,
} from '../wavelengthSampling.js';

function tableToWgslConst(name: string, table: Readonly<Float32Array | Float64Array>): string {
  const vals = Array.from(table)
    .map((v) => Number(v).toFixed(8))
    .join(', ');
  return `const ${name}: array<f32, 82> = array<f32, 82>(${vals});`;
}

export const HERO_WAVELENGTH_TABLES_WGSL = /* wgsl */ `
const HERO_CIE_LAMBDA_MIN: f32 = 380.0;
const HERO_CIE_LAMBDA_STEP: f32 = 5.0;
const HERO_CIE_TABLE_LENGTH: u32 = 81u;
const HERO_X_CMF_INTEGRAL: f32 = ${X_CMF_INTEGRAL.toFixed(8)};
const HERO_Y_CMF_INTEGRAL: f32 = ${Y_CMF_INTEGRAL.toFixed(8)};
const HERO_Z_CMF_INTEGRAL: f32 = ${Z_CMF_INTEGRAL.toFixed(8)};
${tableToWgslConst('HERO_X_CMF', CIE_X_TABLE)}
${tableToWgslConst('HERO_Y_CMF', CIE_Y_TABLE)}
${tableToWgslConst('HERO_Z_CMF', CIE_Z_TABLE)}
${tableToWgslConst('HERO_X_CMF_CDF', X_CMF_CDF)}
${tableToWgslConst('HERO_Y_CMF_CDF', Y_CMF_CDF)}
${tableToWgslConst('HERO_Z_CMF_CDF', Z_CMF_CDF)}
`;
