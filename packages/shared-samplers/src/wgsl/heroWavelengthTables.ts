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
  // Derive the WGSL fixed-size-array length from the actual table length so the
  // declared size always matches the initializer count. The three CMF tables
  // (CIE_X/Y/Z_TABLE) are length 81 (380..780 nm at 5 nm steps); the three CDF
  // tables (X/Y/Z_CMF_CDF) are length 82 (CIE_TABLE_LENGTH + 1). Hardcoding 82
  // produced a constructor count mismatch for the 81-length CMF consts.
  const arr = Array.from(table);
  const vals = arr.map((v) => Number(v).toFixed(8)).join(', ');
  return `const ${name}: array<f32, ${arr.length}> = array<f32, ${arr.length}>(${vals});`;
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
