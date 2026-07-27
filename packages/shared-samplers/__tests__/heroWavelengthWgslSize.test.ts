/**
 * heroWavelengthWgslSize.test.ts — cheap (non-GPU) regression for bug A2.
 *
 * Pins the hero-wavelength WGSL table consts so the declared fixed-size-array
 * length always matches both (a) the source TS table length and (b) the number
 * of constructor initializers emitted.
 *
 * The bug: tableToWgslConst (heroWavelengthTables.ts) hardcoded `array<f32, 82>`
 * for EVERY const, but the three CMF tables (CIE_X/Y/Z_TABLE) are length 81
 * while the three CDF tables (X/Y/Z_CMF_CDF) are length 82. The 81-length CMF
 * consts therefore declared a size-82 array with only 81 initializers — a WGSL
 * fixed-size-array constructor count mismatch that would fail createShaderModule
 * on a conformant implementation. It is latent only because GPU shader
 * compilation is opt-in (default `npm test` never compiles WGSL).
 *
 * This is a string-shape assertion, not a GPU compile; a GPU-gated WGSL-compile
 * test is added separately by the test wave.
 */

import { describe, it, expect } from 'vitest';
import { CIE_X_TABLE, CIE_Y_TABLE, CIE_Z_TABLE } from '../src/cieCmf.js';
import { X_CMF_CDF, Y_CMF_CDF, Z_CMF_CDF } from '../src/wavelengthSampling.js';
import { HERO_WAVELENGTH_TABLES_WGSL } from '../src/wgsl/heroWavelengthTables.js';
import { HERO_WAVELENGTH_WGSL } from '../src/wgsl/heroWavelength.wgsl.js';

/** Maps each emitted const name to the source table it was generated from. */
const TABLE_CONSTS: ReadonlyArray<{ name: string; expectedLength: number }> = [
  // CMF tables: 81 entries (380..780 nm at 5 nm steps).
  { name: 'HERO_X_CMF', expectedLength: CIE_X_TABLE.length },
  { name: 'HERO_Y_CMF', expectedLength: CIE_Y_TABLE.length },
  { name: 'HERO_Z_CMF', expectedLength: CIE_Z_TABLE.length },
  // CDF tables: 82 entries (CIE_TABLE_LENGTH + 1).
  { name: 'HERO_X_CMF_CDF', expectedLength: X_CMF_CDF.length },
  { name: 'HERO_Y_CMF_CDF', expectedLength: Y_CMF_CDF.length },
  { name: 'HERO_Z_CMF_CDF', expectedLength: Z_CMF_CDF.length },
];

/**
 * Parse the emitted const for `name` and return both the declared array length
 * (the `N` in `array<f32, N>`) and the number of constructor initializers.
 *
 * Matches: `const NAME: array<f32, N> = array<f32, M>(v0, v1, ...);`
 */
function parseEmittedConst(
  wgsl: string,
  name: string,
): { declaredType: number; declaredCtor: number; initCount: number } {
  const re = new RegExp(
    `const ${name}: array<f32, (\\d+)> = array<f32, (\\d+)>\\(([^)]*)\\);`,
  );
  const match = wgsl.match(re);
  expect(match, `expected an emitted const for ${name}`).not.toBeNull();
  const declaredTypeStr = match?.[1] ?? '';
  const declaredCtorStr = match?.[2] ?? '';
  const valsStr = match?.[3] ?? '';
  const initCount = valsStr.trim() === '' ? 0 : valsStr.split(',').length;
  return {
    declaredType: Number(declaredTypeStr),
    declaredCtor: Number(declaredCtorStr),
    initCount,
  };
}

describe('hero-wavelength WGSL table sizes (bug A2)', () => {
  it('CMF tables are length 81 and CDF tables are length 82 (source of truth)', () => {
    expect(CIE_X_TABLE.length).toBe(81);
    expect(CIE_Y_TABLE.length).toBe(81);
    expect(CIE_Z_TABLE.length).toBe(81);
    expect(X_CMF_CDF.length).toBe(82);
    expect(Y_CMF_CDF.length).toBe(82);
    expect(Z_CMF_CDF.length).toBe(82);
  });

  for (const { name, expectedLength } of TABLE_CONSTS) {
    it(`${name}: declared array<f32, N> size === source table length (${expectedLength})`, () => {
      const { declaredType, declaredCtor, initCount } = parseEmittedConst(
        HERO_WAVELENGTH_TABLES_WGSL,
        name,
      );
      // Declared type size matches the source table length.
      expect(declaredType).toBe(expectedLength);
      // The array() constructor's declared size matches the type size.
      expect(declaredCtor).toBe(expectedLength);
      // The number of initializers matches the declared size (no count mismatch).
      expect(initCount).toBe(expectedLength);
    });
  }

  it('does not emit any hardcoded size-82 array for the 81-length CMF consts', () => {
    // The regression: HERO_X/Y/Z_CMF must NOT be declared as array<f32, 82>.
    for (const name of ['HERO_X_CMF', 'HERO_Y_CMF', 'HERO_Z_CMF']) {
      expect(HERO_WAVELENGTH_TABLES_WGSL).not.toContain(`const ${name}: array<f32, 82>`);
      expect(HERO_WAVELENGTH_TABLES_WGSL).toContain(`const ${name}: array<f32, 81>`);
    }
  });

  it('inverts the piecewise-linear CMF integral quadratically', () => {
    expect(HERO_WAVELENGTH_WGSL).toContain(
      'let targetIntegral = segmentFraction * segmentIntegral;',
    );
    expect(HERO_WAVELENGTH_WGSL).toContain(
      'let discriminant = max(vLo * vLo + 2.0 * slope * targetIntegral, 0.0);',
    );
    expect(HERO_WAVELENGTH_WGSL).toContain(
      '2.0 * targetIntegral / denominator',
    );
    expect(HERO_WAVELENGTH_WGSL).not.toContain(
      'let t = select(0.0, (uClamped - cdfLo) / (cdfHi - cdfLo)',
    );
  });
});
