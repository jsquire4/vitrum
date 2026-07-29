import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  FRAUNHOFER_C_NM,
  FRAUNHOFER_D_NM,
  FRAUNHOFER_F_NM,
} from '@vitrum/shared-samplers';
import { packMaterialsTexture } from '../scene/materialsTexture.js';
import * as BsdfFns from './shader/bsdf/bsdf_functions.glsl.js';

const BASE_IOR = 1.52;
const DISPERSION_STRENGTH_OFFSET = 15 * 4 + 2;
const bsdfSource = (BsdfFns as unknown as Record<string, string>)['bsdf_functions']!;

function packedDispersionStrength(abbe: number): number {
  const material: MaterialSpec = {
    baseColor: [1, 1, 1],
    roughness: 0,
    metallic: 0,
    transmission: 1,
    ior: BASE_IOR,
    dispersionAbbeNumber: abbe,
  };
  return packMaterialsTexture([material]).data[DISPERSION_STRENGTH_OFFSET]!;
}

function twoTermCauchyIor(lambdaNm: number, bNm2: number): number {
  return Math.max(
    1,
    BASE_IOR +
      bNm2 *
        (
          1 / (lambdaNm * lambdaNm) -
          1 / (FRAUNHOFER_D_NM * FRAUNHOFER_D_NM)
        ),
  );
}

describe('pt-webgl2 material-local Abbe dispersion', () => {
  it('preserves authored d-line IOR and keeps low/high Abbe materials distinct', () => {
    const lowAbbeB = packedDispersionStrength(20);
    const highAbbeB = packedDispersionStrength(90);

    // The shared payload contract stores the two-term Cauchy B coefficient in
    // nm^2. These values intentionally remain much larger than one.
    expect(lowAbbeB).toBeGreaterThan(highAbbeB);
    expect(highAbbeB).toBeGreaterThan(1);

    const lowD = twoTermCauchyIor(FRAUNHOFER_D_NM, lowAbbeB);
    const highD = twoTermCauchyIor(FRAUNHOFER_D_NM, highAbbeB);
    const lowF = twoTermCauchyIor(FRAUNHOFER_F_NM, lowAbbeB);
    const highF = twoTermCauchyIor(FRAUNHOFER_F_NM, highAbbeB);
    const lowC = twoTermCauchyIor(FRAUNHOFER_C_NM, lowAbbeB);
    const highC = twoTermCauchyIor(FRAUNHOFER_C_NM, highAbbeB);

    expect(lowD).toBe(BASE_IOR);
    expect(highD).toBe(BASE_IOR);
    expect(lowF).toBeGreaterThan(highF);
    expect(lowC).toBeLessThan(highC);
    expect(lowF - lowC).toBeGreaterThan(highF - highC);

    // Entry transmission uses eta = 1 / n(lambda), so blue and red ordering
    // must reverse when the IOR ordering reverses.
    expect(1 / lowF).toBeLessThan(1 / highF);
    expect(1 / lowC).toBeGreaterThan(1 / highC);

    expect((BASE_IOR - 1) / (lowF - lowC)).toBeCloseTo(20, 5);
    expect((BASE_IOR - 1) / (highF - highC)).toBeCloseTo(90, 5);
  });

  it('evaluates the packed nm^2 coefficient directly and has no global Cauchy ABI', () => {
    const dispersion = bsdfSource
      .slice(
        bsdfSource.indexOf('float cauchyIORFromDLine'),
        bsdfSource.indexOf('float transmissionEtaAtHero'),
      )
      .replace(/\s+/g, ' ');

    expect(dispersion).toContain(
      'bNm2 * ( 1.0 / lambda2 - 1.0 / dLine2 )',
    );
    expect(dispersion).toContain(
      'cauchyIORFromDLine( heroWavelength, surf.ior, surf.dispersionStrength )',
    );
    expect(dispersion).toContain('uSpectralRendering != 0');
    expect(dispersion).not.toContain('iorCauchyA');
    expect(dispersion).not.toContain('iorCauchyB');
    expect(dispersion).not.toContain('iorCauchyC');
    expect(dispersion).not.toContain('dispersionScale');
  });
});
