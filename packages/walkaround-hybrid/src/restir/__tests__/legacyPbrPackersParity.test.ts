import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  packBVHBeerColorsFromCore,
  packBVHIndexWFromCore,
} from '../packingHelpers.js';
import {
  packBVHBeerColors,
  packBVHIndexW,
  type PbrMaterialLike,
} from './support/legacyPbrPackers.js';

const TRI_INDICES = new Uint32Array([0, 1, 2]);
const TRI_MATERIAL_IDS = new Uint32Array([0]);

describe('legacy structural PBR packers remain byte-identical test oracles', () => {
  it('packs tiny metallic identically to the core packer (dielectric in the 1-bit lane)', () => {
    const metallic = 1e-8;
    const core: MaterialSpec = {
      baseColor: [0.25, 0.5, 0.75],
      roughness: 0.4,
      metallic,
    };
    const structural: PbrMaterialLike = {
      color: { r: 0.25, g: 0.5, b: 0.75 },
      roughness: 0.4,
      metalness: metallic,
    };

    expect(Array.from(packBVHIndexW(
      TRI_INDICES,
      TRI_MATERIAL_IDS,
      [structural],
      1,
    ))).toEqual(Array.from(packBVHIndexWFromCore(
      TRI_INDICES,
      TRI_MATERIAL_IDS,
      [core],
      1,
    )));
  });

  it('packs the double-sided flag identically to the core Beer-color lane', () => {
    const core: MaterialSpec = {
      baseColor: [0.25, 0.5, 0.75],
      roughness: 0.4,
      metallic: 0,
      doubleSided: true,
    };
    const structural: PbrMaterialLike = {
      color: { r: 0.25, g: 0.5, b: 0.75 },
      roughness: 0.4,
      metalness: 0,
      doubleSided: true,
    };

    expect(Array.from(packBVHBeerColors(
      TRI_MATERIAL_IDS,
      [structural],
      1,
    ))).toEqual(Array.from(packBVHBeerColorsFromCore(
      TRI_MATERIAL_IDS,
      [core],
      1,
    )));
  });
});
