// D3 stride-parity guard (2026-06-10). The packer↔shader material stride is the
// repo's recurring upload-gap bug class (H1 / H41 / the D3 85-vs-93 working-tree
// break, which the entire suite stayed GREEN through — this file is the fix for
// that blind spot). The stride is single-sourced in materialStride.js; this test
// asserts every composed-shader fetch site actually carries that value and that
// no stale hardcoded stride survives anywhere in the composed GLSL.
import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { MATERIAL_PIXELS, packMaterialsTexture } from './materialsTexture.js';
import { LIGHT_PIXELS } from './lightsTexture.js';
import { composeTraceGlsl } from '../glsl/composeTraceGlsl.js';
import { DEFAULT_TRACE_FEATURES } from '../featureTypes.js';

function composedShader(): string {
  return composeTraceGlsl({ ...DEFAULT_TRACE_FEATURES });
}

describe('material stride parity (packer ↔ composed GLSL)', () => {
  it('readMaterialInfo fetches at the packer stride', () => {
    expect(composedShader()).toContain(`uint i = index * ${MATERIAL_PIXELS}u;`);
  });

  it('every GLSL MATERIAL_PIXELS const equals the packer stride', () => {
    const matches = [...composedShader().matchAll(/MATERIAL_PIXELS\s*=\s*(\d+)u/g)];
    // thin_film_tmm + inside_fog_volume + util_functions all declare it.
    expect(matches.length).toBeGreaterThanOrEqual(3);
    for (const m of matches) {
      expect(Number(m[1])).toBe(MATERIAL_PIXELS);
    }
  });

  it('no stale hardcoded material stride survives in the composed shader', () => {
    // Every bare `index * <N>u` / `materialIndex * <N>u` multiply (the names the
    // material- and lights-texture decoders use) must carry one of the two
    // packer-exported strides — a mismatched literal is exactly the D3 break.
    // The lookbehind excludes other *Index identifiers (triIndex, faceIndex…).
    const knownStrides = new Set([MATERIAL_PIXELS, LIGHT_PIXELS]);
    const staleFetch = [...composedShader().matchAll(/(?<![A-Za-z])(?:material)?[iI]ndex\s*\*\s*(\d+)u/g)]
      .map((m) => Number(m[1]))
      .filter((n) => !knownStrides.has(n));
    expect(staleFetch).toEqual([]);
  });

  it('packer writes the D3 texels inside the stride (smoke)', () => {
    const mats: MaterialSpec[] = [
      { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, envMapIntensity: 2.5 },
      { baseColor: [1, 0, 0], roughness: 0.1, metallic: 1, aoMapIntensity: 0.25 },
    ];
    const packed = packMaterialsTexture(mats);
    // material 0: texel 85 .a = envMapIntensity; texel 86 .r = aoMapIntensity.
    expect(packed.data[85 * 4 + 3]).toBe(2.5);
    expect(packed.data[86 * 4 + 0]).toBe(1.0); // default aoMapIntensity
    // material 1 decodes at base = MATERIAL_PIXELS — the exact D3 break shape.
    const base1 = MATERIAL_PIXELS * 4;
    expect(packed.data[base1 + 85 * 4 + 3]).toBe(1.0); // default envMapIntensity
    expect(packed.data[base1 + 86 * 4 + 0]).toBe(0.25);
  });
});
