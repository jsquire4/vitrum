import { describe, it, expect } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { SURFACE_TEXTURE_ID } from '@vitrum/stained-glass-extensions';
import { packBVHIndexWFromCore } from '../src/restir/packingHelpers.js';
import { SURFACE_TEXTURES_WGSL } from '../src/shaders/surfaceTextures.wgsl.js';

function materialWithSurfaceTextureId(value: unknown): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 1,
    metallic: 0,
    extensions: { surfaceTextureId: value },
  };
}

function packSurfaceTextureIds(materials: readonly MaterialSpec[]): Uint32Array {
  const triCount = materials.length;
  const indices = new Uint32Array(triCount * 3);
  const triMaterialIds = new Uint32Array(triCount);
  for (let t = 0; t < triCount; t += 1) {
    indices[t * 3 + 0] = t * 3 + 0;
    indices[t * 3 + 1] = t * 3 + 1;
    indices[t * 3 + 2] = t * 3 + 2;
    triMaterialIds[t] = t;
  }
  return packBVHIndexWFromCore(indices, triMaterialIds, materials, triCount);
}

/**
 * Wire-contract pin: the integer values in `SURFACE_TEXTURE_ID` must match
 * the `case <N>u:` labels of `surfaceTextureMod` in surfaceTextures.wgsl.
 * Renumbering either side without the other silently breaks every host
 * that has stamped `MaterialSpec.extensions.surfaceTextureId` for that texture.
 */
describe('SURFACE_TEXTURE_ID — wire contract with surfaceTextures.wgsl', () => {
  it('exports the canonical 0..7 mapping', () => {
    expect(SURFACE_TEXTURE_ID).toEqual({
      smooth: 0,
      hammered: 1,
      ripple: 2,
      granite: 3,
      baroque: 4,
      waterglass: 5,
      catspaw: 6,
      flemish: 7,
    });
  });

  it('every TS value 0..7 has a corresponding `case Nu:` arm in the WGSL switch', () => {
    for (const id of Object.values(SURFACE_TEXTURE_ID)) {
      // The WGSL source contains lines like `case 0u: { m = 1.0; }` —
      // assert each integer appears in that exact pattern.
      const pattern = new RegExp(`case\\s+${id}u\\s*:`);
      expect(SURFACE_TEXTURES_WGSL).toMatch(pattern);
    }
  });

  it('every WGSL `case Nu:` arm (excluding default) is in [0, 7]', () => {
    // Find every `case Nu:` literal in the shader source.
    const matches = [...SURFACE_TEXTURES_WGSL.matchAll(/case\s+(\d+)u\s*:/g)];
    expect(matches.length).toBeGreaterThan(0);
    const ids = matches.map((m) => Number(m[1]));
    const validIds: Set<number> = new Set(Object.values(SURFACE_TEXTURE_ID));
    for (const id of ids) {
      expect(validIds.has(id)).toBe(true);
    }
  });

  it('the WGSL three-bit decode still matches a 0..7 id range', () => {
    // The BVH assigns texTypeId to bits 0..2, so the table cannot
    // grow past 8 entries without a coordinated upgrade in
    // packingHelpers.ts and shade.wgsl's `decodeSurfaceTextureId`.
    expect(Object.keys(SURFACE_TEXTURE_ID).length).toBeLessThanOrEqual(8);
    for (const id of Object.values(SURFACE_TEXTURE_ID)) {
      expect(id).toBeGreaterThanOrEqual(0);
      expect(id).toBeLessThanOrEqual(7);
    }
  });

  it('consumes every canonical package id through the production BVH ingestion path', () => {
    const ids = Object.values(SURFACE_TEXTURE_ID);
    const packed = packSurfaceTextureIds(ids.map(materialWithSurfaceTextureId));

    for (let t = 0; t < ids.length; t += 1) {
      // Mirrors WGSL decodeSurfaceTextureId: extraction is safe because the
      // untyped value was validated before packing.
      expect(packed[t * 4 + 3]! & 0x7).toBe(ids[t]);
    }
  });

  it('uses smooth only when the material extension is absent', () => {
    const withoutExtension: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 1,
      metallic: 0,
    };
    const packed = packSurfaceTextureIds([withoutExtension]);
    expect(packed[3]! & 0x7).toBe(SURFACE_TEXTURE_ID.smooth);
  });

  it('rejects values that previously aliased through low-bit masking', () => {
    for (const invalid of [-1, 8, 15, 1.5, NaN, Infinity, '5', null, true]) {
      expect(
        () => packSurfaceTextureIds([materialWithSurfaceTextureId(invalid)]),
        String(invalid),
      ).toThrow(/materials\[0\]\.extensions\.surfaceTextureId/);
    }
  });
});
