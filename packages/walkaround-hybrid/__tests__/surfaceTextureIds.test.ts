import { describe, it, expect } from 'vitest';
import { SURFACE_TEXTURE_ID } from '../src/surfaceTextureIds.js';
import { SURFACE_TEXTURES_WGSL } from '../src/shaders/surfaceTextures.wgsl.js';

/**
 * Wire-contract pin: the integer values in `SURFACE_TEXTURE_ID` must match
 * the `case <N>u:` labels of `surfaceTextureMod` in surfaceTextures.wgsl.
 * Renumbering either side without the other silently breaks every host
 * that has stamped `userData.surfaceTextureId` for that texture.
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
    const validIds = new Set(Object.values(SURFACE_TEXTURE_ID));
    for (const id of ids) {
      expect(validIds.has(id)).toBe(true);
    }
  });

  it('the WGSL low-3-bit mask still matches a 0..7 id range', () => {
    // The BVH packs texTypeId into 3 bits (`& 0x7`), so the table cannot
    // grow past 8 entries without a coordinated upgrade in
    // packingHelpers.ts and shade.wgsl's `decodeSurfaceTextureId`.
    expect(Object.keys(SURFACE_TEXTURE_ID).length).toBeLessThanOrEqual(8);
    for (const id of Object.values(SURFACE_TEXTURE_ID)) {
      expect(id).toBeGreaterThanOrEqual(0);
      expect(id).toBeLessThanOrEqual(7);
    }
  });
});
