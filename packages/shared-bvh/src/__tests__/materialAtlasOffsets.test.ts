import { describe, expect, it } from 'vitest';
import {
  MATERIAL_ATLAS_OFFSETS,
  buildMaterialAtlasOffsetConstsWGSL,
} from '../wgsl/materialAtlasOffsets.wgsl.js';

/**
 * Pins the single-sourced 62-texel material-atlas offset ABI (T4-2,
 * 2026-07-20). The three WGSL consumers (shade / DDGI / RC) generate their
 * offset-const block from this data; the composed-WGSL byte-identity is pinned
 * downstream (shade/DDGI composed goldens + walkaround-rc
 * `probeRayCastByteIdentity.test.ts`). These tests guard the generator itself.
 */
describe('materialAtlas offset ABI generator', () => {
  it('carries the frozen 62-texel stride and canonical values', () => {
    const byName = new Map(MATERIAL_ATLAS_OFFSETS);
    expect(byName.get('META_TEXELS_PER_TRI')).toBe(62);
    expect(byName.get('SLOT_BASE_COLOR')).toBe(0);
    expect(byName.get('EMISSIVE_TEXEL_OFFSET')).toBe(11);
    expect(byName.get('BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET')).toBe(61);
  });

  it('emits the requested subset in canonical order regardless of input order', () => {
    const out = buildMaterialAtlasOffsetConstsWGSL({
      prefix: 'RC_',
      // deliberately out of canonical order
      include: ['EMISSIVE_TEXEL_OFFSET', 'META_TEXELS_PER_TRI', 'SLOT_BASE_COLOR'],
    });
    expect(out).toBe(
      'const RC_MATERIAL_MAP_META_TEXELS_PER_TRI: u32 = 62u;\n' +
        'const RC_MATERIAL_MAP_SLOT_BASE_COLOR: u32 = 0u;\n' +
        'const RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET: u32 = 11u;',
    );
  });

  it('supports the empty prefix (shade path) verbatim', () => {
    const out = buildMaterialAtlasOffsetConstsWGSL({
      prefix: '',
      include: ['META_TEXELS_PER_TRI'],
    });
    expect(out).toBe('const MATERIAL_MAP_META_TEXELS_PER_TRI: u32 = 62u;');
  });

  it('throws on an unknown offset suffix (single-source drift guard)', () => {
    expect(() =>
      buildMaterialAtlasOffsetConstsWGSL({ prefix: 'DDGI_', include: ['NOT_A_REAL_OFFSET'] }),
    ).toThrow(/unknown offset suffix/);
  });
});
