import { describe, expect, it } from 'vitest';

import { SHADE_WGSL } from '../shade.wgsl.js';

describe('shade SVGF object IDs', () => {
  it('writes sky=0 and hit IDs from instance + triangle into an r32uint target', () => {
    expect(SHADE_WGSL).toMatch(
      /@group\(0\)\s+@binding\(15\)\s+var\s+svgfObjectIdOut:\s+texture_storage_2d<r32uint,\s*write>/,
    );
    expect(SHADE_WGSL).toMatch(/let\s+inst\s*=\s*hit\.instanceIndex\s*\+\s*1u/);
    expect(SHADE_WGSL).toMatch(/let\s+tri\s*=\s*hit\.indices\.w\s*\+\s*1u/);
    expect(SHADE_WGSL).toMatch(/textureStore\(svgfObjectIdOut,\s*pix,\s*vec4u\(0u\)\)/);
    expect(SHADE_WGSL).toMatch(/textureStore\(svgfObjectIdOut,\s*pix,\s*vec4u\(stableSvgfObjectId\(primaryHit\)\)\)/);
  });
});
