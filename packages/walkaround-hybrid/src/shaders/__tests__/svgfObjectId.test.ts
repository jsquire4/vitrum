import { describe, expect, it } from 'vitest';

import { SHADE_WGSL } from '../shade.wgsl.js';

describe('shade SVGF object IDs', () => {
  it('writes sky=0 and hit IDs from instance + triangle into an r32uint target', () => {
    expect(SHADE_WGSL).toMatch(
      /@group\(0\)\s+@binding\(15\)\s+var\s+svgfObjectIdOut:\s+texture_storage_2d<r32uint,\s*write>/,
    );
    expect(SHADE_WGSL).toMatch(/let\s+inst\s*=\s*hit\.instanceIndex\s*\+\s*1u/);
    expect(SHADE_WGSL).toMatch(/let\s+tri\s*=\s*hit\.indices\.w\s*\+\s*1u/);
    expect(SHADE_WGSL).toMatch(/fn\s+storeSvgfObjectId\(pix:\s*vec2u,\s*id:\s*u32\)/);
    expect(SHADE_WGSL).toMatch(/let\s+dims\s*=\s*textureDimensions\(svgfObjectIdOut\)/);
    expect(SHADE_WGSL).toMatch(/pix\.x\s*<\s*dims\.x\s*&&\s*pix\.y\s*<\s*dims\.y/);
    expect(SHADE_WGSL).toMatch(/storeSvgfObjectId\(pix,\s*0u\)/);
    expect(SHADE_WGSL).toMatch(/storeSvgfObjectId\(pix,\s*stableSvgfObjectId\(primaryHit\)\)/);
  });
});
