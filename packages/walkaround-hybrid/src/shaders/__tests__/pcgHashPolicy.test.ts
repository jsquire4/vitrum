import { describe, expect, it } from 'vitest';

import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';
import { SHARED_PRIMITIVES_WGSL } from '../sharedPrimitives.wgsl.js';
import { STAINED_GLASS_SHADE_WGSL } from '../stainedGlassShade.wgsl.js';
import { SURFACE_TEXTURES_WGSL } from '../surfaceTextures.wgsl.js';
import { TRANSPARENT_OIT_WGSL } from '../transparentOit.wgsl.js';

describe('walkaround shader hash policy', () => {
  it('exposes the shared stateless PCG hash through shared primitives', () => {
    expect(SHARED_PRIMITIVES_WGSL).toContain('fn pcgHashToF32');
    expect(SHARED_PRIMITIVES_WGSL).toContain('fn pixelHash2');
    expect(SHARED_PRIMITIVES_WGSL).toContain('fn worldHash2');
  });

  it('keeps deterministic jitter/noise call sites on shared hash helpers', () => {
    const checkedSources = [
      SHADING_TERMS_WGSL,
      STAINED_GLASS_SHADE_WGSL,
      SURFACE_TEXTURES_WGSL,
      TRANSPARENT_OIT_WGSL,
    ].join('\n');

    expect(checkedSources).toContain('pixelHash2');
    expect(checkedSources).toContain('floatCellHash');
    expect(checkedSources).toContain('worldHash2');
  });
});
