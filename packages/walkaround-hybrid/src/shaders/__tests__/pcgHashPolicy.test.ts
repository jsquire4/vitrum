import { describe, expect, it } from 'vitest';

import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';
import { SHARED_PRIMITIVES_WGSL } from '../sharedPrimitives.wgsl.js';
import { REFRACTIVE_CAUSTICS_WGSL } from '../refractiveCaustics.wgsl.js';
import { STAINED_GLASS_SHADE_WGSL } from '../stainedGlassShade.wgsl.js';
import { SURFACE_TEXTURES_WGSL } from '../surfaceTextures.wgsl.js';
import { TRANSPARENT_OIT_WGSL } from '../transparentOit.wgsl.js';

function u32(value: number): number {
  return value >>> 0;
}

function pcgHashToF32(seed: number): number {
  const state = u32(Math.imul(seed, 747_796_405) + 2_891_336_453);
  const shift = (state >>> 28) + 4;
  const word = u32(Math.imul(u32((state >>> shift) ^ state), 277_803_737));
  return (u32((word >>> 22) ^ word) >>> 8) / 16_777_216;
}

function pixelHashLane(x: number, y: number, salt: number): number {
  const seed = u32(
    Math.imul(x, 1_664_525) ^
    Math.imul(y, 1_013_904_223) ^
    Math.imul(salt, 22_695_477),
  );
  return pcgHashToF32(seed);
}

describe('walkaround shader hash policy', () => {
  it('exposes the shared stateless PCG hash through shared primitives', () => {
    expect(SHARED_PRIMITIVES_WGSL).toContain('fn pcgHashToF32');
    expect(SHARED_PRIMITIVES_WGSL).toContain('fn pixelHash2');
    expect(SHARED_PRIMITIVES_WGSL).toContain('fn worldHash2');
    expect(SHARED_PRIMITIVES_WGSL).toContain(
      'return pcgHash2FromSeed(vitrumPcgSeed2(px.x, px.y, salt));',
    );
    expect(SHARED_PRIMITIVES_WGSL).not.toContain('let _ = salt;');
  });

  it('changes a pixel hash when only the salt changes', () => {
    expect(pixelHashLane(37, 91, 0x1234))
      .not.toBe(pixelHashLane(37, 91, 0x5678));
  });

  it('frame-scrambles stochastic caustic and soft-sun candidates', () => {
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'ubo.frameSeed ^ 0x52434658u',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'pixelHash2(gid, ubo.frameSeed ^ 0x53474341u)',
    );
    expect(TRANSPARENT_OIT_WGSL).toContain(
      'worldHash2(hitPos, hit.indices.w ^ ubo.frameSeed ^ 0x4f495431u)',
    );
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

  it('clips stained-glass visibility in the canonical world-distance walk', () => {
    expect(SURFACE_TEXTURES_WGSL).toContain('let hit = traceSceneFirstHit(');
    expect(SURFACE_TEXTURES_WGSL).toContain('if (!hit.didHit || hit.dist >= remaining)');
    expect(SURFACE_TEXTURES_WGSL).toContain('walkRay.origin = origin + dir * traveled;');
    expect(SURFACE_TEXTURES_WGSL).not.toContain('localOrigin, localDir, 1e20');
  });
});
