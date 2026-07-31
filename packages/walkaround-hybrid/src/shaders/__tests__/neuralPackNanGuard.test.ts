/**
 * Item 12 — neuralPack NaN guard (trust-remediation-plan-2026-06-10 §12).
 *
 * `normalize(vec3f(0))` is undefined behaviour per the WGSL spec and produces
 * NaN on real hardware when the normal buffer holds (0,0,0) at sky pixels,
 * background regions, or un-rendered areas.  The fix replaces the bare
 * the decoded vector is first replaced with a safe (0,1,0) fallback, then
 * normalised. The source texture stores the canonical affine-packed normal.
 *
 * This structural test pins the guard in the WGSL source so a future
 * simplification cannot silently reintroduce the NaN path.
 *
 * The runtime naga compile is the pre-push T1 GPU smoke; this is the
 * CPU-side structural proxy.
 */
import { describe, it, expect } from 'vitest';
import { NEURAL_PACK_WGSL } from '../neuralPack.wgsl.js';

describe('neuralPack — world-normal contract and NaN guard', () => {
  it('decodes normalDepth.xyz from affine-packed to signed world normal', () => {
    expect(NEURAL_PACK_WGSL).toContain('textureLoad(normalDepthTex');
    expect(NEURAL_PACK_WGSL).toContain(
      'decodeNormalDepthWorldNormal(textureLoad(normalDepthTex, xy, 0).xyz)',
    );
    expect(NEURAL_PACK_WGSL).toContain(
      'return encoded * 2.0 - vec3f(1.0);',
    );
  });

  it('uses an explicit fallback for zero-length/non-finite normals', () => {
    expect(NEURAL_PACK_WGSL).toMatch(
      /if\s*\(\s*!\(scale > 0\.0\)\s*\|\|\s*!neuralFinite\(scale\)\s*\)\s*\{[\s\S]*return vec3f\(0\.0, 1\.0, 0\.0\);/,
    );
  });

  it('max-component-equilibrates the non-zero safe vector before normalization', () => {
    expect(NEURAL_PACK_WGSL).toContain('let scaled = finiteValue / scale;');
    expect(NEURAL_PACK_WGSL).toContain('return scaled / length(scaled);');
    expect(NEURAL_PACK_WGSL).not.toContain('normalize(finiteValue)');
  });
});
