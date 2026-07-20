import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { PROBE_RAY_CAST_WGSL } from '../src/wgsl/probeRayCast.wgsl.js';

/**
 * Byte-identity golden for the assembled RC probe-ray-cast WGSL.
 *
 * The material-atlas decode helpers and PBR BRDF lobes were extracted into
 * `rcMaterialAtlas.wgsl.ts` / `rcBrdf.wgsl.ts` and re-composed at their original
 * insertion points. This test pins that the composed string is byte-for-byte
 * unchanged by the split (pure template concatenation, no semantic change).
 */
describe('PROBE_RAY_CAST_WGSL byte identity', () => {
  it('matches the captured length + sha256 golden', () => {
    const length = PROBE_RAY_CAST_WGSL.length;
    const sha256 = createHash('sha256').update(PROBE_RAY_CAST_WGSL, 'utf8').digest('hex');
    expect({ length, sha256 }).toEqual({
      length: 103371,
      sha256: 'ac20107b1357c6467332f3a922de67b72c5f05c3b90ff364995d6c495577de9e',
    });
  });
});
