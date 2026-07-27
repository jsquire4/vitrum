import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { PROBE_RAY_CAST_WGSL } from '../src/wgsl/probeRayCast.wgsl.js';

/**
 * Byte-identity golden for the assembled RC probe-ray-cast WGSL.
 *
 * The material-atlas decode helpers and PBR BRDF lobes were extracted into
 * `rcMaterialAtlas.wgsl.ts` / `rcBrdf.wgsl.ts` and re-composed at their original
 * insertion points. The current golden also includes the intentional shared
 * producer/receiver octahedral-stratification fragment, so any later shader
 * composition drift remains explicit. The current capture also includes the
 * shared-BVH value-return loader seam and eight-binding packed scene arena; its
 * traversal, material-atlas, light-evaluation, and binding contracts are pinned
 * independently by the semantic package tests before this hash is updated. The
 * shared any-hit overflow guard is fail-closed in both its canonical and derived
 * shadow-predicate forms so malformed trees cannot leak light.
 */
describe('PROBE_RAY_CAST_WGSL byte identity', () => {
  it('matches the captured length + sha256 golden', () => {
    const length = PROBE_RAY_CAST_WGSL.length;
    const sha256 = createHash('sha256').update(PROBE_RAY_CAST_WGSL, 'utf8').digest('hex');
    expect({ length, sha256 }).toEqual({
      length: 142652,
      sha256: '0a0a8af1e5ccc08668bbec367ed7ddde7a9599d70dbd0dbc92128c39138a5171',
    });
  });
});
