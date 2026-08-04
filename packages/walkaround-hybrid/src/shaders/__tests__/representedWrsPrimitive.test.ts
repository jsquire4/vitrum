import { describe, expect, it } from 'vitest';
import { REPRESENTED_WRS_WGSL } from '@vitrum/shared-samplers';
import { SHARED_PRIMITIVES_WGSL } from '../sharedPrimitives.wgsl.js';

describe('represented WRS shared primitive composition', () => {
  it('includes the canonical helper exactly once and immediately after stateful PCG', () => {
    expect(SHARED_PRIMITIVES_WGSL.split(REPRESENTED_WRS_WGSL)).toHaveLength(2);

    const pcgIndex = SHARED_PRIMITIVES_WGSL.indexOf('fn pcgNext(');
    const wrsIndex = SHARED_PRIMITIVES_WGSL.indexOf('struct RepresentedWrsState');
    const hashIndex = SHARED_PRIMITIVES_WGSL.indexOf('fn pcgHashToF32(');
    expect(pcgIndex).toBeGreaterThanOrEqual(0);
    expect(wrsIndex).toBeGreaterThan(pcgIndex);
    expect(hashIndex).toBeGreaterThan(wrsIndex);
    expect(SHARED_PRIMITIVES_WGSL.match(/struct RepresentedWrsState/g)).toHaveLength(1);
  });
});
