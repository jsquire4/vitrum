import { describe, expect, it } from 'vitest';

import { applyDirectionMatrix4 } from '../worldTransforms.js';

function scaledRotation(scale: number): Float64Array {
  // +90 degrees around Z, followed by a uniform scale.
  return new Float64Array([
    0, scale, 0, 0,
    -scale, 0, 0, 0,
    0, 0, scale, 0,
    0, 0, 0, 1,
  ]);
}

describe('applyDirectionMatrix4 scale contract', () => {
  it.each([1e-30, 1, 1e30])(
    'preserves direction for a finite nonzero transform scale of %s',
    scale => {
      const transformed = applyDirectionMatrix4(scaledRotation(scale), 1, 0, 0);
      expect(transformed[0]).toBeCloseTo(0, 14);
      expect(transformed[1]).toBeCloseTo(1, 14);
      expect(transformed[2]).toBeCloseTo(0, 14);
    },
  );

  it('rejects only exact-zero or nonfinite transformed directions', () => {
    expect(applyDirectionMatrix4(new Float64Array(16), 1, 0, 0)).toEqual([0, 0, 0]);
    const nonfinite = scaledRotation(1);
    nonfinite[1] = Number.POSITIVE_INFINITY;
    expect(applyDirectionMatrix4(nonfinite, 1, 0, 0)).toEqual([0, 0, 0]);
  });
});
