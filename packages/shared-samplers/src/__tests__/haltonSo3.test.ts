import { describe, expect, it } from 'vitest';
import { haltonSO3AxisAngleFromFrameIndex } from '../haltonSo3.js';

describe('haltonSO3AxisAngleFromFrameIndex', () => {
  it('is deterministic for a frame index', () => {
    expect(haltonSO3AxisAngleFromFrameIndex(42)).toEqual(
      haltonSO3AxisAngleFromFrameIndex(42),
    );
  });

  it('pins the first sample so backend extractions cannot rotate the sequence', () => {
    const v = haltonSO3AxisAngleFromFrameIndex(0);
    expect(v[0]).toBeCloseTo(1.694991, 6);
    expect(v[1]).toBeCloseTo(-0.978603, 6);
    expect(v[2]).toBeCloseTo(1.861414, 6);
  });

  it('returns finite axis-angle vectors with angles in [0, pi]', () => {
    for (let frame = 0; frame < 128; frame += 1) {
      const v = haltonSO3AxisAngleFromFrameIndex(frame);
      expect(v.every(Number.isFinite)).toBe(true);
      const angle = Math.hypot(v[0], v[1], v[2]);
      expect(angle).toBeGreaterThanOrEqual(0);
      expect(angle).toBeLessThanOrEqual(Math.PI + 1e-12);
    }
  });
});
