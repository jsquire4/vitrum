import { describe, expect, it } from 'vitest';
import { resolveRCRoomSize } from '../src/HybridEngineRC.js';

describe('RC cascade room-size scale contract', () => {
  it.each([
    [1e-20, 2e-20, 3e-20],
    [1e20, 2e20, 3e20],
  ] as const)('preserves authored non-zero extents %s/%s/%s', (x, y, z) => {
    expect(resolveRCRoomSize([0, 0, 0], [x, y, z])).toEqual([x, y, z]);
  });

  it('adds scale-relative thickness only to a genuinely degenerate axis', () => {
    const size = resolveRCRoomSize([0, 0, 7e-20], [2e-20, 3e-20, 7e-20]);
    expect(size[0]).toBe(2e-20);
    expect(size[1]).toBe(3e-20);
    expect(size[2]).toBe(7e-20 * 2 ** -20);
    expect(size[2]).toBeLessThan(1e-20);
  });

  it('keeps an all-zero point volume representably positive', () => {
    const size = resolveRCRoomSize([0, 0, 0], [0, 0, 0]);
    expect(size).toEqual([2 ** -126, 2 ** -126, 2 ** -126]);
  });

  it('rejects reversed or non-finite bounds', () => {
    expect(() => resolveRCRoomSize([1, 0, 0], [0, 1, 1])).toThrow(/finite and ordered/);
    expect(() => resolveRCRoomSize([0, 0, 0], [Infinity, 1, 1])).toThrow(
      /finite and ordered/,
    );
  });
});
