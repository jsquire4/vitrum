import { describe, expect, it } from 'vitest';
import {
  aabbFromBvhPositions,
  buildPpgUniformGridCells,
  encodePpgCellGpuBytes,
} from '../src/ppg/ppgCellUpload.js';
import { PPG_CELL_BYTE_STRIDE } from '../src/ppg/types.js';

describe('aabbFromBvhPositions', () => {
  it('returns axis-aligned bounds for vec4 strides', () => {
    const f = new Float32Array([
      0, 0, 0, 0,
      2, 4, 6, 0,
      -1, 10, 3, 0,
    ]);
    const b = aabbFromBvhPositions(f.buffer, 3);
    expect(b.min).toEqual([-1, 0, 0]);
    expect(b.max).toEqual([2, 10, 6]);
  });

  it('returns unit fallback when no vertices', () => {
    const f = new Float32Array(0);
    const b = aabbFromBvhPositions(f.buffer, 0);
    expect(b.min).toEqual([0, 0, 0]);
    expect(b.max).toEqual([1, 1, 1]);
  });
});

describe('buildPpgUniformGridCells', () => {
  it('produces at most maxCells centers inside padded volume', () => {
    const cells = buildPpgUniformGridCells([0, 0, 0], [1, 1, 1], 8, 0);
    expect(cells.length).toBeLessThanOrEqual(8);
    expect(cells.length).toBeGreaterThan(0);
    for (const c of cells) {
      expect(c.position[0]).toBeGreaterThanOrEqual(0);
      expect(c.position[0]).toBeLessThanOrEqual(1);
      expect(c.position[1]).toBeGreaterThanOrEqual(0);
      expect(c.position[1]).toBeLessThanOrEqual(1);
      expect(c.position[2]).toBeGreaterThanOrEqual(0);
      expect(c.position[2]).toBeLessThanOrEqual(1);
    }
  });
});

describe('encodePpgCellGpuBytes', () => {
  it('writes leafIndex equal to cell index and zero tail', () => {
    const cells = [
      { position: [1, 2, 3] as const },
      { position: [4, 5, 6] as const },
    ];
    const stride = 32;
    const bytes = encodePpgCellGpuBytes(cells, 2, stride * 3);
    const dv = new DataView(bytes.buffer);
    expect(dv.getFloat32(0, true)).toBe(1);
    expect(dv.getUint32(16, true)).toBe(0);
    expect(dv.getFloat32(stride, true)).toBe(4);
    expect(dv.getUint32(stride + 16, true)).toBe(1);
    expect(bytes[stride * 2 + 0]).toBe(0);
  });

  it('throws when activeCount exceeds buffer capacity', () => {
    expect(() =>
      encodePpgCellGpuBytes([{ position: [0, 0, 0] }], 2, PPG_CELL_BYTE_STRIDE),
    ).toThrow(RangeError);
  });
});
