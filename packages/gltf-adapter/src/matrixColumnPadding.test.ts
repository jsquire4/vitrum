// matrixColumnPadding.test.ts — glTF §3.6.2.4 matrix accessor column-alignment.
//
// Matrix accessors whose component size is < 4 bytes require each MATRIX COLUMN
// to start on a 4-byte boundary. The padding bytes between columns must be
// skipped when unpacking; the decoded Float32Array must contain the tightly
// packed component values in column-major order.
//
// These fixtures are hand-built with the spec-mandated column padding and
// assert the padded layout decodes correctly (and that MAT4 / float matrices,
// which need no padding, are byte-identical to a contiguous read).

import { describe, expect, it } from 'vitest';
import {
  accessorBufferViewRange,
  unpackAccessorFloat,
} from './accessors.js';
import { GltfComponentType, type GltfJson } from './gltfTypes.js';

function bytesBuf(values: number[]): ArrayBuffer {
  return new Uint8Array(values).buffer;
}

function u16Buf(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 2);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setUint16(i * 2, v, true));
  return buf;
}

function f32Buf(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

describe('glTF matrix accessor column padding (§3.6.2.4)', () => {
  it('decodes a MAT3 accessor over BYTE components with per-column 4-byte padding', () => {
    // Column-major: col0 = [1,2,3], col1 = [4,5,6], col2 = [7,8,9].
    // Each 3-byte column padded to 4 bytes: [d,d,d,pad].
    const PAD = 0;
    const buffer = bytesBuf([
      1, 2, 3, PAD,
      4, 5, 6, PAD,
      7, 8, 9, PAD,
    ]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.BYTE,
        count: 1,
        type: 'MAT3',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: buffer.byteLength }],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('decodes two MAT3 / BYTE matrices honoring the padded element stride', () => {
    const PAD = 0;
    const buffer = bytesBuf([
      // matrix 0
      1, 2, 3, PAD,
      4, 5, 6, PAD,
      7, 8, 9, PAD,
      // matrix 1
      10, 11, 12, PAD,
      13, 14, 15, PAD,
      16, 17, 18, PAD,
    ]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.BYTE,
        count: 2,
        type: 'MAT3',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: buffer.byteLength }],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []);
    expect(Array.from(out)).toEqual([
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      10, 11, 12, 13, 14, 15, 16, 17, 18,
    ]);
  });

  it('decodes a MAT2 accessor over BYTE components with per-column 4-byte padding', () => {
    // Column-major: col0 = [1,2], col1 = [3,4]. Each 2-byte column padded to 4.
    const PAD = 0;
    const buffer = bytesBuf([
      1, 2, PAD, PAD,
      3, 4, PAD, PAD,
    ]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.BYTE,
        count: 1,
        type: 'MAT2',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: buffer.byteLength }],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []);
    expect(Array.from(out)).toEqual([1, 2, 3, 4]);
  });

  it('decodes a MAT3 accessor over SHORT components with per-column 8-byte padding', () => {
    // Column-major: col0 = [1,2,3], col1 = [4,5,6], col2 = [7,8,9].
    // Each 6-byte column (3 shorts) padded to 8 bytes: 3 shorts + 1 pad short.
    const buffer = u16Buf([
      1, 2, 3, 0,
      4, 5, 6, 0,
      7, 8, 9, 0,
    ]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.SHORT,
        count: 1,
        type: 'MAT3',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: buffer.byteLength }],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('MAT4 over FLOAT components needs no padding (contiguous read is byte-identical)', () => {
    const values = Array.from({ length: 16 }, (_, i) => i + 1);
    const buffer = f32Buf(values);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.FLOAT,
        count: 1,
        type: 'MAT4',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: buffer.byteLength }],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []);
    expect(Array.from(out)).toEqual(values);
  });

  it('MAT4 over BYTE components needs no padding (4-row columns already 4-byte aligned)', () => {
    // 4 bytes per column, exactly 4-byte aligned — no padding bytes anywhere.
    const values = Array.from({ length: 16 }, (_, i) => i + 1);
    const buffer = bytesBuf(values);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.BYTE,
        count: 1,
        type: 'MAT4',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: buffer.byteLength }],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []);
    expect(Array.from(out)).toEqual(values);
  });

  it('MAT2 over SHORT components needs no padding (2-row columns already 4-byte aligned)', () => {
    const buffer = u16Buf([1, 2, 3, 4]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.SHORT,
        count: 1,
        type: 'MAT2',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: buffer.byteLength }],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []);
    expect(Array.from(out)).toEqual([1, 2, 3, 4]);
  });

  it('applies normalization per padded component (MAT2 / normalized BYTE)', () => {
    const PAD = 0;
    // col0 = [127, -127], col1 = [0, 127]. Normalized signed byte: v/127 clamped.
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setInt8(0, 127);
    view.setInt8(1, -127);
    view.setInt8(2, PAD);
    view.setInt8(3, PAD);
    view.setInt8(4, 0);
    view.setInt8(5, 127);
    view.setInt8(6, PAD);
    view.setInt8(7, PAD);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.BYTE,
        normalized: true,
        count: 1,
        type: 'MAT2',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 8 }],
      buffers: [{ byteLength: 8 }],
    };
    const out = unpackAccessorFloat(gltf, new Map([[0, buf]]), 0, []);
    expect(out[0]).toBeCloseTo(1, 7);
    expect(out[1]).toBeCloseTo(-1, 7);
    expect(out[2]).toBeCloseTo(0, 7);
    expect(out[3]).toBeCloseTo(1, 7);
  });

  describe('accessorBufferViewRange mirrors the padded element byte length', () => {
    it('MAT3 / BYTE element byte length is 12 (4 per column), not 9', () => {
      const range = accessorBufferViewRange(
        { componentType: GltfComponentType.BYTE, count: 2, type: 'MAT3' } as never,
      );
      expect(range.elementByteLength).toBe(12);
      // stride defaults to the padded element length
      expect(range.byteStride).toBe(12);
      // required = (count-1)*stride + elementByteLength
      expect(range.requiredByteLength).toBe(12 + 12);
    });

    it('MAT2 / BYTE element byte length is 8 (4 per column), not 4', () => {
      const range = accessorBufferViewRange(
        { componentType: GltfComponentType.BYTE, count: 1, type: 'MAT2' } as never,
      );
      expect(range.elementByteLength).toBe(8);
    });

    it('MAT3 / SHORT element byte length is 24 (8 per column), not 18', () => {
      const range = accessorBufferViewRange(
        { componentType: GltfComponentType.SHORT, count: 1, type: 'MAT3' } as never,
      );
      expect(range.elementByteLength).toBe(24);
    });

    it('MAT4 / BYTE element byte length is unpadded 16', () => {
      const range = accessorBufferViewRange(
        { componentType: GltfComponentType.BYTE, count: 1, type: 'MAT4' } as never,
      );
      expect(range.elementByteLength).toBe(16);
    });

    it('MAT4 / FLOAT element byte length is unpadded 64', () => {
      const range = accessorBufferViewRange(
        { componentType: GltfComponentType.FLOAT, count: 1, type: 'MAT4' } as never,
      );
      expect(range.elementByteLength).toBe(64);
    });

    it('VEC3 / BYTE (non-matrix) is unaffected by column padding', () => {
      const range = accessorBufferViewRange(
        { componentType: GltfComponentType.BYTE, count: 1, type: 'VEC3' } as never,
      );
      expect(range.elementByteLength).toBe(3);
    });
  });
});
