import { describe, expect, it } from 'vitest';
import { unpackAccessorFloat, unpackAccessorUint32 } from './accessors.js';
import { GltfComponentType, type GltfJson } from './gltfTypes.js';

function bytes(values: number[]): ArrayBuffer {
  return new Uint8Array(values).buffer;
}

function u16(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 2);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setUint16(i * 2, v, true));
  return buf;
}

function f32(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function concat(...parts: ArrayBuffer[]): ArrayBuffer {
  const total = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(new Uint8Array(part), offset);
    offset += part.byteLength;
  }
  return out.buffer;
}

describe('unpackAccessorFloat sparse accessors', () => {
  it('applies a pure sparse accessor with unsigned-byte indices and normalized unsigned-byte values', () => {
    const indices = bytes([2]);
    const values = bytes([255, 128]);
    const buffer = concat(indices, values);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        componentType: GltfComponentType.UNSIGNED_BYTE,
        normalized: true,
        count: 4,
        type: 'VEC2',
        sparse: {
          count: 1,
          indices: { bufferView: 0, componentType: GltfComponentType.UNSIGNED_BYTE },
          values: { bufferView: 1 },
        },
      }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: indices.byteLength },
        { buffer: 0, byteOffset: indices.byteLength, byteLength: values.byteLength },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const warnings: string[] = [];
    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, warnings);
    expect(Array.from(out.slice(0, 4))).toEqual([0, 0, 0, 0]);
    expect(out[4]).toBeCloseTo(1, 7);
    expect(out[5]).toBeCloseTo(128 / 255, 7);
    expect(Array.from(out.slice(6))).toEqual([0, 0]);
    expect(warnings.some((w) => w.includes('sparse storage'))).toBe(true);
  });

  it('applies sparse byte offsets on top of strided base data', () => {
    const base = concat(
      f32([1, 2, 99, 3, 4, 99, 5, 6, 99]),
      bytes([123, 1]),
      bytes([77]),
      f32([8, 9]),
    );
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.FLOAT,
        count: 3,
        type: 'VEC2',
        sparse: {
          count: 1,
          indices: {
            bufferView: 1,
            byteOffset: 1,
            componentType: GltfComponentType.UNSIGNED_BYTE,
          },
          values: {
            bufferView: 2,
            byteOffset: 1,
          },
        },
      }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 36, byteStride: 12 },
        { buffer: 0, byteOffset: 36, byteLength: 2 },
        { buffer: 0, byteOffset: 38, byteLength: 9 },
      ],
      buffers: [{ byteLength: base.byteLength }],
    };
    const out = unpackAccessorFloat(gltf, new Map([[0, base]]), 0, []);
    expect(Array.from(out)).toEqual([1, 2, 8, 9, 5, 6]);
  });

  it('warns and skips sparse entries outside the accessor count', () => {
    const indices = bytes([1, 4]);
    const values = f32([7, 8, 9, 10]);
    const buffer = concat(indices, values);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        componentType: GltfComponentType.FLOAT,
        count: 3,
        type: 'VEC2',
        sparse: {
          count: 2,
          indices: { bufferView: 0, componentType: GltfComponentType.UNSIGNED_BYTE },
          values: { bufferView: 1 },
        },
      }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: indices.byteLength },
        { buffer: 0, byteOffset: indices.byteLength, byteLength: values.byteLength },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const warnings: string[] = [];
    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, warnings);
    expect(Array.from(out)).toEqual([0, 0, 7, 8, 0, 0]);
    expect(warnings.some((w) => w.includes('Sparse index 4 is outside accessor count 3'))).toBe(true);
  });

  it('warns and skips sparse patches with invalid signed index component types', () => {
    const indices = bytes([1]);
    const values = f32([7, 8]);
    const buffer = concat(indices, values);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        componentType: GltfComponentType.FLOAT,
        count: 2,
        type: 'VEC2',
        sparse: {
          count: 1,
          indices: { bufferView: 0, componentType: GltfComponentType.BYTE },
          values: { bufferView: 1 },
        },
      }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: indices.byteLength },
        { buffer: 0, byteOffset: indices.byteLength, byteLength: values.byteLength },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const warnings: string[] = [];
    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, warnings);
    expect(Array.from(out)).toEqual([0, 0, 0, 0]);
    expect(warnings.some((w) => w.includes('Sparse indices componentType'))).toBe(true);
  });
});

describe('unpackAccessorUint32 sparse accessors', () => {
  it('applies sparse integer patches to index accessors', () => {
    const base = u16([0, 1, 2]);
    const indices = bytes([1]);
    const values = u16([7]);
    const buffer = concat(base, indices, values);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.UNSIGNED_SHORT,
        count: 3,
        type: 'SCALAR',
        sparse: {
          count: 1,
          indices: { bufferView: 1, componentType: GltfComponentType.UNSIGNED_BYTE },
          values: { bufferView: 2 },
        },
      }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: base.byteLength },
        { buffer: 0, byteOffset: base.byteLength, byteLength: indices.byteLength },
        { buffer: 0, byteOffset: base.byteLength + indices.byteLength, byteLength: values.byteLength },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const out = unpackAccessorUint32(gltf, new Map([[0, buffer]]), 0);
    expect(Array.from(out)).toEqual([0, 7, 2]);
  });
});
