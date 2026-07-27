import { describe, expect, it } from 'vitest';
import {
  unpackAccessorFloat,
  unpackAccessorUint32,
} from './accessors.js';
import { GltfComponentType, type GltfJson } from './gltfTypes.js';
import { ImportResourceLedger } from './importResourceBudget.js';

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

function scalarFloatAccessorGltf(
  bufferView: NonNullable<GltfJson['bufferViews']>[number],
  count = 1,
): GltfJson {
  return {
    asset: { version: '2.0' },
    accessors: [{
      bufferView: 0,
      componentType: GltfComponentType.FLOAT,
      count,
      type: 'SCALAR',
    }],
    bufferViews: [bufferView],
    buffers: [{ byteLength: 32 }],
  };
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
      bytes([0, 0]),
      bytes([77, 0, 0, 0]),
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
            byteOffset: 4,
          },
        },
      }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 36, byteStride: 12 },
        { buffer: 0, byteOffset: 36, byteLength: 2 },
        { buffer: 0, byteOffset: 40, byteLength: 12 },
      ],
      buffers: [{ byteLength: base.byteLength }],
    };
    const out = unpackAccessorFloat(gltf, new Map([[0, base]]), 0, []);
    expect(Array.from(out)).toEqual([1, 2, 8, 9, 5, 6]);
  });

  it('rejects sparse entries outside the accessor count', () => {
    const indices = bytes([1, 4]);
    const values = f32([7, 8, 9, 10]);
    const buffer = concat(indices, bytes([0, 0]), values);
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
        { buffer: 0, byteOffset: 4, byteLength: values.byteLength },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    expect(() => unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []))
      .toThrow('Sparse index 4 is outside accessor count 3');
  });

  it('rejects sparse patches with invalid signed index component types', () => {
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
    expect(() => unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []))
      .toThrow('Sparse indices componentType 5120 is invalid');
  });

  it('rejects base accessors that read past the declared bufferView byteLength', () => {
    const buffer = f32([1, 2, 3, 4, 5, 6]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.FLOAT,
        count: 3,
        type: 'VEC2',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 8 }],
      buffers: [{ byteLength: buffer.byteLength }],
    };

    expect(() => unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []))
      .toThrow('requires 24 bytes from bufferView 0, but it declares byteLength 8');
  });

  it('rejects sparse patches that read past declared bufferView byteLength', () => {
    const indices = bytes([1]);
    const values = f32([7, 8, 9, 10]);
    const buffer = concat(indices, bytes([0, 0, 0]), values);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        componentType: GltfComponentType.FLOAT,
        count: 2,
        type: 'VEC2',
        sparse: {
          count: 1,
          indices: { bufferView: 0, componentType: GltfComponentType.UNSIGNED_BYTE },
          values: { bufferView: 1 },
        },
      }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: indices.byteLength },
        { buffer: 0, byteOffset: 4, byteLength: 4 },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    expect(() => unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []))
      .toThrow('sparse values requires 8 bytes from bufferView 1, but it declares byteLength 4');
  });
});

describe('unpackAccessorUint32 sparse accessors', () => {
  it('applies sparse integer patches to index accessors', () => {
    const base = u16([0, 1, 2]);
    const indices = bytes([1]);
    const values = u16([7]);
    const buffer = concat(base, indices, bytes([0]), values);
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
        { buffer: 0, byteOffset: 8, byteLength: values.byteLength },
      ],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const out = unpackAccessorUint32(gltf, new Map([[0, buffer]]), 0);
    expect(Array.from(out)).toEqual([0, 7, 2]);
  });

  it('applies pure-sparse index accessors on top of the implicit zero base', () => {
    const indices = bytes([0, 2]);
    const values = u16([5, 9]);
    const buffer = concat(indices, values);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        componentType: GltfComponentType.UNSIGNED_SHORT,
        count: 4,
        type: 'SCALAR',
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
    const diagnostics: unknown[] = [];

    const out = unpackAccessorUint32(gltf, new Map([[0, buffer]]), 0, [], (diagnostic) => {
      diagnostics.push(diagnostic);
    });

    expect(Array.from(out)).toEqual([5, 0, 9, 0]);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: 'sparse-accessor-applied',
      path: 'accessors[0].sparse',
      accessorIndex: 0,
    }));
  });

  it('rejects index accessors that read past the declared bufferView byteLength', () => {
    const buffer = u16([0, 1, 2]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        bufferView: 0,
        componentType: GltfComponentType.UNSIGNED_SHORT,
        count: 3,
        type: 'SCALAR',
      }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 4 }],
      buffers: [{ byteLength: buffer.byteLength }],
    };

    expect(() => unpackAccessorUint32(gltf, new Map([[0, buffer]]), 0))
      .toThrow('requires 6 bytes from bufferView 0, but it declares byteLength 4');
  });
});

describe('accessor range validation and aggregate accounting', () => {
  it('rejects NaN and fractional bufferView range fields before DataView coercion', () => {
    const buffer = new ArrayBuffer(32);
    const nanOffset = scalarFloatAccessorGltf({
      buffer: 0,
      byteOffset: Number.NaN,
      byteLength: 4,
    });
    expect(() => unpackAccessorFloat(nanOffset, new Map([[0, buffer]]), 0, []))
      .toThrow(/byteOffset must be a non-negative safe integer/);

    const fractionalLength = scalarFloatAccessorGltf({
      buffer: 0,
      byteOffset: 0,
      byteLength: 4.5,
    });
    expect(() => unpackAccessorFloat(fractionalLength, new Map([[0, buffer]]), 0, []))
      .toThrow(/byteLength must be a non-negative safe integer/);
  });

  it('uses the intrinsic ArrayBuffer byte length when an own property shadows byteLength', () => {
    const validBuffer = f32([7]);
    Object.defineProperty(validBuffer, 'byteLength', { value: 0 });
    const validGltf = scalarFloatAccessorGltf({
      buffer: 0,
      byteOffset: 0,
      byteLength: 4,
    });
    expect(Array.from(unpackAccessorFloat(
      validGltf,
      new Map([[0, validBuffer]]),
      0,
      [],
    ))).toEqual([7]);

    const truncatedBuffer = f32([7]);
    Object.defineProperty(truncatedBuffer, 'byteLength', { value: 1024 });
    const truncatedGltf = scalarFloatAccessorGltf({
      buffer: 0,
      byteOffset: 0,
      byteLength: 8,
    }, 2);
    expect(() => unpackAccessorFloat(
      truncatedGltf,
      new Map([[0, truncatedBuffer]]),
      0,
      [],
    )).toThrow(/outside buffer length 4/);
  });

  it('rejects a base bufferView offset that is not component-aligned', () => {
    const buffer = new ArrayBuffer(8);
    const gltf = scalarFloatAccessorGltf({
      buffer: 0,
      byteOffset: 2,
      byteLength: 4,
    });

    expect(() => unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []))
      .toThrow(/bufferView 0\.byteOffset 2 is not aligned to component size 4/);
  });

  it('rejects unaligned sparse indices and values byte offsets', () => {
    const indexMisaligned: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        componentType: GltfComponentType.FLOAT,
        count: 1,
        type: 'SCALAR',
        sparse: {
          count: 1,
          indices: {
            bufferView: 0,
            byteOffset: 1,
            componentType: GltfComponentType.UNSIGNED_SHORT,
          },
          values: { bufferView: 1 },
        },
      }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 3 },
        { buffer: 0, byteOffset: 4, byteLength: 4 },
      ],
      buffers: [{ byteLength: 8 }],
    };
    expect(() => unpackAccessorFloat(
      indexMisaligned,
      new Map([[0, new ArrayBuffer(8)]]),
      0,
      [],
    )).toThrow(/sparse\.indices\.byteOffset 1 is not aligned to component size 2/);

    const valueMisaligned: GltfJson = {
      asset: { version: '2.0' },
      accessors: [{
        componentType: GltfComponentType.FLOAT,
        count: 1,
        type: 'SCALAR',
        sparse: {
          count: 1,
          indices: {
            bufferView: 0,
            componentType: GltfComponentType.UNSIGNED_BYTE,
          },
          values: { bufferView: 1, byteOffset: 2 },
        },
      }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 1 },
        { buffer: 0, byteOffset: 4, byteLength: 6 },
      ],
      buffers: [{ byteLength: 10 }],
    };
    expect(() => unpackAccessorFloat(
      valueMisaligned,
      new Map([[0, new ArrayBuffer(10)]]),
      0,
      [],
    )).toThrow(/sparse\.values\.byteOffset 2 is not aligned to component size 4/);
  });

  it('reports duplicate and descending sparse indices with a distinct structured code', () => {
    for (const indices of [[1, 1], [2, 1]]) {
      const sparseIndices = bytes(indices);
      const sparseValues = f32([7, 8]);
      const buffer = concat(sparseIndices, bytes([0, 0]), sparseValues);
      const gltf: GltfJson = {
        asset: { version: '2.0' },
        accessors: [{
          componentType: GltfComponentType.FLOAT,
          count: 3,
          type: 'SCALAR',
          sparse: {
            count: 2,
            indices: {
              bufferView: 0,
              componentType: GltfComponentType.UNSIGNED_BYTE,
            },
            values: { bufferView: 1 },
          },
        }],
        bufferViews: [
          { buffer: 0, byteOffset: 0, byteLength: sparseIndices.byteLength },
          { buffer: 0, byteOffset: 4, byteLength: sparseValues.byteLength },
        ],
        buffers: [{ byteLength: buffer.byteLength }],
      };
      const diagnostics: Array<{ code: string }> = [];

      expect(() => unpackAccessorFloat(
        gltf,
        new Map([[0, buffer]]),
        0,
        [],
        (diagnostic) => diagnostics.push(diagnostic),
      )).toThrow(/not strictly increasing/);
      expect(diagnostics).toContainEqual(expect.objectContaining({
        code: 'sparse-indices-not-strictly-increasing',
      }));
    }
  });

  it('shares one decoded-geometry budget across separate accessor decodes', () => {
    const buffer = f32([1, 2]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      accessors: [0, 1].map(() => ({
        bufferView: 0,
        componentType: GltfComponentType.FLOAT,
        count: 2,
        type: 'SCALAR',
      })),
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: buffer.byteLength }],
      buffers: [{ byteLength: buffer.byteLength }],
    };
    const ledger = new ImportResourceLedger({
      maxDecodedGeometryBytes: 12,
    });

    expect(Array.from(unpackAccessorFloat(
      gltf,
      new Map([[0, buffer]]),
      0,
      [],
      undefined,
      ledger,
    ))).toEqual([1, 2]);
    expect(() => unpackAccessorFloat(
      gltf,
      new Map([[0, buffer]]),
      1,
      [],
      undefined,
      ledger,
    )).toThrow(/decoded-geometry-bytes/);
    expect(ledger.decodedGeometryBytes).toBe(8);
  });
});
