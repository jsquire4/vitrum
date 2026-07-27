import { describe, expect, it } from 'vitest';
import { unpackAccessorFloat } from './accessors.js';
import { GltfComponentType, type GltfJson } from './gltfTypes.js';

function sparseMat3ByteFixture(valueByteLength = 12): {
  gltf: GltfJson;
  buffer: ArrayBuffer;
} {
  const buffer = new ArrayBuffer(4 + valueByteLength);
  const bytes = new Uint8Array(buffer);
  bytes[0] = 1; // patch accessor element 1
  const padded = [
    1, 2, 3, 0,
    4, 5, 6, 0,
    7, 8, 9, 0,
  ];
  bytes.set(padded.slice(0, valueByteLength), 4);

  return {
    buffer,
    gltf: {
      asset: { version: '2.0' },
      buffers: [{ byteLength: buffer.byteLength }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 1 },
        { buffer: 0, byteOffset: 4, byteLength: valueByteLength },
      ],
      accessors: [{
        componentType: GltfComponentType.BYTE,
        count: 2,
        type: 'MAT3',
        sparse: {
          count: 1,
          indices: {
            bufferView: 0,
            componentType: GltfComponentType.UNSIGNED_BYTE,
          },
          values: { bufferView: 1 },
        },
      }],
    },
  };
}

describe('sparse matrix accessor column padding', () => {
  it('decodes sparse MAT3/BYTE values using the padded per-column layout', () => {
    const { gltf, buffer } = sparseMat3ByteFixture();
    const warnings: string[] = [];

    const out = unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, warnings);

    expect(Array.from(out)).toEqual([
      0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 2, 3, 4, 5, 6, 7, 8, 9,
    ]);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain('applying patch');
  });

  it('requires the padded sparse value byte length rather than the tight scalar count', () => {
    // Nine data bytes are enough for a tight MAT3/BYTE, but glTF requires
    // three four-byte-aligned columns (12 bytes total).
    const { gltf, buffer } = sparseMat3ByteFixture(9);
    expect(() => unpackAccessorFloat(gltf, new Map([[0, buffer]]), 0, []))
      .toThrow('sparse values requires 12 bytes from bufferView 1, but it declares byteLength 9');
  });
});
