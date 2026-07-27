import { describe, expect, it } from 'vitest';
import { gltfToScene } from './gltfToScene.js';
import type { GltfJson } from './gltfTypes.js';

function f32(values: readonly number[]): Uint8Array {
  return new Uint8Array(new Float32Array(values).buffer);
}

function sparseJointsAsset(indices: readonly number[]): {
  readonly gltf: GltfJson;
  readonly buffer: ArrayBuffer;
} {
  const positions = f32([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  const weights = f32([
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
  ]);
  const baseJoints = new Uint8Array(12);
  const sparseIndices = new Uint8Array(indices);
  const sparseValues = new Uint8Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
  ]);
  const sparseValuesOffset =
    positions.byteLength +
    weights.byteLength +
    baseJoints.byteLength +
    4;
  const bytes = new Uint8Array(sparseValuesOffset + sparseValues.byteLength);
  let offset = 0;
  bytes.set(positions, offset);
  offset += positions.byteLength;
  bytes.set(weights, offset);
  offset += weights.byteLength;
  bytes.set(baseJoints, offset);
  offset += baseJoints.byteLength;
  bytes.set(sparseIndices, offset);
  bytes.set(sparseValues, sparseValuesOffset);

  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [
        { mesh: 0, skin: 0, children: [1, 2] },
        {},
        {},
      ],
      meshes: [{
        primitives: [{
          attributes: {
            POSITION: 0,
            WEIGHTS_0: 1,
            JOINTS_0: 2,
          },
        }],
      }],
      skins: [{ joints: [1, 2] }],
      accessors: [
        {
          bufferView: 0,
          componentType: 5126,
          count: 3,
          type: 'VEC3',
        },
        {
          bufferView: 1,
          componentType: 5126,
          count: 3,
          type: 'VEC4',
        },
        {
          bufferView: 2,
          componentType: 5121,
          count: 3,
          type: 'VEC4',
          sparse: {
            count: 2,
            indices: { bufferView: 3, componentType: 5121 },
            values: { bufferView: 4 },
          },
        },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        {
          buffer: 0,
          byteOffset: positions.byteLength,
          byteLength: weights.byteLength,
        },
        {
          buffer: 0,
          byteOffset: positions.byteLength + weights.byteLength,
          byteLength: baseJoints.byteLength,
        },
        {
          buffer: 0,
          byteOffset:
            positions.byteLength +
            weights.byteLength +
            baseJoints.byteLength,
          byteLength: sparseIndices.byteLength,
        },
        {
          buffer: 0,
          byteOffset: sparseValuesOffset,
          byteLength: sparseValues.byteLength,
        },
      ],
      buffers: [{ byteLength: bytes.byteLength }],
    },
    buffer: bytes.buffer,
  };
}

describe('sparse JOINTS ordering', () => {
  it.each([
    ['duplicate', [1, 1]],
    ['descending', [2, 1]],
  ] as const)('rejects %s sparse indices with the canonical code', async (_label, indices) => {
    const { gltf, buffer } = sparseJointsAsset(indices);

    await expect(gltfToScene(gltf, {
      buffers: new Map([[0, buffer]]),
    })).rejects.toMatchObject({
      diagnostics: expect.arrayContaining([
        expect.objectContaining({
          code: 'sparse-indices-not-strictly-increasing',
          path: 'accessors[2].sparse.indices[1]',
        }),
      ]),
    });
  });
});
