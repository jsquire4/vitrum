import { describe, expect, it, vi } from 'vitest';
import {
  createGltfSceneController,
  gltfToScene,
  type GltfJson,
  type GltfScenePatchTarget,
} from './index.js';
import { asMat4, type AnimationClip, type MaterialSpec, type Scene, type SkinnedMeshPrimitive } from '@vitrum/core';

const MATERIAL: MaterialSpec = {
  baseColor: [1, 1, 1],
  roughness: 1,
  metallic: 0,
};

function packF32(chunks: readonly number[][]): {
  buffer: ArrayBuffer;
  views: Array<{ buffer: number; byteOffset: number; byteLength: number }>;
} {
  const arrays = chunks.map((chunk) => new Float32Array(chunk));
  const totalBytes = arrays.reduce((sum, array) => sum + array.byteLength, 0);
  const buffer = new ArrayBuffer(totalBytes);
  const views: Array<{ buffer: number; byteOffset: number; byteLength: number }> = [];
  let offset = 0;
  for (const array of arrays) {
    new Uint8Array(buffer, offset, array.byteLength).set(new Uint8Array(array.buffer));
    views.push({ buffer: 0, byteOffset: offset, byteLength: array.byteLength });
    offset += array.byteLength;
  }
  return { buffer, views };
}

function animatedHierarchyGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const packed = packF32([
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1],
    [0, 0, 0, 2, 0, 0],
  ]);
  return {
    buffers: new Map([[0, packed.buffer]]),
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [
        { children: [1] },
        { mesh: 0 },
      ],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 2, componentType: 5126, count: 2, type: 'VEC3' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
      animations: [{
        name: 'parent-slide',
        samplers: [{ input: 1, output: 2, interpolation: 'LINEAR' }],
        channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
      }],
    },
  };
}

function morphGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const packed = packF32([
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1],
    [0, 1],
  ]);
  return {
    buffers: new Map([[0, packed.buffer]]),
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{
        weights: [0],
        primitives: [{ attributes: { POSITION: 0 }, targets: [{ POSITION: 1 }] }],
      }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 2, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 3, componentType: 5126, count: 2, type: 'SCALAR' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
      animations: [{
        name: 'morph-on',
        samplers: [{ input: 2, output: 3, interpolation: 'LINEAR' }],
        channels: [{ sampler: 0, target: { node: 0, path: 'weights' } }],
      }],
    },
  };
}

function identityMat4(): Float32Array {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

function manualSkinnedInput(): {
  gltf: GltfJson;
  scene: Scene;
  animations: readonly AnimationClip[];
  animationTargets: Record<string, readonly string[]>;
  warnings: string[];
} {
  const identity = identityMat4();
  const skinIndices = new Uint32Array(12);
  const skinWeights = new Float32Array(12);
  for (let vertex = 0; vertex < 3; vertex += 1) skinWeights[vertex * 4] = 1;
  const primitive: SkinnedMeshPrimitive = {
    kind: 'skinned-mesh',
    id: 'skin-prim',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: MATERIAL,
    transform: asMat4(new Float32Array(identity)),
    skinIndices,
    skinWeights,
    bones: new Float32Array(identity),
    boneInverses: new Float32Array(identity),
  };
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [
        { mesh: 0, skin: 0, children: [1] },
        {},
      ],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      skins: [{ joints: [1] }],
    },
    scene: {
      primitives: [primitive],
      emitters: [],
      environment: { kind: 'none' },
    },
    animations: [{
      name: 'joint-slide',
      duration: 1,
      channels: [{
        target: { node: 'gltf-node-1', path: 'translation' },
        sampler: {
          times: new Float32Array([0, 1]),
          values: new Float32Array([0, 0, 0, 1, 0, 0]),
        },
      }],
    }],
    animationTargets: { 'gltf-node-0': ['skin-prim'] },
    warnings: [],
  };
}

describe('GltfSceneController', () => {
  it('recomputes child primitive world transforms when an ancestor node is animated', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const result = await gltfToScene(gltf, { buffers });
    const updatePrimitive = vi.fn();
    const engine: GltfScenePatchTarget = { setScene: vi.fn(), updatePrimitive };
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation(0, 0.5, { engine });

    expect(frame.usedSetScene).toBe(false);
    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    const patch = frame.primitivePatches[0]!.patch as { transform: Float32Array };
    expect(patch.transform[12]).toBeCloseTo(1);
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(1);
  });

  it('falls back to setScene when the target has no updatePrimitive method', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const result = await gltfToScene(gltf, { buffers });
    const setScene = vi.fn();
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation('parent-slide', 1, { engine: { setScene } });

    expect(frame.usedSetScene).toBe(true);
    expect(setScene).toHaveBeenCalledTimes(1);
    const scene = setScene.mock.calls[0]![0] as Scene;
    expect((scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(2);
  });

  it('samples morph-weight channels, solves the promoted skinned primitive, and patches deformed geometry', async () => {
    const { gltf, buffers } = morphGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation('morph-on', 1);

    const patch = frame.primitivePatches[0]!.patch as {
      morphWeights: Float32Array;
      positions: Float32Array;
    };
    expect(patch.morphWeights[0]).toBeCloseTo(1);
    expect(Array.from(patch.positions)).toEqual([1, 0, 0, 2, 0, 0, 1, 1, 0]);
    expect((controller.scene.primitives[0] as SkinnedMeshPrimitive).morphWeights![0]).toBeCloseTo(1);
  });

  it('rebuilds skinned bone matrices from animated joint nodes and patches solved vertices', () => {
    const input = manualSkinnedInput();
    const controller = createGltfSceneController(input);

    const frame = controller.applyAnimation('joint-slide', 1);

    const patch = frame.primitivePatches[0]!.patch as {
      bones: Float32Array;
      positions: Float32Array;
    };
    expect(patch.bones[12]).toBeCloseTo(1);
    expect(Array.from(patch.positions)).toEqual([1, 0, 0, 2, 0, 0, 1, 1, 0]);
  });
});

