import { describe, expect, it, vi } from 'vitest';
import {
  createGltfSceneController,
  gltfToScene,
  type GltfJson,
  type GltfScenePatchTarget,
} from './index.js';
import {
  asMat4,
  type AnimationClip,
  type MaterialSpec,
  type MeshPrimitive,
  type Scene,
  type SkinnedMeshPrimitive,
} from '@vitrum/core';

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
    [0, 0, 0, 0, 4, 0],
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
        { bufferView: 3, componentType: 5126, count: 2, type: 'VEC3' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
      animations: [
        {
          name: 'parent-slide',
          samplers: [{ input: 1, output: 2, interpolation: 'LINEAR' }],
          channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
        },
        {
          name: 'parent-lift',
          samplers: [{ input: 1, output: 3, interpolation: 'LINEAR' }],
          channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
        },
      ],
    },
  };
}

function morphGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const packed = packF32([
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1],
    [0, 1],
    [0, 0],
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
        { bufferView: 4, componentType: 5126, count: 2, type: 'SCALAR' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
      animations: [
        {
          name: 'morph-on',
          samplers: [{ input: 2, output: 3, interpolation: 'LINEAR' }],
          channels: [{ sampler: 0, target: { node: 0, path: 'weights' } }],
        },
        {
          name: 'morph-off',
          samplers: [{ input: 2, output: 4, interpolation: 'LINEAR' }],
          channels: [{ sampler: 0, target: { node: 0, path: 'weights' } }],
        },
      ],
    },
  };
}

function materialVariantGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const packed = packF32([
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
  ]);
  return {
    buffers: new Map([[0, packed.buffer]]),
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      extensionsUsed: ['KHR_materials_variants'],
      extensionsRequired: ['KHR_materials_variants'],
      extensions: {
        KHR_materials_variants: {
          variants: [{ name: 'blue' }],
        },
      },
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0 },
          material: 0,
          extensions: {
            KHR_materials_variants: {
              mappings: [{ material: 1, variants: [0] }],
            },
          },
        }],
      }],
      materials: [
        { name: 'base red', pbrMetallicRoughness: { baseColorFactor: [1, 0, 0, 1] } },
        { name: 'variant blue', pbrMetallicRoughness: { baseColorFactor: [0, 0, 1, 1] } },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
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
  diagnostics: [];
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
    diagnostics: [],
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

  it('falls back to setScene when updatePrimitive throws', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const result = await gltfToScene(gltf, { buffers });
    const setScene = vi.fn();
    const updatePrimitive = vi.fn(() => {
      throw new Error('geometry patch rejected');
    });
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation('parent-slide', 1, {
      engine: { setScene, updatePrimitive },
    });

    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(frame.usedSetScene).toBe(true);
    expect(setScene).toHaveBeenCalledTimes(1);
    const scene = setScene.mock.calls[0]![0] as Scene;
    expect((scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(2);
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(2);
    expect(frame.warnings.some((w) => w.includes('falling back to setScene'))).toBe(true);
    expect(frame.warnings.some((w) => w.includes('geometry patch rejected'))).toBe(true);
  });

  it('switches KHR_materials_variants via material-only primitive patches', async () => {
    const { gltf, buffers } = materialVariantGltf();
    const result = await gltfToScene(gltf, { buffers });
    const setScene = vi.fn();
    const updatePrimitive = vi.fn();
    const controller = createGltfSceneController({ gltf, ...result });

    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([1, 0, 0]);

    const frame = controller.setVariant('blue', {
      engine: { setScene, updatePrimitive },
    });

    expect(frame.variantIndex).toBe(0);
    expect(frame.usedSetScene).toBe(false);
    expect(frame.primitivePatches).toHaveLength(1);
    expect(updatePrimitive).toHaveBeenCalledWith(
      'gltf-prim-0',
      expect.objectContaining({
        material: expect.objectContaining({ baseColor: [0, 0, 1] }),
      }),
    );
    expect(setScene).not.toHaveBeenCalled();
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([0, 0, 1]);

    controller.resetPose();
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([0, 0, 1]);

    const reset = controller.setVariant(undefined);
    expect((reset.primitivePatches[0]!.patch as { material: MaterialSpec }).material.baseColor)
      .toEqual([1, 0, 0]);
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([1, 0, 0]);
  });

  it('falls back to setScene when a variant material patch is rejected', async () => {
    const { gltf, buffers } = materialVariantGltf();
    const result = await gltfToScene(gltf, { buffers });
    const setScene = vi.fn();
    const updatePrimitive = vi.fn(() => {
      throw new Error('material fast path unavailable');
    });
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.setVariant(0, {
      engine: { setScene, updatePrimitive },
    });

    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(frame.usedSetScene).toBe(true);
    expect(setScene).toHaveBeenCalledTimes(1);
    const scene = setScene.mock.calls[0]![0] as Scene;
    expect((scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([0, 0, 1]);
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([0, 0, 1]);
    expect(frame.warnings.some((w) => w.includes('falling back to setScene'))).toBe(true);
    expect(frame.warnings.some((w) => w.includes('material fast path unavailable'))).toBe(true);
  });

  it('blends transform clips per channel before patching primitives', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const result = await gltfToScene(gltf, { buffers });
    const updatePrimitive = vi.fn();
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.blend(['parent-slide', 'parent-lift'], [0.25, 0.75], 1, {
      engine: { setScene: vi.fn(), updatePrimitive },
    });

    expect(frame.usedSetScene).toBe(false);
    expect(frame.weights).toEqual([0.25, 0.75]);
    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    const patch = frame.primitivePatches[0]!.patch as { transform: Float32Array };
    expect(patch.transform[12]).toBeCloseTo(0.5);
    expect(patch.transform[13]).toBeCloseTo(3);
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(0.5);
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[13]).toBeCloseTo(3);
  });

  it('blends morph weights before skin solving', async () => {
    const { gltf, buffers } = morphGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.blend(['morph-on', 'morph-off'], [0.25, 0.75], 1);

    const patch = frame.primitivePatches[0]!.patch as {
      morphWeights: Float32Array;
      positions: Float32Array;
    };
    expect(patch.morphWeights[0]).toBeCloseTo(0.25);
    expect(Array.from(patch.positions)).toEqual([0.25, 0, 0, 1.25, 0, 0, 0.25, 1, 0]);
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
