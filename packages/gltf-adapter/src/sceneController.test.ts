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
  type InstancedMeshPrimitive,
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

function animatedCameraGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const packed = packF32([
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
        { name: 'CameraParent', children: [1] },
        { name: 'CameraNode', camera: 0, translation: [0, 1, 3] },
      ],
      cameras: [{
        name: 'HeroCam',
        type: 'perspective',
        perspective: {
          yfov: 0.75,
          znear: 0.1,
          zfar: 100,
          aspectRatio: 1.5,
        },
      }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 1, componentType: 5126, count: 2, type: 'VEC3' },
        { bufferView: 2, componentType: 5126, count: 2, type: 'VEC3' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
      animations: [
        {
          name: 'camera-parent-slide',
          samplers: [{ input: 0, output: 1, interpolation: 'LINEAR' }],
          channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
        },
        {
          name: 'camera-parent-lift',
          samplers: [{ input: 0, output: 2, interpolation: 'LINEAR' }],
          channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
        },
      ],
    },
  };
}

function animatedInstancedGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const packed = packF32([
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [
      2, 0, 0,
      0, 3, 0,
    ],
    [0, 1],
    [
      10, 0, 0,
      14, 0, 0,
    ],
  ]);
  return {
    buffers: new Map([[0, packed.buffer]]),
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      extensionsUsed: ['EXT_mesh_gpu_instancing'],
      extensionsRequired: ['EXT_mesh_gpu_instancing'],
      nodes: [{
        mesh: 0,
        translation: [10, 0, 0],
        extensions: {
          EXT_mesh_gpu_instancing: {
            attributes: { TRANSLATION: 1 },
          },
        },
      }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 2, type: 'VEC3' },
        { bufferView: 2, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 3, componentType: 5126, count: 2, type: 'VEC3' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
      animations: [{
        name: 'instance-slide',
        samplers: [{ input: 2, output: 3, interpolation: 'LINEAR' }],
        channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
      }],
    },
  };
}

function morphGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const packed = packF32([
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [
      1, 0, 0, 1,
      1, 0, 0, 1,
      1, 0, 0, 1,
    ],
    [1, 0, 0, 1, 0, 0, 1, 0, 0],
    [
      0, 1, 0,
      0, 1, 0,
      0, 1, 0,
    ],
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
        primitives: [{ attributes: { POSITION: 0, TANGENT: 1 }, targets: [{ POSITION: 2, TANGENT: 3 }] }],
      }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC4' },
        { bufferView: 2, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 3, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 4, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 5, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 6, componentType: 5126, count: 2, type: 'SCALAR' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
      animations: [
        {
          name: 'morph-on',
          samplers: [{ input: 4, output: 5, interpolation: 'LINEAR' }],
          channels: [{ sampler: 0, target: { node: 0, path: 'weights' } }],
        },
        {
          name: 'morph-off',
          samplers: [{ input: 4, output: 6, interpolation: 'LINEAR' }],
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

function materialPointerAnimationGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const packed = packF32([
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1],
    [
      1, 0, 0, 1,
      0, 1, 0, 0.5,
    ],
    [0.2, 0.8],
    [0.9, 0.1],
    [
      0, 0, 0,
      1, 0.5, 0.25,
    ],
  ]);
  return {
    buffers: new Map([[0, packed.buffer]]),
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      extensionsUsed: ['KHR_animation_pointer'],
      extensionsRequired: ['KHR_animation_pointer'],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{
        pbrMetallicRoughness: {
          baseColorFactor: [1, 0, 0, 1],
          metallicFactor: 0.2,
          roughnessFactor: 0.9,
        },
        emissiveFactor: [0, 0, 0],
        alphaMode: 'BLEND',
      }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 2, componentType: 5126, count: 2, type: 'VEC4' },
        { bufferView: 3, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 4, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 5, componentType: 5126, count: 2, type: 'VEC3' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
      animations: [{
        name: 'material-factors',
        samplers: [
          { input: 1, output: 2, interpolation: 'LINEAR' },
          { input: 1, output: 3, interpolation: 'LINEAR' },
          { input: 1, output: 4, interpolation: 'LINEAR' },
          { input: 1, output: 5, interpolation: 'LINEAR' },
        ],
        channels: [
          {
            sampler: 0,
            target: {
              path: 'pointer',
              extensions: {
                KHR_animation_pointer: {
                  pointer: '/materials/0/pbrMetallicRoughness/baseColorFactor',
                },
              },
            },
          },
          {
            sampler: 1,
            target: {
              path: 'pointer',
              extensions: {
                KHR_animation_pointer: {
                  pointer: '/materials/0/pbrMetallicRoughness/metallicFactor',
                },
              },
            },
          },
          {
            sampler: 2,
            target: {
              path: 'pointer',
              extensions: {
                KHR_animation_pointer: {
                  pointer: '/materials/0/pbrMetallicRoughness/roughnessFactor',
                },
              },
            },
          },
          {
            sampler: 3,
            target: {
              path: 'pointer',
              extensions: {
                KHR_animation_pointer: {
                  pointer: '/materials/0/emissiveFactor',
                },
              },
            },
          },
        ],
      }],
    },
  };
}

function materialVariantHighUvGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const packed = packF32([
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
    [0.25, 0.25, 0.5, 0.25, 0.25, 0.5],
  ]);
  const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
  return {
    buffers: new Map([
      [0, packed.buffer],
      [1, imageBytes.buffer],
    ]),
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      extensionsUsed: ['KHR_materials_variants'],
      extensionsRequired: ['KHR_materials_variants'],
      extensions: {
        KHR_materials_variants: {
          variants: [{ name: 'uv2-textured' }],
        },
      },
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0, TEXCOORD_0: 1, TEXCOORD_2: 2 },
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
        {
          name: 'variant uv2 texture',
          pbrMetallicRoughness: {
            baseColorFactor: [1, 1, 1, 1],
            baseColorTexture: { index: 0, texCoord: 2 },
          },
        },
      ],
      textures: [{ source: 0 }],
      images: [{ bufferView: 3, mimeType: 'image/png' }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
        { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2' },
      ],
      bufferViews: [
        ...packed.views,
        { buffer: 1, byteOffset: 0, byteLength: imageBytes.byteLength },
      ],
      buffers: [
        { byteLength: packed.buffer.byteLength },
        { byteLength: imageBytes.byteLength },
      ],
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

function transformPoint(m: Float32Array | undefined, x: number, y: number, z: number): [number, number, number] {
  if (!m) return [x, y, z];
  return [
    m[0]! * x + m[4]! * y + m[8]! * z + m[12]!,
    m[1]! * x + m[5]! * y + m[9]! * z + m[13]!,
    m[2]! * x + m[6]! * y + m[10]! * z + m[14]!,
  ];
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
  const meshTransform = identityMat4();
  meshTransform[12] = 5;
  const restTangents = new Float32Array([
    1, 0, 0, 1,
    1, 0, 0, 1,
    1, 0, 0, 1,
  ]);
  const skinIndices = new Uint32Array(12);
  const skinWeights = new Float32Array(12);
  for (let vertex = 0; vertex < 3; vertex += 1) skinWeights[vertex * 4] = 1;
  const primitive: SkinnedMeshPrimitive = {
    kind: 'skinned-mesh',
    id: 'skin-prim',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    tangents: restTangents,
    material: MATERIAL,
    transform: asMat4(meshTransform),
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
        { mesh: 0, skin: 0, children: [1], translation: [5, 0, 0] },
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
    const reset = vi.fn();
    const engine: GltfScenePatchTarget = { setScene: vi.fn(), updatePrimitive, reset };
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation(0, 0.5, { engine });

    expect(frame.usedSetScene).toBe(false);
    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(reset).toHaveBeenCalledTimes(1);
    const patch = frame.primitivePatches[0]!.patch as { transform: Float32Array };
    expect(patch.transform[12]).toBeCloseTo(1);
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(1);
  });

  it('updates camera metadata when an authored camera ancestor is animated', async () => {
    const { gltf, buffers } = animatedCameraGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });

    expect(result.scene.primitives).toHaveLength(0);
    expect(result.cameras).toHaveLength(1);
    expect(result.cameras[0]!.name).toBe('HeroCam');
    expect(result.cameras[0]!.nodeName).toBe('CameraNode');
    expect(result.cameras[0]!.worldMatrix[12]).toBeCloseTo(0);
    expect(result.cameras[0]!.worldMatrix[13]).toBeCloseTo(1);
    expect(result.cameras[0]!.worldMatrix[14]).toBeCloseTo(3);

    const frame = controller.applyAnimation('camera-parent-slide', 1);

    expect(frame.primitivePatches).toHaveLength(0);
    expect(frame.cameras).toHaveLength(1);
    expect(frame.cameras[0]!.worldMatrix[12]).toBeCloseTo(2);
    expect(frame.cameras[0]!.worldMatrix[13]).toBeCloseTo(1);
    expect(frame.cameras[0]!.worldMatrix[14]).toBeCloseTo(3);
    expect(frame.cameras[0]!.perspective?.yfov).toBeCloseTo(0.75);
    expect(frame.cameras[0]!.perspective?.aspectRatio).toBeCloseTo(1.5);
    expect(controller.cameras[0]!.worldMatrix[12]).toBeCloseTo(2);

    controller.resetPose();
    expect(controller.cameras[0]!.worldMatrix[12]).toBeCloseTo(0);
    expect(controller.cameras[0]!.worldMatrix[13]).toBeCloseTo(1);
  });

  it('returns blended camera metadata after blending authored camera animations', async () => {
    const { gltf, buffers } = animatedCameraGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.blend(['camera-parent-slide', 'camera-parent-lift'], [0.25, 0.75], 1);

    expect(frame.primitivePatches).toHaveLength(0);
    expect(frame.cameras).toHaveLength(1);
    expect(frame.cameras[0]!.worldMatrix[12]).toBeCloseTo(0.5);
    expect(frame.cameras[0]!.worldMatrix[13]).toBeCloseTo(4);
    expect(frame.cameras[0]!.worldMatrix[14]).toBeCloseTo(3);
    expect(controller.cameras[0]!.worldMatrix[12]).toBeCloseTo(0.5);
    expect(controller.cameras[0]!.worldMatrix[13]).toBeCloseTo(4);
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

  it('applies KHR_animation_pointer material-factor channels through primitive material patches', async () => {
    const { gltf, buffers } = materialPointerAnimationGltf();
    const result = await gltfToScene(gltf, { buffers });
    const updatePrimitive = vi.fn();
    const reset = vi.fn();
    const engine: GltfScenePatchTarget = { setScene: vi.fn(), updatePrimitive, reset };
    const controller = createGltfSceneController({ gltf, ...result });

    expect(result.animations[0]!.channels).toHaveLength(4);
    expect(result.animations[0]!.channels[0]!.target).toMatchObject({
      path: 'pointer',
      pointer: '/materials/0/pbrMetallicRoughness/baseColorFactor',
    });

    const frame = controller.applyAnimation('material-factors', 0.5, { engine });

    expect(frame.usedSetScene).toBe(false);
    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(reset).toHaveBeenCalledTimes(1);
    const patch = frame.primitivePatches[0]!.patch as { material: MaterialSpec };
    expect(patch.material.baseColor).toEqual([0.5, 0.5, 0]);
    expect(patch.material.opacity).toBeCloseTo(0.75);
    expect(patch.material.metallic).toBeCloseTo(0.5);
    expect(patch.material.roughness).toBeCloseTo(0.5);
    expect(patch.material.emissive).toEqual([0.5, 0.25, 0.125]);
    const primitive = controller.scene.primitives[0] as MeshPrimitive;
    expect(primitive.material.baseColor).toEqual([0.5, 0.5, 0]);
    expect(primitive.material.metallic).toBeCloseTo(0.5);
  });

  it('material pointer animation falls back to setScene when primitive patching is unavailable', async () => {
    const { gltf, buffers } = materialPointerAnimationGltf();
    const result = await gltfToScene(gltf, { buffers });
    const setScene = vi.fn();
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation('material-factors', 1, { engine: { setScene } });

    expect(frame.usedSetScene).toBe(true);
    expect(setScene).toHaveBeenCalledTimes(1);
    const scene = setScene.mock.calls[0]![0] as Scene;
    const primitive = scene.primitives[0] as MeshPrimitive;
    expect(primitive.material.baseColor).toEqual([0, 1, 0]);
    expect(primitive.material.opacity).toBeCloseTo(0.5);
    expect(primitive.material.metallic).toBeCloseTo(0.8);
    expect(primitive.material.roughness).toBeCloseTo(0.1);
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
    expect(frame.diagnostics).toEqual([
      expect.objectContaining({
        code: 'controller-update-primitive-failed',
        caller: 'applyAnimation',
        primitiveId: 'gltf-prim-0',
        path: 'scene.primitives["gltf-prim-0"]',
      }),
    ]);
    expect(controller.diagnostics).toEqual(frame.diagnostics);
  });

  it('reports when animated TRS channels override a matrix-imported node', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      matrix: [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        3, 0, 0, 1,
      ],
    };
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });

    const first = controller.applyAnimation('parent-slide', 0.5);
    const second = controller.applyAnimation('parent-slide', 1);

    expect(first.diagnostics).toEqual([
      expect.objectContaining({
        code: 'animation-matrix-overridden',
        caller: 'applyAnimation',
        nodeIndex: 0,
        path: 'nodes[0].matrix',
      }),
    ]);
    expect(second.diagnostics.some((diagnostic) => diagnostic.code === 'animation-matrix-overridden')).toBe(false);
    expect(controller.diagnostics.filter((diagnostic) => diagnostic.code === 'animation-matrix-overridden'))
      .toHaveLength(1);
  });

  it('patches EXT_mesh_gpu_instancing instances when the node animates', async () => {
    const { gltf, buffers } = animatedInstancedGltf();
    const result = await gltfToScene(gltf, { buffers });
    const updatePrimitive = vi.fn();
    const reset = vi.fn();
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation('instance-slide', 0.5, {
      engine: { setScene: vi.fn(), updatePrimitive, reset },
    });

    expect(frame.usedSetScene).toBe(false);
    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(reset).toHaveBeenCalledTimes(1);
    const patch = frame.primitivePatches[0]!.patch as {
      instances: ReadonlyArray<Float32Array>;
      transform?: Float32Array;
    };
    expect(patch.transform).toBeUndefined();
    expect(patch.instances).toHaveLength(2);
    expect(patch.instances[0]![12]).toBeCloseTo(14);
    expect(patch.instances[0]![13]).toBeCloseTo(0);
    expect(patch.instances[1]![12]).toBeCloseTo(12);
    expect(patch.instances[1]![13]).toBeCloseTo(3);
    const primitive = controller.scene.primitives[0] as InstancedMeshPrimitive;
    expect(primitive.kind).toBe('instanced-mesh');
    expect(primitive.instances[0]![12]).toBeCloseTo(14);
    expect(primitive.instances[1]![13]).toBeCloseTo(3);
  });

  it('switches KHR_materials_variants via primitive patches', async () => {
    const { gltf, buffers } = materialVariantGltf();
    const result = await gltfToScene(gltf, { buffers });
    const setScene = vi.fn();
    const updatePrimitive = vi.fn();
    const resetEngine = vi.fn();
    const controller = createGltfSceneController({ gltf, ...result });

    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([1, 0, 0]);

    const frame = controller.setVariant('blue', {
      engine: { setScene, updatePrimitive, reset: resetEngine },
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
    expect(resetEngine).toHaveBeenCalledTimes(1);
    expect(setScene).not.toHaveBeenCalled();
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([0, 0, 1]);

    controller.resetPose();
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([0, 0, 1]);

    const reset = controller.setVariant(undefined);
    expect((reset.primitivePatches[0]!.patch as { material: MaterialSpec }).material.baseColor)
      .toEqual([1, 0, 0]);
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([1, 0, 0]);
  });

  it('reports malformed variant mappings without breaking valid variant switches', async () => {
    const { gltf, buffers } = materialVariantGltf();
    gltf.meshes![0]!.primitives[0]!.extensions!.KHR_materials_variants!.mappings!.push(
      { material: 0 } as unknown as { material: number; variants: number[] },
    );
    const result = await gltfToScene(gltf, { buffers });
    expect(result.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'material-variant-mapping-malformed',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[1].variants',
      }),
    ]));

    const controller = createGltfSceneController({ gltf, ...result });
    const frame = controller.setVariant('blue', {
      engine: { setScene: vi.fn(), updatePrimitive: vi.fn(), reset: vi.fn() },
    });

    expect(frame.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'variant-mapping-malformed',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[1].variants',
      }),
    ]));
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([0, 0, 1]);
  });

  it('reports exact missing-material paths for variant mappings', async () => {
    const { gltf, buffers } = materialVariantGltf();
    gltf.meshes![0]!.primitives[0]!.extensions!.KHR_materials_variants!.mappings![0]!.material = 99;

    const result = await gltfToScene(gltf, { buffers, materialVariant: 'blue' });

    expect(result.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'material-variant-material-missing',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[0].material',
      }),
    ]));
    expect((result.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([1, 0, 0]);

    const controller = createGltfSceneController({ gltf, ...result });
    const frame = controller.setVariant('blue', {
      engine: { setScene: vi.fn(), updatePrimitive: vi.fn(), reset: vi.fn() },
    });

    expect(frame.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'variant-mapping-material-missing',
        path: 'meshes[0].primitives[0].extensions.KHR_materials_variants.mappings[0].material',
      }),
    ]));
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([1, 0, 0]);
  });

  it('reports malformed root material variant lists without throwing', async () => {
    const { gltf, buffers } = materialVariantGltf();
    (gltf.extensions as Record<string, unknown>).KHR_materials_variants = {
      variants: { name: 'blue' },
    };

    const result = await gltfToScene(gltf, { buffers, materialVariant: 'blue' });

    expect(result.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'material-variant-list-malformed',
        path: 'extensions.KHR_materials_variants.variants',
      }),
    ]));
    expect((result.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([1, 0, 0]);

    const controller = createGltfSceneController({ gltf, ...result });
    const frame = controller.setVariant('blue', {
      engine: { setScene: vi.fn(), updatePrimitive: vi.fn(), reset: vi.fn() },
    });

    expect(frame.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'variant-list-malformed',
        path: 'extensions.KHR_materials_variants.variants',
      }),
    ]));
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([1, 0, 0]);
  });

  it('switches high-UV variant materials with matching uv1 patches and clears them on reset', async () => {
    const { gltf, buffers } = materialVariantHighUvGltf();
    const result = await gltfToScene(gltf, { buffers });
    const updatePrimitive = vi.fn();
    const controller = createGltfSceneController({ gltf, ...result });
    const basePrimitive = controller.scene.primitives[0] as MeshPrimitive;

    expect(basePrimitive.uv1).toBeUndefined();
    expect(basePrimitive.material.baseColorMap).toBeUndefined();

    const frame = controller.setVariant('uv2-textured', {
      engine: { setScene: vi.fn(), updatePrimitive },
    });

    expect(frame.usedSetScene).toBe(false);
    const patch = frame.primitivePatches[0]!.patch as Partial<MeshPrimitive>;
    expect((patch.material as MaterialSpec).baseColorMap).toEqual(expect.objectContaining({ texCoord: 1 }));
    expect(patch.uv1).toBeInstanceOf(Float32Array);
    expect(Array.from(patch.uv1 as Float32Array)).toEqual([
      0.25, 0.25,
      0.5, 0.25,
      0.25, 0.5,
    ]);
    expect(updatePrimitive).toHaveBeenCalledWith(
      'gltf-prim-0',
      expect.objectContaining({
        uv1: expect.any(Float32Array),
        material: expect.objectContaining({
          baseColorMap: expect.objectContaining({ texCoord: 1 }),
        }),
      }),
    );
    expect((controller.scene.primitives[0] as MeshPrimitive).uv1).toBeInstanceOf(Float32Array);

    const reset = controller.setVariant(undefined);
    const resetPatch = reset.primitivePatches[0]!.patch as Partial<MeshPrimitive>;
    expect(Object.prototype.hasOwnProperty.call(resetPatch, 'uv1')).toBe(true);
    expect(resetPatch.uv1).toBeUndefined();
    expect((resetPatch.material as MaterialSpec).baseColorMap).toBeUndefined();
    expect((controller.scene.primitives[0] as MeshPrimitive).uv1).toBeUndefined();
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColorMap).toBeUndefined();
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
    expect(frame.diagnostics).toEqual([
      expect.objectContaining({
        code: 'controller-update-primitive-failed',
        caller: 'setVariant',
        primitiveId: 'gltf-prim-0',
        path: 'scene.primitives["gltf-prim-0"]',
      }),
    ]);
    expect(controller.diagnostics).toEqual(frame.diagnostics);
  });

  it('reports missing variant metadata as structured diagnostics', async () => {
    const { gltf, buffers } = materialVariantGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result, materialVariantBindings: [] });

    const frame = controller.setVariant('blue');

    expect(frame.primitivePatches).toHaveLength(0);
    expect(frame.diagnostics).toEqual([
      expect.objectContaining({
        code: 'variant-bindings-missing',
        caller: 'setVariant',
        path: 'materialVariantBindings',
      }),
    ]);
    expect(frame.warnings[0]).toContain('materialVariantBindings metadata');
    expect(controller.diagnostics).toEqual(frame.diagnostics);
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

  it('owns a play/pause/resume clock so hosts can tick animations predictably', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const result = await gltfToScene(gltf, { buffers });
    const updatePrimitive = vi.fn();
    const controller = createGltfSceneController({ gltf, ...result });
    const engine = { setScene: vi.fn(), updatePrimitive };

    const first = controller.play('parent-slide', { time: 0.25, engine });
    expect(first.localTime).toBeCloseTo(0.25);
    expect(controller.playing).toBe(true);
    expect(controller.activeClip?.name).toBe('parent-slide');
    expect(controller.currentTime).toBeCloseTo(0.25);
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(0.5);

    const ticked = controller.tick(0.25, { engine });
    expect(ticked?.localTime).toBeCloseTo(0.5);
    expect(controller.currentTime).toBeCloseTo(0.5);
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(1);
    const callsAfterTick = updatePrimitive.mock.calls.length;

    controller.pause();
    const paused = controller.tick(0.25, { engine });
    expect(paused).toBeUndefined();
    expect(controller.playing).toBe(false);
    expect(controller.currentTime).toBeCloseTo(0.5);
    expect(updatePrimitive).toHaveBeenCalledTimes(callsAfterTick);

    const resumed = controller.resume({ engine });
    expect(resumed.localTime).toBeCloseTo(0.5);
    expect(controller.playing).toBe(true);
    expect(controller.currentTime).toBeCloseTo(0.5);

    const next = controller.tick(0.25, { engine });
    expect(next?.localTime).toBeCloseTo(0.75);
    expect(controller.currentTime).toBeCloseTo(0.75);
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(1.5);
  });

  it('blends morph weights before skin solving', async () => {
    const { gltf, buffers } = morphGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.blend(['morph-on', 'morph-off'], [0.25, 0.75], 1);

    const patch = frame.primitivePatches[0]!.patch as {
      morphWeights: Float32Array;
      positions: Float32Array;
      tangents: Float32Array;
    };
    expect(patch.morphWeights[0]).toBeCloseTo(0.25);
    expect(Array.from(patch.positions)).toEqual([0.25, 0, 0, 1.25, 0, 0, 0.25, 1, 0]);
    expect(patch.tangents).toBeInstanceOf(Float32Array);
    expect(patch.tangents[1]).toBeGreaterThan(0);
  });

  it('samples morph-weight channels, solves the promoted skinned primitive, and patches deformed geometry', async () => {
    const { gltf, buffers } = morphGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation('morph-on', 1);

    const patch = frame.primitivePatches[0]!.patch as {
      morphWeights: Float32Array;
      positions: Float32Array;
      tangents: Float32Array;
    };
    expect(patch.morphWeights[0]).toBeCloseTo(1);
    expect(Array.from(patch.positions)).toEqual([1, 0, 0, 2, 0, 0, 1, 1, 0]);
    expect(patch.tangents).toBeInstanceOf(Float32Array);
    expect(patch.tangents[1]).toBeGreaterThan(0);
    expect((controller.scene.primitives[0] as SkinnedMeshPrimitive).tangents![1]).toBeGreaterThan(0);
    expect((controller.scene.primitives[0] as SkinnedMeshPrimitive).morphWeights![0]).toBeCloseTo(1);
  });

  it('rebuilds skinned bone matrices from animated joint nodes and patches solved vertices', () => {
    const input = manualSkinnedInput();
    const controller = createGltfSceneController(input);

    const frame = controller.applyAnimation('joint-slide', 1);

    const patch = frame.primitivePatches[0]!.patch as {
      bones: Float32Array;
      positions: Float32Array;
      tangents: Float32Array;
    };
    expect(patch.bones[12]).toBeCloseTo(1);
    expect(Array.from(patch.positions)).toEqual([1, 0, 0, 2, 0, 0, 1, 1, 0]);
    expect(Array.from(patch.tangents)).toEqual([
      1, 0, 0, 1,
      1, 0, 0, 1,
      1, 0, 0, 1,
    ]);
    const world = transformPoint(
      (controller.scene.primitives[0] as SkinnedMeshPrimitive).transform,
      patch.positions[0]!,
      patch.positions[1]!,
      patch.positions[2]!,
    );
    expect(world[0]).toBeCloseTo(6);
    expect(world[1]).toBeCloseTo(0);
    expect(world[2]).toBeCloseTo(0);
  });
});
