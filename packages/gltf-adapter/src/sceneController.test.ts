import { describe, expect, it, vi } from 'vitest';
import {
  createGltfSceneController,
  gltfToScene,
  type GltfJson,
  type GltfScenePatchTarget,
} from './index.js';
import {
  asMat4,
  solveSkin,
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

function animatedPunctualLightGltf(): {
  gltf: GltfJson;
  buffers: Map<number, ArrayBuffer>;
} {
  const packed = packF32([
    [0, 1],
    [0, 0, 0, 2, 3, 4],
  ]);
  return {
    buffers: new Map([[0, packed.buffer]]),
    gltf: {
      asset: { version: '2.0' },
      extensionsUsed: ['KHR_lights_punctual'],
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [
        { name: 'LightRig', children: [1] },
        {
          name: 'Lamp',
          translation: [1, 0, 0],
          extensions: { KHR_lights_punctual: { light: 0 } },
        },
      ],
      extensions: {
        KHR_lights_punctual: {
          lights: [{ type: 'point', color: [1, 0.5, 0.25], intensity: 4 }],
        },
      },
      accessors: [
        { bufferView: 0, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 1, componentType: 5126, count: 2, type: 'VEC3' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
      animations: [{
        name: 'move-light-rig',
        samplers: [{ input: 0, output: 1, interpolation: 'LINEAR' }],
        channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
      }],
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

function addMorphTargetToFirstPrimitive(
  fixture: { gltf: GltfJson; buffers: Map<number, ArrayBuffer> },
): void {
  const morph = new Float32Array([
    0, 0, 0.1,
    0, 0, 0.1,
    0, 0, 0.1,
  ]);
  const bufferIndex = fixture.gltf.buffers?.length ?? 0;
  const bufferViewIndex = fixture.gltf.bufferViews?.length ?? 0;
  const accessorIndex = fixture.gltf.accessors?.length ?? 0;
  fixture.gltf.buffers = [
    ...(fixture.gltf.buffers ?? []),
    { byteLength: morph.byteLength },
  ];
  fixture.gltf.bufferViews = [
    ...(fixture.gltf.bufferViews ?? []),
    { buffer: bufferIndex, byteOffset: 0, byteLength: morph.byteLength },
  ];
  fixture.gltf.accessors = [
    ...(fixture.gltf.accessors ?? []),
    { bufferView: bufferViewIndex, componentType: 5126, count: 3, type: 'VEC3' },
  ];
  fixture.gltf.meshes![0]!.primitives[0] = {
    ...fixture.gltf.meshes![0]!.primitives[0]!,
    targets: [{ POSITION: accessorIndex }],
  };
  fixture.gltf.meshes![0]!.weights = [0.5];
  fixture.buffers.set(bufferIndex, morph.buffer.slice(morph.byteOffset, morph.byteOffset + morph.byteLength));
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

function materialVariantPointerAnimationGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const packed = packF32([
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1],
    [
      0, 0, 1, 1,
      0, 1, 0, 1,
    ],
  ]);
  return {
    buffers: new Map([[0, packed.buffer]]),
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      extensionsUsed: ['KHR_materials_variants', 'KHR_animation_pointer'],
      extensionsRequired: ['KHR_materials_variants', 'KHR_animation_pointer'],
      extensions: {
        KHR_materials_variants: {
          variants: [{ name: 'animated' }],
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
        { bufferView: 1, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 2, componentType: 5126, count: 2, type: 'VEC4' },
      ],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
      animations: [{
        name: 'variant-material-factor',
        samplers: [{ input: 1, output: 2, interpolation: 'LINEAR' }],
        channels: [{
          sampler: 0,
          target: {
            path: 'pointer',
            extensions: {
              KHR_animation_pointer: {
                pointer: '/materials/1/pbrMetallicRoughness/baseColorFactor',
              },
            },
          },
        }],
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

function manualInstancedSkinnedInput() {
  const input = manualSkinnedInput();
  const deformationSource = input.scene.primitives[0] as SkinnedMeshPrimitive;
  const secondInstance = identityMat4();
  secondInstance[12] = 10;
  const instances = [
    asMat4(identityMat4()),
    asMat4(secondInstance),
  ];
  const primitive: InstancedMeshPrimitive = {
    kind: 'instanced-mesh',
    id: deformationSource.id,
    positions: deformationSource.positions,
    normals: deformationSource.normals,
    ...(deformationSource.tangents != null
      ? { tangents: deformationSource.tangents }
      : {}),
    instances,
    material: deformationSource.material,
  };
  return {
    ...input,
    scene: {
      ...input.scene,
      primitives: [primitive],
    },
    instancingBindings: [{
      primitiveId: String(primitive.id),
      nodeIndex: 0,
      localInstanceTransforms: instances,
      deformationSource,
    }],
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

  it('resetPose can restore base pose through updatePrimitive and reset playback state', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const result = await gltfToScene(gltf, { buffers });
    const setScene = vi.fn();
    const updatePrimitive = vi.fn();
    const reset = vi.fn();
    const engine: GltfScenePatchTarget = { setScene, updatePrimitive, reset };
    const controller = createGltfSceneController({ gltf, ...result });

    controller.play('parent-slide', { time: 0.5, engine });
    expect(controller.playing).toBe(true);
    expect(controller.currentTime).toBeCloseTo(0.5);
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(1);
    setScene.mockClear();
    updatePrimitive.mockClear();
    reset.mockClear();

    controller.resetPose({ engine, resetPlayback: true });

    expect(setScene).not.toHaveBeenCalled();
    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(updatePrimitive).toHaveBeenCalledWith(
      'gltf-prim-0',
      expect.objectContaining({ transform: expect.any(Float32Array) }),
    );
    const patch = updatePrimitive.mock.calls[0]![1] as { transform: Float32Array };
    expect(patch.transform[12]).toBeCloseTo(0);
    expect(reset).toHaveBeenCalledTimes(1);
    expect(controller.playing).toBe(false);
    expect(controller.currentTime).toBe(0);
    expect(controller.activeClip).toBeUndefined();
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(0);
  });

  it('resetPose uses setScene when forceSetScene is requested', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const result = await gltfToScene(gltf, { buffers });
    const setScene = vi.fn();
    const updatePrimitive = vi.fn();
    const reset = vi.fn();
    const engine: GltfScenePatchTarget = { setScene, updatePrimitive, reset };
    const controller = createGltfSceneController({ gltf, ...result });

    controller.play('parent-slide', { time: 0.5, engine });
    setScene.mockClear();
    updatePrimitive.mockClear();
    reset.mockClear();

    controller.resetPose({ engine, forceSetScene: true, resetPlayback: true });

    expect(updatePrimitive).not.toHaveBeenCalled();
    expect(reset).not.toHaveBeenCalled();
    expect(setScene).toHaveBeenCalledTimes(1);
    const scene = setScene.mock.calls[0]![0] as Scene;
    expect((scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(0);
    expect(controller.playing).toBe(false);
    expect(controller.currentTime).toBe(0);
    expect(controller.activeClip).toBeUndefined();
    expect((controller.scene.primitives[0] as { transform: Float32Array }).transform[12]).toBeCloseTo(0);
  });


  it('resetPose emits diagnostics when updatePrimitive fallback is required', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const result = await gltfToScene(gltf, { buffers });
    const setScene = vi.fn();
    const updatePrimitive = vi.fn(() => {
      throw new Error('reset patch rejected');
    });
    const reset = vi.fn();
    const engine: GltfScenePatchTarget = { setScene, updatePrimitive, reset };
    const controller = createGltfSceneController({ gltf, ...result });

    controller.play('parent-slide', { time: 0.5, engine });
    setScene.mockClear();
    updatePrimitive.mockClear();
    reset.mockClear();

    controller.resetPose({ engine });

    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(reset).not.toHaveBeenCalled();
    expect(setScene).toHaveBeenCalledTimes(1);
    expect(controller.warnings.some((w) => w.includes('reset patch rejected'))).toBe(true);
    expect(controller.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'controller-update-primitive-failed',
        caller: 'resetPose',
        primitiveId: 'gltf-prim-0',
        path: 'scene.primitives["gltf-prim-0"]',
      }),
    ]));
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

  it('animates punctual-light nodes through incremental emitter patches', async () => {
    const { gltf, buffers } = animatedPunctualLightGltf();
    const result = await gltfToScene(gltf, { buffers });
    const updateEmitter = vi.fn();
    const reset = vi.fn();
    const setScene = vi.fn();
    const engine: GltfScenePatchTarget = { setScene, updateEmitter, reset };
    const controller = createGltfSceneController({ gltf, ...result });

    expect(result.punctualEmitterBindings).toEqual([{
      emitterId: 'gltf-light-0',
      nodeIndex: 1,
      lightIndex: 0,
    }]);
    expect(result.scene.emitters[0]).toMatchObject({
      kind: 'point',
      position: [1, 0, 0],
    });

    const frame = controller.applyAnimation('move-light-rig', 1, { engine });

    expect(frame.primitivePatches).toEqual([]);
    expect(frame.emitterPatches).toEqual([{
      id: 'gltf-light-0',
      patch: { position: [3, 3, 4] },
    }]);
    expect(updateEmitter).toHaveBeenCalledWith('gltf-light-0', {
      position: [3, 3, 4],
    });
    expect(reset).toHaveBeenCalledOnce();
    expect(setScene).not.toHaveBeenCalled();
    expect(frame.scene.emitters[0]).toMatchObject({ position: [3, 3, 4] });

    updateEmitter.mockClear();
    reset.mockClear();
    controller.resetPose({ engine });
    expect(updateEmitter).toHaveBeenCalledWith('gltf-light-0', expect.objectContaining({
      position: [1, 0, 0],
    }));
    expect(reset).toHaveBeenCalledOnce();
    expect(controller.scene.emitters[0]).toMatchObject({ position: [1, 0, 0] });
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

  it('routes material-pointer animation to the currently selected material variant', async () => {
    const { gltf, buffers } = materialVariantPointerAnimationGltf();
    const result = await gltfToScene(gltf, { buffers });
    const updatePrimitive = vi.fn();
    const reset = vi.fn();
    const engine: GltfScenePatchTarget = { setScene: vi.fn(), updatePrimitive, reset };
    const controller = createGltfSceneController({ gltf, ...result });

    controller.setVariant('animated', { engine });
    updatePrimitive.mockClear();
    reset.mockClear();

    const frame = controller.applyAnimation('variant-material-factor', 1, { engine });

    expect(frame.usedSetScene).toBe(false);
    expect(frame.diagnostics.some((diagnostic) => diagnostic.code === 'animation-pointer-material-unmapped'))
      .toBe(false);
    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(updatePrimitive).toHaveBeenCalledWith(
      'gltf-prim-0',
      expect.objectContaining({
        material: expect.objectContaining({ baseColor: [0, 1, 0] }),
      }),
    );
    expect(reset).toHaveBeenCalledTimes(1);
    const patch = frame.primitivePatches[0]!.patch as { material: MaterialSpec };
    expect(patch.material.baseColor).toEqual([0, 1, 0]);
    expect((controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([0, 1, 0]);
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
        0.25, 2, 0, 0,
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
    const transformed = (first.scene.primitives[0] as MeshPrimitive).transform!;
    expect(transformed[4]).toBeCloseTo(0.25);
    expect(transformed[5]).toBeCloseTo(2);
  });

  it('animates an invertible matrix-authored node at tiny uniform scale', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const tinyScale = 1e-13;
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      matrix: [
        tinyScale, 0, 0, 0,
        0, tinyScale, 0, 0,
        0, 0, tinyScale, 0,
        3, 0, 0, 1,
      ],
    };
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation('parent-slide', 0.5);

    expect(frame.diagnostics).toEqual([
      expect.objectContaining({
        code: 'animation-matrix-overridden',
        nodeIndex: 0,
      }),
    ]);
    expect(frame.diagnostics.some(
      (diagnostic) => diagnostic.code === 'animation-matrix-trs-unavailable',
    )).toBe(false);
    const transformed = (frame.scene.primitives[0] as MeshPrimitive).transform!;
    expect(transformed[0]).toBeCloseTo(tinyScale, 18);
    expect(transformed[5]).toBeCloseTo(tinyScale, 18);
    expect(transformed[10]).toBeCloseTo(tinyScale, 18);
    expect(transformed[12]).toBeCloseTo(1);
  });

  it('preserves an unrelated singular matrix while other controller animation remains functional', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    gltf.nodes!.push({
      name: 'ZeroScaleDecoration',
      matrix: [
        0, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        4, 0, 0, 1,
      ],
    });
    const result = await gltfToScene(gltf, { buffers });

    expect(() => createGltfSceneController({ gltf, ...result })).not.toThrow();
    const controller = createGltfSceneController({ gltf, ...result });
    const frame = controller.applyAnimation('parent-slide', 0.5);

    expect(frame.diagnostics).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: 'animation-matrix-trs-unavailable' }),
      ]),
    );
    expect((frame.scene.primitives[0] as MeshPrimitive).transform![12]).toBeCloseTo(1);
  });

  it('diagnoses and skips a TRS channel that targets a singular matrix node', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const result = await gltfToScene(gltf, { buffers });
    gltf.nodes!.push({
      name: 'ZeroScaleAnimatedNode',
      matrix: [
        1e-13, 0, 0, 0,
        0, 1e-13, 0, 0,
        0, 0, 0, 0,
        4, 0, 0, 1,
      ],
    });
    const singularTargetClip: AnimationClip = {
      name: 'singular-target',
      duration: 1,
      channels: [{
        target: { node: 'gltf-node-2', path: 'translation' },
        sampler: {
          times: new Float32Array([0, 1]),
          values: new Float32Array([0, 0, 0, 2, 0, 0]),
          interpolation: 'LINEAR',
        },
      }],
    };
    const controller = createGltfSceneController({
      gltf,
      ...result,
      animations: [singularTargetClip],
    });

    const first = controller.applyAnimation('singular-target', 0.5);
    const second = controller.applyAnimation('singular-target', 1);

    expect(first.diagnostics).toEqual([
      expect.objectContaining({
        code: 'animation-matrix-trs-unavailable',
        caller: 'applyAnimation',
        nodeIndex: 2,
        path: 'nodes[2].matrix',
      }),
    ]);
    expect(first.diagnostics.some((diagnostic) => diagnostic.code === 'animation-matrix-overridden'))
      .toBe(false);
    expect(second.diagnostics).toEqual([]);
    expect(controller.diagnostics).toHaveLength(1);
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

  it('patches one shared morphed stream and all native instances atomically when the node animates', async () => {
    const fixture = animatedInstancedGltf();
    addMorphTargetToFirstPrimitive(fixture);
    const result = await gltfToScene(fixture.gltf, { buffers: fixture.buffers });
    const updatePrimitive = vi.fn();
    const reset = vi.fn();
    const controller = createGltfSceneController({ gltf: fixture.gltf, ...result });

    expect(controller.scene.primitives).toHaveLength(1);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({ code: 'generated-flat-normals' }),
    ]);

    const frame = controller.applyAnimation('instance-slide', 0.5, {
      engine: { setScene: vi.fn(), updatePrimitive, reset },
    });

    expect(frame.usedSetScene).toBe(false);
    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(reset).toHaveBeenCalledTimes(1);
    expect(frame.primitivePatches).toHaveLength(1);
    const patch = frame.primitivePatches[0]!.patch as {
      instances: ReadonlyArray<Float32Array>;
      positions: Float32Array;
      normals: Float32Array;
      transform?: Float32Array;
    };
    expect(frame.primitivePatches[0]!.id).toBe('gltf-prim-0');
    expect(patch.transform).toBeUndefined();
    expect(patch.instances).toHaveLength(2);
    expect(patch.instances[0]![12]).toBeCloseTo(14);
    expect(patch.instances[0]![13]).toBeCloseTo(0);
    expect(patch.instances[1]![12]).toBeCloseTo(12);
    expect(patch.instances[1]![13]).toBeCloseTo(3);
    // Unlike a true skinned primitive, an instanced target has nowhere to
    // retain pose state. Its private deformation source must therefore stay
    // resolved into the shared geometry patch.
    expect(patch.positions).toBeInstanceOf(Float32Array);
    expect(patch.normals).toBeInstanceOf(Float32Array);
    expect(patch).not.toHaveProperty('morphWeights');
    expect(Array.from(patch.positions.slice(0, 2))).toEqual([0, 0]);
    expect(Array.from(patch.positions.slice(3, 5))).toEqual([1, 0]);
    expect(Array.from(patch.positions.slice(6, 8))).toEqual([0, 1]);
    expect(patch.positions[2]).toBeCloseTo(0.05);
    expect(patch.positions[5]).toBeCloseTo(0.05);
    expect(patch.positions[8]).toBeCloseTo(0.05);
    expect(updatePrimitive).toHaveBeenCalledWith('gltf-prim-0', patch);
    const primitive = controller.scene.primitives[0] as InstancedMeshPrimitive;
    expect(primitive.kind).toBe('instanced-mesh');
    expect(primitive.instances[0]![12]).toBeCloseTo(14);
    expect(primitive.instances[1]![12]).toBeCloseTo(12);
    expect(Array.from(primitive.positions)).toEqual(Array.from(patch.positions));
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

  it('switches high-UV variant materials without remapping their indexed UV lane', async () => {
    const { gltf, buffers } = materialVariantHighUvGltf();
    const result = await gltfToScene(gltf, { buffers });
    const updatePrimitive = vi.fn();
    const controller = createGltfSceneController({ gltf, ...result });
    const basePrimitive = controller.scene.primitives[0] as MeshPrimitive;

    expect(basePrimitive.uv1).toBeUndefined();
    expect(basePrimitive.uvSets?.[2]).toBeInstanceOf(Float32Array);
    expect(basePrimitive.material.baseColorMap).toBeUndefined();

    const frame = controller.setVariant('uv2-textured', {
      engine: { setScene: vi.fn(), updatePrimitive },
    });

    expect(frame.usedSetScene).toBe(false);
    const patch = frame.primitivePatches[0]!.patch as Partial<MeshPrimitive>;
    expect((patch.material as MaterialSpec).baseColorMap).toEqual(expect.objectContaining({ texCoord: 2 }));
    expect(patch.uv1).toBeUndefined();
    expect(patch.uvSets?.[2]).toBeInstanceOf(Float32Array);
    expect(Array.from(patch.uvSets?.[2] as Float32Array)).toEqual([
      0.25, 0.25,
      0.5, 0.25,
      0.25, 0.5,
    ]);
    expect(updatePrimitive).toHaveBeenCalledWith(
      'gltf-prim-0',
      expect.objectContaining({
        uvSets: expect.any(Array),
        material: expect.objectContaining({
          baseColorMap: expect.objectContaining({ texCoord: 2 }),
        }),
      }),
    );
    expect((controller.scene.primitives[0] as MeshPrimitive).uvSets?.[2]).toBeInstanceOf(Float32Array);

    const reset = controller.setVariant(undefined);
    const resetPatch = reset.primitivePatches[0]!.patch as Partial<MeshPrimitive>;
    expect(resetPatch.uv1).toBeUndefined();
    expect(resetPatch.uvSets?.[2]).toBeInstanceOf(Float32Array);
    expect((resetPatch.material as MaterialSpec).baseColorMap).toBeUndefined();
    expect((controller.scene.primitives[0] as MeshPrimitive).uv1).toBeUndefined();
    expect((controller.scene.primitives[0] as MeshPrimitive).uvSets?.[2]).toBeInstanceOf(Float32Array);
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

  it('hemisphere-corrects and normalizes node-rotation pointer channels while blending', async () => {
    const { gltf, buffers } = animatedHierarchyGltf();
    const result = await gltfToScene(gltf, { buffers });
    const halfSqrt = Math.SQRT1_2;
    const pointer = '/nodes/0/rotation';
    const pointerClip = (
      name: string,
      quaternion: readonly [number, number, number, number],
    ): AnimationClip => ({
      name,
      duration: 1,
      channels: [{
        target: { node: `gltf-pointer:${pointer}`, path: 'pointer', pointer },
        sampler: {
          times: new Float32Array([0, 1]),
          values: new Float32Array([...quaternion, ...quaternion]),
          interpolation: 'LINEAR',
        },
      }],
    });
    const controller = createGltfSceneController({
      gltf,
      ...result,
      animations: [
        pointerClip('positive-hemisphere', [0, halfSqrt, 0, halfSqrt]),
        pointerClip('negative-hemisphere', [0, -halfSqrt, 0, -halfSqrt]),
      ],
    });

    const frame = controller.blend(
      ['positive-hemisphere', 'negative-hemisphere'],
      [0.5, 0.5],
      1,
    );

    const transform = (frame.primitivePatches[0]!.patch as { transform: Float32Array }).transform;
    expect(Array.from(transform).every(Number.isFinite)).toBe(true);
    expect(transform[0]).toBeCloseTo(0, 5);
    expect(Math.abs(transform[2]!)).toBeCloseTo(1, 5);
    expect(Math.abs(transform[8]!)).toBeCloseTo(1, 5);
    expect(transform[10]).toBeCloseTo(0, 5);
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

  it('publishes blended morph state without replacing skinned rest streams', async () => {
    const { gltf, buffers } = morphGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });
    const rest = result.scene.primitives[0] as SkinnedMeshPrimitive;

    const frame = controller.blend(['morph-on', 'morph-off'], [0.25, 0.75], 1);

    const patch = frame.primitivePatches[0]!.patch as {
      morphWeights: Float32Array;
    };
    expect(patch.morphWeights[0]).toBeCloseTo(0.25);
    for (const field of ['positions', 'normals', 'tangents', 'uvs', 'uv1', 'uvSets']) {
      expect(patch).not.toHaveProperty(field);
    }
    const solved = solveSkin({ ...rest, ...patch });
    expect(Array.from(solved.positions)).toEqual([0.25, 0, 0, 1.25, 0, 0, 0.25, 1, 0]);
    expect(solved.tangents).toBeInstanceOf(Float32Array);
    expect(solved.tangents![1]).toBeGreaterThan(0);
    expect((controller.scene.primitives[0] as SkinnedMeshPrimitive).positions)
      .toBe(rest.positions);
  });

  it('samples morph-weight channels while retaining the promoted primitive rest geometry', async () => {
    const { gltf, buffers } = morphGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });
    const rest = result.scene.primitives[0] as SkinnedMeshPrimitive;

    const frame = controller.applyAnimation('morph-on', 1);

    const patch = frame.primitivePatches[0]!.patch as {
      morphWeights: Float32Array;
    };
    expect(patch.morphWeights[0]).toBeCloseTo(1);
    expect(patch).not.toHaveProperty('positions');
    expect(patch).not.toHaveProperty('normals');
    expect(patch).not.toHaveProperty('tangents');
    const solved = solveSkin({ ...rest, ...patch });
    expect(Array.from(solved.positions)).toEqual([1, 0, 0, 2, 0, 0, 1, 1, 0]);
    expect(solved.tangents).toBeInstanceOf(Float32Array);
    expect(solved.tangents![1]).toBeGreaterThan(0);
    expect((controller.scene.primitives[0] as SkinnedMeshPrimitive).tangents)
      .toBe(rest.tangents);
    expect((controller.scene.primitives[0] as SkinnedMeshPrimitive).morphWeights![0]).toBeCloseTo(1);
  });

  it('skips a morphless sibling while continuing weight animation on morph-capable primitives', async () => {
    const { gltf, buffers } = morphGltf();
    const source = gltf.meshes![0]!.primitives[0]!;
    gltf.meshes![0]!.primitives.push({
      attributes: { ...source.attributes },
      ...(source.material !== undefined ? { material: source.material } : {}),
    });
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation('morph-on', 1);

    expect(frame.primitivePatches).toContainEqual(expect.objectContaining({
      id: 'gltf-prim-0',
      patch: expect.objectContaining({
        morphWeights: expect.any(Float32Array),
      }),
    }));
    expect(frame.diagnostics).toContainEqual(expect.objectContaining({
      code: 'animation-morph-target-missing',
      primitiveId: 'gltf-prim-1',
      path: 'scene.primitives["gltf-prim-1"].morphTargets',
    }));
  });

  it('diagnoses and truncates a runtime morph-weight stride mismatch', async () => {
    const { gltf, buffers } = morphGltf();
    const result = await gltfToScene(gltf, { buffers });
    const sourceClip = result.animations.find((clip) => clip.name === 'morph-on')!;
    const mismatchedClip: AnimationClip = {
      ...sourceClip,
      channels: sourceClip.channels.map((channel) => ({
        ...channel,
        sampler: {
          ...channel.sampler,
          // Two weights per keyframe target a primitive with one morph target.
          values: new Float32Array([0, 0, 1, 0.5]),
        },
      })),
    };
    const controller = createGltfSceneController({
      gltf,
      ...result,
      animations: [mismatchedClip],
    });

    const frame = controller.applyAnimation('morph-on', 1);
    const patch = frame.primitivePatches[0]!.patch as {
      morphWeights: Float32Array;
    };

    expect(frame.diagnostics).toEqual([
      expect.objectContaining({
        severity: 'warning',
        caller: 'applyAnimation',
        code: 'morph-weight-count-mismatch',
        path: 'scene.primitives["gltf-prim-0"].morphWeights',
        primitiveId: 'gltf-prim-0',
      }),
    ]);
    expect(Array.from(patch.morphWeights)).toEqual([1]);
    expect(patch).not.toHaveProperty('positions');
    const solved = solveSkin({
      ...(result.scene.primitives[0] as SkinnedMeshPrimitive),
      ...patch,
    });
    expect(Array.from(solved.positions)).toEqual([1, 0, 0, 2, 0, 0, 1, 1, 0]);
  });

  it('diagnoses and zero-fills a short runtime morph-weight stride', async () => {
    const { gltf, buffers } = morphGltf();
    const primitive = gltf.meshes![0]!.primitives[0]!;
    primitive.targets = [primitive.targets![0]!, { ...primitive.targets![0]! }];
    gltf.meshes![0]!.weights = [0, 0];
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController({ gltf, ...result });

    const frame = controller.applyAnimation('morph-on', 1);
    const patch = frame.primitivePatches[0]!.patch as {
      morphWeights: Float32Array;
    };

    expect(frame.diagnostics).toEqual([
      expect.objectContaining({
        severity: 'warning',
        code: 'morph-weight-count-mismatch',
        primitiveId: 'gltf-prim-0',
      }),
    ]);
    expect(Array.from(patch.morphWeights)).toEqual([1, 0]);
    expect(patch).not.toHaveProperty('positions');
    const solved = solveSkin({
      ...(result.scene.primitives[0] as SkinnedMeshPrimitive),
      ...patch,
    });
    expect(Array.from(solved.positions)).toEqual([1, 0, 0, 2, 0, 0, 1, 1, 0]);
  });

  it('publishes joint pose state and leaves backend skin solving single-pass', () => {
    const input = manualSkinnedInput();
    const controller = createGltfSceneController(input);
    const rest = input.scene.primitives[0] as SkinnedMeshPrimitive;

    const frame = controller.applyAnimation('joint-slide', 1);

    const patch = frame.primitivePatches[0]!.patch as {
      bones: Float32Array;
    };
    expect(patch.bones[12]).toBeCloseTo(1);
    for (const field of ['positions', 'normals', 'tangents', 'uvs', 'uv1', 'uvSets']) {
      expect(patch).not.toHaveProperty(field);
    }
    const solved = solveSkin({ ...rest, ...patch });
    expect(Array.from(solved.positions)).toEqual([1, 0, 0, 2, 0, 0, 1, 1, 0]);
    expect(Array.from(solved.tangents!)).toEqual([
      1, 0, 0, 1,
      1, 0, 0, 1,
      1, 0, 0, 1,
    ]);
    expect((controller.scene.primitives[0] as SkinnedMeshPrimitive).positions)
      .toBe(rest.positions);
    const world = transformPoint(
      (controller.scene.primitives[0] as SkinnedMeshPrimitive).transform,
      solved.positions[0]!,
      solved.positions[1]!,
      solved.positions[2]!,
    );
    expect(world[0]).toBeCloseTo(6);
    expect(world[1]).toBeCloseTo(0);
    expect(world[2]).toBeCloseTo(0);
  });

  it('publishes a return-to-bind joint pose instead of skipping against base bones', () => {
    const input = manualSkinnedInput();
    const controller = createGltfSceneController(input);

    const forward = controller.applyAnimation('joint-slide', 1);
    expect(
      (forward.primitivePatches[0]!.patch as { bones: Float32Array }).bones[12],
    ).toBeCloseTo(1);

    const returned = controller.applyAnimation('joint-slide', 0);
    expect(returned.primitivePatches).toHaveLength(1);
    const patch = returned.primitivePatches[0]!.patch as { bones: Float32Array };
    expect(patch.bones[12]).toBeCloseTo(0);
    expect(patch).not.toHaveProperty('positions');
    expect(
      (controller.scene.primitives[0] as SkinnedMeshPrimitive).bones[12],
    ).toBeCloseTo(0);
  });

  it('CPU-solves instanced joint deformation and publishes the return-to-bind geometry', () => {
    const input = manualInstancedSkinnedInput();
    const controller = createGltfSceneController(input);

    const forward = controller.applyAnimation('joint-slide', 1);
    const posedPatch = forward.primitivePatches[0]!.patch as {
      positions: Float32Array;
      normals: Float32Array;
    };
    expect(posedPatch.positions[0]).toBeCloseTo(1);
    expect(posedPatch.normals).toBeInstanceOf(Float32Array);
    expect(posedPatch).not.toHaveProperty('bones');

    const returned = controller.applyAnimation('joint-slide', 0);
    expect(returned.primitivePatches).toHaveLength(1);
    const restPatch = returned.primitivePatches[0]!.patch as {
      positions: Float32Array;
    };
    expect([...restPatch.positions]).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(restPatch).not.toHaveProperty('bones');
    expect(
      (controller.scene.primitives[0] as InstancedMeshPrimitive).positions,
    ).toEqual(restPatch.positions);
  });

  it('publishes combined native joint+morph pose state atomically without geometry', () => {
    const input = manualSkinnedInput();
    const source = input.scene.primitives[0] as SkinnedMeshPrimitive;
    const morphedSource: SkinnedMeshPrimitive = {
      ...source,
      morphTargets: [new Float32Array([
        1, 0, 0,
        1, 0, 0,
        1, 0, 0,
      ])],
      morphWeights: new Float32Array([0]),
    };
    const jointClip = input.animations[0]!;
    const controller = createGltfSceneController({
      ...input,
      gltf: {
        ...input.gltf,
        meshes: [{
          weights: [0],
          primitives: [{
            ...input.gltf.meshes![0]!.primitives[0]!,
            targets: [{ POSITION: 0 }],
          }],
        }],
      },
      scene: {
        ...input.scene,
        primitives: [morphedSource],
      },
      animations: [{
        ...jointClip,
        name: 'joint-and-morph',
        channels: [
          ...jointClip.channels,
          {
            target: { node: 'gltf-node-0', path: 'weights' as const },
            sampler: {
              times: new Float32Array([0, 1]),
              values: new Float32Array([0, 1]),
            },
          },
        ],
      }],
    });

    const frame = controller.applyAnimation('joint-and-morph', 1);
    const patch = frame.primitivePatches[0]!.patch as {
      bones: Float32Array;
      morphWeights: Float32Array;
    };
    expect(Object.keys(patch).sort()).toEqual(['bones', 'morphWeights']);
    expect(patch.bones[12]).toBeCloseTo(1);
    expect([...patch.morphWeights]).toEqual([1]);
    expect([
      ...solveSkin({ ...morphedSource, ...patch }).positions,
    ]).toEqual([2, 0, 0, 3, 0, 0, 2, 1, 0]);
  });

  it('bounds retained diagnostic history without truncating per-operation results', async () => {
    const { gltf, buffers } = materialVariantGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController(
      { gltf, ...result, materialVariantBindings: [] },
      { diagnosticHistoryLimit: 3 },
    );

    let lastFrame = controller.setVariant('blue');
    for (let i = 1; i < 100; i += 1) {
      lastFrame = controller.setVariant('blue');
    }

    expect(lastFrame.diagnostics).toHaveLength(1);
    expect(lastFrame.warnings).toHaveLength(1);
    expect(controller.diagnostics).toHaveLength(3);
    expect(controller.warnings).toHaveLength(3);

    controller.clearDiagnosticHistory();
    expect(controller.diagnostics).toHaveLength(0);
    expect(controller.warnings).toHaveLength(0);
  });

  it('can disable retained history while preserving call-local diagnostics', async () => {
    const { gltf, buffers } = materialVariantGltf();
    const result = await gltfToScene(gltf, { buffers });
    const controller = createGltfSceneController(
      { gltf, ...result, materialVariantBindings: [] },
      { diagnosticHistoryLimit: 0 },
    );

    const frame = controller.setVariant('blue');

    expect(frame.diagnostics).toHaveLength(1);
    expect(frame.warnings).toHaveLength(1);
    expect(controller.diagnostics).toHaveLength(0);
    expect(controller.warnings).toHaveLength(0);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, -1, 1.5])(
    'rejects invalid diagnostic history capacity %s',
    (diagnosticHistoryLimit) => {
      expect(() => createGltfSceneController(
        manualSkinnedInput(),
        { diagnosticHistoryLimit },
      )).toThrow(/diagnosticHistoryLimit.*non-negative safe integer/);
    },
  );

  it('detachEngine releases the retained engine target', () => {
    const input = manualSkinnedInput();
    const updatePrimitive = vi.fn();
    const engine: GltfScenePatchTarget = {
      setScene: vi.fn(),
      updatePrimitive,
    };
    const controller = createGltfSceneController(input, {
      engine,
      setSceneOnAttach: false,
    });
    controller.detachEngine();

    controller.applyAnimation('joint-slide', 1);

    expect(updatePrimitive).not.toHaveBeenCalled();
  });

  it('applies camera/light pointers and rejects invalid sampled cross-property states without clamping', async () => {
    const packed = packF32([[0, 0, 0, 1, 0, 0, 0, 1, 0]]);
    const gltf: GltfJson = {
      asset: { version: '2.0' },
      extensionsUsed: ['KHR_lights_punctual'],
      scene: 0,
      scenes: [{ nodes: [0, 1, 2] }],
      nodes: [
        { mesh: 0 },
        { camera: 0 },
        { extensions: { KHR_lights_punctual: { light: 0 } } },
      ],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{
        pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1] },
        extensions: {
          KHR_materials_iridescence: {
            iridescenceFactor: 1,
            iridescenceThicknessMinimum: 100,
            iridescenceThicknessMaximum: 400,
          },
        },
      }],
      cameras: [{
        type: 'perspective',
        perspective: { yfov: 1, znear: 0.1, zfar: 100, aspectRatio: 1.5 },
      }],
      extensions: {
        KHR_lights_punctual: {
          lights: [{
            type: 'spot',
            color: [1, 1, 1],
            intensity: 1,
            range: 10,
            spot: { innerConeAngle: 0, outerConeAngle: 0.5 },
          }],
        },
      },
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: packed.views,
      buffers: [{ byteLength: packed.buffer.byteLength }],
    };
    const imported = await gltfToScene(gltf, { buffers: new Map([[0, packed.buffer]]) });
    const pointerChannel = (pointer: string, from: number, to: number) => ({
      target: { node: `gltf-pointer:${pointer}`, path: 'pointer' as const, pointer },
      sampler: {
        times: new Float32Array([0, 1]),
        values: new Float32Array([from, to]),
        interpolation: 'LINEAR' as const,
      },
    });
    const animations: AnimationClip[] = [
      {
        name: 'valid-camera-light',
        duration: 1,
        channels: [
          pointerChannel('/cameras/0/perspective/zfar', 100, 200),
          pointerChannel('/extensions/KHR_lights_punctual/lights/0/intensity', 1, 3),
        ],
      },
      {
        name: 'invalid-camera-range',
        duration: 1,
        channels: [pointerChannel('/cameras/0/perspective/znear', 0.1, 150)],
      },
      {
        name: 'invalid-spot-cones',
        duration: 1,
        channels: [pointerChannel(
          '/extensions/KHR_lights_punctual/lights/0/spot/innerConeAngle',
          0,
          0.8,
        )],
      },
      {
        name: 'invalid-iridescence-range',
        duration: 1,
        channels: [pointerChannel(
          '/materials/0/extensions/KHR_materials_iridescence/iridescenceThicknessMinimum',
          100,
          500,
        )],
      },
      {
        name: 'invalid-cubic-light-color',
        duration: 1,
        channels: [{
          target: {
            node:
              'gltf-pointer:/extensions/KHR_lights_punctual/lights/0/color',
            path: 'pointer',
            pointer: '/extensions/KHR_lights_punctual/lights/0/color',
          },
          sampler: {
            times: new Float32Array([0, 1]),
            // in/value/out triplets per key. Endpoints are valid, while the
            // opposing tangents overshoot the [0,1] color domain at t=0.5.
            values: new Float32Array([
              0, 0, 0,
              0.5, 0.5, 0.5,
              10, 10, 10,
              -10, -10, -10,
              0.5, 0.5, 0.5,
              0, 0, 0,
            ]),
            interpolation: 'CUBICSPLINE',
          },
        }],
      },
    ];
    const controller = createGltfSceneController({ gltf, ...imported, animations });

    const valid = controller.applyAnimation('valid-camera-light', 1);
    expect(valid.cameras[0]?.perspective?.zfar).toBeCloseTo(200);
    expect(valid.scene.emitters[0]?.intensity).toBeCloseTo(3);

    const cameraInvalid = controller.applyAnimation('invalid-camera-range', 1);
    expect(cameraInvalid.cameras[0]?.perspective).toMatchObject({ znear: 0.1, zfar: 100 });
    expect(cameraInvalid.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'animation-pointer-value-invalid' }),
    ]));

    const spotBefore = controller.scene.emitters[0];
    const coneInvalid = controller.applyAnimation('invalid-spot-cones', 1);
    expect(coneInvalid.scene.emitters[0]).toMatchObject({
      angle: spotBefore?.kind === 'spot' ? spotBefore.angle : undefined,
      penumbra: spotBefore?.kind === 'spot' ? spotBefore.penumbra : undefined,
    });
    expect(coneInvalid.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'animation-pointer-value-invalid' }),
    ]));

    const rangeInvalid = controller.applyAnimation('invalid-iridescence-range', 1);
    expect((rangeInvalid.scene.primitives[0] as MeshPrimitive).material.iridescenceThicknessRange)
      .toEqual([100, 400]);
    expect(rangeInvalid.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'animation-pointer-value-invalid' }),
    ]));

    const cubicInvalid = controller.applyAnimation('invalid-cubic-light-color', 0.5);
    expect(cubicInvalid.scene.emitters[0]).toMatchObject({ color: [1, 1, 1] });
    expect(cubicInvalid.diagnostics).toContainEqual(expect.objectContaining({
      code: 'animation-pointer-value-invalid',
      path: '/extensions/KHR_lights_punctual/lights/0/color',
    }));
  });
});
