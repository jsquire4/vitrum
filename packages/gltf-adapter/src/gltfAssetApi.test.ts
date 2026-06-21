// gltfAssetApi.test.ts — higher-level arbitrary-asset API tests.
//
// These fixtures exercise the public package root, not private helpers: URL
// loading with external resources, structured feature reporting, and backend
// compatibility ranking against the core promise ledger.

import { describe, expect, it, vi } from 'vitest';
import { PNG } from 'pngjs';
import * as jpeg from 'jpeg-js';
import * as webp from 'webp-wasm';
import {
  analyzeGltfAsset,
  decodeSceneTextures,
  evaluateGltfBackendCompatibility,
  evaluateGltfBackendProfileCompatibility,
  GltfCompatibilityError,
  GltfFetchFailed,
  GltfImportError,
  GltfParseFailed,
  GltfResourceDecodeFailed,
  GltfResourceNotFound,
  loadGltfAndDecodeTextures,
  loadGltfForEngine,
  loadGltfAsset,
  rankGltfBackends,
} from './index.js';
import type { DecodeGltfTexturePixelsFn, GltfAssetFetchResponse, GltfJson } from './index.js';
import type { InstancedMeshPrimitive, MeshPrimitive, Scene, SkinnedMeshPrimitive, TextureRef } from '@vitrum/core';

function f32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function u16Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 2);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setUint16(i * 2, v, true));
  return buf;
}

function concatArrayBuffers(chunks: readonly ArrayBuffer[]): ArrayBuffer {
  const out = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0));
  let offset = 0;
  for (const chunk of chunks) {
    out.set(new Uint8Array(chunk), offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

function bytes(values: number[]): ArrayBuffer {
  return new Uint8Array(values).buffer;
}

function textBuffer(text: string): ArrayBuffer {
  const encoded = new TextEncoder().encode(text);
  return encoded.buffer.slice(encoded.byteOffset, encoded.byteOffset + encoded.byteLength);
}

function glbBuffer(jsonText: string, version = 2): ArrayBuffer {
  const jsonBytes = new TextEncoder().encode(jsonText);
  const jsonLength = Math.ceil(jsonBytes.byteLength / 4) * 4;
  const totalLength = 12 + 8 + jsonLength;
  const buffer = new ArrayBuffer(totalLength);
  const view = new DataView(buffer);
  const bytesOut = new Uint8Array(buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, version, true);
  view.setUint32(8, totalLength, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytesOut.fill(0x20, 20, 20 + jsonLength);
  bytesOut.set(jsonBytes, 20);
  return buffer;
}

function srgbToLinearForTest(value: number): number {
  const c = Math.max(0, Math.min(1, value));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function linearToSrgbForTest(value: number): number {
  const c = Math.max(0, Math.min(1, value));
  return c <= 0.0031308 ? c * 12.92 : 1.055 * (c ** (1 / 2.4)) - 0.055;
}

async function withCreateImageBitmapStub<T>(
  fn: (createImageBitmap: unknown) => Promise<T>,
): Promise<T> {
  const host = globalThis as typeof globalThis & { createImageBitmap?: unknown };
  const hadCreateImageBitmap = Object.prototype.hasOwnProperty.call(host, 'createImageBitmap');
  const previousCreateImageBitmap = host.createImageBitmap;
  const createImageBitmap = vi.fn(async () => ({
    width: 4,
    height: 4,
    close: vi.fn(),
  }));

  Object.defineProperty(host, 'createImageBitmap', {
    configurable: true,
    writable: true,
    value: createImageBitmap,
  });

  try {
    return await fn(createImageBitmap);
  } finally {
    if (hadCreateImageBitmap) {
      Object.defineProperty(host, 'createImageBitmap', {
        configurable: true,
        writable: true,
        value: previousCreateImageBitmap,
      });
    } else {
      delete (host as { createImageBitmap?: unknown }).createImageBitmap;
    }
  }
}

async function withOffscreenCanvasReadbackStub<T>(
  pixels: Uint8ClampedArray,
  fn: (ctx: {
    drawImage: ReturnType<typeof vi.fn>;
    getImageData: ReturnType<typeof vi.fn<[], { data: Uint8ClampedArray }>>;
  }) => Promise<T>,
): Promise<T> {
  const host = globalThis as typeof globalThis & { OffscreenCanvas?: unknown };
  const hadOffscreenCanvas = Object.prototype.hasOwnProperty.call(host, 'OffscreenCanvas');
  const previousOffscreenCanvas = host.OffscreenCanvas;
  const ctx = {
    drawImage: vi.fn(),
    getImageData: vi.fn(() => ({ data: pixels })),
  };
  class OffscreenCanvasStub {
    width: number;
    height: number;

    constructor(width: number, height: number) {
      this.width = width;
      this.height = height;
    }

    getContext(type: string): unknown {
      return type === '2d' ? ctx : null;
    }
  }

  Object.defineProperty(host, 'OffscreenCanvas', {
    configurable: true,
    writable: true,
    value: OffscreenCanvasStub,
  });

  try {
    return await fn(ctx);
  } finally {
    if (hadOffscreenCanvas) {
      Object.defineProperty(host, 'OffscreenCanvas', {
        configurable: true,
        writable: true,
        value: previousOffscreenCanvas,
      });
    } else {
      delete (host as { OffscreenCanvas?: unknown }).OffscreenCanvas;
    }
  }
}

function response(data: ArrayBuffer, contentType = 'application/octet-stream'): GltfAssetFetchResponse {
  return {
    ok: true,
    status: 200,
    headers: { get: (name: string) => (name.toLowerCase() === 'content-type' ? contentType : null) },
    arrayBuffer: async () => data,
  };
}

function makeExternalTexturedGltf(): GltfJson {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  return {
    asset: { version: '2.0', generator: 'vitrum-test' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, material: 0 }] }],
    materials: [{
      pbrMetallicRoughness: {
        baseColorFactor: [1, 1, 1, 1],
        baseColorTexture: { index: 0, texCoord: 0 },
      },
    }],
    textures: [{ source: 0, sampler: 0 }],
    samplers: [{ wrapS: 33071, wrapT: 33648 }],
    images: [{ uri: 'textures/albedo.png', mimeType: 'image/png' }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
      { buffer: 0, byteOffset: positions.byteLength, byteLength: 6 * 4 },
    ],
    buffers: [{ uri: 'mesh.bin', byteLength: positions.byteLength + 6 * 4 }],
  };
}

function makeInlineTriangleGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const vertexData = f32Buffer([
    0, 0, 0, 1, 0, 0, 0, 1, 0,
    0, 0, 1, 0, 0, 1, 0, 0, 1,
  ]);
  const positionByteLength = 9 * 4;
  const normalByteLength = 9 * 4;
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 } }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positionByteLength },
        { buffer: 0, byteOffset: positionByteLength, byteLength: normalByteLength },
      ],
      buffers: [{ byteLength: vertexData.byteLength }],
    },
    buffers: new Map([[0, vertexData]]),
  };
}

function addMorphedGpuInstancing(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
): void {
  const morphDeltas = f32Buffer([0, 0, 0.1, 0, 0, 0.1, 0, 0, 0.1]);
  const instanceTranslations = f32Buffer([
    0, 0, 0,
    2, 0, 0,
  ]);
  const morphAccessor = gltf.accessors?.length ?? 0;
  const instanceAccessor = morphAccessor + 1;
  const morphBufferView = gltf.bufferViews?.length ?? 0;
  const instanceBufferView = morphBufferView + 1;
  const morphBuffer = gltf.buffers?.length ?? 0;
  const instanceBuffer = morphBuffer + 1;
  gltf.extensionsUsed = ['EXT_mesh_gpu_instancing'];
  gltf.extensionsRequired = ['EXT_mesh_gpu_instancing'];
  gltf.nodes![0] = {
    ...gltf.nodes![0]!,
    extensions: {
      EXT_mesh_gpu_instancing: {
        attributes: { TRANSLATION: instanceAccessor },
      },
    },
  };
  gltf.meshes![0]!.primitives[0] = {
    ...gltf.meshes![0]!.primitives[0]!,
    targets: [{ POSITION: morphAccessor }],
  };
  gltf.accessors = [
    ...(gltf.accessors ?? []),
    { bufferView: morphBufferView, componentType: 5126, count: 3, type: 'VEC3' },
    { bufferView: instanceBufferView, componentType: 5126, count: 2, type: 'VEC3' },
  ];
  gltf.bufferViews = [
    ...(gltf.bufferViews ?? []),
    { buffer: morphBuffer, byteOffset: 0, byteLength: morphDeltas.byteLength },
    { buffer: instanceBuffer, byteOffset: 0, byteLength: instanceTranslations.byteLength },
  ];
  gltf.buffers = [
    ...(gltf.buffers ?? []),
    { byteLength: morphDeltas.byteLength },
    { byteLength: instanceTranslations.byteLength },
  ];
  buffers.set(morphBuffer, morphDeltas);
  buffers.set(instanceBuffer, instanceTranslations);
}

function addUnboundSkinAttributes(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  opts: { omitWeights?: boolean } = {},
): void {
  const joints = bytes([
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  ]);
  const weights = f32Buffer([
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0,
  ]);
  const jointsAccessor = gltf.accessors?.length ?? 0;
  const weightsAccessor = jointsAccessor + 1;
  const jointsBufferView = gltf.bufferViews?.length ?? 0;
  const weightsBufferView = jointsBufferView + 1;
  const jointsBuffer = gltf.buffers?.length ?? 0;
  const weightsBuffer = jointsBuffer + 1;
  gltf.meshes![0]!.primitives[0] = {
    ...gltf.meshes![0]!.primitives[0]!,
    attributes: {
      ...gltf.meshes![0]!.primitives[0]!.attributes,
      JOINTS_0: jointsAccessor,
      ...(opts.omitWeights === true ? {} : { WEIGHTS_0: weightsAccessor }),
    },
  };
  gltf.accessors = [
    ...(gltf.accessors ?? []),
    { bufferView: jointsBufferView, componentType: 5121, count: 3, type: 'VEC4' as const },
    ...(opts.omitWeights === true
      ? []
      : [{ bufferView: weightsBufferView, componentType: 5126, count: 3, type: 'VEC4' as const }]),
  ];
  gltf.bufferViews = [
    ...(gltf.bufferViews ?? []),
    { buffer: jointsBuffer, byteOffset: 0, byteLength: joints.byteLength },
    ...(opts.omitWeights === true
      ? []
      : [{ buffer: weightsBuffer, byteOffset: 0, byteLength: weights.byteLength }]),
  ];
  gltf.buffers = [
    ...(gltf.buffers ?? []),
    { byteLength: joints.byteLength },
    ...(opts.omitWeights === true ? [] : [{ byteLength: weights.byteLength }]),
  ];
  buffers.set(jointsBuffer, joints);
  if (opts.omitWeights !== true) buffers.set(weightsBuffer, weights);
}

function addSecondaryVertexColorSet(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
): void {
  const color1 = f32Buffer([
    0.25, 0.25, 0.25,
    0.5, 0.5, 0.5,
    0.75, 0.75, 0.75,
  ]);
  const accessor = gltf.accessors?.length ?? 0;
  const bufferView = gltf.bufferViews?.length ?? 0;
  const buffer = gltf.buffers?.length ?? 0;
  gltf.meshes![0]!.primitives[0] = {
    ...gltf.meshes![0]!.primitives[0]!,
    attributes: {
      ...gltf.meshes![0]!.primitives[0]!.attributes,
      COLOR_1: accessor,
    },
  };
  gltf.accessors = [
    ...(gltf.accessors ?? []),
    { bufferView, componentType: 5126, count: 3, type: 'VEC3' },
  ];
  gltf.bufferViews = [
    ...(gltf.bufferViews ?? []),
    { buffer, byteOffset: 0, byteLength: color1.byteLength },
  ];
  gltf.buffers = [
    ...(gltf.buffers ?? []),
    { byteLength: color1.byteLength },
  ];
  buffers.set(buffer, color1);
}

function addMorphTargetTexcoord(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  opts: {
    semantic?: 'TEXCOORD_0' | 'TEXCOORD_2';
    baseSemantic?: 'TEXCOORD_0' | 'TEXCOORD_2';
    includeBaseUv?: boolean;
    materialTexCoord?: number;
  } = {},
): void {
  const semantic = opts.semantic ?? 'TEXCOORD_0';
  const baseSemantic = opts.baseSemantic ?? 'TEXCOORD_0';
  const includeBaseUv = opts.includeBaseUv ?? true;
  const baseUv = f32Buffer([
    0, 0,
    1, 0,
    0, 1,
  ]);
  const positionDelta = f32Buffer([
    0, 0, 0.1,
    0, 0, 0.1,
    0, 0, 0.1,
  ]);
  const uvDelta = f32Buffer([
    0.1, 0,
    0.1, 0,
    0.1, 0,
  ]);
  const positionAccessor = gltf.accessors?.length ?? 0;
  const baseUvAccessor = includeBaseUv ? positionAccessor + 1 : -1;
  const uvAccessor = includeBaseUv ? positionAccessor + 2 : positionAccessor + 1;
  const positionBufferView = gltf.bufferViews?.length ?? 0;
  const baseUvBufferView = includeBaseUv ? positionBufferView + 1 : -1;
  const uvBufferView = includeBaseUv ? positionBufferView + 2 : positionBufferView + 1;
  const positionBuffer = gltf.buffers?.length ?? 0;
  const baseUvBuffer = includeBaseUv ? positionBuffer + 1 : -1;
  const uvBuffer = includeBaseUv ? positionBuffer + 2 : positionBuffer + 1;
  gltf.meshes![0]!.primitives[0] = {
    ...gltf.meshes![0]!.primitives[0]!,
    attributes: {
      ...gltf.meshes![0]!.primitives[0]!.attributes,
      ...(includeBaseUv ? { [baseSemantic]: baseUvAccessor } : {}),
    },
    ...(opts.materialTexCoord !== undefined ? { material: 0 } : {}),
    targets: [{ POSITION: positionAccessor, [semantic]: uvAccessor }],
  };
  if (opts.materialTexCoord !== undefined) {
    gltf.materials = [{
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0, texCoord: opts.materialTexCoord },
      },
    }];
    gltf.textures = [{ source: 0 }];
    gltf.images = [{ uri: 'data:image/png;base64,AQID', mimeType: 'image/png' }];
  }
  gltf.meshes![0] = {
    ...gltf.meshes![0]!,
    weights: [1],
  };
  gltf.accessors = [
    ...(gltf.accessors ?? []),
    { bufferView: positionBufferView, componentType: 5126, count: 3, type: 'VEC3' },
    ...(includeBaseUv
      ? [{ bufferView: baseUvBufferView, componentType: 5126, count: 3, type: 'VEC2' as const }]
      : []),
    { bufferView: uvBufferView, componentType: 5126, count: 3, type: 'VEC2' },
  ];
  gltf.bufferViews = [
    ...(gltf.bufferViews ?? []),
    { buffer: positionBuffer, byteOffset: 0, byteLength: positionDelta.byteLength },
    ...(includeBaseUv
      ? [{ buffer: baseUvBuffer, byteOffset: 0, byteLength: baseUv.byteLength }]
      : []),
    { buffer: uvBuffer, byteOffset: 0, byteLength: uvDelta.byteLength },
  ];
  gltf.buffers = [
    ...(gltf.buffers ?? []),
    { byteLength: positionDelta.byteLength },
    ...(includeBaseUv ? [{ byteLength: baseUv.byteLength }] : []),
    { byteLength: uvDelta.byteLength },
  ];
  buffers.set(positionBuffer, positionDelta);
  if (includeBaseUv) buffers.set(baseUvBuffer, baseUv);
  buffers.set(uvBuffer, uvDelta);
}

function makeInlineVertexColorGltf(
  colorsValues = [1, 0, 0, 0, 1, 0, 0, 0, 1],
  colorType: 'VEC3' | 'VEC4' = 'VEC3',
): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const colors = f32Buffer(colorsValues);
  const total = new Uint8Array(positions.byteLength + colors.byteLength);
  total.set(new Uint8Array(positions), 0);
  total.set(new Uint8Array(colors), positions.byteLength);
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, COLOR_0: 1 } }] }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: colorType },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 0, byteOffset: positions.byteLength, byteLength: colors.byteLength },
      ],
      buffers: [{ byteLength: total.byteLength }],
    },
    buffers: new Map([[0, total.buffer]]),
  };
}

function makeInlineMaterialVariantGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  return {
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
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
      buffers: [{ byteLength: positions.byteLength }],
    },
    buffers: new Map([[0, positions]]),
  };
}

function makeInlineMaterialPointerAnimationGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const times = f32Buffer([0, 1]);
  const colors = f32Buffer([
    1, 0, 0, 1,
    0, 1, 0, 1,
  ]);
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      extensionsUsed: ['KHR_animation_pointer'],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{
        pbrMetallicRoughness: {
          baseColorFactor: [1, 0, 0, 1],
        },
      }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 2, type: 'SCALAR' },
        { bufferView: 2, componentType: 5126, count: 2, type: 'VEC4' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 1, byteOffset: 0, byteLength: times.byteLength },
        { buffer: 2, byteOffset: 0, byteLength: colors.byteLength },
      ],
      buffers: [
        { byteLength: positions.byteLength },
        { byteLength: times.byteLength },
        { byteLength: colors.byteLength },
      ],
      animations: [{
        name: 'material-color',
        samplers: [{ input: 1, output: 2, interpolation: 'LINEAR' }],
        channels: [{
          sampler: 0,
          target: {
            path: 'pointer',
            extensions: {
              KHR_animation_pointer: {
                pointer: '/materials/0/pbrMetallicRoughness/baseColorFactor',
              },
            },
          },
        }],
      }],
    },
    buffers: new Map([
      [0, positions],
      [1, times],
      [2, colors],
    ]),
  };
}

function makeInlineTexturedVariantGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const imageBytes = bytes([0x89, 0x50, 0x4e, 0x47]);
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      extensionsUsed: ['KHR_materials_variants'],
      extensionsRequired: ['KHR_materials_variants'],
      extensions: {
        KHR_materials_variants: {
          variants: [{ name: 'textured' }],
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
        {
          name: 'variant textured',
          pbrMetallicRoughness: {
            baseColorFactor: [1, 1, 1, 1],
            baseColorTexture: { index: 0 },
          },
        },
      ],
      textures: [{ source: 0 }],
      images: [{ bufferView: 1, mimeType: 'image/png' }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 1, byteOffset: 0, byteLength: imageBytes.byteLength },
      ],
      buffers: [
        { byteLength: positions.byteLength },
        { byteLength: imageBytes.byteLength },
      ],
    },
    buffers: new Map([
      [0, positions],
      [1, imageBytes],
    ]),
  };
}

function makeInlineSpecGlossVariantGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const imageBytes = bytes([0x89, 0x50, 0x4e, 0x47]);
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      extensionsUsed: ['KHR_materials_variants', 'KHR_materials_pbrSpecularGlossiness'],
      extensionsRequired: ['KHR_materials_variants'],
      extensions: {
        KHR_materials_variants: {
          variants: [{ name: 'glossy' }],
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
        {
          name: 'variant spec gloss',
          extensions: {
            KHR_materials_pbrSpecularGlossiness: {
              diffuseFactor: [1, 1, 1, 1],
              specularFactor: [1, 1, 1],
              glossinessFactor: 0.5,
              specularGlossinessTexture: { index: 0 },
            },
          },
        },
      ],
      textures: [{ source: 0 }],
      images: [{ bufferView: 1, mimeType: 'image/png' }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 1, byteOffset: 0, byteLength: imageBytes.byteLength },
      ],
      buffers: [
        { byteLength: positions.byteLength },
        { byteLength: imageBytes.byteLength },
      ],
    },
    buffers: new Map([
      [0, positions],
      [1, imageBytes],
    ]),
  };
}

function makeInlineAnimatedInstancedGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const instanceTranslations = f32Buffer([
    2, 0, 0,
    0, 3, 0,
  ]);
  const times = f32Buffer([0, 1]);
  const nodeTranslations = f32Buffer([
    10, 0, 0,
    14, 0, 0,
  ]);
  return {
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
      bufferViews: [
        { buffer: 0, byteLength: positions.byteLength },
        { buffer: 1, byteLength: instanceTranslations.byteLength },
        { buffer: 2, byteLength: times.byteLength },
        { buffer: 3, byteLength: nodeTranslations.byteLength },
      ],
      buffers: [
        { byteLength: positions.byteLength },
        { byteLength: instanceTranslations.byteLength },
        { byteLength: times.byteLength },
        { byteLength: nodeTranslations.byteLength },
      ],
      animations: [{
        name: 'instance-slide',
        samplers: [{ input: 2, output: 3, interpolation: 'LINEAR' }],
        channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
      }],
    },
    buffers: new Map([
      [0, positions],
      [1, instanceTranslations],
      [2, times],
      [3, nodeTranslations],
    ]),
  };
}

function makeInlineTexturedGltf(
  imageBytes: Uint8Array<ArrayBuffer> = new Uint8Array([0x89, 0x50, 0x4e, 0x47]),
): { gltf: GltfJson; buffers: Map<number, ArrayBuffer>; png: Uint8Array<ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const normals = f32Buffer([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const total = new Uint8Array(positions.byteLength + normals.byteLength + imageBytes.byteLength);
  total.set(new Uint8Array(positions), 0);
  total.set(new Uint8Array(normals), positions.byteLength);
  total.set(imageBytes, positions.byteLength + normals.byteLength);
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, material: 0 }] }],
      materials: [{
        pbrMetallicRoughness: {
          baseColorFactor: [1, 1, 1, 1],
          baseColorTexture: { index: 0 },
        },
      }],
      textures: [{ source: 0 }],
      images: [{ bufferView: 2, mimeType: 'image/png' }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 0, byteOffset: positions.byteLength, byteLength: normals.byteLength },
        { buffer: 0, byteOffset: positions.byteLength + normals.byteLength, byteLength: imageBytes.byteLength },
      ],
      buffers: [{ byteLength: total.byteLength }],
    },
    buffers: new Map([[0, total.buffer]]),
    png: imageBytes,
  };
}

function makePngBytes(
  width: number,
  height: number,
  rgba: readonly number[],
): Uint8Array<ArrayBuffer> {
  const encoded = PNG.sync.write({
    width,
    height,
    data: Uint8Array.from(rgba),
  });
  const buffer = encoded.buffer.slice(
    encoded.byteOffset,
    encoded.byteOffset + encoded.byteLength,
  ) as ArrayBuffer;
  return new Uint8Array(buffer);
}

function makeJpegBytes(
  width: number,
  height: number,
  rgba: readonly number[],
): Uint8Array<ArrayBuffer> {
  const encoded = jpeg.encode({
    width,
    height,
    data: Uint8Array.from(rgba),
  }, 100).data;
  const buffer = encoded.buffer.slice(
    encoded.byteOffset,
    encoded.byteOffset + encoded.byteLength,
  ) as ArrayBuffer;
  return new Uint8Array(buffer);
}

async function makeWebpBytes(
  width: number,
  height: number,
  rgba: readonly number[],
): Promise<Uint8Array<ArrayBuffer>> {
  const encoded = await webp.encode({
    width,
    height,
    data: Uint8ClampedArray.from(rgba),
  }, { quality: 100, lossless: 1, exact: 1 });
  const buffer = encoded.buffer.slice(encoded.byteOffset, encoded.byteOffset + encoded.byteLength) as ArrayBuffer;
  return new Uint8Array(buffer);
}

function makeInlineNormalMappedGltf(
  opts: { readonly texCoord?: number; readonly includeUv0?: boolean } = {},
): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const fixture = makeInlineTriangleGltf();
  const uv = f32Buffer([0, 0, 1, 0, 0, 1]);
  const image = bytes([0x89, 0x50, 0x4e, 0x47]);
  const uvAccessor = fixture.gltf.accessors!.length;
  const uvBufferView = fixture.gltf.bufferViews!.length;
  const imageBufferView = uvBufferView + 1;
  const uvBuffer = fixture.gltf.buffers!.length;
  const imageBuffer = uvBuffer + 1;

  fixture.gltf.meshes![0]!.primitives[0] = {
    ...fixture.gltf.meshes![0]!.primitives[0]!,
    material: 0,
    attributes: {
      ...fixture.gltf.meshes![0]!.primitives[0]!.attributes,
      ...(opts.includeUv0 === false ? {} : { TEXCOORD_0: uvAccessor }),
    },
  };
  fixture.gltf.materials = [{ normalTexture: { index: 0, texCoord: opts.texCoord ?? 0 } }];
  fixture.gltf.textures = [{ source: 0 }];
  fixture.gltf.images = [{ bufferView: imageBufferView, mimeType: 'image/png' }];
  fixture.gltf.accessors = [
    ...fixture.gltf.accessors!,
    { bufferView: uvBufferView, componentType: 5126, count: 3, type: 'VEC2' },
  ];
  fixture.gltf.bufferViews = [
    ...fixture.gltf.bufferViews!,
    { buffer: uvBuffer, byteOffset: 0, byteLength: uv.byteLength },
    { buffer: imageBuffer, byteOffset: 0, byteLength: image.byteLength },
  ];
  fixture.gltf.buffers = [
    ...fixture.gltf.buffers!,
    { byteLength: uv.byteLength },
    { byteLength: image.byteLength },
  ];
  fixture.buffers.set(uvBuffer, uv);
  fixture.buffers.set(imageBuffer, image);
  return fixture;
}

function makeInlineSpecGlossTexturedGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const imageBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
  const total = new Uint8Array(positions.byteLength + imageBytes.byteLength);
  total.set(new Uint8Array(positions), 0);
  total.set(imageBytes, positions.byteLength);
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{
        extensions: {
          KHR_materials_pbrSpecularGlossiness: {
            diffuseFactor: [1, 1, 1, 1],
            specularFactor: [1, 1, 1],
            glossinessFactor: 0.5,
            specularGlossinessTexture: {
              index: 0,
              texCoord: 0,
              extensions: {
                KHR_texture_transform: {
                  texCoord: 1,
                  offset: [0.25, 0.5],
                  scale: [2, 3],
                  rotation: 0.125,
                },
              },
            },
          },
        },
      }],
      textures: [{ source: 0, sampler: 0 }],
      samplers: [{ wrapS: 33071, wrapT: 33648 }],
      images: [{ bufferView: 1, mimeType: 'image/png' }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
        { buffer: 0, byteOffset: positions.byteLength, byteLength: imageBytes.byteLength },
      ],
      buffers: [{ byteLength: total.byteLength }],
    },
    buffers: new Map([[0, total.buffer]]),
  };
}

describe('loadGltfAsset', () => {
  it('fetches JSON glTF, external .bin buffers, and external image bytes', async () => {
    const gltf = makeExternalTexturedGltf();
    const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const uvs = f32Buffer([0, 0, 1, 0, 0, 1]);
    const meshBytes = new Uint8Array(positions.byteLength + uvs.byteLength);
    meshBytes.set(new Uint8Array(positions), 0);
    meshBytes.set(new Uint8Array(uvs), positions.byteLength);
    const imageBytes = bytes([0x89, 0x50, 0x4e, 0x47]);
    const decodedHandle = { kind: 'decoded-image' };

    const fetch = vi.fn(async (url: string) => {
      if (url === 'https://cdn.test/assets/model.gltf') {
        return response(textBuffer(JSON.stringify(gltf)), 'model/gltf+json');
      }
      if (url === 'https://cdn.test/assets/mesh.bin') {
        return response(meshBytes.buffer);
      }
      if (url === 'https://cdn.test/assets/textures/albedo.png') {
        return response(imageBytes, 'image/png');
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    const decodeImage = vi.fn(async (data: Uint8Array, mimeType: string) => {
      expect(Array.from(data)).toEqual([0x89, 0x50, 0x4e, 0x47]);
      expect(mimeType).toBe('image/png');
      return decodedHandle;
    });

    const result = await loadGltfAsset('https://cdn.test/assets/model.gltf', {
      fetch,
      decodeImage,
    });

    expect(fetch.mock.calls.map(([url]) => url)).toEqual([
      'https://cdn.test/assets/model.gltf',
      'https://cdn.test/assets/mesh.bin',
      'https://cdn.test/assets/textures/albedo.png',
    ]);
    expect(decodeImage).toHaveBeenCalledTimes(1);
    expect(result.scene.primitives).toHaveLength(1);
    const prim = result.scene.primitives[0] as MeshPrimitive;
    expect(Array.from(prim.positions)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect((prim.material.baseColorMap as TextureRef).handle).toBe(decodedHandle);
    expect((prim.material.baseColorMap as TextureRef).wrapS).toBe('clamp-to-edge');
    expect((prim.material.baseColorMap as TextureRef).wrapT).toBe('mirrored-repeat');
    expect(result.featureReport.resources.externalBufferCount).toBe(1);
    expect(result.featureReport.resources.externalImageCount).toBe(1);
    expect(result.textureDecodeReport.entries).toEqual(expect.arrayContaining([
      expect.objectContaining({
        primitiveId: 'gltf-prim-0',
        materialField: 'baseColorMap',
        handleKind: 'opaque',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        wrapS: 'clamp-to-edge',
        wrapT: 'mirrored-repeat',
        colorSpace: 'srgb',
      }),
    ]));
    expect(result.recommendedBackend.backend).toBe('pt-webgl2');
    expect(result.recommendedBackend).toMatchObject({
      requiresHookCount: 1,
      issues: expect.arrayContaining([
        expect.objectContaining({
          category: 'texture',
          name: 'texture-readiness:baseColorMap',
          support: 'requires-hook',
          path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        }),
      ]),
    });
  });

  it('forwards pointLineFallbackRadius into generated point/line meshes', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      mode: 0,
    };

    const asset = await loadGltfAsset(gltf, {
      buffers,
      pointLineFallbackRadius: 0.25,
    });

    const primitive = asset.scene.primitives[0] as MeshPrimitive;
    expect(primitive.positions.length).toBeGreaterThan(9);
    expect(Array.from(primitive.positions.slice(0, 3))).toEqual([0.25, -0.25, -0.25]);
  });

  it('fallback-expands EXT_mesh_gpu_instancing on morphed meshes into skinned primitives', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addMorphedGpuInstancing(gltf, buffers);

    const asset = await loadGltfAsset(gltf, { buffers });

    expect(asset.scene.primitives).toHaveLength(2);
    const first = asset.scene.primitives[0] as SkinnedMeshPrimitive;
    const second = asset.scene.primitives[1] as SkinnedMeshPrimitive;
    expect(first.kind).toBe('skinned-mesh');
    expect(second.kind).toBe('skinned-mesh');
    expect(String(first.id)).toMatch(/-instance-0$/);
    expect(String(second.id)).toMatch(/-instance-1$/);
    expect(first.transform?.[12]).toBeCloseTo(0);
    expect(second.transform?.[12]).toBeCloseTo(2);
    expect(first.morphTargets).toHaveLength(1);
    expect(second.morphTargets).toHaveLength(1);
    expect(asset.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'fallback-expanded-gpu-instancing',
        path: 'nodes[0].extensions.EXT_mesh_gpu_instancing',
      }),
    ]));
  });

  it('throws a deterministic error for relative external resources without a baseUri', async () => {
    const gltf = makeExternalTexturedGltf();
    await expect(loadGltfAsset(gltf)).rejects.toBeInstanceOf(GltfResourceNotFound);
    await expect(loadGltfAsset(gltf)).rejects.toMatchObject({
      kind: 'buffer',
      url: 'mesh.bin',
      sourcePath: 'buffers[0].uri',
    });
  });

  it('throws typed decode failures for malformed buffer data URIs', async () => {
    const gltf = makeExternalTexturedGltf();
    gltf.buffers![0]!.uri = 'data:application/octet-stream;base64';

    await expect(loadGltfAsset(gltf)).rejects.toBeInstanceOf(GltfResourceDecodeFailed);
    await expect(loadGltfAsset(gltf)).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_DECODE_FAILED',
      kind: 'buffer',
      reason: 'malformed-data-uri',
      url: 'data:application/octet-stream;base64',
      sourcePath: 'buffers[0].uri',
    });
  });

  it('throws typed decode failures for undecodable buffer data URI payloads', async () => {
    const gltf = makeExternalTexturedGltf();
    gltf.buffers![0]!.uri = 'data:application/octet-stream,%E0%A4%A';

    await expect(loadGltfAsset(gltf)).rejects.toBeInstanceOf(GltfResourceDecodeFailed);
    await expect(loadGltfAsset(gltf)).rejects.toMatchObject({
      code: 'GLTF_RESOURCE_DECODE_FAILED',
      kind: 'buffer',
      reason: 'data-uri-decode-failed',
      url: 'data:application/octet-stream,%E0%A4%A',
      sourcePath: 'buffers[0].uri',
    });
  });

  it('throws typed fetch failures with resource identity', async () => {
    const fetch = vi.fn(async () => ({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      arrayBuffer: async () => new ArrayBuffer(0),
    }));

    await expect(loadGltfAsset('https://cdn.test/missing.gltf', { fetch })).rejects.toBeInstanceOf(GltfFetchFailed);
    await expect(loadGltfAsset('https://cdn.test/missing.gltf', { fetch })).rejects.toMatchObject({
      kind: 'asset',
      url: 'https://cdn.test/missing.gltf',
      status: 404,
      statusText: 'Not Found',
    });
  });

  it('throws typed image fetch failures with glTF source paths', async () => {
    const gltf = makeExternalTexturedGltf();
    const fetch = vi.fn(async () => ({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      arrayBuffer: async () => new ArrayBuffer(0),
    }));

    await expect(loadGltfAsset(gltf, {
      baseUri: 'https://cdn.test/assets/model.gltf',
      buffers: new Map([[0, new ArrayBuffer(gltf.buffers![0]!.byteLength)]]),
      fetch,
    })).rejects.toMatchObject({
      code: 'GLTF_FETCH_FAILED',
      kind: 'image',
      url: 'https://cdn.test/assets/textures/albedo.png',
      sourcePath: 'images[0].uri',
      status: 404,
      statusText: 'Not Found',
    });
  });

  it('throws typed parse failures for malformed JSON assets', async () => {
    await expect(loadGltfAsset(textBuffer('{ "asset": '))).rejects.toBeInstanceOf(GltfParseFailed);
    await expect(loadGltfAsset(textBuffer('{ "asset": '))).rejects.toMatchObject({
      code: 'GLTF_PARSE_FAILED',
      format: 'gltf-json',
      reason: 'json-parse-failed',
    });
  });

  it('throws typed parse failures for malformed GLB containers', async () => {
    await expect(loadGltfAsset(glbBuffer('{"asset":{"version":"2.0"}}', 1))).rejects.toBeInstanceOf(GltfParseFailed);
    await expect(loadGltfAsset(glbBuffer('{"asset":{"version":"2.0"}}', 1))).rejects.toMatchObject({
      code: 'GLTF_PARSE_FAILED',
      format: 'glb',
      reason: 'glb-unsupported-version',
      byteOffset: 4,
      version: 1,
    });

    await expect(loadGltfAsset(glbBuffer('{ "asset": '))).rejects.toMatchObject({
      code: 'GLTF_PARSE_FAILED',
      format: 'glb',
      reason: 'glb-json-parse-failed',
      byteOffset: 20,
    });
  });

  it('uses the cache hook for resolved asset, buffer, and image resources', async () => {
    const gltf = makeExternalTexturedGltf();
    const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const uvs = f32Buffer([0, 0, 1, 0, 0, 1]);
    const meshBytes = new Uint8Array(positions.byteLength + uvs.byteLength);
    meshBytes.set(new Uint8Array(positions), 0);
    meshBytes.set(new Uint8Array(uvs), positions.byteLength);
    const imageBytes = bytes([0x89, 0x50, 0x4e, 0x47]);
    const cacheStore = new Map<string, ArrayBuffer>();
    const cache = {
      get: vi.fn(async (key: { readonly url: string; readonly kind: string }) =>
        cacheStore.get(`${key.kind}:${key.url}`)),
      set: vi.fn(async (key: { readonly url: string; readonly kind: string }, data: ArrayBuffer) => {
        cacheStore.set(`${key.kind}:${key.url}`, data);
      }),
    };
    const fetch = vi.fn(async (url: string) => {
      if (url === 'https://cdn.test/a/model.gltf') {
        return response(textBuffer(JSON.stringify(gltf)), 'model/gltf+json');
      }
      if (url === 'https://cdn.test/a/mesh.bin') {
        return response(meshBytes.buffer);
      }
      if (url === 'https://cdn.test/a/textures/albedo.png') {
        return response(imageBytes, 'image/png');
      }
      throw new Error(`unexpected fetch: ${url}`);
    });

    await loadGltfAsset('model.gltf', {
      baseUri: 'https://cdn.test/a/',
      fetch,
      cache,
      decodeImage: async () => ({ kind: 'decoded' }),
    });
    await loadGltfAsset('model.gltf', {
      baseUri: 'https://cdn.test/a/',
      fetch,
      cache,
      decodeImage: async () => ({ kind: 'decoded' }),
    });

    expect(fetch).toHaveBeenCalledTimes(3);
    expect(cache.set.mock.calls.map(([key]) => `${key.kind}:${key.url}`)).toEqual([
      'asset:https://cdn.test/a/model.gltf',
      'buffer:https://cdn.test/a/mesh.bin',
      'image:https://cdn.test/a/textures/albedo.png',
    ]);
    expect(cache.get.mock.calls.filter(([key]) =>
      key.url === 'https://cdn.test/a/model.gltf' && key.kind === 'asset',
    )).toHaveLength(2);
  });

  it('returns a textureDecodeReport for raw image fallback handles', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const result = await loadGltfAndDecodeTextures(gltf, { buffers });

    expect(result.textureDecodeReport).toMatchObject({
      mapCount: 1,
      uniqueHandleCount: 1,
      rawImageCount: 1,
      opaqueHandleCount: 0,
      cpuReadableCount: 0,
    });
    expect(result.textureDecodeReport.rawImageRefs).toEqual([
      expect.objectContaining({
        primitiveId: 'gltf-prim-0',
        primitiveKind: 'mesh',
        materialField: 'baseColorMap',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        wrapS: 'repeat',
        wrapT: 'repeat',
        colorSpace: 'srgb',
        handleKind: 'raw-image',
        backendReadiness: {
          ptWebgl2: 'opaque',
          ptWebgpu: 'opaque',
          walkaroundHybrid: 'opaque',
        },
      }),
    ]);
  });

  it('reports browser ImageBitmap handles separately from CPU-readable texture payloads', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();

    await withCreateImageBitmapStub(async (createImageBitmap) => {
      const result = await loadGltfAsset(gltf, { buffers });

      expect(createImageBitmap).toHaveBeenCalledTimes(1);
      expect(result.textureDecodeReport).toMatchObject({
        mapCount: 1,
        uniqueHandleCount: 1,
        rawImageCount: 0,
        imageBitmapCount: 1,
        opaqueHandleCount: 0,
        cpuReadableCount: 0,
      });
      expect(result.textureDecodeReport.imageBitmapRefs).toEqual([
        expect.objectContaining({
          primitiveId: 'gltf-prim-0',
          primitiveKind: 'mesh',
          materialField: 'baseColorMap',
          path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
          colorSpace: 'srgb',
          handleKind: 'image-bitmap',
          width: 4,
          height: 4,
          backendReadiness: {
            ptWebgl2: 'opaque',
            ptWebgpu: 'ready',
            walkaroundHybrid: 'opaque',
          },
        }),
      ]);
    });
  });

  it('loadGltfAndDecodeTextures normalizes raw images when a pixel decoder is supplied', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => ({
      width: 1,
      height: 1,
      data: new Uint8Array([128, 64, 255, 128]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: context.colorSpace,
    }));

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      decodePixels,
    });

    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(decodePixels.mock.calls[0]?.[1]).toMatchObject({
      materialField: 'baseColorMap',
      path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
      colorSpace: 'srgb',
      primitiveId: 'gltf-prim-0',
      primitiveIndex: 0,
      textureIndex: 0,
      imageIndex: 0,
      imageMimeType: 'image/png',
    });
    expect(result.decodedTextureCount).toBe(1);
    expect(result.unchangedTextureCount).toBe(0);
    expect(result.textureDecodeDiagnostics).toEqual([]);
    expect(result.textureDecodeWarnings).toEqual([]);
    expect(result.textureDecodeReport).toMatchObject({
      mapCount: 1,
      uniqueHandleCount: 1,
      rawImageCount: 0,
      opaqueHandleCount: 0,
      cpuReadableCount: 1,
    });
    expect(result.textureDecodeReport.entries).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        width: 1,
        height: 1,
        isPowerOfTwo: true,
        originalWidth: 1,
        originalHeight: 1,
        wasResized: false,
        textureIndex: 0,
        imageIndex: 0,
        imageMimeType: 'image/png',
      }),
    ]);

    const primitive = result.scene.primitives[0] as MeshPrimitive;
    const ref = primitive.material.baseColorMap as TextureRef;
    const handle = ref.handle as { data: Float32Array; __vitrum_hint__: { colorSpace: string } };
    expect(handle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    expect(handle.data[0]).toBeCloseTo(srgbToLinearForTest(128 / 255));
    expect(handle.data[1]).toBeCloseTo(srgbToLinearForTest(64 / 255));
    expect(handle.data[2]).toBeCloseTo(1);
    expect(handle.data[3]).toBeCloseTo(128 / 255);
  });

  it('reports host decodePixels failures as texture diagnostics instead of throwing', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const decodePixels = vi.fn(() => {
      throw new Error('decoder unavailable');
    });

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      decodePixels,
    });

    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(result.decodedTextureCount).toBe(0);
    expect(result.unchangedTextureCount).toBe(1);
    expect(result.textureDecodeDiagnostics).toEqual([
      expect.objectContaining({
        code: 'decode-pixels-failed',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        materialField: 'baseColorMap',
        textureIndex: 0,
        imageIndex: 0,
        imageMimeType: 'image/png',
        causeMessage: 'decoder unavailable',
      }),
    ]);
    expect(result.textureDecodeWarnings[0]).toContain('decodePixels hook failed');
  });

  it('reports invalid decodePixels dimensions as texture diagnostics instead of throwing', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      decodePixels: () => ({
        width: 0,
        height: 1,
        data: new Uint8Array([255, 255, 255, 255]),
        channels: 4,
        dataType: 'uint8',
      }),
    });

    expect(result.decodedTextureCount).toBe(0);
    expect(result.unchangedTextureCount).toBe(1);
    expect(result.textureDecodeDiagnostics).toEqual([
      expect.objectContaining({
        code: 'decode-pixels-invalid',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        materialField: 'baseColorMap',
        width: 0,
        height: 1,
        textureIndex: 0,
        imageIndex: 0,
      }),
    ]);
    expect(result.textureDecodeWarnings[0]).toContain('invalid pixels');
  });

  it('reports too-short decodePixels payloads as texture diagnostics instead of padding missing texels', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      decodePixels: () => ({
        width: 2,
        height: 2,
        data: new Uint8Array([
          255, 0, 0, 255,
          0, 255, 0, 255,
          0, 0, 255,
        ]),
        channels: 4,
        dataType: 'uint8',
      }),
    });

    expect(result.decodedTextureCount).toBe(0);
    expect(result.unchangedTextureCount).toBe(1);
    expect(result.textureDecodeDiagnostics).toEqual([
      expect.objectContaining({
        code: 'decode-pixels-invalid',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        materialField: 'baseColorMap',
        width: 2,
        height: 2,
        textureIndex: 0,
        imageIndex: 0,
      }),
    ]);
    expect(result.textureDecodeWarnings[0]).toContain('data length 11 is too short for 2x2x4');
  });

  it('loadGltfAndDecodeTextures bypasses browser ImageBitmap handles when a pixel decoder is supplied', async () => {
    const { gltf, buffers, png } = makeInlineTexturedGltf();
    const decodePixels = vi.fn((
      handle: Parameters<DecodeGltfTexturePixelsFn>[0],
      context: Parameters<DecodeGltfTexturePixelsFn>[1],
    ) => {
      expect(handle).toMatchObject({
        kind: 'raw-image',
        mimeType: 'image/png',
      });
      expect(handle.data).toEqual(png);
      return {
        width: 1,
        height: 1,
        data: new Uint8Array([255, 255, 255, 255]),
        channels: 4 as const,
        dataType: 'uint8' as const,
        colorSpace: context.colorSpace,
      };
    });

    await withCreateImageBitmapStub(async (createImageBitmap) => {
      const result = await loadGltfAndDecodeTextures(gltf, {
        buffers,
        decodePixels,
      });

      expect(createImageBitmap).not.toHaveBeenCalled();
      expect(decodePixels).toHaveBeenCalledTimes(1);
      expect(result.decodedTextureCount).toBe(1);
      expect(result.unchangedTextureCount).toBe(0);
      expect(result.textureDecodeDiagnostics).toEqual([]);
      expect(result.textureDecodeReport).toMatchObject({
        mapCount: 1,
        uniqueHandleCount: 1,
        rawImageCount: 0,
        opaqueHandleCount: 0,
        cpuReadableCount: 1,
      });
      expect(result.textureDecodeReport.entries).toEqual([
        expect.objectContaining({
          materialField: 'baseColorMap',
          handleKind: 'pixel-data',
          textureIndex: 0,
          imageIndex: 0,
          imageMimeType: 'image/png',
        }),
      ]);
    });
  });

  it('loadGltfAndDecodeTextures uses browser image and canvas readback when no pixel decoder is supplied', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const rgba = new Uint8ClampedArray(4 * 4 * 4);
    rgba.set([128, 64, 255, 128], 0);
    rgba.fill(255, 4);

    await withCreateImageBitmapStub(async (createImageBitmap) => {
      await withOffscreenCanvasReadbackStub(rgba, async (ctx) => {
        const result = await loadGltfAndDecodeTextures(gltf, { buffers });

        expect(createImageBitmap).toHaveBeenCalledTimes(1);
        expect(ctx.drawImage).toHaveBeenCalledTimes(1);
        expect(ctx.getImageData).toHaveBeenCalledWith(0, 0, 4, 4);
        expect(result.decodedTextureCount).toBe(1);
        expect(result.unchangedTextureCount).toBe(0);
        expect(result.textureDecodeDiagnostics).toEqual([]);
        expect(result.textureDecodeReport).toMatchObject({
          mapCount: 1,
          uniqueHandleCount: 1,
          rawImageCount: 0,
          opaqueHandleCount: 0,
          cpuReadableCount: 1,
        });

        const primitive = result.scene.primitives[0] as MeshPrimitive;
        const ref = primitive.material.baseColorMap as TextureRef;
        const handle = ref.handle as {
          width: number;
          height: number;
          data: Float32Array;
          __vitrum_hint__: { colorSpace: string };
        };
        expect(handle.width).toBe(4);
        expect(handle.height).toBe(4);
        expect(handle.__vitrum_hint__.colorSpace).toBe('linear');
        expect(handle.data[0]).toBeCloseTo(srgbToLinearForTest(128 / 255));
        expect(handle.data[1]).toBeCloseTo(srgbToLinearForTest(64 / 255));
        expect(handle.data[2]).toBeCloseTo(1);
        expect(handle.data[3]).toBeCloseTo(128 / 255);
        expect(result.textureDecodeReport.entries).toEqual([
          expect.objectContaining({
            materialField: 'baseColorMap',
            handleKind: 'pixel-data',
            handleColorSpace: 'linear',
            width: 4,
            height: 4,
            textureIndex: 0,
            imageIndex: 0,
          }),
        ]);
      });
    });
  });

  it('loadGltfAndDecodeTextures decodes embedded PNG bytes in Node without a host pixel decoder', async () => {
    const png = makePngBytes(2, 1, [
      128, 64, 255, 128,
      255, 0, 0, 255,
    ]);
    const { gltf, buffers } = makeInlineTexturedGltf(png);

    const result = await loadGltfAndDecodeTextures(gltf, { buffers });

    expect(result.textureDecodeDiagnostics).toEqual([]);
    expect(result.decodedTextureCount).toBe(1);
    expect(result.unchangedTextureCount).toBe(0);
    expect(result.textureDecodeWarnings).toEqual([]);
    expect(result.textureDecodeReport).toMatchObject({
      mapCount: 1,
      uniqueHandleCount: 1,
      rawImageCount: 0,
      opaqueHandleCount: 0,
      cpuReadableCount: 1,
    });

    const primitive = result.scene.primitives[0] as MeshPrimitive;
    const ref = primitive.material.baseColorMap as TextureRef;
    const handle = ref.handle as {
      width: number;
      height: number;
      data: Float32Array;
      __vitrum_hint__: { channels: number; dataType: string; colorSpace: string };
    };
    expect(handle.width).toBe(2);
    expect(handle.height).toBe(1);
    expect(handle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    expect(handle.data[0]).toBeCloseTo(srgbToLinearForTest(128 / 255));
    expect(handle.data[1]).toBeCloseTo(srgbToLinearForTest(64 / 255));
    expect(handle.data[2]).toBeCloseTo(1);
    expect(handle.data[3]).toBeCloseTo(128 / 255);
    expect(handle.data[4]).toBeCloseTo(1);
    expect(handle.data[5]).toBeCloseTo(0);
    expect(handle.data[6]).toBeCloseTo(0);
    expect(handle.data[7]).toBeCloseTo(1);
    expect(result.textureDecodeReport.entries).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        handleKind: 'pixel-data',
        handleColorSpace: 'linear',
        width: 2,
        height: 1,
        isPowerOfTwo: true,
        textureIndex: 0,
        imageIndex: 0,
        imageMimeType: 'image/png',
        backendReadiness: {
          ptWebgl2: 'ready',
          ptWebgpu: 'ready',
          walkaroundHybrid: 'ready',
        },
      }),
    ]);
  });

  it('loadGltfAndDecodeTextures decodes embedded JPEG bytes in Node without a host pixel decoder', async () => {
    const jpegBytes = makeJpegBytes(2, 2, [
      200, 100, 50, 255,
      200, 100, 50, 255,
      200, 100, 50, 255,
      200, 100, 50, 255,
    ]);
    const { gltf, buffers } = makeInlineTexturedGltf(jpegBytes);
    gltf.images![0] = {
      ...gltf.images![0]!,
      mimeType: 'image/jpeg',
    };

    const result = await loadGltfAndDecodeTextures(gltf, { buffers });

    expect(result.textureDecodeDiagnostics).toEqual([]);
    expect(result.decodedTextureCount).toBe(1);
    expect(result.unchangedTextureCount).toBe(0);
    expect(result.textureDecodeWarnings).toEqual([]);

    const primitive = result.scene.primitives[0] as MeshPrimitive;
    const ref = primitive.material.baseColorMap as TextureRef;
    const handle = ref.handle as {
      width: number;
      height: number;
      data: Float32Array;
      __vitrum_hint__: { channels: number; dataType: string; colorSpace: string };
    };
    expect(handle.width).toBe(2);
    expect(handle.height).toBe(2);
    expect(handle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    expect(handle.data[0]).toBeCloseTo(srgbToLinearForTest(200 / 255), 1);
    expect(handle.data[1]).toBeCloseTo(srgbToLinearForTest(100 / 255), 1);
    expect(handle.data[2]).toBeCloseTo(srgbToLinearForTest(50 / 255), 1);
    expect(handle.data[3]).toBeCloseTo(1);
    expect(result.textureDecodeReport.entries).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        handleKind: 'pixel-data',
        handleColorSpace: 'linear',
        width: 2,
        height: 2,
        textureIndex: 0,
        imageIndex: 0,
        imageMimeType: 'image/jpeg',
        backendReadiness: {
          ptWebgl2: 'ready',
          ptWebgpu: 'ready',
          walkaroundHybrid: 'ready',
        },
      }),
    ]);
  });

  it('loadGltfAndDecodeTextures decodes embedded WebP bytes in Node without a host pixel decoder', async () => {
    const webpBytes = await makeWebpBytes(2, 2, [
      32, 160, 224, 255,
      32, 160, 224, 255,
      32, 160, 224, 255,
      32, 160, 224, 255,
    ]);
    const { gltf, buffers } = makeInlineTexturedGltf(webpBytes);
    gltf.images![0] = {
      ...gltf.images![0]!,
      mimeType: 'image/webp',
    };

    const result = await loadGltfAndDecodeTextures(gltf, { buffers });

    expect(result.textureDecodeDiagnostics).toEqual([]);
    expect(result.decodedTextureCount).toBe(1);
    expect(result.unchangedTextureCount).toBe(0);
    expect(result.textureDecodeWarnings).toEqual([]);

    const primitive = result.scene.primitives[0] as MeshPrimitive;
    const ref = primitive.material.baseColorMap as TextureRef;
    const handle = ref.handle as {
      width: number;
      height: number;
      data: Float32Array;
      __vitrum_hint__: { channels: number; dataType: string; colorSpace: string };
    };
    expect(handle.width).toBe(2);
    expect(handle.height).toBe(2);
    expect(handle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    expect(handle.data[0]).toBeCloseTo(srgbToLinearForTest(32 / 255), 0);
    expect(handle.data[1]).toBeCloseTo(srgbToLinearForTest(160 / 255), 0);
    expect(handle.data[2]).toBeCloseTo(srgbToLinearForTest(224 / 255), 0);
    expect(handle.data[3]).toBeCloseTo(1);
    expect(result.textureDecodeReport.entries).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        handleKind: 'pixel-data',
        handleColorSpace: 'linear',
        width: 2,
        height: 2,
        textureIndex: 0,
        imageIndex: 0,
        imageMimeType: 'image/webp',
        backendReadiness: {
          ptWebgl2: 'ready',
          ptWebgpu: 'ready',
          walkaroundHybrid: 'ready',
        },
      }),
    ]);
  });

  it('keeps raw images with a structured diagnostic when browser pixel readback is unavailable', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();

    await withCreateImageBitmapStub(async (createImageBitmap) => {
      const result = await loadGltfAndDecodeTextures(gltf, { buffers });

      expect(createImageBitmap).toHaveBeenCalledTimes(1);
      expect(result.decodedTextureCount).toBe(0);
      expect(result.unchangedTextureCount).toBe(1);
      expect(result.textureDecodeDiagnostics).toEqual([
        expect.objectContaining({
          code: 'platform-image-readback-unavailable',
          path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
          materialField: 'baseColorMap',
          handleKind: 'raw-image',
        }),
      ]);
      expect(result.textureDecodeReport).toMatchObject({
        mapCount: 1,
        rawImageCount: 1,
        cpuReadableCount: 0,
      });
    });
  });

  it('bakes spec-gloss alpha into a CPU-linear roughnessMap when decoding textures', async () => {
    const { gltf, buffers } = makeInlineSpecGlossTexturedGltf();
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => {
      expect(context).toMatchObject({
        materialField: 'specularColorMap',
        path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
        colorSpace: 'srgb',
      });
      return {
        width: 2,
        height: 1,
        data: new Uint8Array([
          255, 0, 0, 128,
          0, 255, 0, 64,
        ]),
        channels: 4 as const,
        dataType: 'uint8' as const,
        colorSpace: context.colorSpace,
      };
    });

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      decodePixels,
    });

    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(result.decodedTextureCount).toBe(2);
    expect(result.unchangedTextureCount).toBe(0);
    expect(result.textureDecodeDiagnostics).toEqual([]);
    expect(result.textureDecodeWarnings).toEqual([]);
    expect(result.textureDecodeReport.entries.map((entry) => entry.materialField).sort()).toEqual([
      'roughnessMap',
      'specularColorMap',
    ]);
    const compatibilityIssues = result.backendCompatibility.flatMap((candidate) => candidate.issues);
    expect(compatibilityIssues).not.toContainEqual(expect.objectContaining({
      name: 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha',
    }));

    const primitive = result.scene.primitives[0] as MeshPrimitive;
    const specular = primitive.material.specularColorMap as TextureRef;
    const roughness = primitive.material.roughnessMap as TextureRef;
    expect(roughness).toBeDefined();
    expect(roughness.handle).not.toBe(specular.handle);
    expect(roughness.texCoord).toBe(1);
    expect(roughness.transform).toEqual({
      offset: [0.25, 0.5],
      scale: [2, 3],
      rotation: 0.125,
    });
    expect(roughness.wrapS).toBe('clamp-to-edge');
    expect(roughness.wrapT).toBe('mirrored-repeat');

    const handle = roughness.handle as { width: number; height: number; data: Float32Array; __vitrum_hint__: unknown };
    expect(handle.width).toBe(2);
    expect(handle.height).toBe(1);
    expect(handle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    const first = 1 - 0.5 * (128 / 255);
    const second = 1 - 0.5 * (64 / 255);
    expect(Array.from(handle.data.slice(0, 4))).toEqual([
      expect.closeTo(first),
      expect.closeTo(first),
      expect.closeTo(first),
      1,
    ]);
    expect(Array.from(handle.data.slice(4, 8))).toEqual([
      expect.closeTo(second),
      expect.closeTo(second),
      expect.closeTo(second),
      1,
    ]);
  });

  it('keeps spec-gloss alpha degradation until every authored spec-gloss texture path is baked', async () => {
    const { gltf, buffers } = makeInlineSpecGlossTexturedGltf();
    gltf.meshes![0]!.primitives.push({ attributes: { POSITION: 0 }, material: 1 });
    gltf.materials!.push({
      extensions: {
        KHR_materials_pbrSpecularGlossiness: {
          diffuseFactor: [1, 1, 1, 1],
          specularFactor: [1, 1, 1],
          glossinessFactor: 0.25,
          specularGlossinessTexture: { index: 99 },
        },
      },
    });
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => ({
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 128]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: context.colorSpace,
    }));

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      decodePixels,
    });

    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(result.textureDecodeReport.entries).toEqual(expect.arrayContaining([
      expect.objectContaining({
        materialField: 'roughnessMap',
        path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
        handleKind: 'pixel-data',
        handleColorSpace: 'linear',
      }),
    ]));
    expect(result.textureDecodeReport.entries.some((entry) =>
      entry.materialField === 'roughnessMap' &&
      entry.path === 'materials[1].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture'
    )).toBe(false);
    expect(result.backendCompatibility.flatMap((candidate) => candidate.issues)).toContainEqual(
      expect.objectContaining({
        name: 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha',
      }),
    );
  });
});

describe('decodeSceneTextures', () => {
  it('bakes spec-gloss roughness from already CPU-readable pixel handles', async () => {
    const pixelHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 128]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'srgb' as const,
    };
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'spec-gloss-pixel-mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 0.75,
            metallic: 0,
            specularColorMap: {
              handle: pixelHandle,
              texCoord: 1,
              wrapS: 'clamp-to-edge',
            },
            extensions: {
              KHR_materials_pbrSpecularGlossiness: {
                glossinessFactor: 0.25,
                specularGlossinessTexture: { index: 0 },
              },
            },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const result = await decodeSceneTextures(scene, { target: 'cpu-linear' });

    expect(result.decodedCount).toBe(2);
    expect(result.unchangedCount).toBe(0);
    expect(result.diagnostics).toEqual([]);
    const material = (result.scene.primitives[0] as MeshPrimitive).material;
    const specular = material.specularColorMap as TextureRef;
    const specularHandle = specular.handle as {
      data: Float32Array;
      __vitrum_hint__: { colorSpace: string };
    };
    expect(specular.handle).not.toBe(pixelHandle);
    expect(specularHandle.__vitrum_hint__.colorSpace).toBe('linear');
    expect(specularHandle.data[0]).toBeCloseTo(1);
    const roughness = material.roughnessMap as TextureRef;
    expect(roughness.texCoord).toBe(1);
    expect(roughness.wrapS).toBe('clamp-to-edge');
    const handle = roughness.handle as { data: Float32Array };
    const expected = 1 - 0.25 * (128 / 255);
    expect(handle.data[0]).toBeCloseTo(expected);
    expect(handle.data[1]).toBeCloseTo(expected);
    expect(handle.data[2]).toBeCloseTo(expected);
    expect(handle.data[3]).toBe(1);
  });

  it('normalizes already CPU-readable texture refs to the requested CPU-linear target', async () => {
    const pixelHandle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 64, 255, 128]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'srgb' as const,
    };
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'cpu-readable-srgb-mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 1,
            metallic: 0,
            baseColorMap: { handle: pixelHandle },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const result = await decodeSceneTextures(scene, { target: 'cpu-linear' });

    expect(result.decodedCount).toBe(1);
    expect(result.unchangedCount).toBe(0);
    expect(result.diagnostics).toEqual([]);
    const material = (result.scene.primitives[0] as MeshPrimitive).material;
    const ref = material.baseColorMap as TextureRef;
    const handle = ref.handle as {
      width: number;
      height: number;
      data: Float32Array;
      __vitrum_hint__: { colorSpace: string };
    };
    expect(handle).not.toBe(pixelHandle);
    expect(handle.width).toBe(1);
    expect(handle.height).toBe(1);
    expect(handle.__vitrum_hint__.colorSpace).toBe('linear');
    expect(handle.data[0]).toBeCloseTo(srgbToLinearForTest(128 / 255));
    expect(handle.data[1]).toBeCloseTo(srgbToLinearForTest(64 / 255));
    expect(handle.data[2]).toBeCloseTo(1);
    expect(handle.data[3]).toBeCloseTo(128 / 255);
    expect(result.report.entries).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        colorSpace: 'srgb',
        handleColorSpace: 'linear',
        handleKind: 'pixel-data',
        backendReadiness: expect.objectContaining({
          ptWebgl2: 'ready',
          ptWebgpu: 'ready',
          walkaroundHybrid: 'ready',
        }),
      }),
    ]);
  });

  it('reports decoded lightMap handles as walkaround-ready', async () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'lightmap-mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 1,
            metallic: 0,
            lightMap: { handle: { kind: 'raw-image', uri: 'light.png' } },
            lightMapIntensity: 2,
            bumpMap: { handle: { kind: 'raw-image', uri: 'bump.png' } },
            bumpScale: 0.5,
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const result = await decodeSceneTextures(scene, {
      target: 'cpu-linear',
      decodePixels: () => ({
        width: 1,
        height: 1,
        data: new Uint8Array([64, 128, 255, 255]),
        channels: 4,
        dataType: 'uint8',
      }),
    });

    expect(result.report.entries).toHaveLength(2);
    expect(result.report.entries).toEqual(expect.arrayContaining([
      expect.objectContaining({
        primitiveId: 'lightmap-mesh',
        materialField: 'lightMap',
        colorSpace: 'linear',
        handleKind: 'pixel-data',
        backendReadiness: {
          ptWebgl2: 'ready',
          ptWebgpu: 'ready',
          walkaroundHybrid: 'ready',
        },
      }),
      expect.objectContaining({
        primitiveId: 'lightmap-mesh',
        materialField: 'bumpMap',
        colorSpace: 'linear',
        handleKind: 'pixel-data',
        backendReadiness: {
          ptWebgl2: 'ready',
          ptWebgpu: 'ready',
          walkaroundHybrid: 'ready',
        },
      }),
    ]));
  });

  it('normalizes raw-image texture refs to linear CPU pixel handles with field color-space policy', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.materials![0] = {
      ...gltf.materials![0]!,
      normalTexture: { index: 0 },
    };
    const asset = await loadGltfAsset(gltf, { buffers });
    const decoderColorSpaces: string[] = [];
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => {
      decoderColorSpaces.push(context.colorSpace);
      return {
        width: 1,
        height: 1,
        data: new Uint8Array([128, 64, 255, 128]),
        channels: 4 as const,
        dataType: 'uint8' as const,
      };
    });

    const result = await decodeSceneTextures(asset.scene, {
      target: 'cpu-linear',
      decodePixels,
    });

    expect(decodePixels).toHaveBeenCalledTimes(2);
    expect(decoderColorSpaces).toEqual(['srgb', 'linear']);
    expect(result.decodedCount).toBe(2);
    expect(result.unchangedCount).toBe(0);
    expect(result.diagnostics).toEqual([]);
    expect(result.warnings).toEqual([]);

    const before = asset.scene.primitives[0] as MeshPrimitive;
    const after = result.scene.primitives[0] as MeshPrimitive;
    const baseColor = after.material.baseColorMap as TextureRef;
    const normal = after.material.normalMap as TextureRef;
    const baseHandle = baseColor.handle as { data: Float32Array; __vitrum_hint__: { colorSpace: string } };
    const normalHandle = normal.handle as { data: Float32Array; __vitrum_hint__: { colorSpace: string } };

    expect(baseColor.handle).not.toBe((before.material.baseColorMap as TextureRef).handle);
    expect(baseHandle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    expect(baseHandle.data[0]).toBeCloseTo(srgbToLinearForTest(128 / 255));
    expect(baseHandle.data[1]).toBeCloseTo(srgbToLinearForTest(64 / 255));
    expect(baseHandle.data[2]).toBeCloseTo(1);
    expect(baseHandle.data[3]).toBeCloseTo(128 / 255);
    expect(normalHandle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    expect(normalHandle.data[0]).toBeCloseTo(128 / 255);
    expect(normalHandle.data[1]).toBeCloseTo(64 / 255);
    expect(normalHandle.data[2]).toBeCloseTo(1);
    expect(normalHandle.data[3]).toBeCloseTo(128 / 255);
    expect(result.report.entries).toEqual(expect.arrayContaining([
      expect.objectContaining({
        materialField: 'baseColorMap',
        colorSpace: 'srgb',
        handleColorSpace: 'linear',
        handleKind: 'pixel-data',
        backendReadiness: expect.objectContaining({
          ptWebgl2: 'ready',
          ptWebgpu: 'ready',
          walkaroundHybrid: 'ready',
        }),
      }),
      expect.objectContaining({
        materialField: 'normalMap',
        colorSpace: 'linear',
        handleColorSpace: 'linear',
        handleKind: 'pixel-data',
        backendReadiness: expect.objectContaining({
          ptWebgl2: 'ready',
          ptWebgpu: 'ready',
          walkaroundHybrid: 'ready',
        }),
      }),
    ]));
  });

  it('warns and preserves raw-image texture refs when no CPU decode hook is supplied', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const asset = await loadGltfAsset(gltf, { buffers });
    const warnings: string[] = [];

    const result = await decodeSceneTextures(asset.scene, {
      target: 'cpu-linear',
      onWarning: (message) => warnings.push(message),
    });

    const before = asset.scene.primitives[0] as MeshPrimitive;
    const after = result.scene.primitives[0] as MeshPrimitive;
    expect(result.decodedCount).toBe(0);
    expect(result.unchangedCount).toBe(1);
    expect(result.warnings).toEqual(warnings);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        severity: 'warning',
        code: 'raw-image-decoder-missing',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        materialField: 'baseColorMap',
        primitiveId: 'gltf-prim-0',
        primitiveIndex: 0,
        handleKind: 'raw-image',
      }),
    ]);
    expect(warnings[0]).toContain('materials[0].pbrMetallicRoughness.baseColorTexture');
    expect((after.material.baseColorMap as TextureRef).handle).toBe((before.material.baseColorMap as TextureRef).handle);
    expect(result.report.rawImageCount).toBe(1);
  });

  it('guards throwing texture decode diagnostic callbacks', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const asset = await loadGltfAsset(gltf, { buffers });
    const onDiagnostic = vi.fn(() => {
      throw new Error('host diagnostic callback failed');
    });
    const onWarning = vi.fn(() => {
      throw new Error('host warning callback failed');
    });

    const result = await decodeSceneTextures(asset.scene, {
      target: 'cpu-linear',
      onDiagnostic,
      onWarning,
    });

    expect(onDiagnostic).toHaveBeenCalledTimes(1);
    expect(onWarning).toHaveBeenCalledTimes(1);
    expect(result.decodedCount).toBe(0);
    expect(result.unchangedCount).toBe(1);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        severity: 'warning',
        code: 'raw-image-decoder-missing',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        materialField: 'baseColorMap',
      }),
    ]);
  });

  it('reports selected compressed texture-source provenance when no pixel decoder is supplied', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const ktx = bytes([0xab, 0x4b, 0x54, 0x58, 0x20]);
    const bufferIndex = gltf.buffers!.length;
    const bufferViewIndex = gltf.bufferViews!.length;
    const imageIndex = gltf.images!.length;
    buffers.set(bufferIndex, ktx);
    gltf.buffers!.push({ byteLength: ktx.byteLength });
    gltf.bufferViews!.push({ buffer: bufferIndex, byteOffset: 0, byteLength: ktx.byteLength });
    gltf.images!.push({ bufferView: bufferViewIndex, mimeType: 'image/ktx2' });
    gltf.extensionsUsed = ['KHR_texture_basisu'];
    gltf.textures![0] = {
      ...gltf.textures![0]!,
      extensions: { KHR_texture_basisu: { source: imageIndex } },
    };

    const asset = await loadGltfAsset(gltf, {
      buffers,
      textureSourceExtensions: ['KHR_texture_basisu'],
    });
    const result = await decodeSceneTextures(asset.scene, { target: 'cpu-linear' });

    expect(result.decodedCount).toBe(0);
    expect(result.unchangedCount).toBe(1);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        severity: 'warning',
        code: 'raw-image-decoder-missing',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        materialField: 'baseColorMap',
        primitiveId: 'gltf-prim-0',
        primitiveIndex: 0,
        handleKind: 'raw-image',
        textureIndex: 0,
        imageIndex,
        imageMimeType: 'image/ktx2',
        textureSourceExtension: 'KHR_texture_basisu',
        message: expect.stringContaining('KHR_texture_basisu'),
      }),
    ]);
  });

  it('resizes decoded textures to maxTextureSize before backend upload', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const asset = await loadGltfAsset(gltf, { buffers });
    const diagnostics: unknown[] = [];
    const pixels = new Float32Array(4 * 2 * 4);
    for (let p = 0; p < 8; p += 1) {
      pixels[p * 4] = p / 10;
      pixels[p * 4 + 1] = 0.25;
      pixels[p * 4 + 2] = 0.5;
      pixels[p * 4 + 3] = 1;
    }

    const result = await decodeSceneTextures(asset.scene, {
      target: 'cpu-linear',
      maxTextureSize: 2,
      decodePixels: () => ({
        width: 4,
        height: 2,
        data: pixels,
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      }),
      onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
    });

    expect(result.decodedCount).toBe(1);
    expect(result.diagnostics).toEqual(diagnostics);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        severity: 'warning',
        code: 'decoded-texture-exceeds-max-size',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        materialField: 'baseColorMap',
        primitiveId: 'gltf-prim-0',
        primitiveIndex: 0,
        width: 4,
        height: 2,
        maxTextureSize: 2,
        resizedWidth: 2,
        resizedHeight: 1,
      }),
    ]);
    expect(result.warnings).toHaveLength(1);
    expect(result.warnings[0]).toContain('exceeds maxTextureSize=2');
    expect(result.warnings[0]).toContain('resized to 2x1');

    const primitive = result.scene.primitives[0] as MeshPrimitive;
    const handle = (primitive.material.baseColorMap as TextureRef).handle as {
      width: number;
      height: number;
      data: Float32Array;
    };
    expect(handle.width).toBe(2);
    expect(handle.height).toBe(1);
    expect(handle.data[0]).toBeCloseTo(0);
    expect(handle.data[1]).toBeCloseTo(0.25);
    expect(handle.data[2]).toBeCloseTo(0.5);
    expect(handle.data[3]).toBeCloseTo(1);
    expect(handle.data[4]).toBeCloseTo(0.2);
    expect(handle.data[5]).toBeCloseTo(0.25);
    expect(handle.data[6]).toBeCloseTo(0.5);
    expect(handle.data[7]).toBeCloseTo(1);
    expect(result.report.entries).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        width: 2,
        height: 1,
        isPowerOfTwo: true,
        originalWidth: 4,
        originalHeight: 2,
        wasResized: true,
        maxTextureSize: 2,
        textureIndex: 0,
        imageIndex: 0,
      }),
    ]);
  });

  it('applies maxTextureSize diagnostics to already CPU-readable handles', async () => {
    const pixels = new Float32Array(4 * 2 * 4);
    for (let p = 0; p < 8; p += 1) {
      pixels[p * 4] = p / 10;
      pixels[p * 4 + 1] = 0.25;
      pixels[p * 4 + 2] = 0.5;
      pixels[p * 4 + 3] = 1;
    }
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'cpu-readable-large-mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 1,
            metallic: 0,
            normalMap: {
              handle: {
                width: 4,
                height: 2,
                data: pixels,
                channels: 4,
                dataType: 'float32',
                colorSpace: 'linear',
              },
            },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const result = await decodeSceneTextures(scene, {
      target: 'cpu-linear',
      maxTextureSize: 2,
    });

    expect(result.decodedCount).toBe(1);
    expect(result.unchangedCount).toBe(0);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        severity: 'warning',
        code: 'decoded-texture-exceeds-max-size',
        path: 'scene.primitives[0].material.normalMap',
        materialField: 'normalMap',
        primitiveId: 'cpu-readable-large-mesh',
        primitiveIndex: 0,
        width: 4,
        height: 2,
        maxTextureSize: 2,
        resizedWidth: 2,
        resizedHeight: 1,
      }),
    ]);
    const material = (result.scene.primitives[0] as MeshPrimitive).material;
    const handle = (material.normalMap as TextureRef).handle as {
      width: number;
      height: number;
      data: Float32Array;
      __vitrum_hint__: {
        originalWidth?: number;
        originalHeight?: number;
        maxTextureSize?: number;
      };
    };
    expect(handle.width).toBe(2);
    expect(handle.height).toBe(1);
    expect(handle.__vitrum_hint__).toMatchObject({
      originalWidth: 4,
      originalHeight: 2,
      maxTextureSize: 2,
    });
  });

  it('emits structured NPOT-repeat diagnostics after host decoding', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const asset = await loadGltfAsset(gltf, { buffers });

    const result = await decodeSceneTextures(asset.scene, {
      target: 'cpu-linear',
      warnOnNpotRepeatWrap: true,
      decodePixels: () => ({
        width: 3,
        height: 5,
        data: new Float32Array(3 * 5 * 4).fill(1),
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      }),
    });

    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        severity: 'warning',
        code: 'decoded-texture-npot-repeat-wrap',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        materialField: 'baseColorMap',
        primitiveId: 'gltf-prim-0',
        primitiveIndex: 0,
        width: 3,
        height: 5,
        wrapS: 'repeat',
        wrapT: 'repeat',
      }),
    ]);
    expect(result.warnings).toEqual([
      expect.stringContaining('NPOT 3x5'),
    ]);
  });

  it('promotes NPOT repeat-wrap decode diagnostics into degraded compatibility issues', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      warnOnNpotRepeatWrap: true,
      decodePixels: () => ({
        width: 3,
        height: 5,
        data: new Float32Array(3 * 5 * 4).fill(1),
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      }),
    });

    expect(result.textureDecodeDiagnostics).toEqual([
      expect.objectContaining({
        code: 'decoded-texture-npot-repeat-wrap',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        materialField: 'baseColorMap',
      }),
    ]);
    expect(result.backendCompatibility).toEqual(expect.arrayContaining([
      expect.objectContaining({
        backend: 'pt-webgl2',
        issues: expect.arrayContaining([
          expect.objectContaining({
            category: 'texture',
            name: 'texture-decode:decoded-texture-npot-repeat-wrap:baseColorMap',
            support: 'approximate',
            path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
          }),
        ]),
      }),
    ]));
  });

  it('emits NPOT-repeat diagnostics for already CPU-readable handles', async () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'cpu-readable-npot-mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 1,
            metallic: 0,
            metallicMap: {
              handle: {
                width: 3,
                height: 5,
                data: new Float32Array(3 * 5 * 4).fill(1),
                channels: 4,
                dataType: 'float32',
                colorSpace: 'linear',
              },
              wrapS: 'repeat',
              wrapT: 'mirrored-repeat',
            },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const result = await decodeSceneTextures(scene, {
      target: 'cpu-linear',
      warnOnNpotRepeatWrap: true,
    });

    expect(result.decodedCount).toBe(1);
    expect(result.unchangedCount).toBe(0);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        severity: 'warning',
        code: 'decoded-texture-npot-repeat-wrap',
        path: 'scene.primitives[0].material.metallicMap',
        materialField: 'metallicMap',
        primitiveId: 'cpu-readable-npot-mesh',
        primitiveIndex: 0,
        width: 3,
        height: 5,
        wrapS: 'repeat',
        wrapT: 'mirrored-repeat',
      }),
    ]);
  });

  it('decodes raw-image handles for the webgpu target while preserving backend color space', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const asset = await loadGltfAsset(gltf, { buffers });
    const decodePixels = vi.fn(() => ({
      width: 1,
      height: 1,
      data: new Float32Array([0.25, 0.5, 0.75, 1]),
      channels: 4 as const,
      dataType: 'float32' as const,
      colorSpace: 'linear' as const,
    }));

    const result = await decodeSceneTextures(asset.scene, {
      target: 'webgpu',
      decodePixels,
    });

    const before = asset.scene.primitives[0] as MeshPrimitive;
    const after = result.scene.primitives[0] as MeshPrimitive;
    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(result.decodedCount).toBe(1);
    expect(result.unchangedCount).toBe(0);
    expect(result.diagnostics).toEqual([]);
    expect(result.warnings).toEqual([]);
    const handle = (after.material.baseColorMap as TextureRef).handle as {
      width: number;
      height: number;
      data: Float32Array;
      __vitrum_hint__: { colorSpace: string };
    };
    expect(handle).not.toBe((before.material.baseColorMap as TextureRef).handle);
    expect(handle.width).toBe(1);
    expect(handle.height).toBe(1);
    expect(handle.__vitrum_hint__.colorSpace).toBe('srgb');
    expect(handle.data[0]).toBeCloseTo(linearToSrgbForTest(0.25));
    expect(handle.data[1]).toBeCloseTo(linearToSrgbForTest(0.5));
    expect(handle.data[2]).toBeCloseTo(linearToSrgbForTest(0.75));
    expect(handle.data[3]).toBeCloseTo(1);
    expect(result.report.cpuReadableCount).toBe(1);
    expect(result.report.rawImageCount).toBe(0);
    expect(result.report.entries).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        colorSpace: 'srgb',
        handleColorSpace: 'srgb',
        handleKind: 'pixel-data',
        backendReadiness: expect.objectContaining({
          ptWebgpu: 'ready',
        }),
      }),
    ]);
  });

  it('bakes spec-gloss alpha roughness for the webgpu texture target', async () => {
    const { gltf, buffers } = makeInlineSpecGlossTexturedGltf();
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => ({
      width: 2,
      height: 1,
      data: new Uint8Array([
        255, 0, 0, 128,
        0, 255, 0, 64,
      ]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: context.colorSpace,
    }));

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      textureTarget: 'webgpu',
      decodePixels,
    });

    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(result.decodedTextureCount).toBe(2);
    expect(result.unchangedTextureCount).toBe(0);
    expect(result.textureDecodeDiagnostics).toEqual([]);
    expect(result.textureDecodeWarnings).toEqual([]);
    expect(result.textureDecodeReport.entries).toEqual(expect.arrayContaining([
      expect.objectContaining({
        materialField: 'specularColorMap',
        handleColorSpace: 'srgb',
        handleKind: 'pixel-data',
        backendReadiness: expect.objectContaining({ ptWebgpu: 'ready' }),
      }),
      expect.objectContaining({
        materialField: 'roughnessMap',
        handleColorSpace: 'linear',
        handleKind: 'pixel-data',
        backendReadiness: expect.objectContaining({ ptWebgpu: 'ready' }),
      }),
    ]));

    const primitive = result.scene.primitives[0] as MeshPrimitive;
    const specular = primitive.material.specularColorMap as TextureRef;
    const roughness = primitive.material.roughnessMap as TextureRef;
    expect(roughness.handle).not.toBe(specular.handle);
    expect(roughness.texCoord).toBe(1);
    expect(roughness.transform).toEqual({
      offset: [0.25, 0.5],
      scale: [2, 3],
      rotation: 0.125,
    });
    const specularHandle = specular.handle as { __vitrum_hint__: { colorSpace: string } };
    const roughnessHandle = roughness.handle as {
      data: Float32Array;
      __vitrum_hint__: { colorSpace: string };
    };
    expect(specularHandle.__vitrum_hint__.colorSpace).toBe('srgb');
    expect(roughnessHandle.__vitrum_hint__.colorSpace).toBe('linear');

    const first = 1 - 0.5 * (128 / 255);
    const second = 1 - 0.5 * (64 / 255);
    expect(Array.from(roughnessHandle.data.slice(0, 4))).toEqual([
      expect.closeTo(first),
      expect.closeTo(first),
      expect.closeTo(first),
      1,
    ]);
    expect(Array.from(roughnessHandle.data.slice(4, 8))).toEqual([
      expect.closeTo(second),
      expect.closeTo(second),
      expect.closeTo(second),
      1,
    ]);
  });

  it('reports source provenance when spec-gloss alpha roughness baking cannot run', async () => {
    const { gltf, buffers } = makeInlineSpecGlossTexturedGltf();

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      textureTarget: 'webgpu',
      decodeImage: async () => ({ kind: 'decoded-texture' }),
    });

    const diagnostic = result.textureDecodeDiagnostics.find((entry) =>
      entry.code === 'spec-gloss-alpha-bake-unavailable'
    );
    expect(diagnostic).toEqual(expect.objectContaining({
      code: 'spec-gloss-alpha-bake-unavailable',
      path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
      materialField: 'roughnessMap',
      primitiveId: 'gltf-prim-0',
      primitiveIndex: 0,
      textureIndex: 0,
      imageIndex: 0,
      samplerIndex: 0,
      imageMimeType: 'image/png',
    }));
  });

  it('preserves resize provenance on generated spec-gloss roughness reports', async () => {
    const { gltf, buffers } = makeInlineSpecGlossTexturedGltf();
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => ({
      width: 4,
      height: 2,
      data: new Uint8Array(4 * 2 * 4).fill(255),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: context.colorSpace,
    }));

    const result = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      textureTarget: 'webgpu',
      decodePixels,
      maxTextureSize: 2,
    });

    const specularEntry = result.textureDecodeReport.entries.find((entry) =>
      entry.materialField === 'specularColorMap'
    );
    const roughnessEntry = result.textureDecodeReport.entries.find((entry) =>
      entry.materialField === 'roughnessMap'
    );
    expect(specularEntry).toMatchObject({
      path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
      textureIndex: 0,
      imageIndex: 0,
      samplerIndex: 0,
      colorSpace: 'srgb',
      width: 2,
      height: 1,
      originalWidth: 4,
      originalHeight: 2,
      wasResized: true,
      maxTextureSize: 2,
    });
    expect(roughnessEntry).toMatchObject({
      path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
      textureIndex: 0,
      imageIndex: 0,
      samplerIndex: 0,
      colorSpace: 'linear',
      width: 2,
      height: 1,
      originalWidth: 4,
      originalHeight: 2,
      wasResized: true,
      maxTextureSize: 2,
    });

    const primitive = result.scene.primitives[0] as MeshPrimitive;
    const roughness = primitive.material.roughnessMap as TextureRef;
    expect((roughness.handle as {
      __vitrum_hint__?: { originalWidth?: number; originalHeight?: number; maxTextureSize?: number };
    }).__vitrum_hint__).toMatchObject({
      originalWidth: 4,
      originalHeight: 2,
      maxTextureSize: 2,
    });
  });
});

describe('analyzeGltfAsset and compatibility ranking', () => {
  it('reports material fields, unsupported extensions, resources, and animation paths', () => {
    const gltf = makeExternalTexturedGltf();
    gltf.extensionsUsed = ['KHR_materials_unlit', 'KHR_materials_dispersion', 'KHR_materials_pbrSpecularGlossiness'];
    gltf.materials![0] = {
      ...gltf.materials![0]!,
      normalTexture: { index: 0, texCoord: 1 },
      extensions: {
        KHR_materials_unlit: {},
        KHR_materials_volume: {
          thicknessFactor: 0.5,
          thicknessTexture: { index: 0 },
        },
        KHR_materials_dispersion: {
          dispersion: 0.05,
        },
        KHR_materials_pbrSpecularGlossiness: {
          specularGlossinessTexture: { index: 0 },
        },
      },
    };
    gltf.accessors = [
      ...gltf.accessors!,
      { bufferView: 0, componentType: 5126, count: 2, type: 'SCALAR' },
      { bufferView: 0, componentType: 5126, count: 2, type: 'VEC3' },
    ];
    gltf.animations = [{
      samplers: [{ input: 2, output: 3, interpolation: 'BEZIER' as never }],
      channels: [
        { sampler: 0, target: { node: 0, path: 'translation' } },
        { sampler: 0, target: { node: 0, path: 'pointer' as never } },
      ],
    }];

    const report = analyzeGltfAsset(gltf);
    expect(report.extensions.supported).toContain('KHR_materials_unlit');
    expect(report.extensions.unsupportedOptional).not.toContain('KHR_materials_unlit');
    expect(report.materials.unsupportedKnownExtensions).not.toContain('KHR_materials_unlit');
    expect(report.materials.materialFields).toEqual(
      expect.arrayContaining([
        'baseColor',
        'baseColorMap',
        'dispersionAbbeNumber',
        'normalMap',
        'shadingModel',
        'thickness',
        'thicknessMap',
      ]),
    );
    expect(report.materials.textureFields).toEqual(
      expect.arrayContaining(['baseColorMap', 'normalMap', 'thicknessMap']),
    );
    expect(report.materials.uvSets).toEqual([0, 1]);
    expect(report.materials.volumeThicknessTextureCount).toBe(1);
    expect(report.materials.specularGlossinessMaterialCount).toBe(1);
    expect(report.materials.specularGlossinessTextureCount).toBe(1);
    expect(report.resources.externalBufferCount).toBe(1);
    expect(report.resources.externalImageCount).toBe(1);
    expect(report.animations.paths).toEqual(['pointer', 'translation']);
    expect(report.animations.unsupportedTargetPaths).toEqual(['pointer']);
    expect(report.animations.issuePaths).toEqual({
      'degradedInterpolation:BEZIER': ['animations[0].samplers[0].interpolation'],
      'unsupportedTargetPath:pointer': ['animations[0].channels[1].target.path'],
    });
    expect(report.animations.interpolations).toEqual(['BEZIER']);
    expect(report.animations.degradedInterpolations).toEqual(['BEZIER']);
    expect(report.animations.malformedChannels).toEqual([]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues.some((issue) =>
      issue.name === 'thicknessMap' &&
      issue.support === 'approximate',
    )).toBe(true);
    expect(compatibility.issues.some((issue) =>
      issue.category === 'material' &&
      issue.name === 'shadingModel' &&
      issue.support === 'approximate',
    )).toBe(true);
    expect(compatibility.issues.some((issue) =>
      issue.name === 'KHR_materials_pbrSpecularGlossiness' &&
      issue.support === 'approximate',
    )).toBe(true);
    expect(compatibility.issues.some((issue) =>
      issue.name === 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha' &&
      issue.support === 'approximate',
    )).toBe(true);
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'animation',
      name: 'target.path:pointer',
      support: 'unsupported',
      path: 'animations[0].channels[1].target.path',
    }));
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'animation',
      name: 'sampler.interpolation:BEZIER',
      support: 'approximate',
      path: 'animations[0].samplers[0].interpolation',
    }));
    const webgpuCompatibility = evaluateGltfBackendCompatibility(report, 'pt-webgpu');
    expect(webgpuCompatibility.issues.some((issue) =>
      issue.category === 'material' &&
      issue.name === 'shadingModel' &&
      issue.support === 'approximate',
    )).toBe(true);
    const walkaroundCompatibility = evaluateGltfBackendCompatibility(report, 'walkaround-hybrid');
    expect(walkaroundCompatibility.issues).toContainEqual(expect.objectContaining({
      category: 'material',
      name: 'dispersionAbbeNumber',
      support: 'unsupported',
      path: 'materials[0].extensions.KHR_materials_dispersion.dispersion',
    }));
  });

  it('reports malformed punctual-light references before skipped-emitter import fallback', () => {
    const { gltf } = makeInlineTriangleGltf();
    gltf.extensionsUsed = ['KHR_lights_punctual'];
    gltf.extensions = {
      KHR_lights_punctual: {
        lights: [
          { type: 'point' },
          { type: 'tube' },
          null,
        ],
      },
    } as never;
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      extensions: { KHR_lights_punctual: { light: 99 } },
    };
    gltf.nodes!.push({
      extensions: { KHR_lights_punctual: { light: 1 } },
    });
    gltf.nodes!.push({
      extensions: { KHR_lights_punctual: { light: 2 } },
    });
    gltf.scenes![0] = { nodes: [0, 1, 2] };

    const report = analyzeGltfAsset(gltf);

    expect(report.sceneGraph.punctualLightIssues).toEqual(expect.arrayContaining([
      {
        kind: 'missing-light',
        path: 'nodes[0].extensions.KHR_lights_punctual.light',
        nodeIndex: 0,
        lightIndex: 99,
      },
      {
        kind: 'unsupported-light-type',
        path: 'extensions.KHR_lights_punctual.lights[1].type',
        lightIndex: 1,
        lightType: 'tube',
      },
      {
        kind: 'unsupported-light-type',
        path: 'extensions.KHR_lights_punctual.lights[2].type',
        lightIndex: 2,
      },
    ]));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'scene',
        name: 'KHR_lights_punctual.missing-light',
        support: 'approximate',
        path: 'nodes[0].extensions.KHR_lights_punctual.light',
      }),
      expect.objectContaining({
        category: 'scene',
        name: 'KHR_lights_punctual.unsupported-light-type',
        support: 'approximate',
        path: 'extensions.KHR_lights_punctual.lights[1].type',
      }),
      expect.objectContaining({
        category: 'scene',
        name: 'KHR_lights_punctual.unsupported-light-type',
        support: 'approximate',
        path: 'extensions.KHR_lights_punctual.lights[2].type',
      }),
    ]));
  });

  it('reports malformed animation channels with compatibility source paths', () => {
    const gltf = makeExternalTexturedGltf();
    const timeAccessor = gltf.accessors!.length;
    const outputAccessor = timeAccessor + 1;
    const invalidTypeAccessor = timeAccessor + 2;
    const missingInputBufferViewAccessor = timeAccessor + 3;
    const missingOutputBufferAccessor = timeAccessor + 4;
    const invalidInputComponentAccessor = timeAccessor + 5;
    const invalidOutputComponentAccessor = timeAccessor + 6;
    const missingInputBufferAccessor = timeAccessor + 7;
    const missingOutputBufferViewAccessor = timeAccessor + 8;
    const missingAnimationBufferView = gltf.bufferViews!.length;
    gltf.bufferViews = [
      ...gltf.bufferViews!,
      { buffer: 99, byteOffset: 0, byteLength: 36 },
    ];
    gltf.accessors = [
      ...gltf.accessors!,
      { bufferView: 0, componentType: 5126, count: 2, type: 'SCALAR' },
      { bufferView: 0, componentType: 5126, count: 1, type: 'VEC3' },
      { bufferView: 0, componentType: 5126, count: 1, type: 'VEC9' as never },
      { bufferView: 99, componentType: 5126, count: 2, type: 'SCALAR' },
      { bufferView: missingAnimationBufferView, componentType: 5126, count: 1, type: 'VEC3' },
      { bufferView: 0, componentType: 999 as never, count: 2, type: 'SCALAR' },
      { bufferView: 0, componentType: 999 as never, count: 1, type: 'VEC3' },
      { bufferView: missingAnimationBufferView, componentType: 5126, count: 2, type: 'SCALAR' },
      { bufferView: 99, componentType: 5126, count: 1, type: 'VEC3' },
    ];
    gltf.animations = [{
      samplers: [
        { input: timeAccessor, output: outputAccessor },
        { input: 42, output: outputAccessor },
        { input: timeAccessor, output: 43 },
        { input: timeAccessor, output: invalidTypeAccessor },
        { input: invalidTypeAccessor, output: outputAccessor },
        { input: missingInputBufferViewAccessor, output: outputAccessor },
        { input: timeAccessor, output: missingOutputBufferAccessor },
        { input: invalidInputComponentAccessor, output: outputAccessor },
        { input: timeAccessor, output: invalidOutputComponentAccessor },
        { input: missingInputBufferAccessor, output: outputAccessor },
        { input: timeAccessor, output: missingOutputBufferViewAccessor },
      ],
      channels: [
        { sampler: 99, target: { node: 0, path: 'translation' } },
        { sampler: 0, target: { path: 'rotation' } },
        { sampler: 0, target: { node: 99, path: 'scale' } },
        { sampler: 0, target: { node: 0, path: 'translation' } },
        { sampler: 1, target: { node: 0, path: 'translation' } },
        { sampler: 2, target: { node: 0, path: 'translation' } },
        { sampler: 3, target: { node: 0, path: 'translation' } },
        { sampler: 4, target: { node: 0, path: 'translation' } },
        { sampler: 5, target: { node: 0, path: 'translation' } },
        { sampler: 6, target: { node: 0, path: 'translation' } },
        { sampler: 7, target: { node: 0, path: 'translation' } },
        { sampler: 8, target: { node: 0, path: 'translation' } },
        { sampler: 9, target: { node: 0, path: 'translation' } },
        { sampler: 10, target: { node: 0, path: 'translation' } },
      ],
    }];

    const report = analyzeGltfAsset(gltf);
    expect(report.animations.malformedChannels).toHaveLength(14);
    expect(report.animations.malformedChannels).toEqual(expect.arrayContaining([
      {
        kind: 'missing-target-node',
        path: 'animations[0].channels[1].target.node',
        animationIndex: 0,
        channelIndex: 1,
        targetPath: 'rotation',
      },
      {
        kind: 'target-node-not-found',
        path: 'animations[0].channels[2].target.node',
        animationIndex: 0,
        channelIndex: 2,
        targetPath: 'scale',
        nodeIndex: 99,
      },
      {
        kind: 'invalid-output-count',
        path: 'animations[0].channels[3].sampler',
        animationIndex: 0,
        channelIndex: 3,
        targetPath: 'translation',
        samplerIndex: 0,
        nodeIndex: 0,
        expectedOutputFloats: 6,
        actualOutputFloats: 3,
      },
      {
        kind: 'missing-input-accessor',
        path: 'animations[0].samplers[1].input',
        animationIndex: 0,
        channelIndex: 4,
        targetPath: 'translation',
        samplerIndex: 1,
        nodeIndex: 0,
        accessorIndex: 42,
        accessorRole: 'input',
      },
      {
        kind: 'missing-output-accessor',
        path: 'animations[0].samplers[2].output',
        animationIndex: 0,
        channelIndex: 5,
        targetPath: 'translation',
        samplerIndex: 2,
        nodeIndex: 0,
        accessorIndex: 43,
        accessorRole: 'output',
      },
      {
        kind: 'invalid-output-accessor-type',
        path: 'animations[0].samplers[3].output',
        animationIndex: 0,
        channelIndex: 6,
        targetPath: 'translation',
        samplerIndex: 3,
        nodeIndex: 0,
        accessorIndex: invalidTypeAccessor,
        accessorRole: 'output',
        accessorType: 'VEC9',
      },
      {
        kind: 'invalid-input-accessor-type',
        path: 'animations[0].samplers[4].input',
        animationIndex: 0,
        channelIndex: 7,
        targetPath: 'translation',
        samplerIndex: 4,
        nodeIndex: 0,
        accessorIndex: invalidTypeAccessor,
        accessorRole: 'input',
        accessorType: 'VEC9',
      },
      {
        kind: 'missing-input-buffer-view',
        path: `accessors[${missingInputBufferViewAccessor}].bufferView`,
        animationIndex: 0,
        channelIndex: 8,
        targetPath: 'translation',
        samplerIndex: 5,
        nodeIndex: 0,
        accessorIndex: missingInputBufferViewAccessor,
        accessorRole: 'input',
        bufferViewIndex: 99,
      },
      {
        kind: 'missing-output-buffer',
        path: `bufferViews[${missingAnimationBufferView}].buffer`,
        animationIndex: 0,
        channelIndex: 9,
        targetPath: 'translation',
        samplerIndex: 6,
        nodeIndex: 0,
        accessorIndex: missingOutputBufferAccessor,
        accessorRole: 'output',
        bufferViewIndex: missingAnimationBufferView,
        bufferIndex: 99,
      },
      {
        kind: 'invalid-input-accessor-component-type',
        path: `accessors[${invalidInputComponentAccessor}].componentType`,
        animationIndex: 0,
        channelIndex: 10,
        targetPath: 'translation',
        samplerIndex: 7,
        nodeIndex: 0,
        accessorIndex: invalidInputComponentAccessor,
        accessorRole: 'input',
        componentType: 999,
      },
      {
        kind: 'invalid-output-accessor-component-type',
        path: `accessors[${invalidOutputComponentAccessor}].componentType`,
        animationIndex: 0,
        channelIndex: 11,
        targetPath: 'translation',
        samplerIndex: 8,
        nodeIndex: 0,
        accessorIndex: invalidOutputComponentAccessor,
        accessorRole: 'output',
        componentType: 999,
      },
      {
        kind: 'missing-input-buffer',
        path: `bufferViews[${missingAnimationBufferView}].buffer`,
        animationIndex: 0,
        channelIndex: 12,
        targetPath: 'translation',
        samplerIndex: 9,
        nodeIndex: 0,
        accessorIndex: missingInputBufferAccessor,
        accessorRole: 'input',
        bufferViewIndex: missingAnimationBufferView,
        bufferIndex: 99,
      },
      {
        kind: 'missing-output-buffer-view',
        path: `accessors[${missingOutputBufferViewAccessor}].bufferView`,
        animationIndex: 0,
        channelIndex: 13,
        targetPath: 'translation',
        samplerIndex: 10,
        nodeIndex: 0,
        accessorIndex: missingOutputBufferViewAccessor,
        accessorRole: 'output',
        bufferViewIndex: 99,
      },
      {
        kind: 'missing-sampler',
        path: 'animations[0].samplers[99]',
        animationIndex: 0,
        channelIndex: 0,
        targetPath: 'translation',
        samplerIndex: 99,
        nodeIndex: 0,
      },
    ]));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'animation',
        name: 'channel.missing-target-node',
        support: 'unsupported',
        path: 'animations[0].channels[1].target.node',
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.target-node-not-found',
        support: 'unsupported',
        path: 'animations[0].channels[2].target.node',
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.invalid-output-count',
        support: 'unsupported',
        path: 'animations[0].channels[3].sampler',
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.missing-sampler',
        support: 'unsupported',
        path: 'animations[0].samplers[99]',
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.missing-input-accessor',
        support: 'unsupported',
        path: 'animations[0].samplers[1].input',
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.missing-output-accessor',
        support: 'unsupported',
        path: 'animations[0].samplers[2].output',
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.invalid-output-accessor-type',
        support: 'unsupported',
        path: 'animations[0].samplers[3].output',
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.invalid-input-accessor-type',
        support: 'unsupported',
        path: 'animations[0].samplers[4].input',
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.missing-input-buffer-view',
        support: 'unsupported',
        path: `accessors[${missingInputBufferViewAccessor}].bufferView`,
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.missing-output-buffer',
        support: 'unsupported',
        path: `bufferViews[${missingAnimationBufferView}].buffer`,
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.missing-input-buffer',
        support: 'unsupported',
        path: `bufferViews[${missingAnimationBufferView}].buffer`,
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.missing-output-buffer-view',
        support: 'unsupported',
        path: `accessors[${missingOutputBufferViewAccessor}].bufferView`,
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.invalid-input-accessor-component-type',
        support: 'unsupported',
        path: `accessors[${invalidInputComponentAccessor}].componentType`,
      }),
      expect.objectContaining({
        category: 'animation',
        name: 'channel.invalid-output-accessor-component-type',
        support: 'unsupported',
        path: `accessors[${invalidOutputComponentAccessor}].componentType`,
      }),
    ]));
  });

  it('treats supported KHR_animation_pointer material-factor channels as importable', () => {
    const { gltf } = makeInlineTriangleGltf();
    gltf.extensionsUsed = ['KHR_animation_pointer'];
    gltf.extensionsRequired = ['KHR_animation_pointer'];
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      material: 0,
    };
    gltf.materials = [{
      pbrMetallicRoughness: {
        baseColorFactor: [1, 0, 0, 1],
      },
    }];

    const timeAccessor = gltf.accessors!.length;
    const outputAccessor = timeAccessor + 1;
    const badOutputAccessor = timeAccessor + 2;
    const packed = concatArrayBuffers([
      f32Buffer([0, 1]),
      f32Buffer([1, 0, 0, 1, 0, 1, 0, 1]),
      f32Buffer([0, 1]),
    ]);
    const bufferIndex = gltf.buffers!.length;
    const timeView = gltf.bufferViews!.length;
    const outputView = timeView + 1;
    const badOutputView = timeView + 2;
    gltf.buffers!.push({ byteLength: packed.byteLength });
    gltf.bufferViews!.push(
      { buffer: bufferIndex, byteOffset: 0, byteLength: 8 },
      { buffer: bufferIndex, byteOffset: 8, byteLength: 32 },
      { buffer: bufferIndex, byteOffset: 40, byteLength: 8 },
    );
    gltf.accessors!.push(
      { bufferView: timeView, componentType: 5126, count: 2, type: 'SCALAR' },
      { bufferView: outputView, componentType: 5126, count: 2, type: 'VEC4' },
      { bufferView: badOutputView, componentType: 5126, count: 2, type: 'SCALAR' },
    );
    gltf.animations = [{
      samplers: [
        { input: timeAccessor, output: outputAccessor },
        { input: timeAccessor, output: badOutputAccessor },
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
                pointer: '/materials/0/pbrMetallicRoughness/baseColorFactor',
              },
            },
          },
        },
        {
          sampler: 0,
          target: {
            path: 'pointer',
            extensions: {
              KHR_animation_pointer: {
                pointer: '/materials/0/unknownMutableProperty',
              },
            },
          },
        },
      ],
    }];

    const report = analyzeGltfAsset(gltf);
    expect(report.extensions.unsupportedRequired).not.toContain('KHR_animation_pointer');
    expect(report.extensions.supported).toContain('KHR_animation_pointer');
    expect(report.animations.paths).toEqual(['pointer']);
    expect(report.animations.unsupportedTargetPaths).toEqual(['pointer']);
    expect(report.animations.issuePaths).toMatchObject({
      'unsupportedTargetPath:pointer': [
        'animations[0].channels[2].target.extensions.KHR_animation_pointer.pointer',
      ],
    });
    expect(report.animations.malformedChannels).toContainEqual(expect.objectContaining({
      kind: 'invalid-output-count',
      path: 'animations[0].channels[1].sampler',
      expectedOutputFloats: 8,
      actualOutputFloats: 2,
    }));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'animation',
      name: 'target.path:pointer',
      support: 'unsupported',
      path: 'animations[0].channels[2].target.extensions.KHR_animation_pointer.pointer',
    }));
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'animation',
      name: 'channel.invalid-output-count',
      support: 'unsupported',
      path: 'animations[0].channels[1].sampler',
    }));
  });

  it('reports primitive import blockers with compatibility source paths', () => {
    const gltf = makeExternalTexturedGltf();
    const shortPositionAccessor = gltf.accessors!.length;
    const invalidTypeAccessor = shortPositionAccessor + 1;
    const invalidComponentAccessor = shortPositionAccessor + 2;
    const missingPositionBufferViewAccessor = shortPositionAccessor + 3;
    const missingBufferViewIndex = gltf.bufferViews!.length;
    const missingPositionBufferAccessor = shortPositionAccessor + 4;
    const invalidIndexAccessor = shortPositionAccessor + 5;
    const missingIndexBufferViewAccessor = shortPositionAccessor + 6;
    const missingIndexBufferAccessor = shortPositionAccessor + 7;
    const truncatedPositionAccessor = shortPositionAccessor + 8;
    const truncatedIndexAccessor = shortPositionAccessor + 9;
    const truncatedPositionBufferView = gltf.bufferViews!.length + 1;
    const truncatedIndexBufferView = truncatedPositionBufferView + 1;
    gltf.bufferViews = [
      ...gltf.bufferViews!,
      { buffer: 99, byteOffset: 0, byteLength: 36 },
      { buffer: 0, byteOffset: 0, byteLength: 8 },
      { buffer: 0, byteOffset: 0, byteLength: 4 },
    ];
    gltf.accessors = [
      ...gltf.accessors!,
      { bufferView: 0, componentType: 5126, count: 2, type: 'VEC3' },
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC9' as never },
      { bufferView: 0, componentType: 999 as never, count: 3, type: 'VEC3' },
      { bufferView: 99, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: missingBufferViewIndex, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 0, componentType: 5126, count: 3, type: 'SCALAR' },
      { bufferView: 99, componentType: 5123, count: 3, type: 'SCALAR' },
      { bufferView: missingBufferViewIndex, componentType: 5123, count: 3, type: 'SCALAR' },
      { bufferView: truncatedPositionBufferView, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: truncatedIndexBufferView, componentType: 5123, count: 3, type: 'SCALAR' },
    ];
    gltf.meshes![0]!.primitives = [
      { attributes: {}, material: 0 },
      { attributes: { POSITION: 99 }, material: 0 },
      { attributes: { POSITION: invalidTypeAccessor }, material: 0 },
      { attributes: { POSITION: invalidComponentAccessor }, material: 0 },
      { attributes: { POSITION: missingPositionBufferViewAccessor }, material: 0 },
      { attributes: { POSITION: missingPositionBufferAccessor }, material: 0 },
      { attributes: { POSITION: 0 }, indices: 99, material: 0 },
      { attributes: { POSITION: 0 }, indices: invalidIndexAccessor, material: 0 },
      { attributes: { POSITION: 0 }, indices: missingIndexBufferViewAccessor, material: 0 },
      { attributes: { POSITION: 0 }, indices: missingIndexBufferAccessor, material: 0 },
      { mode: 5, attributes: { POSITION: shortPositionAccessor }, material: 0 },
      { attributes: { POSITION: truncatedPositionAccessor }, material: 0 },
      { attributes: { POSITION: 0 }, indices: truncatedIndexAccessor, material: 0 },
    ];

    const report = analyzeGltfAsset(gltf);
    expect(report.primitives.malformedPrimitives).toHaveLength(13);
    expect(report.primitives.malformedPrimitives).toEqual(expect.arrayContaining([
      expect.objectContaining({
        kind: 'missing-position',
        path: 'meshes[0].primitives[0].attributes.POSITION',
      }),
      expect.objectContaining({
        kind: 'missing-position-accessor',
        path: 'meshes[0].primitives[1].attributes.POSITION',
        accessorIndex: 99,
      }),
      expect.objectContaining({
        kind: 'invalid-position-accessor-type',
        path: 'meshes[0].primitives[2].attributes.POSITION',
        accessorIndex: invalidTypeAccessor,
        accessorType: 'VEC9',
      }),
      expect.objectContaining({
        kind: 'invalid-position-accessor-component-type',
        path: 'meshes[0].primitives[3].attributes.POSITION',
        accessorIndex: invalidComponentAccessor,
        componentType: 999,
      }),
      expect.objectContaining({
        kind: 'missing-position-buffer-view',
        path: 'meshes[0].primitives[4].attributes.POSITION',
        accessorIndex: missingPositionBufferViewAccessor,
      }),
      expect.objectContaining({
        kind: 'missing-position-buffer',
        path: `bufferViews[${missingBufferViewIndex}].buffer`,
        accessorIndex: missingPositionBufferAccessor,
        bufferViewIndex: missingBufferViewIndex,
        bufferIndex: 99,
      }),
      expect.objectContaining({
        kind: 'missing-index-accessor',
        path: 'meshes[0].primitives[6].indices',
        accessorIndex: 99,
      }),
      expect.objectContaining({
        kind: 'invalid-index-accessor',
        path: 'meshes[0].primitives[7].indices',
        accessorIndex: invalidIndexAccessor,
        componentType: 5126,
      }),
      expect.objectContaining({
        kind: 'missing-index-buffer-view',
        path: 'meshes[0].primitives[8].indices',
        accessorIndex: missingIndexBufferViewAccessor,
      }),
      expect.objectContaining({
        kind: 'missing-index-buffer',
        path: `bufferViews[${missingBufferViewIndex}].buffer`,
        accessorIndex: missingIndexBufferAccessor,
        bufferViewIndex: missingBufferViewIndex,
        bufferIndex: 99,
      }),
      expect.objectContaining({
        kind: 'empty-triangulated-primitive',
        path: 'meshes[0].primitives[10]',
        mode: 5,
      }),
      expect.objectContaining({
        kind: 'truncated-position-buffer-view',
        path: `accessors[${truncatedPositionAccessor}].bufferView`,
        accessorIndex: truncatedPositionAccessor,
        bufferViewIndex: truncatedPositionBufferView,
        requiredByteLength: 36,
        byteLength: 8,
      }),
      expect.objectContaining({
        kind: 'truncated-index-buffer-view',
        path: `accessors[${truncatedIndexAccessor}].bufferView`,
        accessorIndex: truncatedIndexAccessor,
        bufferViewIndex: truncatedIndexBufferView,
        requiredByteLength: 6,
        byteLength: 4,
      }),
    ]));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'primitive',
        name: 'malformed.missing-position',
        support: 'unsupported',
        path: 'meshes[0].primitives[0].attributes.POSITION',
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'malformed.invalid-position-accessor-component-type',
        support: 'unsupported',
        path: 'meshes[0].primitives[3].attributes.POSITION',
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'malformed.missing-index-buffer-view',
        support: 'unsupported',
        path: 'meshes[0].primitives[8].indices',
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'malformed.missing-position-buffer',
        support: 'unsupported',
        path: `bufferViews[${missingBufferViewIndex}].buffer`,
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'malformed.missing-index-buffer',
        support: 'unsupported',
        path: `bufferViews[${missingBufferViewIndex}].buffer`,
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'malformed.empty-triangulated-primitive',
        support: 'unsupported',
        path: 'meshes[0].primitives[10]',
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'malformed.truncated-position-buffer-view',
        support: 'unsupported',
        path: `accessors[${truncatedPositionAccessor}].bufferView`,
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'malformed.truncated-index-buffer-view',
        support: 'unsupported',
        path: `accessors[${truncatedIndexAccessor}].bufferView`,
      }),
    ]));
  });

  it('reports malformed texture sampler policies with compatibility source paths', () => {
    const gltf = makeExternalTexturedGltf();
    gltf.materials![0] = {
      ...gltf.materials![0]!,
      normalTexture: { index: 1 },
    };
    gltf.textures = [
      { source: 0, sampler: 0 },
      { source: 0, sampler: 7 },
    ];
    gltf.samplers = [{
      wrapS: 123 as never,
      wrapT: 456 as never,
      magFilter: 111 as never,
      minFilter: 222 as never,
    }];

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.malformedSamplerPolicies).toEqual(expect.arrayContaining([
      expect.objectContaining({
        kind: 'invalid-wrap-s',
        materialField: 'baseColorMap',
        textureIndex: 0,
        samplerIndex: 0,
        path: 'samplers[0].wrapS',
        value: 123,
      }),
      expect.objectContaining({
        kind: 'invalid-wrap-t',
        materialField: 'baseColorMap',
        textureIndex: 0,
        samplerIndex: 0,
        path: 'samplers[0].wrapT',
        value: 456,
      }),
      expect.objectContaining({
        kind: 'invalid-mag-filter',
        materialField: 'baseColorMap',
        textureIndex: 0,
        samplerIndex: 0,
        path: 'samplers[0].magFilter',
        value: 111,
      }),
      expect.objectContaining({
        kind: 'invalid-min-filter',
        materialField: 'baseColorMap',
        textureIndex: 0,
        samplerIndex: 0,
        path: 'samplers[0].minFilter',
        value: 222,
      }),
      expect.objectContaining({
        kind: 'missing-sampler',
        materialField: 'normalMap',
        textureIndex: 1,
        samplerIndex: 7,
        path: 'textures[1].sampler',
      }),
    ]));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'material',
        name: 'baseColorMap.samplerPolicy.invalid-wrap-s',
        support: 'approximate',
        path: 'samplers[0].wrapS',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'baseColorMap.samplerPolicy.invalid-min-filter',
        support: 'approximate',
        path: 'samplers[0].minFilter',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'normalMap.samplerPolicy.missing-sampler',
        support: 'approximate',
        path: 'textures[1].sampler',
      }),
    ]));
  });

  it('reports malformed material texture references with compatibility source paths', () => {
    const gltf = makeExternalTexturedGltf();
    const unavailableImageBufferView = gltf.bufferViews!.length;
    gltf.bufferViews = [
      ...gltf.bufferViews!,
      { buffer: 99, byteOffset: 0, byteLength: 4 },
    ];
    gltf.materials![0] = {
      pbrMetallicRoughness: {
        baseColorTexture: { index: 99 },
        metallicRoughnessTexture: { index: 1 },
      },
      normalTexture: { index: 2 },
      occlusionTexture: { index: 3 },
      emissiveTexture: { index: 4 },
      extensions: {
        KHR_materials_specular: {
          specularTexture: { index: 5 },
        },
      },
    };
    gltf.textures = [
      { source: 0 },
      {},
      { source: 99 },
      { source: 1 },
      { source: 2 },
      { source: 3 },
    ];
    gltf.images = [
      { uri: 'textures/albedo.png', mimeType: 'image/png' },
      { mimeType: 'image/png' },
      { bufferView: 99, mimeType: 'image/png' },
      { bufferView: unavailableImageBufferView, mimeType: 'image/png' },
    ];

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.textureReferenceIssues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        kind: 'missing-texture',
        materialField: 'baseColorMap',
        textureIndex: 99,
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture.index',
      }),
      expect.objectContaining({
        kind: 'missing-texture-source',
        materialField: 'roughnessMap',
        textureIndex: 1,
        path: 'textures[1].source',
      }),
      expect.objectContaining({
        kind: 'missing-image',
        materialField: 'normalMap',
        textureIndex: 2,
        imageIndex: 99,
        path: 'textures[2].source',
      }),
      expect.objectContaining({
        kind: 'missing-image-source',
        materialField: 'aoMap',
        textureIndex: 3,
        imageIndex: 1,
        path: 'images[1]',
      }),
      expect.objectContaining({
        kind: 'missing-image-buffer-view',
        materialField: 'emissiveMap',
        textureIndex: 4,
        imageIndex: 2,
        bufferViewIndex: 99,
        path: 'images[2].bufferView',
      }),
      expect.objectContaining({
        kind: 'image-buffer-unavailable',
        materialField: 'specularIntensityMap',
        textureIndex: 5,
        imageIndex: 3,
        bufferViewIndex: unavailableImageBufferView,
        bufferIndex: 99,
        path: `bufferViews[${unavailableImageBufferView}].buffer`,
      }),
    ]));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'material',
        name: 'baseColorMap.textureRef.missing-texture',
        support: 'approximate',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture.index',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'roughnessMap.textureRef.missing-texture-source',
        support: 'approximate',
        path: 'textures[1].source',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'normalMap.textureRef.missing-image',
        support: 'approximate',
        path: 'textures[2].source',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'aoMap.textureRef.missing-image-source',
        support: 'approximate',
        path: 'images[1]',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'emissiveMap.textureRef.missing-image-buffer-view',
        support: 'approximate',
        path: 'images[2].bufferView',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'specularIntensityMap.textureRef.image-buffer-unavailable',
        support: 'approximate',
        path: `bufferViews[${unavailableImageBufferView}].buffer`,
      }),
    ]));
  });

  it('reports extension-only material texture sources as host-hook requirements', () => {
    const gltf = makeExternalTexturedGltf();
    gltf.extensionsUsed = ['EXT_texture_webp'];
    gltf.textures = [{
      extensions: { EXT_texture_webp: { source: 0 } },
    }];

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.textureReferenceIssues).toEqual([expect.objectContaining({
      kind: 'disabled-texture-source-extension',
      materialField: 'baseColorMap',
      textureIndex: 0,
      path: 'textures[0].extensions.EXT_texture_webp',
      textureSourceExtensions: ['EXT_texture_webp'],
    })]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'material',
        name: 'baseColorMap.textureRef.disabled-texture-source-extension',
        support: 'requires-hook',
        path: 'textures[0].extensions.EXT_texture_webp',
      }),
    ]));

    const selectedReport = analyzeGltfAsset(gltf, { textureSourceExtensions: ['EXT_texture_webp'] });
    expect(selectedReport.materials.textureReferenceIssues).toEqual([]);
  });

  it('uses the backend promise ledger to rank textured assets by fidelity tier', () => {
    const report = analyzeGltfAsset(makeExternalTexturedGltf());
    const ranked = rankGltfBackends(report, 'fidelity');
    const walkaround = evaluateGltfBackendCompatibility(report, 'walkaround-hybrid');

    expect(ranked[0]!.backend).toBe('pt-webgl2');
    expect(ranked.map((entry) => entry.profileId)).toEqual([
      'pt-webgl2',
      'pt-webgpu',
      'walkaround-hybrid',
      'pt-webgpu-lite',
    ]);
    expect(walkaround.issues.some((issue) =>
      issue.category === 'material' &&
      issue.name === 'baseColorMap' &&
      issue.support === 'approximate',
    )).toBe(true);
  });

  it('reports emissiveTexture emitter texel-PDF as an approximate compatibility edge', () => {
    const gltf = makeExternalTexturedGltf();
    gltf.materials![0] = {
      emissiveFactor: [1, 1, 1],
      emissiveTexture: { index: 0 },
    };

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.textureFields).toContain('emissiveMap');

    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    const webgl2Issue = webgl2.issues.find((issue) => issue.name === 'emissiveMap.texelPdf');
    expect(webgl2Issue).toEqual(expect.objectContaining({
      category: 'material',
      support: 'approximate',
      path: 'materials[0].emissiveTexture',
    }));
    expect(webgl2Issue?.message).toContain('GI/probe hit shading samples readable texels at the hit UV');
    expect(webgl2Issue?.message).toContain('Exact global texel-space emitter alias tables/PDFs');

    const webgpuFull = evaluateGltfBackendProfileCompatibility(report, 'pt-webgpu');
    expect(webgpuFull.issues).toContainEqual(expect.objectContaining({
      category: 'material',
      name: 'emissiveMap.texelPdf',
      support: 'approximate',
      path: 'materials[0].emissiveTexture',
    }));

    const walkaround = evaluateGltfBackendCompatibility(report, 'walkaround-hybrid');
    expect(walkaround.issues).toContainEqual(expect.objectContaining({
      category: 'material',
      name: 'emissiveMap.texelPdf',
      support: 'approximate',
      path: 'materials[0].emissiveTexture',
    }));

    const lite = evaluateGltfBackendProfileCompatibility(report, 'pt-webgpu-lite');
    expect(lite.issues.some((issue) => issue.name === 'emissiveMap.texelPdf')).toBe(false);
    expect(lite.issues).toContainEqual(expect.objectContaining({
      category: 'material',
      name: 'emissiveMap',
      support: 'unsupported',
      path: 'materials[0].emissiveTexture',
    }));
  });

  it('clears emissiveTexture texel-PDF degradation for PT backends after CPU texture decode', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.materials![0] = {
      emissiveFactor: [1, 1, 1],
      emissiveTexture: { index: 0 },
    };
    const preDecode = await loadGltfAsset(gltf, { buffers });
    const preDecodePtWebgl2 = preDecode.backendCompatibility.find((entry) =>
      entry.profileId === 'pt-webgl2'
    );
    expect(preDecodePtWebgl2?.issues.some((issue) => issue.name === 'emissiveMap.texelPdf')).toBe(true);

    const decodePixelsImpl: DecodeGltfTexturePixelsFn = () => ({
      width: 2,
      height: 2,
      data: new Uint8Array([
        255, 0, 0, 255,
        0, 255, 0, 255,
        0, 0, 255, 255,
        255, 255, 255, 255,
      ]),
      channels: 4,
      dataType: 'uint8',
      colorSpace: 'srgb',
    });
    const decodePixels = vi.fn(decodePixelsImpl);

    const decoded = await loadGltfAndDecodeTextures(gltf, { buffers, decodePixels });
    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(decoded.textureDecodeReport.entries).toContainEqual(expect.objectContaining({
      materialField: 'emissiveMap',
      handleKind: 'pixel-data',
      backendReadiness: {
        ptWebgl2: 'ready',
        ptWebgpu: 'ready',
        walkaroundHybrid: 'ready',
      },
    }));

    const ptWebgl2 = decoded.backendCompatibility.find((entry) => entry.profileId === 'pt-webgl2');
    const ptWebgpu = decoded.backendCompatibility.find((entry) => entry.profileId === 'pt-webgpu');
    const walkaround = decoded.backendCompatibility.find((entry) => entry.profileId === 'walkaround-hybrid');
    expect(ptWebgl2?.issues.some((issue) => issue.name === 'emissiveMap.texelPdf')).toBe(false);
    expect(ptWebgpu?.issues.some((issue) => issue.name === 'emissiveMap.texelPdf')).toBe(false);
    expect(walkaround?.issues).toContainEqual(expect.objectContaining({
      category: 'material',
      name: 'emissiveMap.texelPdf',
      support: 'approximate',
    }));
    expect((ptWebgl2?.approximateCount ?? 0)).toBeLessThan(preDecodePtWebgl2?.approximateCount ?? 0);
  });

  it('keeps emissiveTexture texel-PDF degradation for PT backends after sRGB webgpu-target decode', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.materials![0] = {
      emissiveFactor: [1, 1, 1],
      emissiveTexture: { index: 0 },
    };
    const decodePixels = vi.fn(() => ({
      width: 2,
      height: 2,
      data: new Uint8Array([
        255, 0, 0, 255,
        0, 255, 0, 255,
        0, 0, 255, 255,
        255, 255, 255, 255,
      ]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'srgb' as const,
    }));

    const decoded = await loadGltfAndDecodeTextures(gltf, {
      buffers,
      decodePixels,
      textureTarget: 'webgpu',
    });
    expect(decoded.textureDecodeReport.entries).toContainEqual(expect.objectContaining({
      materialField: 'emissiveMap',
      handleKind: 'pixel-data',
      handleColorSpace: 'srgb',
    }));

    const ptWebgl2 = decoded.backendCompatibility.find((entry) => entry.profileId === 'pt-webgl2');
    const ptWebgpu = decoded.backendCompatibility.find((entry) => entry.profileId === 'pt-webgpu');
    expect(ptWebgl2?.issues).toContainEqual(expect.objectContaining({
      name: 'emissiveMap.texelPdf',
      support: 'approximate',
    }));
    expect(ptWebgpu?.issues).toContainEqual(expect.objectContaining({
      name: 'emissiveMap.texelPdf',
      support: 'approximate',
    }));
  });

  it('reports material textures that require UV sets beyond the core Scene contract', () => {
    const gltf = makeExternalTexturedGltf();
    gltf.materials![0]!.pbrMetallicRoughness!.baseColorTexture = { index: 0, texCoord: 2 };

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.uvSets).toEqual([2]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    const uvIssue = compatibility.issues.find((issue) => issue.name === 'TEXCOORD_2');
    expect(uvIssue).toEqual(expect.objectContaining({
      category: 'material',
      support: 'unsupported',
      path: 'materials[0].pbrMetallicRoughness.baseColorTexture.texCoord',
    }));
    expect(uvIssue?.message).toContain('only UV sets 0 and 1');
  });

  it('does not reject a single high material UV set that can be remapped into uv1', () => {
    const gltf = makeExternalTexturedGltf();
    gltf.materials![0]!.pbrMetallicRoughness!.baseColorTexture = { index: 0, texCoord: 2 };
    gltf.meshes![0]!.primitives[0]!.attributes.TEXCOORD_2 = 1;

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.uvSets).toEqual([2]);
    expect(report.materials.unrepresentableUvSets).toEqual([]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues.some((issue) => issue.name === 'TEXCOORD_2')).toBe(false);
  });

  it('does not reject a variant material high UV set that can be remapped on its mapped primitive', () => {
    const gltf = makeExternalTexturedGltf();
    gltf.extensionsUsed = ['KHR_materials_variants'];
    gltf.extensions = {
      KHR_materials_variants: {
        variants: [{ name: 'uv2-variant' }],
      },
    };
    gltf.materials!.push({
      pbrMetallicRoughness: {
        baseColorTexture: { index: 0, texCoord: 2 },
      },
    });
    gltf.meshes![0]!.primitives[0]!.attributes.TEXCOORD_2 = 1;
    gltf.meshes![0]!.primitives[0]!.extensions = {
      KHR_materials_variants: {
        mappings: [{ material: 1, variants: [0] }],
      },
    };

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.uvSets).toEqual([0, 2]);
    expect(report.materials.unrepresentableUvSets).toEqual([]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues.some((issue) => issue.name === 'TEXCOORD_2')).toBe(false);
  });

  it('reports KHR_texture_transform texCoord overrides beyond uv1 at the override source path', () => {
    const gltf = makeExternalTexturedGltf();
    gltf.materials![0]!.pbrMetallicRoughness!.baseColorTexture = {
      index: 0,
      texCoord: 0,
      extensions: {
        KHR_texture_transform: {
          texCoord: 3,
          offset: [0, 0],
          scale: [1, 1],
        },
      },
    };

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.uvSets).toEqual([3]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'walkaround-hybrid');
    const uvIssue = compatibility.issues.find((issue) => issue.name === 'TEXCOORD_3');
    expect(uvIssue).toEqual(expect.objectContaining({
      category: 'material',
      support: 'unsupported',
      path: 'materials[0].pbrMetallicRoughness.baseColorTexture.extensions.KHR_texture_transform.texCoord',
    }));
  });

  it('scores pt-webgpu full and lite as distinct planner profiles', () => {
    const gltf = makeExternalTexturedGltf();
    gltf.materials![0] = {
      ...gltf.materials![0]!,
      alphaMode: 'BLEND',
      normalTexture: { index: 0, scale: 0.5 },
    };
    const report = analyzeGltfAsset(gltf);
    const full = evaluateGltfBackendCompatibility(report, 'pt-webgpu');
    const lite = evaluateGltfBackendProfileCompatibility(report, 'pt-webgpu-lite');

    expect(full.profileId).toBe('pt-webgpu');
    expect(full.traceTier).toBe('full');
    expect(lite.backend).toBe('pt-webgpu');
    expect(lite.profileId).toBe('pt-webgpu-lite');
    expect(lite.traceTier).toBe('lite');
    expect(full.issues.some((issue) =>
      issue.category === 'material' &&
      issue.name === 'baseColorMap' &&
      issue.support === 'unsupported',
    )).toBe(false);
    expect(lite.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'material',
        name: 'baseColorMap',
        support: 'unsupported',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'normalMap',
        support: 'unsupported',
        path: 'materials[0].normalTexture',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'alphaMode',
        support: 'unsupported',
        path: 'materials[0].alphaMode',
      }),
    ]));
    expect(lite.unsupportedCount).toBeGreaterThan(full.unsupportedCount);
  });

  it('keeps the full pt-webgpu profile ahead of lite for scalar-only assets', () => {
    const report = analyzeGltfAsset(makeInlineTriangleGltf().gltf);
    const ranked = rankGltfBackends(report, 'realtime');
    const webgpuRows = ranked.filter((entry) => entry.backend === 'pt-webgpu');

    expect(webgpuRows.map((entry) => entry.profileId)).toEqual(['pt-webgpu', 'pt-webgpu-lite']);
    expect(webgpuRows[0]!.unsupportedCount).toBe(0);
    expect(webgpuRows[1]!.unsupportedCount).toBe(0);
  });

  it('routes glTF dispersion materials away from walkaround in the planner', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.extensionsUsed = ['KHR_materials_dispersion'];
    gltf.materials = [{
      pbrMetallicRoughness: {
        baseColorFactor: [1, 1, 1, 1],
      },
      extensions: {
        KHR_materials_dispersion: {
          dispersion: 0.08,
        },
      },
    }];
    gltf.meshes![0]!.primitives[0]!.material = 0;

    const report = analyzeGltfAsset(gltf);
    expect(report.materials.materialFields).toContain('dispersionAbbeNumber');

    const realtimeRanked = rankGltfBackends(report, 'realtime');
    const walkaround = realtimeRanked.find((entry) => entry.profileId === 'walkaround-hybrid');
    expect(realtimeRanked[0]).toMatchObject({
      backend: 'pt-webgpu',
      profileId: 'pt-webgpu',
      unsupportedCount: 0,
    });
    expect(walkaround).toMatchObject({
      backend: 'walkaround-hybrid',
      unsupportedCount: 1,
      issues: expect.arrayContaining([
        expect.objectContaining({
          category: 'material',
          name: 'dispersionAbbeNumber',
          support: 'unsupported',
          path: 'materials[0].extensions.KHR_materials_dispersion.dispersion',
        }),
      ]),
    });

    let error: unknown;
    try {
      await loadGltfForEngine(gltf, {
        buffers,
        backend: 'walkaround-hybrid',
        compatibilityMode: 'reject-unsupported',
        createEngine: async () => ({ setScene: vi.fn() }),
      });
    } catch (err) {
      error = err;
    }

    expect(error).toBeInstanceOf(GltfCompatibilityError);
    expect((error as GltfCompatibilityError).failures).toContain(
      'material:dispersionAbbeNumber=unsupported at materials[0].extensions.KHR_materials_dispersion.dispersion',
    );
    expect((error as GltfCompatibilityError).failureDetails).toContainEqual(expect.objectContaining({
      source: 'compatibility-issue',
      category: 'material',
      name: 'dispersionAbbeNumber',
      support: 'unsupported',
      path: 'materials[0].extensions.KHR_materials_dispersion.dispersion',
    }));
  });

  it('reports malformed sparse POSITION storage before backend selection', () => {
    const { gltf } = makeInlineTriangleGltf();
    gltf.accessors![0] = {
      ...gltf.accessors![0]!,
      sparse: {
        count: 1,
        indices: { bufferView: 2, componentType: 5123 },
        values: { bufferView: 3 },
      },
    };
    gltf.bufferViews!.push({ buffer: 0, byteOffset: 0, byteLength: 2 });

    const report = analyzeGltfAsset(gltf);

    expect(report.primitives.malformedPrimitives).toContainEqual(expect.objectContaining({
      kind: 'invalid-position-sparse-accessor',
      path: 'accessors[0].sparse.values.bufferView',
      accessorIndex: 0,
      sparseIssueKind: 'missing-sparse-values-buffer-view',
      bufferViewIndex: 3,
    }));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'primitive',
      name: 'malformed.invalid-position-sparse-accessor',
      support: 'unsupported',
      path: 'accessors[0].sparse.values.bufferView',
    }));
  });

  it('reports malformed sparse storage on optional primitive streams as degraded import issues', () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addMorphedGpuInstancing(gltf, buffers);
    const inverseBindAccessor = gltf.accessors!.length;
    gltf.nodes![0] = { ...gltf.nodes![0]!, skin: 0 };
    gltf.nodes!.push({ name: 'joint' });
    gltf.skins = [{ joints: [1], inverseBindMatrices: inverseBindAccessor }];
    gltf.accessors!.push({
      bufferView: 0,
      componentType: 5126,
      count: 1,
      type: 'MAT4',
      sparse: {
        count: 1,
        indices: { bufferView: 0, componentType: 5123 },
        values: { bufferView: 102 },
      },
    });
    gltf.accessors![1] = {
      ...gltf.accessors![1]!,
      sparse: {
        count: 1,
        indices: { bufferView: 0, componentType: 5123 },
        values: { bufferView: 99 },
      },
    };
    gltf.accessors![2] = {
      ...gltf.accessors![2]!,
      sparse: {
        count: 1,
        indices: { bufferView: 0, componentType: 5123 },
        values: { bufferView: 100 },
      },
    };
    gltf.accessors![3] = {
      ...gltf.accessors![3]!,
      sparse: {
        count: 1,
        indices: { bufferView: 0, componentType: 5123 },
        values: { bufferView: 101 },
      },
    };

    const report = analyzeGltfAsset(gltf);

    expect(report.primitives.accessorStorageIssues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        semantic: 'attributes.NORMAL',
        accessorIndex: 1,
        path: 'accessors[1].sparse.values.bufferView',
        sparseIssueKind: 'missing-sparse-values-buffer-view',
        meshIndex: 0,
        primitiveIndex: 0,
        bufferViewIndex: 99,
      }),
      expect.objectContaining({
        semantic: 'targets.POSITION',
        accessorIndex: 2,
        path: 'accessors[2].sparse.values.bufferView',
        sparseIssueKind: 'missing-sparse-values-buffer-view',
        meshIndex: 0,
        primitiveIndex: 0,
        targetIndex: 0,
        bufferViewIndex: 100,
      }),
      expect.objectContaining({
        semantic: 'instancing.TRANSLATION',
        accessorIndex: 3,
        path: 'accessors[3].sparse.values.bufferView',
        sparseIssueKind: 'missing-sparse-values-buffer-view',
        nodeIndex: 0,
        bufferViewIndex: 101,
      }),
      expect.objectContaining({
        semantic: 'skin.inverseBindMatrices',
        accessorIndex: inverseBindAccessor,
        path: `accessors[${inverseBindAccessor}].sparse.values.bufferView`,
        sparseIssueKind: 'missing-sparse-values-buffer-view',
        skinIndex: 0,
        bufferViewIndex: 102,
      }),
    ]));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'primitive',
        name: 'accessor.attributes.NORMAL.missing-sparse-values-buffer-view',
        support: 'approximate',
        path: 'accessors[1].sparse.values.bufferView',
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'accessor.targets.POSITION.missing-sparse-values-buffer-view',
        support: 'approximate',
        path: 'accessors[2].sparse.values.bufferView',
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'accessor.instancing.TRANSLATION.missing-sparse-values-buffer-view',
        support: 'approximate',
        path: 'accessors[3].sparse.values.bufferView',
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'accessor.skin.inverseBindMatrices.missing-sparse-values-buffer-view',
        support: 'approximate',
        path: `accessors[${inverseBindAccessor}].sparse.values.bufferView`,
      }),
    ]));
  });

  it('reports malformed auxiliary primitive accessors before import drops or downgrades them', () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addUnboundSkinAttributes(gltf, buffers);
    const colorAccessor = gltf.accessors!.length;
    const truncatedTangentAccessor = colorAccessor + 1;
    const truncatedTangentBufferView = gltf.bufferViews!.length;
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      attributes: {
        ...gltf.meshes![0]!.primitives[0]!.attributes,
        COLOR_0: colorAccessor,
        TANGENT: truncatedTangentAccessor,
      },
    };
    gltf.accessors!.push(
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: truncatedTangentBufferView, componentType: 5126, count: 3, type: 'VEC4' },
    );
    gltf.bufferViews!.push({ buffer: 0, byteOffset: 0, byteLength: 8 });

    const report = analyzeGltfAsset(gltf);

    expect(report.primitives.accessorImportIssues).toContainEqual(expect.objectContaining({
      kind: 'invalid-accessor-type',
      support: 'approximate',
      semantic: 'attributes.COLOR_0',
      accessorIndex: colorAccessor,
      path: `accessors[${colorAccessor}].type`,
      expectedTypes: ['VEC3', 'VEC4'],
      accessorType: 'VEC2',
      meshIndex: 0,
      primitiveIndex: 0,
    }));
    expect(report.primitives.accessorImportIssues).toContainEqual(expect.objectContaining({
      kind: 'truncated-buffer-view',
      support: 'approximate',
      semantic: 'attributes.TANGENT',
      accessorIndex: truncatedTangentAccessor,
      path: `accessors[${truncatedTangentAccessor}].bufferView`,
      bufferViewIndex: truncatedTangentBufferView,
      requiredByteLength: 48,
      byteLength: 8,
      meshIndex: 0,
      primitiveIndex: 0,
    }));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'primitive',
      name: 'accessor.attributes.COLOR_0.invalid-accessor-type',
      support: 'approximate',
      path: `accessors[${colorAccessor}].type`,
    }));
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'primitive',
      name: 'accessor.attributes.TANGENT.truncated-buffer-view',
      support: 'approximate',
      path: `accessors[${truncatedTangentAccessor}].bufferView`,
    }));
  });

  it('reports malformed bound skin and instancing accessors as unsupported primitive fallbacks', () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addUnboundSkinAttributes(gltf, buffers);
    const jointsAccessor = gltf.meshes![0]!.primitives[0]!.attributes.JOINTS_0!;
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      skin: 0,
      extensions: {
        EXT_mesh_gpu_instancing: {
          attributes: { ROTATION: 1 },
        },
      },
    };
    gltf.nodes!.push({ name: 'joint' });
    gltf.skins = [{ joints: [1] }];
    gltf.extensionsUsed = ['EXT_mesh_gpu_instancing'];
    gltf.accessors![jointsAccessor] = {
      ...gltf.accessors![jointsAccessor]!,
      componentType: 5126,
    };

    const report = analyzeGltfAsset(gltf);

    expect(report.primitives.accessorImportIssues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        kind: 'invalid-accessor-component-type',
        support: 'unsupported',
        semantic: 'attributes.JOINTS_0',
        accessorIndex: jointsAccessor,
        path: `accessors[${jointsAccessor}].componentType`,
        componentType: 5126,
        meshIndex: 0,
        primitiveIndex: 0,
      }),
      expect.objectContaining({
        kind: 'invalid-accessor-type',
        support: 'unsupported',
        semantic: 'instancing.ROTATION',
        accessorIndex: 1,
        path: 'accessors[1].type',
        expectedTypes: ['VEC4'],
        accessorType: 'VEC3',
        nodeIndex: 0,
      }),
    ]));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'primitive',
        name: 'accessor.attributes.JOINTS_0.invalid-accessor-component-type',
        support: 'unsupported',
        path: `accessors[${jointsAccessor}].componentType`,
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'accessor.instancing.ROTATION.invalid-accessor-type',
        support: 'unsupported',
        path: 'accessors[1].type',
      }),
    ]));
  });

  it('reports malformed sparse animation storage before backend selection', () => {
    const { gltf } = makeInlineTriangleGltf();
    const inputAccessor = gltf.accessors!.length;
    const outputAccessor = inputAccessor + 1;
    gltf.accessors!.push(
      { bufferView: 0, componentType: 5126, count: 2, type: 'SCALAR' },
      {
        componentType: 5126,
        count: 2,
        type: 'VEC3',
        sparse: {
          count: 1,
          indices: { bufferView: 2, componentType: 5126 as never },
          values: { bufferView: 3 },
        },
      },
    );
    gltf.bufferViews!.push(
      { buffer: 0, byteOffset: 0, byteLength: 4 },
      { buffer: 0, byteOffset: 4, byteLength: 12 },
    );
    gltf.animations = [{
      samplers: [{ input: inputAccessor, output: outputAccessor }],
      channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
    }];

    const report = analyzeGltfAsset(gltf);

    expect(report.animations.malformedChannels).toContainEqual(expect.objectContaining({
      kind: 'invalid-output-sparse-accessor',
      path: `accessors[${outputAccessor}].sparse.indices.componentType`,
      accessorIndex: outputAccessor,
      accessorRole: 'output',
      sparseIssueKind: 'invalid-sparse-indices-component-type',
      componentType: 5126,
    }));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'animation',
      name: 'channel.invalid-output-sparse-accessor',
      support: 'unsupported',
      path: `accessors[${outputAccessor}].sparse.indices.componentType`,
    }));
  });

  it('reports truncated animation accessor storage before backend selection', () => {
    const { gltf } = makeInlineTriangleGltf();
    const inputAccessor = gltf.accessors!.length;
    const outputAccessor = inputAccessor + 1;
    const inputBufferView = gltf.bufferViews!.length;
    const outputBufferView = inputBufferView + 1;
    gltf.accessors!.push(
      { bufferView: inputBufferView, componentType: 5126, count: 2, type: 'SCALAR' },
      { bufferView: outputBufferView, componentType: 5126, count: 2, type: 'VEC3' },
    );
    gltf.bufferViews!.push(
      { buffer: 0, byteOffset: 0, byteLength: 4 },
      { buffer: 0, byteOffset: 0, byteLength: 24 },
    );
    gltf.animations = [{
      samplers: [{ input: inputAccessor, output: outputAccessor }],
      channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
    }];

    const report = analyzeGltfAsset(gltf);

    expect(report.animations.malformedChannels).toContainEqual(expect.objectContaining({
      kind: 'truncated-input-buffer-view',
      path: `accessors[${inputAccessor}].bufferView`,
      accessorIndex: inputAccessor,
      accessorRole: 'input',
      bufferViewIndex: inputBufferView,
      requiredByteLength: 8,
      byteLength: 4,
    }));

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'animation',
      name: 'channel.truncated-input-buffer-view',
      support: 'unsupported',
      path: `accessors[${inputAccessor}].bufferView`,
    }));
  });

  it('reports morph tangent deltas as an approximate primitive compatibility issue', () => {
    const report = analyzeGltfAsset({
      asset: { version: '2.0' },
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0 },
          targets: [{ TANGENT: 1 }],
        }],
      }],
    });

    expect(report.primitives.hasMorphTargetTangents).toBe(true);
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues.some((issue) =>
      issue.category === 'primitive' &&
      issue.name === 'morphTargetTangents' &&
      issue.support === 'approximate',
    )).toBe(true);
  });

  it('treats morph TEXCOORD_0 deltas with a matching base UV stream as supported', () => {
    const report = analyzeGltfAsset({
      asset: { version: '2.0' },
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0, TEXCOORD_0: 1 },
          targets: [{ TEXCOORD_0: 1 }],
        }],
      }],
    });

    expect(report.primitives.hasMorphTargetTexcoords).toBe(true);
    expect(report.primitives.hasUnsupportedMorphTargetTexcoords).toBe(false);
    expect(report.primitives.issuePaths.morphTargetTexcoords).toEqual([
      'meshes[0].primitives[0].targets[0].TEXCOORD_0',
    ]);
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues.some((issue) => issue.name === 'morphTargetTexcoords')).toBe(false);
  });

  it('treats morph TEXCOORD_2 deltas as supported when the matching material UV set remaps into uv1', () => {
    const report = analyzeGltfAsset({
      asset: { version: '2.0' },
      materials: [{
        pbrMetallicRoughness: {
          baseColorTexture: { index: 0, texCoord: 2 },
        },
      }],
      textures: [{ source: 0 }],
      images: [{ uri: 'base-color.png' }],
      meshes: [{
        primitives: [{
          material: 0,
          attributes: { POSITION: 0, TEXCOORD_2: 1 },
          targets: [{ TEXCOORD_2: 2 }],
        }],
      }],
    });

    expect(report.primitives.hasMorphTargetTexcoords).toBe(true);
    expect(report.primitives.hasUnsupportedMorphTargetTexcoords).toBe(false);
    expect(report.primitives.issuePaths.morphTargetTexcoords).toEqual([
      'meshes[0].primitives[0].targets[0].TEXCOORD_2',
    ]);
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues.some((issue) => issue.name === 'morphTargetTexcoords')).toBe(false);
  });

  it('reports morph TEXCOORD_2 deltas as an unsupported primitive compatibility issue', () => {
    const report = analyzeGltfAsset({
      asset: { version: '2.0' },
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0, TEXCOORD_0: 1 },
          targets: [{ TEXCOORD_2: 2 }],
        }],
      }],
    });

    expect(report.primitives.hasMorphTargetTexcoords).toBe(true);
    expect(report.primitives.hasUnsupportedMorphTargetTexcoords).toBe(true);
    expect(report.primitives.issuePaths.unsupportedMorphTargetTexcoords).toEqual([
      'meshes[0].primitives[0].targets[0].TEXCOORD_2',
    ]);
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'primitive',
      name: 'morphTargetTexcoords',
      support: 'unsupported',
      path: 'meshes[0].primitives[0].targets[0].TEXCOORD_2',
    }));
  });

  it('reports COLOR_0 vertex-color compatibility by backend instead of silently recommending unsupported paths', () => {
    const report = analyzeGltfAsset({
      asset: { version: '2.0' },
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0, COLOR_0: 1 },
        }],
      }],
    });

    expect(report.primitives.hasVertexColors).toBe(true);
    expect(report.primitives.issuePaths.vertexColors).toEqual(['meshes[0].primitives[0].attributes.COLOR_0']);

    const webgl2 = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(webgl2.issues.some((issue) => issue.name === 'vertexColors')).toBe(false);

    const webgpuFull = evaluateGltfBackendProfileCompatibility(report, 'pt-webgpu');
    expect(webgpuFull.issues.some((issue) => issue.name === 'vertexColors')).toBe(false);

    const lite = evaluateGltfBackendProfileCompatibility(report, 'pt-webgpu-lite');
    expect(lite.issues).toContainEqual(expect.objectContaining({
      category: 'primitive',
      name: 'vertexColors',
      support: 'unsupported',
      path: 'meshes[0].primitives[0].attributes.COLOR_0',
    }));

    const walkaround = evaluateGltfBackendProfileCompatibility(report, 'walkaround-hybrid');
    expect(walkaround.issues).toContainEqual(expect.objectContaining({
      category: 'primitive',
      name: 'vertexColors',
      support: 'approximate',
      path: 'meshes[0].primitives[0].attributes.COLOR_0',
    }));
  });

  it('reports EXT_mesh_gpu_instancing as native instanced-mesh input with node source path', () => {
    const report = analyzeGltfAsset({
      asset: { version: '2.0' },
      extensionsUsed: ['EXT_mesh_gpu_instancing'],
      extensionsRequired: ['EXT_mesh_gpu_instancing'],
      nodes: [{
        mesh: 0,
        extensions: {
          EXT_mesh_gpu_instancing: {
            attributes: {
              TRANSLATION: 1,
            },
          },
        },
      }],
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0 },
        }],
      }],
    });

    expect(report.extensions.supported).toContain('EXT_mesh_gpu_instancing');
    expect(report.extensions.unsupportedOptional).not.toContain('EXT_mesh_gpu_instancing');
    expect(report.extensions.unsupportedRequired).not.toContain('EXT_mesh_gpu_instancing');
    expect(report.extensions.sourcePaths.EXT_mesh_gpu_instancing).toContain(
      'nodes[0].extensions.EXT_mesh_gpu_instancing',
    );
    expect(report.primitives.expectedPrimitiveKinds).toEqual(
      expect.arrayContaining(['mesh', 'instanced-mesh']),
    );
    expect(report.primitives.issuePaths['kind:instanced-mesh']).toEqual([
      'nodes[0].extensions.EXT_mesh_gpu_instancing',
    ]);
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues.some((issue) =>
      issue.category === 'extension' &&
      issue.name === 'EXT_mesh_gpu_instancing',
    )).toBe(false);
    expect(compatibility.issues.some((issue) =>
      issue.category === 'primitive' &&
      issue.name === 'instanced-mesh',
    )).toBe(false);
  });

  it('reports malformed EXT_mesh_gpu_instancing shapes as unsupported base-mesh fallbacks', () => {
    const cases = [
      {
        name: 'missing-attributes',
        extension: {},
        expected: {
          kind: 'missing-attributes',
          path: 'nodes[0].extensions.EXT_mesh_gpu_instancing',
        },
      },
      {
        name: 'missing-transform-attributes',
        extension: { attributes: { _CUSTOM: 1 } },
        expected: {
          kind: 'missing-transform-attributes',
          path: 'nodes[0].extensions.EXT_mesh_gpu_instancing.attributes',
        },
      },
      {
        name: 'invalid-attribute-accessor-index',
        extension: { attributes: { TRANSLATION: -1 } },
        expected: {
          kind: 'invalid-attribute-accessor-index',
          path: 'nodes[0].extensions.EXT_mesh_gpu_instancing.attributes.TRANSLATION',
          attribute: 'TRANSLATION',
          value: -1,
        },
      },
    ] as const;

    for (const testCase of cases) {
      const { gltf } = makeInlineTriangleGltf();
      gltf.extensionsUsed = ['EXT_mesh_gpu_instancing'];
      gltf.nodes![0] = {
        ...gltf.nodes![0]!,
        extensions: {
          EXT_mesh_gpu_instancing: testCase.extension,
        },
      };

      const report = analyzeGltfAsset(gltf);
      expect(report.primitives.instancingIssues, testCase.name).toContainEqual(expect.objectContaining({
        ...testCase.expected,
        nodeIndex: 0,
      }));

      const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
      expect(compatibility.issues, testCase.name).toContainEqual(expect.objectContaining({
        category: 'primitive',
        name: `EXT_mesh_gpu_instancing.${testCase.expected.kind}`,
        support: 'unsupported',
        path: testCase.expected.path,
      }));
    }
  });

  it('reports EXT_mesh_gpu_instancing on morphed meshes as a fallback-expanded combined primitive case', () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addMorphedGpuInstancing(gltf, buffers);

    const report = analyzeGltfAsset(gltf);

    expect(report.extensions.supported).toContain('EXT_mesh_gpu_instancing');
    expect(report.primitives.hasMorphTargets).toBe(true);
    expect(report.primitives.hasInstancedSkinnedOrMorphed).toBe(true);
    expect(report.primitives.expectedPrimitiveKinds).toEqual(
      expect.arrayContaining(['mesh', 'skinned-mesh', 'instanced-mesh']),
    );
    expect(report.primitives.issuePaths.instancedSkinnedOrMorphed).toEqual([
      'nodes[0].extensions.EXT_mesh_gpu_instancing',
    ]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'primitive',
      name: 'EXT_mesh_gpu_instancing.skinnedOrMorphed',
      support: 'fallback-generated-mesh',
      path: 'nodes[0].extensions.EXT_mesh_gpu_instancing',
    }));
  });

  it('reports skin attributes without node.skin as unsupported ignored primitive data', () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addUnboundSkinAttributes(gltf, buffers);

    const report = analyzeGltfAsset(gltf);

    expect(report.primitives.hasSkins).toBe(false);
    expect(report.primitives.expectedPrimitiveKinds).toEqual(['mesh']);
    expect(report.primitives.hasIgnoredSkinAttributes).toBe(true);
    expect(report.primitives.issuePaths.ignoredSkinAttributes).toEqual([
      'meshes[0].primitives[0].attributes.JOINTS_0',
      'meshes[0].primitives[0].attributes.WEIGHTS_0',
    ]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'primitive',
      name: 'skinAttributesWithoutNodeSkin',
      support: 'unsupported',
      path: 'meshes[0].primitives[0].attributes.JOINTS_0',
    }));
  });

  it('reports skin nodes with incomplete JOINTS_0/WEIGHTS_0 streams as unsupported', () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addUnboundSkinAttributes(gltf, buffers, { omitWeights: true });
    gltf.nodes![0] = { ...gltf.nodes![0]!, skin: 0 };
    gltf.nodes!.push({ name: 'joint' });
    gltf.skins = [{ joints: [1] }];

    const report = analyzeGltfAsset(gltf);

    expect(report.primitives.hasSkins).toBe(false);
    expect(report.primitives.expectedPrimitiveKinds).toEqual(['mesh']);
    expect(report.primitives.hasIncompleteSkinAttributes).toBe(true);
    expect(report.primitives.issuePaths.incompleteSkinAttributes).toEqual([
      'meshes[0].primitives[0]',
    ]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'primitive',
      name: 'incompleteSkinAttributes',
      support: 'unsupported',
      path: 'meshes[0].primitives[0]',
    }));
  });

  it('reports skin nodes with no JOINTS_0/WEIGHTS_0 streams as unsupported', () => {
    const { gltf } = makeInlineTriangleGltf();
    gltf.nodes![0] = { ...gltf.nodes![0]!, skin: 0 };
    gltf.nodes!.push({ name: 'joint' });
    gltf.skins = [{ joints: [1] }];

    const report = analyzeGltfAsset(gltf);

    expect(report.primitives.hasSkins).toBe(false);
    expect(report.primitives.expectedPrimitiveKinds).toEqual(['mesh']);
    expect(report.primitives.hasIncompleteSkinAttributes).toBe(true);
    expect(report.primitives.issuePaths.incompleteSkinAttributes).toEqual([
      'meshes[0].primitives[0]',
    ]);

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'primitive',
      name: 'incompleteSkinAttributes',
      support: 'unsupported',
      path: 'meshes[0].primitives[0]',
    }));
  });

  it('attaches source paths to compatibility issues, including cameras and double-sided materials', () => {
    const report = analyzeGltfAsset({
      asset: { version: '2.0' },
      extensionsUsed: ['EXT_unknown_feature', 'KHR_draco_mesh_compression'],
      cameras: [{ type: 'perspective' }],
      meshes: [{
        primitives: [{
          attributes: { POSITION: 0 },
          mode: 1,
          targets: [{ TANGENT: 1 }],
          material: 0,
        }],
      }],
      materials: [{
        doubleSided: true,
        extensions: {
          KHR_materials_pbrSpecularGlossiness: {
            specularGlossinessTexture: { index: 0 },
          },
        },
      }],
    });

    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');

    expect(compatibility.issues.length).toBeGreaterThan(0);
    expect(compatibility.issues.every((issue) => typeof issue.path === 'string' && issue.path.length > 0)).toBe(true);
    expect(compatibility.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'extension',
        name: 'EXT_unknown_feature',
        support: 'unsupported',
        path: 'extensionsUsed[0]',
      }),
      expect.objectContaining({
        category: 'scene',
        name: 'cameras',
        support: 'approximate',
        path: 'cameras[0]',
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'mode:1',
        support: 'fallback-generated-mesh',
        path: 'meshes[0].primitives[0].mode',
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'morphTargetTangents',
        support: 'approximate',
        path: 'meshes[0].primitives[0].targets[0].TANGENT',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'doubleSided',
        support: 'approximate',
        path: 'materials[0].doubleSided',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'KHR_materials_pbrSpecularGlossiness',
        support: 'approximate',
        path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness',
      }),
      expect.objectContaining({
        category: 'material',
        name: 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha',
        support: 'approximate',
        path: 'materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
      }),
    ]));
    expect(compatibility.issues.some((issue) =>
      issue.category === 'extension' &&
      issue.name === 'KHR_draco_mesh_compression' &&
      issue.support === 'requires-hook',
    )).toBe(false);
  });
});

describe('loadGltfForEngine', () => {
  it('loads, selects the recommended backend, constructs an injected engine, and attaches a controller', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const engine = { setScene: vi.fn(), updatePrimitive: vi.fn() };
    const createEngine = vi.fn(async ({ scene, backend, asset, options }) => {
      expect(scene).toBe(asset.scene);
      expect(backend).toBe('pt-webgl2');
      expect(options).toEqual({ label: 'viewer' });
      return engine;
    });

    const result = await loadGltfForEngine(gltf, {
      buffers,
      createEngine,
      engineOptions: { label: 'viewer' },
    });

    expect(result.backend).toBe('pt-webgl2');
    expect(result.engine).toBe(engine);
    expect(result.attached).toBe(true);
    expect(engine.setScene).toHaveBeenCalledWith(result.asset.scene);
    expect(result.controller.scene.primitives).toHaveLength(1);
  });

  it('scopes strict compatibility to the selected scene instead of unused scenes', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.scenes = [{ nodes: [0] }, { nodes: [1] }];
    gltf.nodes = [
      { mesh: 0 },
      { mesh: 1, camera: 0 },
    ];
    gltf.cameras = [{}];
    gltf.materials = [
      { pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1] } },
      { doubleSided: true, pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1] } },
    ];
    gltf.meshes = [
      { primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, material: 0 }] },
      { primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, material: 1, mode: 0 }] },
    ];

    const cleanEngine = { backendId: 'pt-webgl2' as const, setScene: vi.fn() };
    const clean = await loadGltfForEngine(gltf, {
      buffers,
      sceneIndex: 0,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine: async () => cleanEngine,
    });

    expect(clean.asset.scene.primitives).toHaveLength(1);
    expect(clean.asset.diagnostics.some((diagnostic) =>
      diagnostic.code === 'ignored-camera' ||
      diagnostic.code === 'double-sided-material',
    )).toBe(false);
    const selectedCompatibility = clean.asset.backendCompatibility.find((entry) =>
      entry.profileId === 'pt-webgl2'
    );
    expect(selectedCompatibility?.issues.some((issue) =>
      issue.name === 'doubleSided' ||
      issue.name === 'cameras' ||
      issue.name === 'mode:0',
    )).toBe(false);

    await expect(loadGltfForEngine(gltf, {
      buffers,
      sceneIndex: 1,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine: async () => ({ backendId: 'pt-webgl2' as const, setScene: vi.fn() }),
    })).rejects.toThrow('Selected backend "pt-webgl2" does not satisfy reject-degraded');
  });

  it('omits unused non-variant materials from selected-scene textureDecodeReport', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.materials = [
      { pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1] } },
      { pbrMetallicRoughness: { baseColorTexture: { index: 0 } } },
    ];
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      material: 0,
    };
    gltf.textures = [{ source: 0 }];
    gltf.images = [{ uri: 'data:image/png;base64,AQID', mimeType: 'image/png' }];

    const result = await loadGltfAsset(gltf, { buffers, sceneIndex: 0 });

    expect(result.textureDecodeReport.entries).toHaveLength(0);
    expect(result.backendCompatibility.flatMap((entry) => entry.issues).some((issue) =>
      issue.path.includes('materials[1]') ||
      issue.path.includes('textures[0]') ||
      issue.name === 'baseColorMap',
    )).toBe(false);
  });

  it('does not fetch external buffers or images reachable only from unused scenes/materials', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const originalBuffer = buffers.get(0)!;
    gltf.scenes = [{ nodes: [0] }, { nodes: [1] }];
    gltf.nodes = [{ mesh: 0 }, { mesh: 1 }];
    gltf.materials = [
      { pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1] } },
      { pbrMetallicRoughness: { baseColorTexture: { index: 0 } } },
    ];
    gltf.meshes = [
      { primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, material: 0 }] },
      { primitives: [{ attributes: { POSITION: 2, NORMAL: 3 }, material: 1 }] },
    ];
    gltf.accessors = [
      ...(gltf.accessors ?? []),
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 3, componentType: 5126, count: 3, type: 'VEC3' },
    ];
    gltf.bufferViews = [
      ...(gltf.bufferViews ?? []),
      { buffer: 1, byteOffset: 0, byteLength: 9 * 4 },
      { buffer: 1, byteOffset: 9 * 4, byteLength: 9 * 4 },
    ];
    gltf.buffers = [
      { byteLength: originalBuffer.byteLength },
      { uri: 'unused-scene.bin', byteLength: originalBuffer.byteLength },
    ];
    gltf.textures = [{ source: 0 }];
    gltf.images = [{ uri: 'unused-material.png', mimeType: 'image/png' }];
    const fetch = vi.fn(async (url: string) => {
      throw new Error(`unexpected fetch: ${url}`);
    });

    const result = await loadGltfAsset(gltf, {
      buffers,
      sceneIndex: 0,
      baseUri: 'https://assets.example/model.gltf',
      fetch,
    });

    expect(fetch).not.toHaveBeenCalled();
    expect(result.scene.primitives).toHaveLength(1);
    expect(result.textureDecodeReport.entries).toHaveLength(0);
    expect(result.diagnostics.some((diagnostic) =>
      diagnostic.path.includes('materials[1]') ||
      diagnostic.path.includes('textures[0]') ||
      diagnostic.path.includes('images[0]'),
    )).toBe(false);
  });

  it('does not import or warn on animations reachable only from unused scenes', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const originalBuffer = buffers.get(0)!;
    gltf.scenes = [{ nodes: [0] }, { nodes: [1] }];
    gltf.nodes = [{ mesh: 0 }, { mesh: 0 }];
    gltf.accessors = [
      ...(gltf.accessors ?? []),
      { bufferView: 2, componentType: 5126, count: 2, type: 'SCALAR' },
      { bufferView: 3, componentType: 5126, count: 2, type: 'VEC3' },
    ];
    gltf.bufferViews = [
      ...(gltf.bufferViews ?? []),
      { buffer: 1, byteOffset: 0, byteLength: 2 * 4 },
      { buffer: 1, byteOffset: 2 * 4, byteLength: 2 * 3 * 4 },
    ];
    gltf.buffers = [
      { byteLength: originalBuffer.byteLength },
      { uri: 'unused-animation.bin', byteLength: 8 * 4 },
    ];
    gltf.animations = [{
      name: 'unused-scene-motion',
      samplers: [{ input: 2, output: 3 }],
      channels: [{ sampler: 0, target: { node: 1, path: 'translation' } }],
    }];
    const fetch = vi.fn(async (url: string) => {
      throw new Error(`unexpected fetch: ${url}`);
    });

    const result = await loadGltfAsset(gltf, {
      buffers,
      sceneIndex: 0,
      baseUri: 'https://assets.example/model.gltf',
      fetch,
    });

    expect(fetch).not.toHaveBeenCalled();
    expect(result.featureReport.animations.count).toBe(0);
    expect(result.featureReport.animations.channelCount).toBe(0);
    expect(result.animations).toEqual([]);
    expect(result.diagnostics.some((diagnostic) =>
      diagnostic.path.startsWith('animations[0]') ||
      diagnostic.code === 'unreadable-animation-sampler' ||
      diagnostic.code === 'dropped-animation',
    )).toBe(false);
  });

  it('fetches selected-scene sparse animation patch buffers before conversion', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const originalBuffer = buffers.get(0)!;
    const times = f32Buffer([0, 1]);
    const baseValues = f32Buffer([0, 0, 0, 0, 0, 0]);
    const baseBuffer = concatArrayBuffers([originalBuffer, times, baseValues]);
    const timeOffset = originalBuffer.byteLength;
    const valueOffset = timeOffset + times.byteLength;
    const sparseIndices = u16Buffer([1]);
    const sparseValues = f32Buffer([2, 3, 4]);
    const sparseBuffer = concatArrayBuffers([sparseIndices, sparseValues]);
    buffers.set(0, baseBuffer);
    gltf.accessors = [
      ...(gltf.accessors ?? []),
      { bufferView: 2, componentType: 5126, count: 2, type: 'SCALAR' },
      {
        bufferView: 3,
        componentType: 5126,
        count: 2,
        type: 'VEC3',
        sparse: {
          count: 1,
          indices: { bufferView: 4, componentType: 5123 },
          values: { bufferView: 5 },
        },
      },
    ];
    gltf.bufferViews = [
      ...(gltf.bufferViews ?? []),
      { buffer: 0, byteOffset: timeOffset, byteLength: times.byteLength },
      { buffer: 0, byteOffset: valueOffset, byteLength: baseValues.byteLength },
      { buffer: 1, byteOffset: 0, byteLength: sparseIndices.byteLength },
      { buffer: 1, byteOffset: sparseIndices.byteLength, byteLength: sparseValues.byteLength },
    ];
    gltf.buffers = [
      { byteLength: baseBuffer.byteLength },
      { uri: 'selected-animation-sparse.bin', byteLength: sparseBuffer.byteLength },
    ];
    gltf.animations = [{
      name: 'selected-sparse-motion',
      samplers: [{ input: 2, output: 3 }],
      channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
    }];
    const fetch = vi.fn(async (url: string) => {
      if (!url.endsWith('/selected-animation-sparse.bin')) {
        throw new Error(`unexpected fetch: ${url}`);
      }
      return response(sparseBuffer);
    });

    const result = await loadGltfAsset(gltf, {
      buffers,
      sceneIndex: 0,
      baseUri: 'https://assets.example/model.gltf',
      fetch,
    });

    expect(fetch.mock.calls.map(([url]) => url)).toEqual([
      'https://assets.example/selected-animation-sparse.bin',
    ]);
    expect(result.animations).toHaveLength(1);
    expect(Array.from(result.animations[0]!.channels[0]!.sampler.values)).toEqual([
      0, 0, 0,
      2, 3, 4,
    ]);
    expect(result.diagnostics.some((diagnostic) =>
      diagnostic.code === 'sparse-values-buffer-unavailable' ||
      diagnostic.code === 'dropped-animation',
    )).toBe(false);
  });

  it('fetches selected-scene KHR_animation_pointer material sampler buffers before conversion', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const originalBuffer = buffers.get(0)!;
    gltf.extensionsUsed = ['KHR_animation_pointer'];
    gltf.extensionsRequired = ['KHR_animation_pointer'];
    gltf.materials = [{ pbrMetallicRoughness: { baseColorFactor: [1, 0, 0, 1] } }];
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      material: 0,
    };
    const times = f32Buffer([0, 1]);
    const baseValues = f32Buffer([
      1, 0, 0, 1,
      0, 0, 1, 1,
    ]);
    const samplerBuffer = concatArrayBuffers([times, baseValues]);
    const timeOffset = 0;
    const valueOffset = times.byteLength;
    const sparseIndices = u16Buffer([1]);
    const sparseValues = f32Buffer([0, 1, 0, 1]);
    const sparseBuffer = concatArrayBuffers([sparseIndices, sparseValues]);
    gltf.accessors = [
      ...(gltf.accessors ?? []),
      { bufferView: 2, componentType: 5126, count: 2, type: 'SCALAR' },
      {
        bufferView: 3,
        componentType: 5126,
        count: 2,
        type: 'VEC4',
        sparse: {
          count: 1,
          indices: { bufferView: 4, componentType: 5123 },
          values: { bufferView: 5 },
        },
      },
    ];
    gltf.bufferViews = [
      ...(gltf.bufferViews ?? []),
      { buffer: 1, byteOffset: timeOffset, byteLength: times.byteLength },
      { buffer: 1, byteOffset: valueOffset, byteLength: baseValues.byteLength },
      { buffer: 2, byteOffset: 0, byteLength: sparseIndices.byteLength },
      { buffer: 2, byteOffset: sparseIndices.byteLength, byteLength: sparseValues.byteLength },
    ];
    gltf.buffers = [
      { byteLength: originalBuffer.byteLength },
      { uri: 'selected-pointer-animation.bin', byteLength: samplerBuffer.byteLength },
      { uri: 'selected-pointer-animation-sparse.bin', byteLength: sparseBuffer.byteLength },
    ];
    gltf.animations = [{
      name: 'selected-material-pointer',
      samplers: [{ input: 2, output: 3 }],
      channels: [{
        sampler: 0,
        target: {
          path: 'pointer',
          extensions: {
            KHR_animation_pointer: {
              pointer: '/materials/0/pbrMetallicRoughness/baseColorFactor',
            },
          },
        },
      }],
    }];
    const fetch = vi.fn(async (url: string) => {
      if (url.endsWith('/selected-pointer-animation.bin')) return response(samplerBuffer);
      if (url.endsWith('/selected-pointer-animation-sparse.bin')) return response(sparseBuffer);
      throw new Error(`unexpected fetch: ${url}`);
    });

    const result = await loadGltfAsset(gltf, {
      buffers,
      sceneIndex: 0,
      baseUri: 'https://assets.example/model.gltf',
      fetch,
    });

    expect(fetch.mock.calls.map(([url]) => url)).toEqual([
      'https://assets.example/selected-pointer-animation.bin',
      'https://assets.example/selected-pointer-animation-sparse.bin',
    ]);
    expect(result.animations).toHaveLength(1);
    expect(result.animations[0]!.channels[0]!.target).toMatchObject({
      path: 'pointer',
      pointer: '/materials/0/pbrMetallicRoughness/baseColorFactor',
    });
    expect(Array.from(result.animations[0]!.channels[0]!.sampler.values)).toEqual([
      1, 0, 0, 1,
      0, 1, 0, 1,
    ]);
    expect(result.diagnostics.some((diagnostic) =>
      diagnostic.code === 'sparse-values-buffer-unavailable' ||
      diagnostic.code === 'unreadable-animation-sampler' ||
      diagnostic.code === 'dropped-animation',
    )).toBe(false);
  });

  it('does not fetch KHR_animation_pointer sampler buffers for materials reachable only from unused scenes', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const originalBuffer = buffers.get(0)!;
    gltf.extensionsUsed = ['KHR_animation_pointer'];
    gltf.extensionsRequired = ['KHR_animation_pointer'];
    gltf.scenes = [{ nodes: [0] }, { nodes: [1] }];
    gltf.nodes = [{ mesh: 0 }, { mesh: 1 }];
    gltf.materials = [
      { pbrMetallicRoughness: { baseColorFactor: [1, 0, 0, 1] } },
      { pbrMetallicRoughness: { baseColorFactor: [0, 1, 0, 1] } },
    ];
    gltf.meshes = [
      { primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, material: 0 }] },
      { primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, material: 1 }] },
    ];
    const times = f32Buffer([0, 1]);
    const values = f32Buffer([
      0, 1, 0, 1,
      0, 0, 1, 1,
    ]);
    const animationBuffer = concatArrayBuffers([times, values]);
    gltf.accessors = [
      ...(gltf.accessors ?? []),
      { bufferView: 2, componentType: 5126, count: 2, type: 'SCALAR' },
      { bufferView: 3, componentType: 5126, count: 2, type: 'VEC4' },
    ];
    gltf.bufferViews = [
      ...(gltf.bufferViews ?? []),
      { buffer: 1, byteOffset: 0, byteLength: times.byteLength },
      { buffer: 1, byteOffset: times.byteLength, byteLength: values.byteLength },
    ];
    gltf.buffers = [
      { byteLength: originalBuffer.byteLength },
      { uri: 'unused-pointer-animation.bin', byteLength: animationBuffer.byteLength },
    ];
    gltf.animations = [{
      name: 'unused-material-pointer',
      samplers: [{ input: 2, output: 3 }],
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
    }];
    const fetch = vi.fn(async (url: string) => {
      throw new Error(`unexpected fetch: ${url}`);
    });

    const result = await loadGltfAsset(gltf, {
      buffers,
      sceneIndex: 0,
      baseUri: 'https://assets.example/model.gltf',
      fetch,
    });

    expect(fetch).not.toHaveBeenCalled();
    expect(result.featureReport.animations.count).toBe(0);
    expect(result.featureReport.animations.channelCount).toBe(0);
    expect(result.animations).toEqual([]);
    expect(result.diagnostics.some((diagnostic) =>
      diagnostic.path.startsWith('animations[0]') ||
      diagnostic.code === 'unreadable-animation-sampler' ||
      diagnostic.code === 'dropped-animation',
    )).toBe(false);
  });

  it('fetches selected-scene variant material textures without disabled texture-source sidecars', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.extensionsUsed = ['KHR_materials_variants', 'EXT_texture_webp'];
    gltf.extensions = {
      KHR_materials_variants: { variants: [{ name: 'variant' }] },
    };
    gltf.materials = [
      { pbrMetallicRoughness: { baseColorTexture: { index: 0 } } },
      { pbrMetallicRoughness: { baseColorTexture: { index: 1 } } },
    ];
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      material: 0,
      extensions: {
        KHR_materials_variants: {
          mappings: [{ material: 1, variants: [0] }],
        },
      },
    };
    gltf.textures = [
      { source: 0, extensions: { EXT_texture_webp: { source: 2 } } },
      { source: 1 },
    ];
    gltf.images = [
      { uri: 'base.png', mimeType: 'image/png' },
      { uri: 'variant.png', mimeType: 'image/png' },
      { uri: 'disabled-sidecar.webp', mimeType: 'image/webp' },
    ];
    const fetched: string[] = [];
    const fetch = vi.fn(async (url: string) => {
      fetched.push(url);
      if (url.includes('disabled-sidecar.webp')) {
        throw new Error(`disabled sidecar should not be fetched: ${url}`);
      }
      return response(bytes([1, 2, 3, 4]), url.endsWith('.png') ? 'image/png' : 'application/octet-stream');
    });

    const result = await loadGltfAsset(gltf, {
      buffers,
      sceneIndex: 0,
      baseUri: 'https://assets.example/model.gltf',
      fetch,
    });

    expect(fetched).toEqual([
      'https://assets.example/base.png',
      'https://assets.example/variant.png',
    ]);
    expect(result.textureDecodeReport.entries.map((entry) => entry.textureIndex).sort()).toEqual([0, 1]);
  });

  it('lets direct callers target the pt-webgpu-lite profile while factories receive pt-webgpu', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const engine = { backendId: 'pt-webgpu' as const, setScene: vi.fn() };
    const createEngine = vi.fn(async ({ backend }) => {
      expect(backend).toBe('pt-webgpu');
      return engine;
    });

    const result = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgpu-lite',
      createEngine,
    });

    expect(result.backend).toBe('pt-webgpu');
    expect(result.profileId).toBe('pt-webgpu-lite');
    expect(result.engine).toBe(engine);
    expect(createEngine).toHaveBeenCalledTimes(1);
  });

  it('lets adapter-only hosts narrow a pt-webgpu selection to the runtime lite profile', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const engine = { backendId: 'pt-webgpu' as const, setScene: vi.fn() };
    const createEngine = vi.fn(async ({ backend }) => {
      expect(backend).toBe('pt-webgpu');
      return engine;
    });

    const result = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgpu',
      runtimeProfile: 'pt-webgpu-lite',
      createEngine,
    });

    expect(result.backend).toBe('pt-webgpu');
    expect(result.profileId).toBe('pt-webgpu-lite');
    expect(createEngine).toHaveBeenCalledTimes(1);
  });

  it('allows direct pt-webgpu-lite strict loads for primitive-constant RGB COLOR_0 scenes', async () => {
    const { gltf, buffers } = makeInlineVertexColorGltf([
      0.5, 0.25, 1,
      0.5, 0.25, 1,
      0.5, 0.25, 1,
    ]);
    const engine = { backendId: 'pt-webgpu' as const, setScene: vi.fn() };
    const createEngine = vi.fn(async () => engine);

    const result = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgpu-lite',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    });

    const liteCompatibility = result.asset.backendCompatibility.find((entry) =>
      entry.profileId === 'pt-webgpu-lite'
    );
    expect(liteCompatibility?.issues.some((issue) => issue.name === 'vertexColors')).toBe(false);
    expect(result.profileId).toBe('pt-webgpu-lite');
    expect(result.engine).toBe(engine);
    expect(createEngine).toHaveBeenCalledTimes(1);
  });

  it('rejects direct pt-webgpu-lite strict loads before constructing nonconstant COLOR_0 scenes', async () => {
    const { gltf, buffers } = makeInlineVertexColorGltf();
    const createEngine = vi.fn(async () => ({ backendId: 'pt-webgpu' as const, setScene: vi.fn() }));

    const promise = loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgpu-lite',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    });

    await expect(promise).rejects.toThrow(
      'Selected backend "pt-webgpu" profile "pt-webgpu-lite" does not satisfy reject-unsupported: primitive:vertexColors=unsupported',
    );
    await expect(promise).rejects.toBeInstanceOf(GltfCompatibilityError);
    await expect(promise).rejects.toMatchObject({
      code: 'GLTF_COMPATIBILITY_REJECTED',
      backend: 'pt-webgpu',
      profileId: 'pt-webgpu-lite',
      compatibilityMode: 'reject-unsupported',
      failures: ['primitive:vertexColors=unsupported at meshes[0].primitives[0].attributes.COLOR_0'],
    });

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects direct pt-webgpu-lite strict loads for constant COLOR_0 with alpha below one', async () => {
    const { gltf, buffers } = makeInlineVertexColorGltf([
      0.5, 0.25, 1, 0.5,
      0.5, 0.25, 1, 0.5,
      0.5, 0.25, 1, 0.5,
    ], 'VEC4');
    const createEngine = vi.fn(async () => ({ backendId: 'pt-webgpu' as const, setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgpu-lite',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).rejects.toThrow(
      'Selected backend "pt-webgpu" profile "pt-webgpu-lite" does not satisfy reject-unsupported: primitive:vertexColors=unsupported',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('uses runtime pt-webgpu-lite compatibility before constructing map-heavy pt-webgpu scenes', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const createEngine = vi.fn(async () => ({ backendId: 'pt-webgpu' as const, setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgpu',
      runtimeProfile: 'pt-webgpu-lite',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).rejects.toThrow(
      'Selected backend "pt-webgpu" profile "pt-webgpu-lite" does not satisfy reject-unsupported: material:baseColorMap=unsupported',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rechecks strict compatibility against an engine-reported runtime profile before attaching', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const engine = {
      backendId: 'pt-webgpu' as const,
      backendProfileId: 'pt-webgpu-lite' as const,
      setScene: vi.fn(),
    };
    const createEngine = vi.fn(async () => engine);

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgpu',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).rejects.toThrow(
      'Actual engine profile "pt-webgpu" profile "pt-webgpu-lite" does not satisfy reject-unsupported: material:baseColorMap=unsupported',
    );

    expect(createEngine).toHaveBeenCalledTimes(1);
    expect(engine.setScene).not.toHaveBeenCalled();
  });

  it('rechecks strict compatibility against pt-webgpu lite engines that expose only capabilities', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const engine = {
      backendId: 'pt-webgpu' as const,
      capabilities: {
        experimentalFeatures: new Set(['pt-webgpu-lite-tier']),
      },
      setScene: vi.fn(),
    };

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgpu',
      compatibilityMode: 'reject-unsupported',
      engine,
    })).rejects.toThrow(
      'Actual engine profile "pt-webgpu" profile "pt-webgpu-lite" does not satisfy reject-unsupported: material:baseColorMap=unsupported',
    );

    expect(engine.setScene).not.toHaveBeenCalled();
  });

  it('keeps existing pt-webgpu engines on the full profile when no lite marker is present', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const engine = {
      backendId: 'pt-webgpu' as const,
      capabilities: {
        experimentalFeatures: new Set<string>(),
      },
      setScene: vi.fn(),
    };

    const result = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgpu',
      engine,
    });

    expect(result.backend).toBe('pt-webgpu');
    expect(result.profileId).toBe('pt-webgpu');
    expect(result.engine).toBe(engine);
    expect(engine.setScene).toHaveBeenCalledTimes(1);
  });

  it('rejects a runtime profile from a different backend family', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    const promise = loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      runtimeProfile: 'pt-webgpu-lite',
      createEngine,
    });

    await expect(promise).rejects.toThrow(
      'runtimeProfile "pt-webgpu" profile "pt-webgpu-lite" does not match selected backend "pt-webgl2"',
    );
    await expect(promise).rejects.toBeInstanceOf(GltfCompatibilityError);
    await expect(promise).rejects.toMatchObject({
      code: 'GLTF_RUNTIME_PROFILE_MISMATCH',
      backend: 'pt-webgl2',
      profileId: 'pt-webgl2',
      runtimeProfile: 'pt-webgpu-lite',
    });

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('attaches an existing engine without invoking a factory', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const engine = { setScene: vi.fn() };
    const createEngine = vi.fn();

    const result = await loadGltfForEngine(gltf, {
      buffers,
      engine,
      createEngine,
      attachScene: false,
    });

    expect(result.engine).toBe(engine);
    expect(result.attached).toBe(true);
    expect(createEngine).not.toHaveBeenCalled();
    expect(engine.setScene).not.toHaveBeenCalled();
  });

  it('reports an existing engine backendId instead of the planned backend', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    const engine = { backendId: 'pt-webgpu' as const, setScene: vi.fn() };

    const result = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      engine,
    });

    expect(result.backend).toBe('pt-webgpu');
    expect(result.engine).toBe(engine);
    expect(engine.setScene).toHaveBeenCalledWith(result.asset.scene);
  });

  it('rechecks strict compatibility against the actual factory backend before attaching', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const engine = { backendId: 'walkaround-hybrid' as const, setScene: vi.fn() };
    const createEngine = vi.fn(async () => engine);

    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine,
    })).rejects.toThrow('Actual engine backend "walkaround-hybrid" does not satisfy reject-degraded');

    expect(createEngine).toHaveBeenCalledTimes(1);
    expect(engine.setScene).not.toHaveBeenCalled();
  });

  it('rejects structural import diagnostics in strict mode before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    delete gltf.meshes![0]!.primitives[0]!.attributes.POSITION;
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).rejects.toThrow(
      'import:missing-position=unsupported at meshes[0].primitives[0].attributes.POSITION',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects ignored secondary vertex-color diagnostics in reject-unsupported mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addSecondaryVertexColorSet(gltf, buffers);
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).rejects.toThrow(
      'import:ignored-vertex-color-set=unsupported at meshes[0].primitives[0].attributes.COLOR_1',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('accepts supported morph-target TEXCOORD_0 diagnostics in reject-unsupported mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addMorphTargetTexcoord(gltf, buffers);
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).resolves.toBeDefined();

    expect(createEngine).toHaveBeenCalledOnce();
  });

  it('rejects unsupported high morph-target TEXCOORD diagnostics in reject-unsupported mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addMorphTargetTexcoord(gltf, buffers, { semantic: 'TEXCOORD_2' });
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).rejects.toThrow(
      'import:ignored-morph-target-texcoord=unsupported at meshes[0].primitives[0].targets[0].TEXCOORD_2',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('accepts remappable high morph-target TEXCOORD diagnostics in reject-unsupported mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addMorphTargetTexcoord(gltf, buffers, {
      semantic: 'TEXCOORD_2',
      baseSemantic: 'TEXCOORD_2',
      materialTexCoord: 2,
    });
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).resolves.toBeDefined();

    expect(createEngine).toHaveBeenCalledOnce();
  });

  it('rejects malformed animation diagnostics in reject-unsupported mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.animations = [{
      name: 'bad-walk',
      channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
      samplers: [],
    }];
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).rejects.toThrow(
      'import:missing-animation-sampler=unsupported at animations[0].samplers[0]',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('allows double-sided diagnostics in reject-unsupported mode but rejects them in reject-degraded mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      material: 0,
    };
    gltf.materials = [{ doubleSided: true }];

    const createAcceptedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    const accepted = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine: createAcceptedEngine,
    });
    expect(accepted.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'double-sided-material',
        path: 'materials[0].doubleSided',
      }),
    ]));
    expect(createAcceptedEngine).toHaveBeenCalledTimes(1);

    const createRejectedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine: createRejectedEngine,
    })).rejects.toThrow(
      'import:double-sided-material=approximate at materials[0].doubleSided',
    );

    expect(createRejectedEngine).not.toHaveBeenCalled();
  });

  it('allows skin rest-pose diagnostics in reject-unsupported mode but rejects them in reject-degraded mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.skins = [{ joints: [0] }];
    gltf.nodes![0] = { ...gltf.nodes![0]!, skin: 0 };
    addUnboundSkinAttributes(gltf, buffers);

    const createAcceptedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    const accepted = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine: createAcceptedEngine,
    });
    expect(accepted.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'skin-rest-pose',
        path: 'skins[0]',
      }),
    ]));
    expect(createAcceptedEngine).toHaveBeenCalledTimes(1);

    const createRejectedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine: createRejectedEngine,
    })).rejects.toThrow('import:skin-rest-pose=approximate at skins[0]');

    expect(createRejectedEngine).not.toHaveBeenCalled();
  });

  it('allows glTF cameras in reject-unsupported mode but rejects them in reject-degraded mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.cameras = [{
      type: 'orthographic',
      name: 'Ortho Preview',
      orthographic: { xmag: 2, ymag: 1, znear: 0.01, zfar: 20 },
    }];
    gltf.nodes![0] = {
      ...gltf.nodes![0]!,
      name: 'Preview Camera Node',
      camera: 0,
      translation: [0, 4, 8],
    };

    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    const accepted = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    });

    expect(accepted.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'ignored-camera',
        path: 'cameras[0]',
      }),
    ]));
    expect(accepted.asset.cameras).toEqual([
      expect.objectContaining({
        cameraIndex: 0,
        nodeIndex: 0,
        type: 'orthographic',
        name: 'Ortho Preview',
        nodeName: 'Preview Camera Node',
        orthographic: {
          xmag: 2,
          ymag: 1,
          znear: 0.01,
          zfar: 20,
        },
      }),
    ]);
    expect(accepted.controller.cameras).toHaveLength(1);
    expect(accepted.controller.cameras[0]!.worldMatrix[13]).toBeCloseTo(4);
    expect(accepted.controller.cameras[0]!.worldMatrix[14]).toBeCloseTo(8);
    expect(createEngine).toHaveBeenCalledTimes(1);

    const createRejectedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine: createRejectedEngine,
    })).rejects.toThrow('scene:cameras=approximate at cameras[0]');

    expect(createRejectedEngine).not.toHaveBeenCalled();
  });

  it('allows generated tangents in reject-unsupported mode but rejects them in reject-degraded mode', async () => {
    const { gltf, buffers } = makeInlineNormalMappedGltf();

    const createAcceptedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    const accepted = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine: createAcceptedEngine,
    });
    expect(accepted.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'generated-tangents',
        path: 'meshes[0].primitives[0].attributes.TANGENT',
      }),
    ]));
    expect(createAcceptedEngine).toHaveBeenCalledTimes(1);

    const createRejectedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine: createRejectedEngine,
    })).rejects.toThrow(
      'import:generated-tangents=approximate at meshes[0].primitives[0].attributes.TANGENT',
    );

    expect(createRejectedEngine).not.toHaveBeenCalled();
  });

  it('classifies malformed authored tangents as degraded import diagnostics', async () => {
    const { gltf, buffers } = makeInlineNormalMappedGltf();
    const tangentData = f32Buffer([
      1, 0, 0,
      1, 0, 0,
      1, 0, 0,
    ]);
    const tangentAccessor = gltf.accessors!.length;
    const tangentBufferView = gltf.bufferViews!.length;
    const tangentBuffer = gltf.buffers!.length;
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      attributes: {
        ...gltf.meshes![0]!.primitives[0]!.attributes,
        TANGENT: tangentAccessor,
      },
    };
    gltf.accessors!.push({
      bufferView: tangentBufferView,
      componentType: 5126,
      count: 3,
      type: 'VEC3',
    });
    gltf.bufferViews!.push({
      buffer: tangentBuffer,
      byteOffset: 0,
      byteLength: tangentData.byteLength,
    });
    gltf.buffers!.push({ byteLength: tangentData.byteLength });
    buffers.set(tangentBuffer, tangentData);

    const createAcceptedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    const accepted = await loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine: createAcceptedEngine,
    });

    expect(accepted.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'invalid-primitive-attribute',
        path: 'meshes[0].primitives[0].attributes.TANGENT',
      }),
      expect.objectContaining({
        code: 'generated-tangents',
        path: 'meshes[0].primitives[0].attributes.TANGENT',
      }),
    ]));
    expect(createAcceptedEngine).toHaveBeenCalledTimes(1);

    const createRejectedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine: createRejectedEngine,
    })).rejects.toThrow(
      'primitive:accessor.attributes.TANGENT.invalid-accessor-type=approximate ' +
      `at accessors[${tangentAccessor}].type`,
    );

    expect(createRejectedEngine).not.toHaveBeenCalled();
  });

  it('allows generated flat normals in reject-unsupported mode but rejects them in reject-degraded mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    delete gltf.meshes![0]!.primitives[0]!.attributes.NORMAL;

    const createAcceptedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    const accepted = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine: createAcceptedEngine,
    });
    expect(accepted.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'generated-flat-normals',
        path: 'meshes[0].primitives[0].attributes.NORMAL',
      }),
    ]));
    expect(createAcceptedEngine).toHaveBeenCalledTimes(1);

    const createRejectedEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine: createRejectedEngine,
    })).rejects.toThrow(
      'import:generated-flat-normals=approximate at meshes[0].primitives[0].attributes.NORMAL',
    );

    expect(createRejectedEngine).not.toHaveBeenCalled();
  });

  it('rejects missing tangent texcoord diagnostics in reject-degraded mode before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineNormalMappedGltf({ includeUv0: false });
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine,
    })).rejects.toThrow(
      'import:missing-tangent-texcoord=approximate at meshes[0].primitives[0].attributes.TEXCOORD_0',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects embedded image acquisition diagnostics in reject-degraded mode before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineNormalMappedGltf();
    gltf.images![0]!.bufferView = 99;
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine,
    })).rejects.toThrow(
      'import:image-buffer-view-not-found=approximate at images[0].bufferView',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects missing material texture indices in reject-degraded mode before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineNormalMappedGltf();
    gltf.materials![0]!.normalTexture = { index: 99 };
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine,
    })).rejects.toThrow(
      'import:material-texture-not-found=approximate at materials[0].normalTexture.index',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('reports and rejects missing base primitive material indices in reject-degraded mode', async () => {
    const { gltf, buffers } = makeInlineNormalMappedGltf();
    gltf.meshes![0]!.primitives[0]!.material = 99;
    const report = analyzeGltfAsset(gltf);
    expect(report.materials.primitiveMaterialReferenceIssues).toEqual([
      expect.objectContaining({
        kind: 'missing-material',
        path: 'meshes[0].primitives[0].material',
        meshIndex: 0,
        primitiveIndex: 0,
        materialIndex: 99,
      }),
    ]);
    expect(evaluateGltfBackendCompatibility(report, 'pt-webgl2').issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        category: 'material',
        name: 'primitive.material.missing-material',
        support: 'approximate',
        path: 'meshes[0].primitives[0].material',
      }),
    ]));
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine,
    })).rejects.toThrow(
      'import:material-not-found=approximate at meshes[0].primitives[0].material',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects ignored high-UV material texture diagnostics in reject-degraded mode before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineNormalMappedGltf({ texCoord: 2 });
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine,
    })).rejects.toThrow(
      'import:ignored-material-texcoord=approximate at meshes[0].primitives[0].material',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects opaque texture handles in reject-degraded mode unless the host opts in', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    })).rejects.toThrow(
      'texture:baseColorMap=requires-hook at materials[0].pbrMetallicRoughness.baseColorTexture (opaque)',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects malformed data URI texture acquisition diagnostics in reject-degraded mode', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.images![0] = { uri: 'data:image/png;base64' };
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine,
    })).rejects.toThrow('import:malformed-data-uri=approximate at images[0].uri');

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects malformed sparse accessor diagnostics in reject-degraded mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.accessors![0] = {
      ...gltf.accessors![0]!,
      sparse: {
        count: 1,
        indices: { bufferView: 0, componentType: 5120 },
        values: { bufferView: 0 },
      },
    };
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    })).rejects.toThrow(
      'import:invalid-sparse-indices-component-type=approximate at accessors[0].sparse.indices.componentType',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects unknown material extension diagnostics in reject-degraded mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      material: 0,
    };
    gltf.materials = [{
      extensions: {
        VENDOR_material_magic: { mode: 'mystery' },
      },
    }];
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    })).rejects.toThrow(
      'import:unknown-material-extension=approximate at materials[0].extensions.VENDOR_material_magic',
    );

    expect(createEngine).not.toHaveBeenCalled();
  });

  it('allows opaque texture handles in strict mode when the host asserts backend readiness', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const engine = { setScene: vi.fn() };

    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine: async () => engine,
    })).resolves.toMatchObject({
      backend: 'pt-webgl2',
      attached: true,
    });

    expect(engine.setScene).toHaveBeenCalledTimes(1);
  });

  it('can decode textures before engine attachment and surface decode diagnostics', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const engine = { setScene: vi.fn() };
    const decodePixels = vi.fn((
      _handle: Parameters<DecodeGltfTexturePixelsFn>[0],
      context: Parameters<DecodeGltfTexturePixelsFn>[1],
    ) => ({
      width: 4,
      height: 2,
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: context.colorSpace,
      data: new Uint8Array(4 * 2 * 4).fill(255),
    }));

    const result = await loadGltfForEngine(gltf, {
      buffers,
      engine,
      decodeTextures: true,
      decodeImage: async (data: Uint8Array, mimeType: string) => ({ kind: 'raw-image', mimeType, data }),
      decodePixels,
      maxTextureSize: 2,
    });

    expect(result.attached).toBe(true);
    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(result.decodedTextureCount).toBe(1);
    expect(result.unchangedTextureCount).toBe(0);
    expect(result.textureDecodeReport.cpuReadableCount).toBe(1);
    expect(result.textureDecodeDiagnostics).toEqual([
      expect.objectContaining({
        code: 'decoded-texture-exceeds-max-size',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        materialField: 'baseColorMap',
        resizedWidth: 2,
        resizedHeight: 1,
      }),
    ]);
    expect(result.textureDecodeWarnings).toEqual([
      expect.stringContaining('exceeds maxTextureSize=2'),
    ]);
    expect(result.warnings).toEqual(expect.arrayContaining([...result.textureDecodeWarnings]));
    expect(result.asset.backendCompatibility).toEqual(expect.arrayContaining([
      expect.objectContaining({
        backend: 'pt-webgl2',
        issues: expect.arrayContaining([
          expect.objectContaining({
            category: 'texture',
            name: 'texture-decode:decoded-texture-exceeds-max-size:baseColorMap',
            support: 'approximate',
            path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
          }),
        ]),
      }),
    ]));

    const attachedScene = engine.setScene.mock.calls[0]![0] as Scene;
    const primitive = attachedScene.primitives[0] as MeshPrimitive;
    const handle = (primitive.material.baseColorMap as TextureRef).handle as {
      width: number;
      height: number;
      data: Float32Array;
    };
    expect(handle.width).toBe(2);
    expect(handle.height).toBe(1);
    expect(handle.data).toBeInstanceOf(Float32Array);
    expect(result.asset.scene).toBe(attachedScene);
  });

  it('rejects degraded texture decode diagnostics before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodeTextures: true,
      decodeImage: async (data: Uint8Array, mimeType: string) => ({ kind: 'raw-image', mimeType, data }),
      decodePixels: () => ({
        width: 4,
        height: 2,
        channels: 4 as const,
        dataType: 'uint8' as const,
        colorSpace: 'srgb' as const,
        data: new Uint8Array(4 * 2 * 4).fill(255),
      }),
      maxTextureSize: 2,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    })).rejects.toThrow(
      'texture:texture-decode:decoded-texture-exceeds-max-size:baseColorMap=approximate at materials[0].pbrMetallicRoughness.baseColorTexture',
    );
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('loadGltfForEngine decodes browser image bytes without requiring a custom decodeImage hook', async () => {
    const { gltf, buffers, png } = makeInlineTexturedGltf();
    const engine = { setScene: vi.fn() };
    const decodePixels = vi.fn((
      handle: Parameters<DecodeGltfTexturePixelsFn>[0],
      context: Parameters<DecodeGltfTexturePixelsFn>[1],
    ) => {
      expect(handle).toMatchObject({
        kind: 'raw-image',
        mimeType: 'image/png',
      });
      expect(handle.data).toEqual(png);
      return {
        width: 2,
        height: 2,
        channels: 4 as const,
        dataType: 'uint8' as const,
        colorSpace: context.colorSpace,
        data: new Uint8Array(2 * 2 * 4).fill(255),
      };
    });

    await withCreateImageBitmapStub(async (createImageBitmap) => {
      const result = await loadGltfForEngine(gltf, {
        buffers,
        engine,
        decodeTextures: true,
        decodePixels,
        attachScene: false,
      });

      expect(createImageBitmap).not.toHaveBeenCalled();
      expect(decodePixels).toHaveBeenCalledTimes(1);
      expect(engine.setScene).not.toHaveBeenCalled();
      expect(result.attached).toBe(true);
      expect(result.decodedTextureCount).toBe(1);
      expect(result.unchangedTextureCount).toBe(0);
      expect(result.textureDecodeDiagnostics).toEqual([]);
      expect(result.textureDecodeReport).toMatchObject({
        rawImageCount: 0,
        opaqueHandleCount: 0,
        cpuReadableCount: 1,
      });
    });
  });

  it('preserves KHR_materials_variants metadata on bridge-created controllers', async () => {
    const { gltf, buffers } = makeInlineMaterialVariantGltf();
    const engine = { setScene: vi.fn(), updatePrimitive: vi.fn() };

    const result = await loadGltfForEngine(gltf, {
      buffers,
      engine,
      backend: 'pt-webgl2',
    });

    expect((result.controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([1, 0, 0]);
    const frame = result.controller.setVariant('blue');

    expect(frame.variantIndex).toBe(0);
    expect(frame.usedSetScene).toBe(false);
    expect(engine.updatePrimitive).toHaveBeenCalledWith(
      'gltf-prim-0',
      expect.objectContaining({
        material: expect.objectContaining({ baseColor: [0, 0, 1] }),
      }),
    );
    expect((result.controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([0, 0, 1]);
  });

  it('preserves KHR_animation_pointer material bindings on bridge-created controllers', async () => {
    const { gltf, buffers } = makeInlineMaterialPointerAnimationGltf();
    const engine = { setScene: vi.fn(), updatePrimitive: vi.fn(), reset: vi.fn() };

    const result = await loadGltfForEngine(gltf, {
      buffers,
      engine,
      backend: 'pt-webgl2',
      attachScene: false,
    });

    const frame = result.controller.applyAnimation('material-color', 1);

    expect(frame.diagnostics).toEqual([]);
    expect(frame.usedSetScene).toBe(false);
    expect(engine.setScene).not.toHaveBeenCalled();
    expect(engine.updatePrimitive).toHaveBeenCalledWith(
      'gltf-prim-0',
      expect.objectContaining({
        material: expect.objectContaining({ baseColor: [0, 1, 0] }),
      }),
    );
    expect(engine.reset).toHaveBeenCalledTimes(1);
    expect((result.controller.scene.primitives[0] as MeshPrimitive).material.baseColor).toEqual([0, 1, 0]);
  });

  it('rejects malformed root KHR_materials_variants lists before constructing engines', async () => {
    const { gltf, buffers } = makeInlineMaterialVariantGltf();
    (gltf.extensions as Record<string, unknown>).KHR_materials_variants = {
      variants: { name: 'blue' },
    };
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    const promise = loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
      materialVariant: 'blue',
    });

    await expect(promise).rejects.toThrow(
      'material:KHR_materials_variants.variants.malformed-list=unsupported at extensions.KHR_materials_variants.variants',
    );
    await expect(promise).rejects.toBeInstanceOf(GltfCompatibilityError);
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('keeps inactive material variants decoded before controller variant patches', async () => {
    const { gltf, buffers } = makeInlineTexturedVariantGltf();
    const engine = { setScene: vi.fn(), updatePrimitive: vi.fn(), reset: vi.fn() };
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => ({
      width: 1,
      height: 1,
      data: new Uint8Array([128, 64, 255, 255]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: context.colorSpace,
    }));

    const result = await loadGltfForEngine(gltf, {
      buffers,
      engine,
      backend: 'pt-webgl2',
      decodePixels,
      attachScene: false,
    });

    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(result.decodedTextureCount).toBe(1);
    expect(result.textureDecodeDiagnostics).toEqual([]);
    expect(result.textureDecodeReport.entries).toEqual(expect.arrayContaining([
      expect.objectContaining({
        primitiveId: 'gltf-material-1',
        materialField: 'baseColorMap',
        path: 'materials[1].pbrMetallicRoughness.baseColorTexture',
        handleKind: 'pixel-data',
        handleColorSpace: 'linear',
      }),
    ]));

    const frame = result.controller.setVariant('textured');

    expect(frame.usedSetScene).toBe(false);
    expect(engine.updatePrimitive).toHaveBeenCalledTimes(1);
    const patch = engine.updatePrimitive.mock.calls[0]![1] as { material: MeshPrimitive['material'] };
    const baseColorMap = patch.material.baseColorMap as TextureRef;
    const handle = baseColorMap.handle as { data: Float32Array; __vitrum_hint__: { colorSpace: string } };
    expect(handle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    expect(handle.data[0]).toBeCloseTo(srgbToLinearForTest(128 / 255));
    expect(handle.data[1]).toBeCloseTo(srgbToLinearForTest(64 / 255));
    expect(handle.data[2]).toBeCloseTo(1);
  });

  it('keeps generated spec-gloss roughness maps in controller variant patches', async () => {
    const { gltf, buffers } = makeInlineSpecGlossVariantGltf();
    const engine = { setScene: vi.fn(), updatePrimitive: vi.fn(), reset: vi.fn() };
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => ({
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 128]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: context.colorSpace,
    }));

    const result = await loadGltfForEngine(gltf, {
      buffers,
      engine,
      backend: 'pt-webgl2',
      decodePixels,
      attachScene: false,
    });

    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(result.decodedTextureCount).toBe(2);
    expect(result.textureDecodeDiagnostics).toEqual([]);
    expect(result.textureDecodeReport.entries.map((entry) => entry.materialField).sort()).toEqual([
      'roughnessMap',
      'specularColorMap',
    ]);

    const frame = result.controller.setVariant('glossy');

    expect(frame.usedSetScene).toBe(false);
    expect(engine.updatePrimitive).toHaveBeenCalledTimes(1);
    const patch = engine.updatePrimitive.mock.calls[0]![1] as { material: MeshPrimitive['material'] };
    const roughness = patch.material.roughnessMap as TextureRef | undefined;
    expect(roughness).toBeDefined();
    const handle = roughness!.handle as { width: number; height: number; data: Float32Array; __vitrum_hint__: unknown };
    expect(handle.width).toBe(1);
    expect(handle.height).toBe(1);
    expect(handle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    const expected = 1 - 0.5 * (128 / 255);
    expect(Array.from(handle.data.slice(0, 4))).toEqual([
      expect.closeTo(expected),
      expect.closeTo(expected),
      expect.closeTo(expected),
      1,
    ]);
  });

  it('reports inactive material variant textures before decode', async () => {
    const { gltf, buffers } = makeInlineTexturedVariantGltf();

    const result = await loadGltfAsset(gltf, { buffers });

    expect(result.textureDecodeReport.entries).toEqual(expect.arrayContaining([
      expect.objectContaining({
        primitiveId: 'gltf-material-1',
        materialField: 'baseColorMap',
        path: 'materials[1].pbrMetallicRoughness.baseColorTexture',
        handleKind: 'raw-image',
      }),
    ]));
  });

  it('preserves EXT_mesh_gpu_instancing metadata on bridge-created controllers', async () => {
    const { gltf, buffers } = makeInlineAnimatedInstancedGltf();
    const engine = { setScene: vi.fn(), updatePrimitive: vi.fn(), reset: vi.fn() };

    const result = await loadGltfForEngine(gltf, {
      buffers,
      engine,
      backend: 'pt-webgl2',
      attachScene: false,
    });

    const primitive = result.controller.scene.primitives[0] as InstancedMeshPrimitive;
    expect(primitive.kind).toBe('instanced-mesh');

    const frame = result.controller.applyAnimation('instance-slide', 0.5);

    expect(frame.usedSetScene).toBe(false);
    expect(engine.setScene).not.toHaveBeenCalled();
    expect(engine.updatePrimitive).toHaveBeenCalledTimes(1);
    expect(engine.reset).toHaveBeenCalledTimes(1);
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
  });

  it('can reject a selected backend before construction when compatibility would drop material fidelity', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();

    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'walkaround-hybrid',
      compatibilityMode: 'reject-degraded',
      createEngine: async () => ({ setScene: vi.fn() }),
    })).rejects.toThrow('baseColorMap');
  });

  it('deduplicates strict texture-readiness failures into the canonical decode-report message', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();

    let error: unknown;
    try {
      await loadGltfForEngine(gltf, {
        buffers,
        decodeImage: async () => ({ kind: 'decoded-texture' }),
        backend: 'pt-webgl2',
        compatibilityMode: 'reject-degraded',
        createEngine: async () => ({ setScene: vi.fn() }),
      });
    } catch (err) {
      error = err;
    }

    expect(error).toBeInstanceOf(GltfCompatibilityError);
    const failures = (error as GltfCompatibilityError).failures;
    expect(failures).toContain(
      'texture:baseColorMap=requires-hook at materials[0].pbrMetallicRoughness.baseColorTexture (opaque)',
    );
    expect(failures).not.toContain(
      'texture:texture-readiness:baseColorMap=requires-hook at materials[0].pbrMetallicRoughness.baseColorTexture',
    );
    expect(failures.filter((failure) => failure.includes('baseColorMap'))).toHaveLength(1);
    expect((error as GltfCompatibilityError).failureDetails).toContainEqual(expect.objectContaining({
      source: 'texture-readiness',
      category: 'texture',
      name: 'baseColorMap',
      support: 'requires-hook',
      path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
      materialField: 'baseColorMap',
      status: 'opaque',
    }));
  });

  it('accepts authored sampler policies on pt-webgl2 before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.textures![0] = { ...gltf.textures![0]!, sampler: 0 };
    gltf.samplers = [{ magFilter: 9728, minFilter: 9984 }];
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    const decodePixels = vi.fn(() => ({
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 255]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'srgb' as const,
    }));

    await loadGltfForEngine(gltf, {
      buffers,
      decodePixels,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    });

    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(createEngine).toHaveBeenCalledTimes(1);
  });

  it('keeps structured import diagnostic details on strict compatibility errors', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.nodes = [{ mesh: 0, camera: 0 }];
    gltf.cameras = [{}];
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    let error: unknown;
    try {
      await loadGltfForEngine(gltf, {
        buffers,
        backend: 'pt-webgl2',
        compatibilityMode: 'reject-degraded',
        createEngine,
      });
    } catch (err) {
      error = err;
    }

    expect(error).toBeInstanceOf(GltfCompatibilityError);
    expect((error as GltfCompatibilityError).failureDetails).toContainEqual(expect.objectContaining({
      source: 'import-diagnostic',
      category: 'import',
      code: 'ignored-camera',
      support: 'approximate',
      path: 'cameras[0]',
    }));
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('reports malformed material texture samplers during best-effort import', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.textures![0] = { ...gltf.textures![0]!, sampler: 0 };
    gltf.samplers = [{
      wrapS: 123 as never,
      magFilter: 111 as never,
      minFilter: 222 as never,
    }];

    const result = await loadGltfAsset(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
    });

    expect(result.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'invalid-material-texture-sampler',
        path: 'samplers[0].wrapS',
        materialIndex: 0,
        textureIndex: 0,
        samplerIndex: 0,
        samplerProperty: 'wrapS',
        samplerValue: 123,
      }),
      expect.objectContaining({
        code: 'invalid-material-texture-sampler',
        path: 'samplers[0].minFilter',
        materialIndex: 0,
        textureIndex: 0,
        samplerIndex: 0,
        samplerProperty: 'minFilter',
        samplerValue: 222,
      }),
    ]));
    const baseColor = result.scene.primitives[0]!.material.baseColorMap as TextureRef;
    expect(baseColor.wrapS).toBeUndefined();
    expect(baseColor.magFilter).toBeUndefined();
    expect(baseColor.minFilter).toBeUndefined();
  });

  it('keeps malformed sampler diagnostics visible while reject-unsupported accepts the approximation', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.textures![0] = { ...gltf.textures![0]!, sampler: 0 };
    gltf.samplers = [{
      wrapS: 123 as never,
      minFilter: 222 as never,
    }];
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    const result = await loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    });

    expect(createEngine).toHaveBeenCalledTimes(1);
    expect(result.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: 'invalid-material-texture-sampler',
        path: 'samplers[0].wrapS',
        samplerProperty: 'wrapS',
        samplerValue: 123,
      }),
      expect.objectContaining({
        code: 'invalid-material-texture-sampler',
        path: 'samplers[0].minFilter',
        samplerProperty: 'minFilter',
        samplerValue: 222,
      }),
    ]));
  });

  it('rejects malformed texture sampler policies before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.textures![0] = { ...gltf.textures![0]!, sampler: 0 };
    gltf.samplers = [{ minFilter: 222 as never }];
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    })).rejects.toThrow('material:baseColorMap.samplerPolicy.invalid-min-filter=approximate at samplers[0].minFilter');
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects spec-gloss alpha degradation before construction when no CPU-linear bake is available', async () => {
    const { gltf, buffers } = makeInlineSpecGlossTexturedGltf();
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    let error: unknown;
    try {
      await loadGltfForEngine(gltf, {
        buffers,
        decodeImage: async () => ({ kind: 'decoded-texture' }),
        backend: 'pt-webgl2',
        compatibilityMode: 'reject-degraded',
        createEngine,
      });
    } catch (err) {
      error = err;
    }

    expect(error).toBeInstanceOf(GltfCompatibilityError);
    const failures = (error as GltfCompatibilityError).failures;
    expect(failures).toContain(
      'material:KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha=approximate at materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
    );
    expect(failures).toContain(
      'import:spec-gloss-approximation=approximate at materials[0].extensions.KHR_materials_pbrSpecularGlossiness',
    );
    expect(failures).toContain(
      'import:spec-gloss-texture-alpha-approximation=approximate at materials[0].extensions.KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture',
    );
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('does not keep the spec-gloss alpha degradation after the CPU-linear roughness bake succeeds', async () => {
    const { gltf, buffers } = makeInlineSpecGlossTexturedGltf();
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => ({
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 128]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: context.colorSpace,
    }));

    let message = '';
    try {
      await loadGltfForEngine(gltf, {
        buffers,
        decodePixels,
        backend: 'pt-webgl2',
        compatibilityMode: 'reject-degraded',
        createEngine,
      });
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }

    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(createEngine).not.toHaveBeenCalled();
    expect(message).toContain('material:KHR_materials_pbrSpecularGlossiness=approximate');
    expect(message).not.toContain('specularGlossinessTexture.glossinessAlpha');
    expect(message).not.toContain('spec-gloss-texture-alpha-approximation');
  });

  it('does not satisfy spec-gloss alpha degradation with an unrelated decoded roughnessMap', async () => {
    const { gltf, buffers } = makeInlineSpecGlossTexturedGltf();
    const specGloss = gltf.materials![0]!.extensions!.KHR_materials_pbrSpecularGlossiness!;
    specGloss.specularGlossinessTexture = { index: 99 };
    gltf.meshes![0]!.primitives.push({ attributes: { POSITION: 0 }, material: 1 });
    gltf.materials!.push({
      pbrMetallicRoughness: {
        baseColorFactor: [1, 1, 1, 1],
        metallicRoughnessTexture: { index: 0 },
      },
    });
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => ({
      width: 1,
      height: 1,
      data: new Uint8Array([255, 255, 255, 128]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: context.colorSpace,
    }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodePixels,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    })).rejects.toThrow(
      'material:KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha=approximate',
    );

    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('attaches the decoded spec-gloss roughness bake to the engine scene in best-effort mode', async () => {
    const { gltf, buffers } = makeInlineSpecGlossTexturedGltf();
    const engine = { setScene: vi.fn(), updatePrimitive: vi.fn() };
    const decodePixels = vi.fn((...[, context]: Parameters<DecodeGltfTexturePixelsFn>) => ({
      width: 2,
      height: 1,
      data: new Uint8Array([
        255, 0, 0, 128,
        0, 255, 0, 64,
      ]),
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: context.colorSpace,
    }));

    const result = await loadGltfForEngine(gltf, {
      buffers,
      engine,
      backend: 'pt-webgl2',
      decodePixels,
    });

    expect(result.attached).toBe(true);
    expect(decodePixels).toHaveBeenCalledTimes(1);
    expect(result.decodedTextureCount).toBe(2);
    expect(result.unchangedTextureCount).toBe(0);
    expect(result.textureDecodeDiagnostics).toEqual([]);
    expect(result.textureDecodeReport.entries.map((entry) => entry.materialField).sort()).toEqual([
      'roughnessMap',
      'specularColorMap',
    ]);
    expect(engine.setScene).toHaveBeenCalledTimes(1);
    expect(engine.setScene).toHaveBeenCalledWith(result.controller.scene);

    const attachedScene = engine.setScene.mock.calls[0]![0] as Scene;
    const primitive = attachedScene.primitives[0] as MeshPrimitive;
    const specular = primitive.material.specularColorMap as TextureRef;
    const roughness = primitive.material.roughnessMap as TextureRef;
    expect(roughness.handle).not.toBe(specular.handle);
    expect(roughness.texCoord).toBe(1);
    expect(roughness.transform).toEqual({
      offset: [0.25, 0.5],
      scale: [2, 3],
      rotation: 0.125,
    });

    const handle = roughness.handle as { data: Float32Array; __vitrum_hint__: unknown };
    const first = 1 - 0.5 * (128 / 255);
    const second = 1 - 0.5 * (64 / 255);
    expect(handle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    expect(Array.from(handle.data.slice(0, 4))).toEqual([
      expect.closeTo(first),
      expect.closeTo(first),
      expect.closeTo(first),
      1,
    ]);
    expect(Array.from(handle.data.slice(4, 8))).toEqual([
      expect.closeTo(second),
      expect.closeTo(second),
      expect.closeTo(second),
      1,
    ]);
  });

  it('allows point/line fallback meshes in reject-unsupported mode before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      mode: 1,
    };
    const setScene = vi.fn();
    const createEngine = vi.fn(async () => ({ setScene }));

    await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    });
    expect(createEngine).toHaveBeenCalledTimes(1);
    expect(setScene).toHaveBeenCalledTimes(1);
    const scene = setScene.mock.calls[0]?.[0] as { primitives?: Array<{ positions: Float32Array }> };
    expect(scene.primitives?.[0]?.positions.length).toBeGreaterThan(9);
  });

  it('rejects point/line fallback meshes in reject-degraded mode before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      mode: 1,
    };
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    })).rejects.toThrow('primitive:mode:1=fallback-generated-mesh at meshes[0].primitives[0].mode');
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('allows fallback-expanded instanced morphed meshes in reject-unsupported mode before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addMorphedGpuInstancing(gltf, buffers);
    const setScene = vi.fn();
    const createEngine = vi.fn(async () => ({ setScene }));

    const result = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    });

    expect(createEngine).toHaveBeenCalledTimes(1);
    expect(setScene).toHaveBeenCalledTimes(1);
    expect(result.asset.scene.primitives).toHaveLength(2);
    expect(result.asset.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: 'fallback-expanded-gpu-instancing' }),
    ]));
  });

  it('rejects fallback-expanded instanced morphed meshes in reject-degraded mode before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addMorphedGpuInstancing(gltf, buffers);
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    })).rejects.toThrow(
      'primitive:EXT_mesh_gpu_instancing.skinnedOrMorphed=fallback-generated-mesh at nodes[0].extensions.EXT_mesh_gpu_instancing',
    );
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects unbound skin attributes in reject-unsupported mode before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addUnboundSkinAttributes(gltf, buffers);
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).rejects.toThrow(
      'primitive:skinAttributesWithoutNodeSkin=unsupported at meshes[0].primitives[0].attributes.JOINTS_0',
    );
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('returns structured diagnostics for unbound skin attributes in best-effort mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    addUnboundSkinAttributes(gltf, buffers);

    const asset = await loadGltfAsset(gltf, { buffers });

    expect(asset.scene.primitives[0]?.kind).toBe('mesh');
    expect(asset.diagnostics).toContainEqual(expect.objectContaining({
      code: 'ignored-skin-attributes',
      path: 'meshes[0].primitives[0].attributes.JOINTS_0',
    }));
  });

  it('returns structured diagnostics for skin nodes with missing skin streams in best-effort mode', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.nodes![0] = { ...gltf.nodes![0]!, skin: 0 };
    gltf.nodes!.push({ name: 'joint' });
    gltf.skins = [{ joints: [1] }];

    const asset = await loadGltfAsset(gltf, { buffers });

    expect(asset.scene.primitives[0]?.kind).toBe('mesh');
    expect(asset.diagnostics).toContainEqual(expect.objectContaining({
      code: 'incomplete-skin-attributes',
      path: 'meshes[0].primitives[0].attributes.JOINTS_0',
    }));
  });

  it('allows reject-degraded to use an optional texture-source extension fallback without a hook', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.extensionsUsed = ['KHR_texture_basisu'];
    gltf.textures![0] = {
      ...gltf.textures![0]!,
      extensions: { KHR_texture_basisu: { source: 0 } },
    };

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine: async () => ({ setScene: vi.fn() }),
    })).resolves.toMatchObject({
      backend: 'pt-webgl2',
      attached: true,
    });
  });

  it.each(['reject-unsupported', 'reject-degraded'] as const)(
    'rejects unselected required texture-source extensions with a canonical compatibility error under %s',
    async (compatibilityMode) => {
      const { gltf, buffers } = makeInlineTexturedGltf();
      gltf.extensionsUsed = ['KHR_texture_basisu'];
      gltf.extensionsRequired = ['KHR_texture_basisu'];
      gltf.textures![0] = {
        ...gltf.textures![0]!,
        extensions: { KHR_texture_basisu: { source: 0 } },
      };
      const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

      let error: unknown;
      try {
        await loadGltfForEngine(gltf, {
          buffers,
          backend: 'pt-webgl2',
          compatibilityMode,
          createEngine,
        });
      } catch (caught) {
        error = caught;
      }

      expect(error).toBeInstanceOf(GltfCompatibilityError);
      expect(error).not.toBeInstanceOf(GltfImportError);
      expect(error).toMatchObject({
        code: 'GLTF_COMPATIBILITY_REJECTED',
        backend: 'pt-webgl2',
        profileId: 'pt-webgl2',
        compatibilityMode,
        failures: [
          'extension:KHR_texture_basisu=requires-hook at textures[0].extensions.KHR_texture_basisu',
        ],
      });
      expect(createEngine).not.toHaveBeenCalled();
    },
  );

  it('rejects selected optional texture-source extensions without an image decode hook', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.extensionsUsed = ['EXT_texture_webp'];
    gltf.textures![0] = {
      ...gltf.textures![0]!,
      extensions: { EXT_texture_webp: { source: 0 } },
    };
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      textureSourceExtensions: ['EXT_texture_webp'],
      createEngine,
    })).rejects.toThrow('extension:EXT_texture_webp=requires-hook at textures[0].extensions.EXT_texture_webp');
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('accepts selected optional texture-source extensions with an explicit image decode hook', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.extensionsUsed = ['EXT_texture_webp'];
    gltf.textures![0] = {
      ...gltf.textures![0]!,
      extensions: { EXT_texture_webp: { source: 0 } },
    };

    const result = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      textureSourceExtensions: ['EXT_texture_webp'],
      decodeImage: async () => ({ kind: 'decoded-webp' }),
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine: async () => ({ setScene: vi.fn() }),
    });

    expect(result).toMatchObject({
      backend: 'pt-webgl2',
      attached: true,
    });
    expect(result.textureDecodeReport.entries).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        textureIndex: 0,
        imageIndex: 0,
        textureSourceExtension: 'EXT_texture_webp',
      }),
    ]);
  });

  it('accepts selected texture-source extensions when the texture decode bridge returns CPU pixels', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.extensionsUsed = ['EXT_texture_webp'];
    gltf.images![0] = {
      ...gltf.images![0]!,
      mimeType: 'image/webp',
    };
    gltf.textures![0] = {
      ...gltf.textures![0]!,
      extensions: { EXT_texture_webp: { source: 0 } },
    };
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));
    let decodedHandle: unknown;
    const decodePixels: DecodeGltfTexturePixelsFn = (handle) => {
      decodedHandle = handle;
      return {
        width: 1,
        height: 1,
        data: new Uint8Array([255, 128, 0, 255]),
        channels: 4 as const,
        dataType: 'uint8' as const,
        colorSpace: 'srgb' as const,
      };
    };
    const decodePixelsMock = vi.fn(decodePixels);
    const preDecodeAsset = await loadGltfAsset(gltf, {
      buffers,
      textureSourceExtensions: ['EXT_texture_webp'],
    });
    const preDecodePtWebgl2 = preDecodeAsset.backendCompatibility.find((entry) =>
      entry.backend === 'pt-webgl2' && entry.profileId === 'pt-webgl2'
    );

    const result = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      textureSourceExtensions: ['EXT_texture_webp'],
      decodeTextures: true,
      decodePixels: decodePixelsMock,
      createEngine,
    });

    expect(result).toMatchObject({
      backend: 'pt-webgl2',
      attached: true,
      decodedTextureCount: 1,
      unchangedTextureCount: 0,
      textureDecodeDiagnostics: [],
      textureDecodeWarnings: [],
    });
    expect(createEngine).toHaveBeenCalledTimes(1);
    expect(decodePixelsMock).toHaveBeenCalledTimes(1);
    expect(decodedHandle).toMatchObject({
      kind: 'raw-image',
      mimeType: 'image/webp',
    });
    expect(result.textureDecodeReport.entries).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        textureIndex: 0,
        imageIndex: 0,
        textureSourceExtension: 'EXT_texture_webp',
        handleKind: 'pixel-data',
        backendReadiness: {
          ptWebgl2: 'ready',
          ptWebgpu: 'ready',
          walkaroundHybrid: 'ready',
        },
      }),
    ]);
    const postDecodePtWebgl2 = result.asset.backendCompatibility.find((entry) =>
      entry.backend === 'pt-webgl2' && entry.profileId === 'pt-webgl2'
    );
    expect(preDecodePtWebgl2).toMatchObject({
      requiresHookCount: 2,
    });
    expect(postDecodePtWebgl2).toMatchObject({
      requiresHookCount: 0,
      nativeCount: preDecodePtWebgl2?.nativeCount,
      approximateCount: preDecodePtWebgl2?.approximateCount,
      unsupportedCount: preDecodePtWebgl2?.unsupportedCount,
    });
    expect(postDecodePtWebgl2?.nativeCount).toBeGreaterThan(0);
  });

  it('accepts selected WebP texture-source extensions through the built-in Node decode bridge', async () => {
    const webpBytes = await makeWebpBytes(2, 2, [
      224, 96, 32, 255,
      224, 96, 32, 255,
      224, 96, 32, 255,
      224, 96, 32, 255,
    ]);
    const { gltf, buffers } = makeInlineTexturedGltf(webpBytes);
    gltf.extensionsUsed = ['EXT_texture_webp'];
    gltf.images![0] = {
      ...gltf.images![0]!,
      mimeType: 'image/webp',
    };
    gltf.textures![0] = {
      ...gltf.textures![0]!,
      extensions: { EXT_texture_webp: { source: 0 } },
    };
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    const result = await loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      textureSourceExtensions: ['EXT_texture_webp'],
      decodeTextures: true,
      createEngine,
    });

    expect(result).toMatchObject({
      backend: 'pt-webgl2',
      attached: true,
      decodedTextureCount: 1,
      unchangedTextureCount: 0,
      textureDecodeDiagnostics: [],
      textureDecodeWarnings: [],
    });
    expect(createEngine).toHaveBeenCalledTimes(1);
    const primitive = result.asset.scene.primitives[0] as MeshPrimitive;
    const ref = primitive.material.baseColorMap as TextureRef;
    const handle = ref.handle as {
      data: Float32Array;
      __vitrum_hint__: { channels: number; dataType: string; colorSpace: string };
    };
    expect(handle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    expect(handle.data[0]).toBeCloseTo(srgbToLinearForTest(224 / 255), 0);
    expect(handle.data[1]).toBeCloseTo(srgbToLinearForTest(96 / 255), 0);
    expect(handle.data[2]).toBeCloseTo(srgbToLinearForTest(32 / 255), 0);
    expect(handle.data[3]).toBeCloseTo(1);
    expect(result.textureDecodeReport.entries).toEqual([
      expect.objectContaining({
        materialField: 'baseColorMap',
        textureIndex: 0,
        imageIndex: 0,
        imageMimeType: 'image/webp',
        textureSourceExtension: 'EXT_texture_webp',
        handleKind: 'pixel-data',
        backendReadiness: {
          ptWebgl2: 'ready',
          ptWebgpu: 'ready',
          walkaroundHybrid: 'ready',
        },
      }),
    ]);
    expect(result.asset.backendCompatibility.find((entry) =>
      entry.backend === 'pt-webgl2' && entry.profileId === 'pt-webgl2'
    )).toMatchObject({
      requiresHookCount: 0,
    });
  });

  it('does not treat an explicitly enabled texture-source extension as a missing host hook', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.extensionsUsed = ['KHR_texture_basisu'];
    gltf.extensionsRequired = ['KHR_texture_basisu'];
    gltf.textures![0] = {
      ...gltf.textures![0]!,
      extensions: { KHR_texture_basisu: { source: 0 } },
    };

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      textureSourceExtensions: ['KHR_texture_basisu'],
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine: async () => ({ setScene: vi.fn() }),
    })).resolves.toMatchObject({
      backend: 'pt-webgl2',
      attached: true,
    });
  });
});
