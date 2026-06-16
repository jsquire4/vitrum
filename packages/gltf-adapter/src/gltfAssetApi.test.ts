// gltfAssetApi.test.ts — higher-level arbitrary-asset API tests.
//
// These fixtures exercise the public package root, not private helpers: URL
// loading with external resources, structured feature reporting, and backend
// compatibility ranking against the core promise ledger.

import { describe, expect, it, vi } from 'vitest';
import {
  analyzeGltfAsset,
  decodeSceneTextures,
  evaluateGltfBackendCompatibility,
  evaluateGltfBackendProfileCompatibility,
  GltfFetchFailed,
  GltfResourceNotFound,
  loadGltfAndDecodeTextures,
  loadGltfForEngine,
  loadGltfAsset,
  rankGltfBackends,
} from './index.js';
import type { DecodeGltfTexturePixelsFn, GltfAssetFetchResponse, GltfJson } from './index.js';
import type { MeshPrimitive, Scene, TextureRef } from '@vitrum/core';

function f32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function bytes(values: number[]): ArrayBuffer {
  return new Uint8Array(values).buffer;
}

function textBuffer(text: string): ArrayBuffer {
  const encoded = new TextEncoder().encode(text);
  return encoded.buffer.slice(encoded.byteOffset, encoded.byteOffset + encoded.byteLength);
}

function srgbToLinearForTest(value: number): number {
  const c = Math.max(0, Math.min(1, value));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function linearToSrgbForTest(value: number): number {
  const c = Math.max(0, Math.min(1, value));
  return c <= 0.0031308 ? c * 12.92 : 1.055 * (c ** (1 / 2.4)) - 0.055;
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
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  return {
    gltf: {
      asset: { version: '2.0' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0 }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
      buffers: [{ byteLength: positions.byteLength }],
    },
    buffers: new Map([[0, positions]]),
  };
}

function makeInlineVertexColorGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
  const positions = f32Buffer([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const colors = f32Buffer([1, 0, 0, 0, 1, 0, 0, 0, 1]);
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
        { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
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

function makeInlineTexturedGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
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
        pbrMetallicRoughness: {
          baseColorFactor: [1, 1, 1, 1],
          baseColorTexture: { index: 0 },
        },
      }],
      textures: [{ source: 0 }],
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
  });

  it('throws a deterministic error for relative external resources without a baseUri', async () => {
    const gltf = makeExternalTexturedGltf();
    await expect(loadGltfAsset(gltf)).rejects.toBeInstanceOf(GltfResourceNotFound);
    await expect(loadGltfAsset(gltf)).rejects.toMatchObject({
      kind: 'buffer',
      url: 'mesh.bin',
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

    const primitive = result.scene.primitives[0] as MeshPrimitive;
    const ref = primitive.material.baseColorMap as TextureRef;
    const handle = ref.handle as { data: Float32Array; __vitrum_hint__: { colorSpace: string } };
    expect(handle.__vitrum_hint__).toEqual({ channels: 4, dataType: 'float32', colorSpace: 'linear' });
    expect(handle.data[0]).toBeCloseTo(srgbToLinearForTest(128 / 255));
    expect(handle.data[1]).toBeCloseTo(srgbToLinearForTest(64 / 255));
    expect(handle.data[2]).toBeCloseTo(1);
    expect(handle.data[3]).toBeCloseTo(128 / 255);
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
        } as MeshPrimitive,
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const result = await decodeSceneTextures(scene, { target: 'cpu-linear' });

    expect(result.decodedCount).toBe(1);
    expect(result.unchangedCount).toBe(1);
    expect(result.diagnostics).toEqual([]);
    const material = (result.scene.primitives[0] as MeshPrimitive).material;
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
        } as MeshPrimitive,
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
    gltf.animations = [{
      samplers: [{ input: 2, output: 3, interpolation: 'STEP' }],
      channels: [{ sampler: 0, target: { node: 0, path: 'translation' } }],
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
    expect(report.animations.paths).toEqual(['translation']);
    expect(report.animations.interpolations).toEqual(['STEP']);

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

  it('reports EXT_mesh_gpu_instancing as an explicit unsupported extension with node source path', () => {
    const report = analyzeGltfAsset({
      asset: { version: '2.0' },
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

    expect(report.extensions.unsupportedOptional).toContain('EXT_mesh_gpu_instancing');
    const compatibility = evaluateGltfBackendCompatibility(report, 'pt-webgl2');
    expect(compatibility.issues).toContainEqual(expect.objectContaining({
      category: 'extension',
      name: 'EXT_mesh_gpu_instancing',
      support: 'unsupported',
      path: 'nodes[0].extensions.EXT_mesh_gpu_instancing',
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
        category: 'extension',
        name: 'KHR_draco_mesh_compression',
        support: 'requires-hook',
        path: 'extensionsUsed[1]',
      }),
      expect.objectContaining({
        category: 'scene',
        name: 'cameras',
        support: 'unsupported',
        path: 'cameras[0]',
      }),
      expect.objectContaining({
        category: 'primitive',
        name: 'mode:1',
        support: 'unsupported',
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

  it('rejects direct pt-webgpu-lite strict loads before constructing unsupported COLOR_0 scenes', async () => {
    const { gltf, buffers } = makeInlineVertexColorGltf();
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

  it('allows opaque texture handles in strict mode when the host asserts backend readiness', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    const engine = { setScene: vi.fn() };

    await expect(loadGltfForEngine(gltf, {
      buffers,
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

  it('rejects degraded authored sampler policies before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTexturedGltf();
    gltf.textures![0] = { ...gltf.textures![0]!, sampler: 0 };
    gltf.samplers = [{ magFilter: 9728, minFilter: 9984 }];
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    })).rejects.toThrow('material:baseColorMap.samplerPolicy=approximate at samplers[0].minFilter');
    expect(createEngine).not.toHaveBeenCalled();
  });

  it('rejects spec-gloss alpha degradation before construction when no CPU-linear bake is available', async () => {
    const { gltf, buffers } = makeInlineSpecGlossTexturedGltf();
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      decodeImage: async () => ({ kind: 'decoded-texture' }),
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      createEngine,
    })).rejects.toThrow(
      'material:KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha=approximate',
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

  it('rejects unsupported point/line primitive modes before constructing an engine', async () => {
    const { gltf, buffers } = makeInlineTriangleGltf();
    gltf.meshes![0]!.primitives[0] = {
      ...gltf.meshes![0]!.primitives[0]!,
      mode: 1,
    };
    const createEngine = vi.fn(async () => ({ setScene: vi.fn() }));

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-unsupported',
      createEngine,
    })).rejects.toThrow('primitive:mode:1=unsupported at meshes[0].primitives[0].mode');
    expect(createEngine).not.toHaveBeenCalled();
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

    await expect(loadGltfForEngine(gltf, {
      buffers,
      backend: 'pt-webgl2',
      compatibilityMode: 'reject-degraded',
      textureSourceExtensions: ['EXT_texture_webp'],
      decodeImage: async () => ({ kind: 'decoded-webp' }),
      opaqueTextureHandlesReady: ['pt-webgl2'],
      createEngine: async () => ({ setScene: vi.fn() }),
    })).resolves.toMatchObject({
      backend: 'pt-webgl2',
      attached: true,
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
