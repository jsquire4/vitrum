// gltfAssetApi.test.ts — higher-level arbitrary-asset API tests.
//
// These fixtures exercise the public package root, not private helpers: URL
// loading with external resources, structured feature reporting, and backend
// compatibility ranking against the core promise ledger.

import { describe, expect, it, vi } from 'vitest';
import {
  analyzeGltfAsset,
  evaluateGltfBackendCompatibility,
  evaluateGltfBackendProfileCompatibility,
  GltfFetchFailed,
  GltfResourceNotFound,
  loadGltfAndDecodeTextures,
  loadGltfForEngine,
  loadGltfAsset,
  rankGltfBackends,
} from './index.js';
import type { GltfAssetFetchResponse, GltfJson } from './index.js';
import type { MeshPrimitive, TextureRef } from '@vitrum/core';

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
        path: 'scene.primitives[0].material.baseColorMap',
        wrapS: 'clamp-to-edge',
        wrapT: 'mirrored-repeat',
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
        path: 'scene.primitives[0].material.baseColorMap',
        wrapS: 'repeat',
        wrapT: 'repeat',
        handleKind: 'raw-image',
        backendReadiness: {
          ptWebgl2: 'opaque',
          ptWebgpu: 'opaque',
          walkaroundHybrid: 'ignored',
        },
      }),
    ]);
  });
});

describe('analyzeGltfAsset and compatibility ranking', () => {
  it('reports material fields, unsupported extensions, resources, and animation paths', () => {
    const gltf = makeExternalTexturedGltf();
    gltf.extensionsUsed = ['KHR_materials_unlit', 'KHR_materials_pbrSpecularGlossiness'];
    gltf.materials![0] = {
      ...gltf.materials![0]!,
      normalTexture: { index: 0, texCoord: 1 },
      extensions: {
        KHR_materials_unlit: {},
        KHR_materials_volume: {
          thicknessFactor: 0.5,
          thicknessTexture: { index: 0 },
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
      expect.arrayContaining(['baseColor', 'baseColorMap', 'normalMap', 'shadingModel', 'thickness', 'thicknessMap']),
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
  });

  it('uses the backend promise ledger to keep rich textured assets off walkaround in fidelity mode', () => {
    const report = analyzeGltfAsset(makeExternalTexturedGltf());
    const ranked = rankGltfBackends(report, 'fidelity');
    const walkaround = evaluateGltfBackendCompatibility(report, 'walkaround-hybrid');

    expect(ranked[0]!.backend).toBe('pt-webgl2');
    expect(ranked.map((entry) => entry.profileId)).toEqual([
      'pt-webgl2',
      'pt-webgpu',
      'pt-webgpu-lite',
      'walkaround-hybrid',
    ]);
    expect(walkaround.issues.some((issue) =>
      issue.category === 'material' &&
      issue.name === 'baseColorMap' &&
      issue.support === 'unsupported',
    )).toBe(true);
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

  it('reports morph tangent deltas as an unsupported primitive compatibility issue', () => {
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
      issue.support === 'unsupported',
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

    for (const backend of ['pt-webgpu', 'walkaround-hybrid'] as const) {
      const compatibility = evaluateGltfBackendCompatibility(report, backend);
      expect(compatibility.issues).toContainEqual(expect.objectContaining({
        category: 'primitive',
        name: 'vertexColors',
        support: 'unsupported',
        path: 'meshes[0].primitives[0].attributes.COLOR_0',
      }));
    }
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
        support: 'unsupported',
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
      compatibilityMode: 'reject-unsupported',
      createEngine: async () => ({ setScene: vi.fn() }),
    })).rejects.toThrow('baseColorMap');
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
      createEngine: async () => ({ setScene: vi.fn() }),
    })).resolves.toMatchObject({
      backend: 'pt-webgl2',
      attached: true,
    });
  });
});
