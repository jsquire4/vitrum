// gltfAssetApi.test.ts — higher-level arbitrary-asset API tests.
//
// These fixtures exercise the public package root, not private helpers: URL
// loading with external resources, structured feature reporting, and backend
// compatibility ranking against the core promise ledger.

import { describe, expect, it, vi } from 'vitest';
import {
  analyzeGltfAsset,
  evaluateGltfBackendCompatibility,
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
    textures: [{ source: 0 }],
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
    expect(result.featureReport.resources.externalBufferCount).toBe(1);
    expect(result.featureReport.resources.externalImageCount).toBe(1);
    expect(result.recommendedBackend.backend).toBe('pt-webgl2');
  });

  it('throws a deterministic error for relative external resources without a baseUri', async () => {
    const gltf = makeExternalTexturedGltf();
    await expect(loadGltfAsset(gltf)).rejects.toThrow('without a baseUri');
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
      issue.support === 'unsupported',
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
    expect(walkaround.issues.some((issue) =>
      issue.category === 'material' &&
      issue.name === 'baseColorMap' &&
      issue.support === 'unsupported',
    )).toBe(true);
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
