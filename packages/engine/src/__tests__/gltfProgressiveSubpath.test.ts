import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GltfJson } from '@vitrum/gltf-adapter';
import type { MeshPrimitive, TextureRef } from '@vitrum/core';

const createProgressiveEngineMock = vi.hoisted(() => vi.fn());
const probeAdapterProfileMock = vi.hoisted(() => vi.fn());

vi.mock('../createProgressiveEngine.js', () => ({
  createProgressiveEngine: createProgressiveEngineMock,
}));

vi.mock('../adapterProfile.js', () => ({
  probeAdapterProfile: probeAdapterProfileMock,
}));

import {
  loadGltfWithProgressiveEngine,
  type LoadGltfWithProgressiveEngineOptions,
} from '../gltf.js';

function f32Buffer(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

function makeInlineTexturedTriangleGltf(): { gltf: GltfJson; buffers: Map<number, ArrayBuffer> } {
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
      nodes: [{ mesh: 0, camera: 0 }],
      cameras: [{ type: 'perspective' }],
      meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
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

describe('@vitrum/engine/gltf progressive helper', () => {
  beforeEach(() => {
    createProgressiveEngineMock.mockReset();
    probeAdapterProfileMock.mockReset();
    probeAdapterProfileMock.mockResolvedValue({
      hasWebGPU: true,
      hybridCapable: true,
      hybridLiteCapable: true,
      ptWebgpuTier: 'lite',
      maxStorageBuffersPerStage: 16,
      maxStorageTexturesPerStage: 8,
      isSoftwareAdapter: false,
      adapterKind: 'hardware',
      hasWebGL2: true,
      recommendedRealtimeTier: 'ultra',
      recommendedHeroBackend: 'pt-webgpu-lite',
      limits: Object.freeze({}),
    });
  });

  it('loads a glTF scene and wires its controller into createProgressiveEngine', async () => {
    const { gltf, buffers } = makeInlineTexturedTriangleGltf();
    const canvas = {} as HTMLCanvasElement;
    const handle = {
      coordinator: {},
      realtime: {},
      converged: {},
      dispose: vi.fn(),
    };
    createProgressiveEngineMock.mockResolvedValueOnce(handle);

    const options: LoadGltfWithProgressiveEngineOptions = {
      buffers,
      decodeTextures: true,
      decodeImage: async (data: Uint8Array, mimeType: string) => ({ kind: 'raw-image', mimeType, data }),
      decodePixels: (_handle, context) => ({
        width: 4,
        height: 2,
        channels: 4,
        dataType: 'uint8',
        colorSpace: context.colorSpace,
        data: new Uint8Array(4 * 2 * 4).fill(255),
      }),
      maxTextureSize: 2,
      engineOptions: {
        canvas,
        seedFromRealtime: false,
      },
    };
    const result = await loadGltfWithProgressiveEngine(gltf, options);

    expect(result.backend).toBe('pt-webgpu');
    expect(result.profileId).toBe('pt-webgpu');
    expect(result.engine).toBe(handle);
    expect(result.attached).toBe(true);
    expect(result.textureDecodeReport).toBe(result.asset.textureDecodeReport);
    expect(result.textureDecodeReport).toMatchObject({
      mapCount: 1,
      rawImageCount: 0,
      cpuReadableCount: 1,
      entries: [
        expect.objectContaining({
          primitiveId: 'gltf-prim-0',
          materialField: 'baseColorMap',
          path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
          colorSpace: 'srgb',
          handleColorSpace: 'linear',
          handleKind: 'pixel-data',
        }),
      ],
    });
    expect(result.decodedTextureCount).toBe(1);
    expect(result.unchangedTextureCount).toBe(0);
    expect(result.textureDecodeDiagnostics).toEqual([
      expect.objectContaining({
        code: 'decoded-texture-exceeds-max-size',
        path: 'materials[0].pbrMetallicRoughness.baseColorTexture',
        resizedWidth: 2,
        resizedHeight: 1,
      }),
    ]);
    expect(result.textureDecodeWarnings).toEqual([
      expect.stringContaining('exceeds maxTextureSize=2'),
    ]);
    expect(result.warnings).toEqual(expect.arrayContaining([
      expect.stringContaining('exceeds maxTextureSize=2'),
      expect.stringContaining('reported on result.cameras'),
    ]));
    expect(result.diagnostics).toEqual(expect.arrayContaining([
      expect.objectContaining({
        severity: 'warning',
        code: 'ignored-camera',
        path: 'cameras[0]',
      }),
    ]));
    expect(createProgressiveEngineMock).toHaveBeenCalledTimes(1);
    expect(createProgressiveEngineMock.mock.calls[0]![0]).toEqual(expect.objectContaining({
      canvas,
      seedFromRealtime: false,
      scene: result.asset.scene,
      controller: result.controller,
    }));
    const primitive = createProgressiveEngineMock.mock.calls[0]![0].scene.primitives[0] as MeshPrimitive;
    const textureHandle = (primitive.material.baseColorMap as TextureRef).handle as {
      width: number;
      height: number;
      data: Float32Array;
    };
    expect(textureHandle.width).toBe(2);
    expect(textureHandle.height).toBe(1);
    expect(textureHandle.data).toBeInstanceOf(Float32Array);
  });

  it('validates progressive glTF against the full pt-webgpu profile instead of a standalone lite probe', async () => {
    const { gltf, buffers } = makeInlineTexturedTriangleGltf();
    delete gltf.cameras;
    const handle = {
      coordinator: {},
      realtime: {},
      converged: {},
      dispose: vi.fn(),
    };
    createProgressiveEngineMock.mockResolvedValueOnce(handle);

    const result = await loadGltfWithProgressiveEngine(gltf, {
      buffers,
      compatibilityMode: 'reject-unsupported',
      engineOptions: {
        canvas: {} as HTMLCanvasElement,
        seedFromRealtime: false,
      },
    });

    expect(result.profileId).toBe('pt-webgpu');
    expect(result.engine).toBe(handle);
    expect(probeAdapterProfileMock).not.toHaveBeenCalled();
    expect(createProgressiveEngineMock).toHaveBeenCalledTimes(1);
  });

  it('reports the full runtime profile for full-tier progressive loads', async () => {
    probeAdapterProfileMock.mockResolvedValueOnce({
      hasWebGPU: true,
      hybridCapable: true,
      hybridLiteCapable: true,
      ptWebgpuTier: 'full',
      maxStorageBuffersPerStage: 16,
      maxStorageTexturesPerStage: 8,
      isSoftwareAdapter: false,
      adapterKind: 'hardware',
      hasWebGL2: true,
      recommendedRealtimeTier: 'ultra',
      recommendedHeroBackend: 'pt-webgpu',
      limits: Object.freeze({}),
    });
    const { gltf, buffers } = makeInlineTexturedTriangleGltf();
    delete gltf.cameras;
    const handle = {
      coordinator: {},
      realtime: {},
      converged: {},
      dispose: vi.fn(),
    };
    createProgressiveEngineMock.mockResolvedValueOnce(handle);

    const result = await loadGltfWithProgressiveEngine(gltf, {
      buffers,
      engineOptions: {
        canvas: {} as HTMLCanvasElement,
        seedFromRealtime: false,
      },
    });

    expect(result.backend).toBe('pt-webgpu');
    expect(result.profileId).toBe('pt-webgpu');
    expect(result.engine).toBe(handle);
  });
});
