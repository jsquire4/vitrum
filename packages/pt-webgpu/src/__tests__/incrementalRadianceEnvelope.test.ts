import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

const F32_MAX = 3.4028234663852886e38;

function mappedScene(withMeshEmitter: boolean): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'panel',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [0.8, 0.2, 0.1],
          roughness: 0.3,
          metallic: 0.1,
          emissive: [1, 0, 0],
          emissiveIntensity: 1,
          emissiveMap: {
            handle: {
              width: 2,
              height: 1,
              // Mean red is one (so average-factor checks pass), but the
              // first exact texel still overflows when multiplied by F32_MAX.
              data: new Float32Array([2, 0, 0, 1, 0, 0, 0, 1]),
            },
          },
        },
      },
    ],
    emitters: withMeshEmitter
      ? [{
          kind: 'mesh-area',
          id: 'panel-light',
          meshId: 'panel',
          color: [1, 0, 0],
          intensity: 1,
        }]
      : [],
    environment: { kind: 'none' },
  };
}

function lightMappedScene(): Scene {
  const scene = mappedScene(false);
  const primitive = scene.primitives[0]!;
  return {
    ...scene,
    primitives: [{
      ...primitive,
      material: {
        baseColor: [0.8, 0.2, 0.1],
        roughness: 0.3,
        metallic: 0.1,
        lightMapIntensity: 1,
        lightMap: {
          handle: {
            width: 2,
            height: 1,
            data: new Float32Array([2, 0, 0, 1, 0, 0, 0, 1]),
          },
        },
      },
    }],
  };
}

function makeStubDevice(): GPUDevice {
  const copyBufferToBuffer = vi.fn();
  const copyTextureToTexture = vi.fn();
  const finish = vi.fn(() => ({}));
  return {
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
      submit: vi.fn(),
    },
    createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
      label: desc.label ?? '',
      size: Number(desc.size),
      usage: Number(desc.usage),
      destroy: vi.fn(),
    })),
    ...textureStubMethods(),
    createCommandEncoder: vi.fn(() => ({
      copyBufferToBuffer,
      copyTextureToTexture,
      finish,
    })),
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 8,
      maxTextureDimension2D: 8192,
      maxTextureArrayLayers: 256,
      maxBufferSize: 1_073_741_824,
      maxStorageBufferBindingSize: 1_073_741_824,
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

describe('pt-webgpu incremental outgoing-radiance envelope', () => {
  it('revalidates exact emissive-map texels after updatePrimitive changes their factor', async () => {
    installGpuConstStubs();
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice() });
    engine.setScene(mappedScene(false));
    const liveScene = engine.getScene?.();

    expect(() => engine.updatePrimitive?.('panel', {
      material: { emissive: [F32_MAX, 0, 0] },
    })).toThrow(/emissiveMap layer 0 mip 0 texel 0 radiance must remain finite/);
    expect(engine.getScene?.()).toBe(liveScene);
  });

  it('revalidates exact light-map texels after updatePrimitive changes their factor', async () => {
    installGpuConstStubs();
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice() });
    engine.setScene(lightMappedScene());
    const liveScene = engine.getScene?.();

    expect(() => engine.updatePrimitive?.('panel', {
      material: { lightMapIntensity: F32_MAX },
    })).toThrow(/lightMap layer 0 mip 0 texel 0 radiance must remain finite/);
    expect(engine.getScene?.()).toBe(liveScene);
  });

  it('revalidates the mapped mesh-light proposal after updateEmitter changes its factor', async () => {
    installGpuConstStubs();
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice() });
    engine.setScene(mappedScene(true));
    const liveScene = engine.getScene?.();

    expect(() => engine.updateEmitter?.('panel-light', {
      color: [F32_MAX, 0, 0],
    })).toThrow(/emissiveMap layer 0 mip 0 texel 0 radiance must remain finite/);
    expect(engine.getScene?.()).toBe(liveScene);
  });

  it('revalidates mapped mesh-light factors during an atomic updateLighting replacement', async () => {
    installGpuConstStubs();
    const engine = await createPTEngine_WebGPU({ device: makeStubDevice() });
    engine.setScene(mappedScene(true));
    const liveScene = engine.getScene?.();

    expect(() => engine.updateLighting?.({
      emitters: [{
        kind: 'mesh-area',
        id: 'panel-light',
        meshId: 'panel',
        color: [F32_MAX, 0, 0],
        intensity: 1,
      }],
    })).toThrow(/emissiveMap layer 0 mip 0 texel 0 radiance must remain finite/);
    expect(engine.getScene?.()).toBe(liveScene);
  });
});
