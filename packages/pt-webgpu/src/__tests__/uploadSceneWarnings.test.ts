import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning, Scene } from '@vitrum/core';
import { createPTEngine_WebGPU } from '../index.js';
import { uploadPackedScene, buildPackedScene } from '../scene/uploadSceneBuffers.js';
import { PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE } from '../webgpuLimits.js';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

function oneMeshSceneWithBadBaseColorTexture(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
          baseColorMap: { handle: { image: { width: 0, height: 0 } } },
        },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function oneMeshSceneWithBumpSamplerPolicy(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
          bumpScale: 0.2,
          bumpMap: {
            handle: { image: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) } },
            magFilter: 'nearest',
            minFilter: 'nearest',
            mipFilter: 'linear',
          },
        },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}


function oneMeshSceneWithHighUvBaseColorTexture(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
          baseColorMap: { handle: { image: { width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) } }, texCoord: 3 },
        },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function makeDevice(): GPUDevice {
  return {
    queue: {
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
      copyExternalImageToTexture: vi.fn(),
    },
    createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
      label: desc.label,
      size: desc.size,
      destroy: vi.fn(),
    })),
    ...textureStubMethods(),
    createCommandEncoder: vi.fn(),
    limits: {
      maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      maxStorageTexturesPerShaderStage: 8,
      maxTextureDimension2D: 8192,
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

describe('uploadPackedScene warning propagation', () => {

  it('surfaces direct-core high-UV material map drops as structured warnings', () => {
    installGpuConstStubs();
    const device = makeDevice();
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const uploaded = uploadPackedScene(device, buildPackedScene(oneMeshSceneWithHighUvBaseColorTexture()));
      expect(uploaded.structuredWarnings).toEqual(expect.arrayContaining([
        expect.objectContaining({
          code: 'pt-webgpu.material-texture-unsupported-texcoord',
          backend: 'pt-webgpu',
          phase: 'setScene',
          method: 'setScene',
          details: expect.objectContaining({
            materialIndex: 0,
            field: 'baseColorMap',
            texCoord: 3,
            supportedTexCoords: [0, 1],
            fallback: 'map-ignored',
          }),
        }),
      ]));
    } finally {
      consoleWarn.mockRestore();
    }
  });
  it('T1-6 — uploads emissive to a dedicated rgba16float array; HDR emissive > 1.0 survives packing', () => {
    installGpuConstStubs();
    // Device that records createTexture formats + writeTexture payloads keyed by texture.
    const writes: Array<{ format: string; data: unknown }> = [];
    const device = {
      queue: {
        writeBuffer: vi.fn(),
        writeTexture: vi.fn((dst: { texture: { format: string } }, data: unknown) => {
          writes.push({ format: dst.texture.format, data });
        }),
        copyExternalImageToTexture: vi.fn(),
        submit: vi.fn(),
      },
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({ label: desc.label, size: desc.size, destroy: vi.fn() })),
      createTexture: vi.fn((desc: GPUTextureDescriptor) => {
        const size = desc.size as { width: number; height: number; depthOrArrayLayers?: number };
        return {
          label: desc.label, format: desc.format,
          width: size.width, height: size.height, depthOrArrayLayers: size.depthOrArrayLayers ?? 1,
          createView: vi.fn(() => ({})), destroy: vi.fn(),
        };
      }),
      createSampler: vi.fn(() => ({})),
      createShaderModule: vi.fn(() => ({})),
      createRenderPipeline: vi.fn(() => ({ getBindGroupLayout: vi.fn(() => ({})) })),
      createBindGroup: vi.fn(() => ({})),
      createCommandEncoder: vi.fn(() => ({
        beginRenderPass: vi.fn(() => ({ setPipeline: vi.fn(), setViewport: vi.fn(), setBindGroup: vi.fn(), draw: vi.fn(), end: vi.fn() })),
        finish: vi.fn(() => ({})),
      })),
      limits: {
        maxStorageBuffersPerShaderStage: PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
        maxStorageTexturesPerShaderStage: 8,
        maxTextureDimension2D: 8192,
      },
      addEventListener: vi.fn(), removeEventListener: vi.fn(), lost: new Promise<never>(() => {}),
    } as unknown as GPUDevice;

    const hdrEmissive: Scene = {
      primitives: [{
        kind: 'mesh', id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1], roughness: 0.5, metallic: 0,
          // Linear HDR emissive texel (float32 → linear pass-through, > 1.0).
          emissiveMap: { handle: { image: { width: 1, height: 1, data: new Float32Array([12.0, 4.0, 0.5, 1.0]) } } },
        },
      }],
      emitters: [], environment: { kind: 'none' },
    };

    const uploaded = uploadPackedScene(device, buildPackedScene(hdrEmissive));
    // A rgba16float emissive array exists and is bound distinctly from the sRGB array.
    expect(uploaded.materialEmissiveTexture.format).toBe('rgba16float');
    expect(uploaded.materialEmissiveTexture).not.toBe(uploaded.materialTexture);
    // The HDR emissive texel was written to the rgba16float target as half-floats,
    // and the > 1.0 red channel survives (half-float encoding of 12.0).
    const emissiveWrite = writes.find((w) => w.format === 'rgba16float' && w.data instanceof Uint16Array);
    expect(emissiveWrite).toBeDefined();
    const half = emissiveWrite!.data as Uint16Array;
    const halfToFloat = (h: number): number => {
      const sign = (h & 0x8000) ? -1 : 1;
      const exp = (h & 0x7c00) >> 10;
      const frac = h & 0x03ff;
      if (exp === 0) return sign * frac * 2 ** -24;
      if (exp === 0x1f) return frac ? NaN : sign * Infinity;
      return sign * (1 + frac / 1024) * 2 ** (exp - 15);
    };
    expect(halfToFloat(half[0]!)).toBeCloseTo(12.0, 1); // > 1.0 emissive NOT clamped
    expect(halfToFloat(half[1]!)).toBeCloseTo(4.0, 1);
    expect(halfToFloat(half[2]!)).toBeCloseTo(0.5, 2);
  });

  it('routes material texture upload warnings through UploadedSceneBuffers warnings', () => {
    installGpuConstStubs();
    const device = makeDevice();
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const uploaded = uploadPackedScene(device, buildPackedScene(oneMeshSceneWithBadBaseColorTexture()));
      expect(uploaded.warnings).toContain(
        'material texture array (sRGB): [materialTextureArray] source 0 has no usable image; layer left black.',
      );
      expect(uploaded.structuredWarnings).toEqual([
        expect.objectContaining({
          code: 'pt-webgpu.material-texture-unreadable',
          backend: 'pt-webgpu',
          phase: 'setScene',
          method: 'setScene',
          details: expect.objectContaining({
            warning:
              'material texture array (sRGB): [materialTextureArray] source 0 has no usable image; layer left black.',
            colorSpace: 'sRGB',
            layer: 0,
            fields: ['baseColorMap'],
            materialIndices: [0],
            fallback: 'black-layer',
            uses: [{ materialIndex: 0, field: 'baseColorMap', colorSpace: 'srgb', texCoord: 0 }],
          }),
        }),
      ]);
      expect(consoleWarn).not.toHaveBeenCalled();
    } finally {
      consoleWarn.mockRestore();
    }
  });

  it('does not warn for bump sampler policy because bump height reads consume the policy natively', () => {
    installGpuConstStubs();
    const device = makeDevice();
    const uploaded = uploadPackedScene(device, buildPackedScene(oneMeshSceneWithBumpSamplerPolicy()));

    expect(uploaded.structuredWarnings.some((warning) =>
      warning.code === 'pt-webgpu.material-texture-sampler-policy-approximation',
    )).toBe(false);
  });

  it('emits upload-time material texture failures through Engine.onWarning', async () => {
    installGpuConstStubs();
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const structured: EngineWarning[] = [];
    try {
      const engine = await createPTEngine_WebGPU({
        device: makeDevice(),
        onWarning: (warning) => structured.push(warning),
      });
      engine.setScene(oneMeshSceneWithBadBaseColorTexture());

      const warning = structured.find((w) => w.code === 'pt-webgpu.material-texture-unreadable');
      expect(warning).toEqual(expect.objectContaining({
        phase: 'setScene',
        method: 'setScene',
        details: expect.objectContaining({
          fields: ['baseColorMap'],
          materialIndices: [0],
          fallback: 'black-layer',
        }),
      }));
      expect(structured.some((w) =>
        w.code === 'pt-webgpu.scene-pack-warning' &&
        w.details?.warning ===
          'material texture array (sRGB): [materialTextureArray] source 0 has no usable image; layer left black.',
      )).toBe(false);
      engine.dispose();
    } finally {
      consoleWarn.mockRestore();
    }
  });
});
