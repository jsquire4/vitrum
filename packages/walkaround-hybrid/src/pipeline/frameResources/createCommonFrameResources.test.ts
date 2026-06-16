import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createCommonFrameResources } from './createCommonFrameResources.js';
import { createFrameResources } from '../resourceManager.js';

const GPU_TEXTURE_USAGE = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
} as const;

const GPU_BUFFER_USAGE = {
  COPY_DST: 0x08,
  UNIFORM: 0x40,
  STORAGE: 0x80,
  COPY_SRC: 0x04,
} as const;

let previousGpuTextureUsage: unknown;
let previousGpuBufferUsage: unknown;

interface FakeTexture {
  readonly label?: string;
  readonly width: number;
  readonly height: number;
  readonly depthOrArrayLayers: number;
  readonly format: GPUTextureFormat;
  readonly usage: number;
  readonly createView: ReturnType<typeof vi.fn>;
  readonly destroy: ReturnType<typeof vi.fn>;
}

function makeFakeDevice(): GPUDevice {
  return {
    createTexture(desc: GPUTextureDescriptor): GPUTexture {
      const size = desc.size as unknown as
        | readonly [number, number, number?]
        | { width: number; height: number; depthOrArrayLayers?: number };
      let width: number;
      let height: number;
      let depthOrArrayLayers: number;
      if (Array.isArray(size)) {
        const tuple = size as readonly [number, number, number?];
        width = tuple[0];
        height = tuple[1];
        depthOrArrayLayers = tuple[2] ?? 1;
      } else {
        const extent = size as { width: number; height: number; depthOrArrayLayers?: number };
        width = extent.width;
        height = extent.height;
        depthOrArrayLayers = extent.depthOrArrayLayers ?? 1;
      }
      return {
        label: desc.label,
        width,
        height,
        depthOrArrayLayers,
        format: desc.format,
        usage: desc.usage,
        createView: vi.fn(() => ({ label: `${desc.label ?? 'texture'}-view` })),
        destroy: vi.fn(),
      } as unknown as GPUTexture;
    },
    createBuffer(desc: GPUBufferDescriptor): GPUBuffer {
      return {
        label: desc.label,
        size: desc.size,
        usage: desc.usage,
        destroy: vi.fn(),
      } as unknown as GPUBuffer;
    },
    createSampler(desc?: GPUSamplerDescriptor): GPUSampler {
      return { label: desc?.label } as unknown as GPUSampler;
    },
    queue: {
      writeTexture: vi.fn(),
      writeBuffer: vi.fn(),
    },
  } as unknown as GPUDevice;
}

function textureSize(texture: GPUTexture): readonly [number, number] {
  const t = texture as unknown as FakeTexture;
  return [t.width, t.height];
}

beforeEach(() => {
  previousGpuTextureUsage = (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage;
  previousGpuBufferUsage = (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  Object.assign(globalThis, {
    GPUTextureUsage: GPU_TEXTURE_USAGE,
    GPUBufferUsage: GPU_BUFFER_USAGE,
  });
});

afterEach(() => {
  if (previousGpuTextureUsage === undefined) {
    delete (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage;
  } else {
    (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage = previousGpuTextureUsage;
  }
  if (previousGpuBufferUsage === undefined) {
    delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  } else {
    (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage = previousGpuBufferUsage;
  }
});

describe('createCommonFrameResources Welford allocation policy', () => {
  it('keeps full-size Welford ping-pong resources for the legacy/default path', () => {
    const resources = createCommonFrameResources(makeFakeDevice(), 64, 32);

    expect(textureSize(resources.varianceBuffer)).toEqual([64, 32]);
    expect(textureSize(resources.varianceBufferAux)).toEqual([64, 32]);
    expect(textureSize(resources.atrousVarianceEstimateTexture)).toEqual([64, 32]);
  });

  it('collapses unused Welford aux resources to 1x1 when ping-pong is disabled', () => {
    const resources = createCommonFrameResources(makeFakeDevice(), 64, 32, {
      welfordPingPong: false,
    });

    expect(textureSize(resources.varianceBuffer)).toEqual([64, 32]);
    expect(textureSize(resources.varianceBufferAux)).toEqual([1, 1]);
    expect(textureSize(resources.atrousVarianceEstimateTexture)).toEqual([1, 1]);
  });

  it('threads the policy through createFrameResources without changing the resource shape', () => {
    const resources = createFrameResources(makeFakeDevice(), 64, 32, {
      svgfEnabled: false,
      welfordPingPong: false,
    });

    expect(textureSize(resources.common.varianceBuffer)).toEqual([64, 32]);
    expect(textureSize(resources.common.varianceBufferAux)).toEqual([1, 1]);
    expect(textureSize(resources.common.atrousVarianceEstimateTexture)).toEqual([1, 1]);
    expect(Object.keys(resources.common)).toContain('varianceBufferAux');
    expect(Object.keys(resources.common)).toContain('atrousVarianceEstimateTexture');
  });
});
