// gpuStub.ts — shared WebGPU mock-device support for the scene-upload unit tests.
// (Not a *.test.ts file, so vitest does not collect it as a suite.)
//
// The upload path references the WebGPU constant globals and, since P2, creates a
// material texture_2d_array + sampler. These helpers let the bespoke per-file
// stub devices model that without a real GPU.
import { vi } from 'vitest';

type SizeValidatingStubLimits = Partial<
  Pick<GPUSupportedLimits, 'maxBufferSize' | 'maxTextureDimension1D' | 'maxTextureDimension2D' | 'maxTextureDimension3D' | 'maxTextureArrayLayers'>
>;

export interface SizeValidatingGpuStub {
  device: GPUDevice;
  buffers: Array<{ label: string | undefined; size: number; destroy: ReturnType<typeof vi.fn> }>;
  textures: Array<{
    label: string | undefined;
    width: number;
    height: number;
    depthOrArrayLayers: number;
    destroy: ReturnType<typeof vi.fn>;
  }>;
  createBuffer: unknown;
  createTexture: unknown;
}

/** Install the WebGPU constant globals the upload path reads. Idempotent. */
export function installGpuConstStubs(): void {
  const g = globalThis as unknown as {
    GPUBufferUsage?: Record<string, number>;
    GPUTextureUsage?: Record<string, number>;
    GPUShaderStage?: Record<string, number>;
  };
  if (g.GPUBufferUsage == null) {
    g.GPUBufferUsage = { STORAGE: 1 << 0, COPY_DST: 1 << 1, UNIFORM: 1 << 2, COPY_SRC: 1 << 3 };
  }
  if (g.GPUTextureUsage == null) {
    g.GPUTextureUsage = {
      COPY_SRC: 1 << 0,
      COPY_DST: 1 << 1,
      TEXTURE_BINDING: 1 << 2,
      STORAGE_BINDING: 1 << 3,
      RENDER_ATTACHMENT: 1 << 4,
    };
  }
  if (g.GPUShaderStage == null) {
    g.GPUShaderStage = { VERTEX: 1 << 0, FRAGMENT: 1 << 1, COMPUTE: 1 << 2 };
  }
}

/** Stub GPUDevice texture/sampler methods (P2 material-texture upload), to spread
 *  into a mock device literal next to its buffer stubs. `createTexture` returns an
 *  object with `createView` + `destroy` so the upload's view/dispose paths work. */
export function textureStubMethods() {
  return {
    createTexture: vi.fn((desc?: { label?: string }) => ({
      label: desc?.label ?? '',
      createView: vi.fn(() => ({})),
      destroy: vi.fn(),
    })),
    createSampler: vi.fn(() => ({})),
  };
}

function textureExtent(size: GPUTextureDescriptor['size']): [number, number, number] {
  if (typeof size === 'number') return [size, 1, 1];
  if (Array.isArray(size)) return [size[0] ?? 1, size[1] ?? 1, size[2] ?? 1];
  if (typeof size === 'object' && size != null && 'width' in size) {
    const dict = size as { width: number; height?: number; depthOrArrayLayers?: number };
    return [dict.width, dict.height ?? 1, dict.depthOrArrayLayers ?? 1];
  }
  const iterable = Array.from(size as Iterable<number>);
  return [iterable[0] ?? 1, iterable[1] ?? 1, iterable[2] ?? 1];
}

function finitePositiveInt(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 1 || Math.floor(value) !== value) {
    throw new Error(`GPU stub rejected invalid ${name}: ${value}`);
  }
  return value;
}

export function createSizeValidatingGpuDeviceStub(
  limits: SizeValidatingStubLimits = {},
): SizeValidatingGpuStub {
  installGpuConstStubs();

  const resolvedLimits = {
    maxBufferSize: limits.maxBufferSize ?? 256 * 1024 * 1024,
    maxTextureDimension1D: limits.maxTextureDimension1D ?? 8192,
    maxTextureDimension2D: limits.maxTextureDimension2D ?? 8192,
    maxTextureDimension3D: limits.maxTextureDimension3D ?? 2048,
    maxTextureArrayLayers: limits.maxTextureArrayLayers ?? 256,
  };
  const buffers: SizeValidatingGpuStub['buffers'] = [];
  const textures: SizeValidatingGpuStub['textures'] = [];

  const createBuffer = vi.fn((desc: GPUBufferDescriptor) => {
    const size = finitePositiveInt(Number(desc.size), 'buffer size');
    if (size > resolvedLimits.maxBufferSize) {
      throw new Error(
        `GPU stub rejected buffer "${desc.label ?? '<unlabeled>'}" size ${size}; ` +
          `maxBufferSize=${resolvedLimits.maxBufferSize}`,
      );
    }
    const buffer = { label: desc.label, size, destroy: vi.fn() };
    buffers.push(buffer);
    return buffer;
  });

  const createTexture = vi.fn((desc: GPUTextureDescriptor) => {
    const [width, height, depthOrArrayLayers] = textureExtent(desc.size).map((n, i) =>
      finitePositiveInt(Number(n), ['texture width', 'texture height', 'texture depthOrArrayLayers'][i] ?? 'texture extent'),
    ) as [number, number, number];
    const dimension = desc.dimension ?? '2d';
    const maxDim =
      dimension === '1d'
        ? resolvedLimits.maxTextureDimension1D
        : dimension === '3d'
          ? resolvedLimits.maxTextureDimension3D
          : resolvedLimits.maxTextureDimension2D;
    const maxAxis = Math.max(width, height);
    if (maxAxis > maxDim) {
      throw new Error(
        `GPU stub rejected texture "${desc.label ?? '<unlabeled>'}" ${width}x${height}; ` +
          `maxTextureDimension${dimension.toUpperCase()}=${maxDim}`,
      );
    }
    if (dimension !== '3d' && depthOrArrayLayers > resolvedLimits.maxTextureArrayLayers) {
      throw new Error(
        `GPU stub rejected texture "${desc.label ?? '<unlabeled>'}" array layers ${depthOrArrayLayers}; ` +
          `maxTextureArrayLayers=${resolvedLimits.maxTextureArrayLayers}`,
      );
    }
    const destroy = vi.fn();
    const texture = {
      label: desc.label,
      width,
      height,
      depthOrArrayLayers,
      destroy,
      createView: vi.fn(() => ({})),
    };
    textures.push(texture);
    return texture;
  });

  const encoder = { clearBuffer: vi.fn(), finish: vi.fn(() => ({})) };
  const device = {
    limits: resolvedLimits,
    createBuffer,
    createTexture,
    createSampler: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => encoder),
    queue: { submit: vi.fn(), writeBuffer: vi.fn(), writeTexture: vi.fn() },
  } as unknown as GPUDevice;

  return { device, buffers, textures, createBuffer, createTexture };
}
