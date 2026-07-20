import { describe, expect, it, vi } from 'vitest';
import {
  createMaterialTextureArray,
  materialTextureMipLevelCount,
} from '../scene/materialTextureArray.js';
import { installGpuConstStubs } from './gpuStub.js';

function makeDevice() {
  const writeTexture = vi.fn();
  const beginRenderPass = vi.fn(() => ({
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    draw: vi.fn(),
    end: vi.fn(),
  }));
  const submit = vi.fn();
  const createTexture = vi.fn((desc: GPUTextureDescriptor) => {
    const size = desc.size as { width: number; height: number; depthOrArrayLayers?: number };
    return {
      label: desc.label,
      width: size.width,
      height: size.height,
      depthOrArrayLayers: size.depthOrArrayLayers ?? 1,
      format: desc.format,
      createView: vi.fn(() => ({})),
      destroy: vi.fn(),
    };
  });
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    queue: {
      writeTexture,
      copyExternalImageToTexture: vi.fn(),
      submit,
    },
    createTexture,
    createSampler: vi.fn(() => ({})),
    createShaderModule: vi.fn(() => ({})),
    createRenderPipeline: vi.fn(() => ({
      getBindGroupLayout: vi.fn(() => ({})),
    })),
    createBindGroup: vi.fn(() => ({})),
    createCommandEncoder: vi.fn(() => ({
      beginRenderPass,
      finish: vi.fn(() => ({})),
    })),
  } as unknown as GPUDevice;
  return { device, writeTexture, createTexture, beginRenderPass, submit };
}

function rawImage(width: number, height: number): { width: number; height: number; data: Uint8Array } {
  return { width, height, data: new Uint8Array(width * height * 4) };
}

describe('createMaterialTextureArray', () => {
  it('computes full mip chain lengths from the max texture dimension', () => {
    expect(materialTextureMipLevelCount(1, 1)).toBe(1);
    expect(materialTextureMipLevelCount(2, 1)).toBe(2);
    expect(materialTextureMipLevelCount(4, 2)).toBe(3);
    expect(materialTextureMipLevelCount(8, 8)).toBe(4);
  });

  it('allocates and renders a mip chain for each material texture array layer', () => {
    installGpuConstStubs();
    const { device, createTexture, beginRenderPass, submit } = makeDevice();
    const array = createMaterialTextureArray(device, [
      rawImage(4, 4),
      rawImage(4, 4),
    ]);

    expect(array.mipLevelCount).toBe(3);
    expect(createTexture).toHaveBeenCalledWith(expect.objectContaining({
      mipLevelCount: 3,
      size: { width: 4, height: 4, depthOrArrayLayers: 2 },
    }));
    expect(beginRenderPass).toHaveBeenCalledTimes(4);
    expect(submit).toHaveBeenCalledTimes(1);
  });

  it('reports per-layer UV-fit scales for heterogeneous source dimensions', () => {
    installGpuConstStubs();
    const { device } = makeDevice();
    const array = createMaterialTextureArray(device, [
      rawImage(2, 4),
      rawImage(4, 4),
      rawImage(4, 2),
    ]);

    expect(array.layerUvScales).toEqual([
      [0.5, 1],
      [1, 1],
      [1, 0.5],
    ]);
    expect(array.warnings).toContain(
      '[materialTextureArray] source 0 is 2×4 but the array is 4×4; copied at native size and sampled through a per-layer UV-fit scale. Use same-size textures when exact mip/border filtering parity is required.',
    );
    expect(array.warnings).toContain(
      '[materialTextureArray] source 2 is 4×2 but the array is 4×4; copied at native size and sampled through a per-layer UV-fit scale. Use same-size textures when exact mip/border filtering parity is required.',
    );
    expect(array.structuredWarnings).toEqual([
      expect.objectContaining({
        code: 'texture-size-mismatch',
        layer: 0,
        width: 2,
        height: 4,
        arrayWidth: 4,
        arrayHeight: 4,
      }),
      expect.objectContaining({
        code: 'texture-size-mismatch',
        layer: 2,
        width: 4,
        height: 2,
        arrayWidth: 4,
        arrayHeight: 4,
      }),
    ]);
  });

  it('expands 8-bit RGB raw data to RGBA8 rows before writeTexture', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    const array = createMaterialTextureArray(device, [
      { width: 2, height: 1, data: new Uint8Array([10, 20, 30, 40, 50, 60]) },
    ]);

    expect(array.warnings).toEqual([]);
    expect(writeTexture).toHaveBeenCalledTimes(1);
    const call = writeTexture.mock.calls[0] as [
      GPUImageCopyTexture,
      Uint8Array,
      GPUImageDataLayout,
      GPUExtent3D,
    ];
    expect(Array.from(call[1])).toEqual([10, 20, 30, 255, 40, 50, 60, 255]);
    expect(call[2]).toEqual({ bytesPerRow: 8, rowsPerImage: 1 });
    expect(call[3]).toEqual({ width: 2, height: 1 });
  });

  it('normalizes Float32 raw data to RGBA8 rows before writeTexture', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    const array = createMaterialTextureArray(device, [
      { width: 2, height: 1, data: new Float32Array([1, 0.5, 0, 1, 0.25, 0, 0.75, 0.5]) },
    ]);

    expect(array.warnings).toEqual([]);
    expect(writeTexture).toHaveBeenCalledTimes(1);
    const call = writeTexture.mock.calls[0] as [
      GPUImageCopyTexture,
      Uint8Array,
      GPUImageDataLayout,
      GPUExtent3D,
    ];
    expect(Array.from(call[1])).toEqual([255, 128, 0, 255, 64, 0, 191, 128]);
    expect(call[2]).toEqual({ bytesPerRow: 8, rowsPerImage: 1 });
    expect(call[3]).toEqual({ width: 2, height: 1 });
  });

  it('normalizes Uint16 raw data to RGBA8 rows before writeTexture', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    const array = createMaterialTextureArray(device, [
      { width: 1, height: 1, data: new Uint16Array([65535, 32768, 0, 65535]) },
    ]);

    expect(array.warnings).toEqual([]);
    expect(writeTexture).toHaveBeenCalledTimes(1);
    const call = writeTexture.mock.calls[0] as [
      GPUImageCopyTexture,
      Uint8Array,
      GPUImageDataLayout,
      GPUExtent3D,
    ];
    expect(Array.from(call[1])).toEqual([255, 128, 0, 255]);
    expect(call[2]).toEqual({ bytesPerRow: 4, rowsPerImage: 1 });
    expect(call[3]).toEqual({ width: 1, height: 1 });
  });

  it('warns and leaves unsupported raw typed-array shapes black', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    const array = createMaterialTextureArray(device, [
      { width: 1, height: 1, data: new Float32Array([1, 0, 0, 1, 0]) },
    ]);

    expect(writeTexture).not.toHaveBeenCalled();
    expect(array.warnings).toContain(
      '[materialTextureArray] source 0 has raw data with unsupported byte layout (20 bytes for 1×1); expected 1, 2, 3, or 4 8-bit or normalized numeric channel(s) per pixel. Layer left black.',
    );
    expect(array.structuredWarnings).toEqual([
      expect.objectContaining({
        code: 'texture-unsupported-layout',
        layer: 0,
        width: 1,
        height: 1,
        byteLength: 20,
        fallback: 'black-layer',
      }),
    ]);
  });

  it('attaches source-layer uses to structured upload warnings', () => {
    installGpuConstStubs();
    const { device } = makeDevice();
    const array = createMaterialTextureArray(
      device,
      [{ image: { width: 0, height: 0 } }],
      'rgba8unorm-srgb',
      [{
        layer: 0,
        uses: [{ materialIndex: 3, field: 'baseColorMap', colorSpace: 'srgb', texCoord: 1 }],
      }],
    );

    expect(array.structuredWarnings).toEqual([
      expect.objectContaining({
        code: 'texture-unreadable',
        layer: 0,
        fallback: 'black-layer',
        uses: [{ materialIndex: 3, field: 'baseColorMap', colorSpace: 'srgb', texCoord: 1 }],
      }),
    ]);
  });

  it('accepts regular material-map sampler policy requests consumed by the pt-webgpu descriptor', () => {
    installGpuConstStubs();
    const { device } = makeDevice();
    const array = createMaterialTextureArray(
      device,
      [rawImage(1, 1)],
      'rgba8unorm-srgb',
      [{
        layer: 0,
        uses: [{
          materialIndex: 2,
          field: 'baseColorMap',
          colorSpace: 'srgb',
          texCoord: 0,
          magFilter: 'nearest',
          minFilter: 'nearest',
          mipFilter: 'none',
        }],
      }],
    );

    expect(array.structuredWarnings.filter((warning) =>
      warning.code === 'texture-sampler-policy-approximation',
    )).toEqual([]);
    expect(array.warnings.some((warning) => warning.includes('sampler policy'))).toBe(false);
  });

  it('accepts authored mip policies when mag/min filtering stays linear', () => {
    installGpuConstStubs();
    const { device } = makeDevice();
    const array = createMaterialTextureArray(
      device,
      [rawImage(1, 1), rawImage(1, 1)],
      'rgba8unorm-srgb',
      [
        {
          layer: 0,
          uses: [{
            materialIndex: 0,
            field: 'baseColorMap',
            colorSpace: 'srgb',
            texCoord: 0,
            magFilter: 'linear',
            minFilter: 'linear',
            mipFilter: 'none',
          }],
        },
        {
          layer: 1,
          uses: [{
            materialIndex: 1,
            field: 'emissiveMap',
            colorSpace: 'srgb',
            texCoord: 0,
            magFilter: 'linear',
            minFilter: 'linear',
            mipFilter: 'nearest',
          }],
        },
      ],
    );

    expect(array.structuredWarnings.filter((warning) =>
      warning.code === 'texture-sampler-policy-approximation',
    )).toEqual([]);
  });

  // ── T1-6 — rgba16float emissive array (HDR emissive) ────────────────────────
  // Decode a half-float (Uint16 bit pattern) back to a JS number for assertions.
  function halfToFloat(h: number): number {
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h & 0x7c00) >> 10;
    const frac = h & 0x03ff;
    if (exp === 0) return sign * frac * 2 ** -24;
    if (exp === 0x1f) return frac ? NaN : sign * Infinity;
    return sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  const srgbToLinear = (u: number): number =>
    u <= 0.04045 ? u / 12.92 : ((u + 0.055) / 1.055) ** 2.4;

  it('uploads an rgba16float emissive array with a half-float dummy for empty scenes', () => {
    installGpuConstStubs();
    const { device, writeTexture, createTexture } = makeDevice();
    const array = createMaterialTextureArray(device, [], 'rgba16float');
    expect(createTexture).toHaveBeenCalledWith(expect.objectContaining({ format: 'rgba16float' }));
    // Linear-white dummy texel: 1.0 → half-float 0x3c00 in all four channels.
    const call = writeTexture.mock.calls[0] as [unknown, Uint16Array, unknown, unknown];
    expect(Array.from(call[1])).toEqual([0x3c00, 0x3c00, 0x3c00, 0x3c00]);
  });

  it('sRGB-decodes 8-bit LDR emissive into linear half-floats (visually identical round-trip)', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    // A mid-gray + white LDR emissive texel pair (sRGB-encoded 8-bit bytes).
    createMaterialTextureArray(
      device,
      [{ width: 2, height: 1, data: new Uint8Array([128, 128, 128, 255, 255, 255, 255, 255]) }],
      'rgba16float',
    );
    const call = writeTexture.mock.calls[0] as [unknown, Uint16Array, GPUImageDataLayout, unknown];
    // 8 half-floats (2 texels × RGBA); RGB is sRGB-decoded, alpha linear.
    expect(call[1]).toBeInstanceOf(Uint16Array);
    const px = Array.from(call[1]).map(halfToFloat);
    // Texel 0: rgb = srgbToLinear(128/255) ≈ 0.2158, a = 1.0.
    const expectMid = srgbToLinear(128 / 255);
    expect(px[0]).toBeCloseTo(expectMid, 2);
    expect(px[1]).toBeCloseTo(expectMid, 2);
    expect(px[2]).toBeCloseTo(expectMid, 2);
    expect(px[3]).toBeCloseTo(1, 3);
    // Texel 1: rgb = srgbToLinear(1) = 1, a = 1.
    expect(px[4]).toBeCloseTo(1, 3);
    expect(px[7]).toBeCloseTo(1, 3);
    // 4 half-floats × 2 texels = 8 bytes/row × wait: 2 texels × 4 ch × 2 bytes = 16.
    expect(call[2]).toEqual({ bytesPerRow: 16, rowsPerImage: 1 });
  });

  it('preserves HDR float emissive values > 1.0 (the linear pass-through path)', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    // Linear HDR radiance: a bright emitter (>1) + a dim one. Float32 → linear pass-through.
    createMaterialTextureArray(
      device,
      [{ width: 2, height: 1, data: new Float32Array([8.5, 2.0, 0.5, 1.0, 100.0, 0.25, 0.0, 1.0]) }],
      'rgba16float',
    );
    const call = writeTexture.mock.calls[0] as [unknown, Uint16Array, unknown, unknown];
    const px = Array.from(call[1]).map(halfToFloat);
    // NO sRGB decode, NO [0,1] clamp — HDR survives (half-float carries up to ~65504).
    expect(px[0]).toBeCloseTo(8.5, 1);
    expect(px[1]).toBeCloseTo(2.0, 2);
    expect(px[2]).toBeCloseTo(0.5, 2);
    expect(px[3]).toBeCloseTo(1.0, 3);
    expect(px[4]).toBeCloseTo(100.0, 0); // >> 1.0 survives packing
    expect(px[5]).toBeCloseTo(0.25, 2);
  });

  it('accepts explicit mipmapped bump policies because bump height samples are policy-aware', () => {
    installGpuConstStubs();
    const { device } = makeDevice();
    const array = createMaterialTextureArray(
      device,
      [rawImage(1, 1)],
      'rgba8unorm',
      [{
        layer: 0,
        uses: [{
          materialIndex: 4,
          field: 'bumpMap',
          colorSpace: 'linear',
          texCoord: 0,
          magFilter: 'linear',
          minFilter: 'linear',
          mipFilter: 'linear',
        }],
      }],
    );

    expect(array.structuredWarnings.filter((warning) =>
      warning.code === 'texture-sampler-policy-approximation',
    )).toEqual([]);
    expect(array.warnings.some((warning) => warning.includes('sampler policy'))).toBe(false);
  });
});
