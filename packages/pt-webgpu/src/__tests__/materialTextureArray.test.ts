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
  const copyTextureToTexture = vi.fn();
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
    limits: { maxTextureDimension2D: 8192, maxTextureArrayLayers: 256 },
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
      copyTextureToTexture,
      finish: vi.fn(() => ({})),
    })),
  } as unknown as GPUDevice;
  return {
    device,
    writeTexture,
    createTexture,
    beginRenderPass,
    copyTextureToTexture,
    submit,
  };
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
    expect(submit).toHaveBeenCalledTimes(4);
  });

  it('builds independent native mip rectangles for heterogeneous source dimensions', () => {
    installGpuConstStubs();
    const { device, copyTextureToTexture } = makeDevice();
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
    expect(array.warnings).toEqual([]);
    expect(array.structuredWarnings).toEqual([]);
    expect(copyTextureToTexture).toHaveBeenCalledTimes(9);
    expect(copyTextureToTexture).toHaveBeenCalledWith(
      expect.objectContaining({ mipLevel: 1 }),
      expect.objectContaining({ mipLevel: 1, origin: { x: 0, y: 0, z: 0 } }),
      { width: 1, height: 2, depthOrArrayLayers: 1 },
    );
    expect(copyTextureToTexture).toHaveBeenCalledWith(
      expect.objectContaining({ mipLevel: 1 }),
      expect.objectContaining({ mipLevel: 1, origin: { x: 0, y: 0, z: 2 } }),
      { width: 2, height: 1, depthOrArrayLayers: 1 },
    );
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

  it('normalizes one-channel as RRR and two-channel as RG0 with opaque alpha', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    createMaterialTextureArray(device, [
      { width: 1, height: 1, data: new Uint8Array([25]) },
      { width: 1, height: 1, data: new Uint8Array([10, 20]) },
    ], 'rgba8unorm');

    const first = writeTexture.mock.calls[0] as [unknown, Uint8Array, unknown, unknown];
    const second = writeTexture.mock.calls[1] as [unknown, Uint8Array, unknown, unknown];
    expect(Array.from(first[1])).toEqual([25, 25, 25, 255]);
    expect(Array.from(second[1])).toEqual([10, 20, 0, 255]);
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

  it('rejects unsupported raw typed-array shapes before GPU allocation', () => {
    installGpuConstStubs();
    const { device, writeTexture, createTexture } = makeDevice();
    expect(() => createMaterialTextureArray(device, [
      { width: 1, height: 1, data: new Float32Array([1, 0, 0, 1, 0]) },
    ])).toThrow(/unsupported raw layout/);
    expect(writeTexture).not.toHaveBeenCalled();
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('surfaces unreadable source dimensions as an explicit black-layer fallback', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    const array = createMaterialTextureArray(
      device,
      [{ image: { width: 0, height: 0 } }],
      'rgba8unorm-srgb',
      [{
        layer: 0,
        uses: [{ materialIndex: 3, field: 'baseColorMap', colorSpace: 'srgb', texCoord: 1 }],
      }],
    );
    expect(createTexture).toHaveBeenCalledTimes(1);
    expect(array.structuredWarnings).toEqual([expect.objectContaining({
      code: 'texture-unreadable',
      layer: 0,
      fallback: 'black-layer',
      uses: [{ materialIndex: 3, field: 'baseColorMap', colorSpace: 'srgb', texCoord: 1 }],
    })]);
  });

  it('rejects non-finite Float32 samples before GPU allocation', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    expect(() => createMaterialTextureArray(device, [{
      width: 1,
      height: 1,
      data: new Float32Array([1, Number.NaN, 0, 1]),
    }], 'rgba16float')).toThrow(/raw HDR sample 1 must be finite/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects lossy out-of-range Float32 values for rgba8 arrays', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    expect(() => createMaterialTextureArray(device, [{
      width: 1,
      height: 1,
      data: new Float32Array([2, 0, 0, 1]),
    }], 'rgba8unorm')).toThrow(/must be in \[0, 1\]/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects unsupported numeric element types instead of coercing them', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    expect(() => createMaterialTextureArray(device, [{
      width: 1,
      height: 1,
      data: new Float64Array([1, 0, 0, 1]),
    }])).toThrow(/unsupported raw layout/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects dimensions above device limits rather than truncating them', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    expect(() => createMaterialTextureArray(device, [{
      width: 8193,
      height: 1,
    }])).toThrow(/truncation is not permitted/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects aggregate mip allocation above the explicit peak budget', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    expect(() => createMaterialTextureArray(device, [
      { width: 8192, height: 8192 },
      { width: 8192, height: 8192 },
    ], 'rgba16float')).toThrow(/estimated array upload peak/);
    expect(createTexture).not.toHaveBeenCalled();
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

    expect(array.structuredWarnings).toEqual([]);
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

    expect(array.structuredWarnings).toEqual([]);
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
    createMaterialTextureArray(device, [], 'rgba16float');
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

  it('uses the same RG0 rule for two-channel HDR emissive payloads', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    createMaterialTextureArray(
      device,
      [{ width: 1, height: 1, data: new Float32Array([0.25, 0.5]) }],
      'rgba16float',
    );
    const call = writeTexture.mock.calls[0] as [unknown, Uint16Array, unknown, unknown];
    const px = Array.from(call[1]).map(halfToFloat);
    expect(px).toEqual([0.25, 0.5, 0, 1]);
  });

  it('rejects a bright emissive texel that overflows an otherwise finite packed factor', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    const f32Max = 3.4028234663852886e38;
    expect(() => createMaterialTextureArray(
      device,
      [{
        width: 2,
        height: 1,
        // The mean red texel is one, so an average-only classifier would
        // accept this map. The exact bright texel must still fail.
        data: new Float32Array([2, 0, 0, 1, 0, 0, 0, 1]),
      }],
      'rgba16float',
      [{
        layer: 0,
        uses: [{
          materialIndex: 0,
          field: 'emissiveMap',
          colorSpace: 'srgb',
          texCoord: 0,
        }],
      }],
      new Set(),
      { emissiveMap: [[[f32Max, 0, 0]]] },
    )).toThrow(/emissiveMap layer 0 texel 0 radiance must remain finite/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects exact positive light-map texel products that disappear in Float32', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    const f32Min = 1.401298464324817e-45;
    expect(() => createMaterialTextureArray(
      device,
      [{ width: 1, height: 1, data: new Uint8Array([1, 0, 0, 255]) }],
      'rgba8unorm',
      [{
        layer: 0,
        uses: [{
          materialIndex: 0,
          field: 'lightMap',
          colorSpace: 'linear',
          texCoord: 0,
        }],
      }],
      new Set(),
      { lightMap: [[[f32Min, f32Min, f32Min]]] },
    )).toThrow(/positive .*lightMap layer 0 texel 0 radiance.*underflow/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('keeps opaque light-map sources supported through the shader finite guard', () => {
    installGpuConstStubs();
    const { device } = makeDevice();
    const external = { width: 1, height: 1 };
    const array = createMaterialTextureArray(
      device,
      [external],
      'rgba8unorm',
      [{
        layer: 0,
        uses: [{
          materialIndex: 0,
          field: 'lightMap',
          colorSpace: 'linear',
          texCoord: 0,
        }],
      }],
      new Set(),
      { lightMap: [[[1, 1, 1]]] },
    );
    expect(array.layerCount).toBe(1);
  });

  it('fails closed for active emissive envelopes without exact CPU texels', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    expect(() => createMaterialTextureArray(
      device,
      [{ width: 1, height: 1 }],
      'rgba16float',
      [{
        layer: 0,
        uses: [{
          materialIndex: 0,
          field: 'emissiveMap',
          colorSpace: 'srgb',
          texCoord: 0,
        }],
      }],
      new Set(),
      { emissiveMap: [[[1, 1, 1]]] },
    )).toThrow(/no exact CPU-readable texels/);
    expect(createTexture).not.toHaveBeenCalled();
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

    expect(array.structuredWarnings).toEqual([]);
    expect(array.warnings.some((warning) => warning.includes('sampler policy'))).toBe(false);
  });
});
