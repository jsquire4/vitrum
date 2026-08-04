import { describe, expect, it, vi } from 'vitest';
import {
  createMaterialTextureArray,
  createMaterialTextureArrayFromStaged,
  materialTextureMipLevelCount,
  stageMaterialTextureUploadPlan,
} from '../scene/materialTextureArray.js';
import { createPtWebgpuTextureSource } from '../materialTextureSource.js';
import { installGpuConstStubs } from './gpuStub.js';

function makeDevice() {
  const writeTexture = vi.fn();
  const beginRenderPass = vi.fn(() => ({
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    setViewport: vi.fn(),
    draw: vi.fn(),
    end: vi.fn(),
  }));
  const submit = vi.fn();
  const copyTextureToTexture = vi.fn();
  const createTexture = vi.fn((desc: GPUTextureDescriptor): GPUTexture => {
    const size = desc.size as { width: number; height: number; depthOrArrayLayers?: number };
    return {
      label: desc.label,
      width: size.width,
      height: size.height,
      depthOrArrayLayers: size.depthOrArrayLayers ?? 1,
      format: desc.format,
      createView: vi.fn(() => ({})),
      destroy: vi.fn(),
    } as unknown as GPUTexture;
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

function mockGpuTexture(format: GPUTextureFormat = 'rgba8unorm'): GPUTexture {
  return {
    width: 1,
    height: 1,
    depthOrArrayLayers: 1,
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: '2d',
    format,
    usage: GPUTextureUsage.TEXTURE_BINDING,
    label: '',
    createView: vi.fn(() => ({} as GPUTextureView)),
    destroy: vi.fn(),
  } as unknown as GPUTexture;
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

  it('rejects unreadable authored sources before GPU allocation', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    expect(() => createMaterialTextureArray(
      device,
      [{ image: { width: 0, height: 0 } }],
      'rgba8unorm-srgb',
      [{
        layer: 0,
        uses: [{ materialIndex: 3, field: 'baseColorMap', colorSpace: 'srgb', texCoord: 1 }],
      }],
    )).toThrow(/authored source 0 has no usable image/);
    expect(createTexture).not.toHaveBeenCalled();
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
    }])).toThrow(/raw material data must be Uint8/);
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

  it('budgets every eagerly retained CPU radiance mip chain across layers', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    const dimension = 2048;
    const sources = Array.from({ length: 6 }, () => ({
      width: dimension,
      height: dimension,
      // One input byte per texel keeps the fixture modest; normalization still
      // expands each layer into a retained rgba16float mip chain.
      data: new Uint8Array(dimension * dimension),
    }));

    expect(() => createMaterialTextureArray(
      device,
      sources,
      'rgba16float',
    )).toThrow(/estimated array upload peak/);
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

  it('carries the top half-subnormal into minimum-normal half on both signs', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    const carryBoundary = 2 ** -14 - 2 ** -25;
    createMaterialTextureArray(
      device,
      [{ width: 2, height: 1, data: new Float32Array([carryBoundary, -carryBoundary]) }],
      'rgba16float',
    );

    const base = writeTexture.mock.calls[0]?.[1] as Uint16Array;
    expect(Array.from(base)).toEqual([
      0x0400, 0x0400, 0x0400, 0x3c00,
      0x8400, 0x8400, 0x8400, 0x3c00,
    ]);
  });

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

  it('keeps integer light-map texels linear in the shared HDR radiance array', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    createMaterialTextureArray(
      device,
      [{ width: 1, height: 1, data: new Uint8Array([128, 64, 32, 255]) }],
      'rgba16float',
      [{
        layer: 0,
        uses: [{ materialIndex: 0, field: 'lightMap', colorSpace: 'linear', texCoord: 0 }],
      }],
      new Set(),
      { lightMap: [[[1, 1, 1]]] },
    );
    const call = writeTexture.mock.calls[0] as [unknown, Uint16Array, unknown, unknown];
    const px = Array.from(call[1]).map(halfToFloat);
    expect(px[0]).toBeCloseTo(128 / 255, 3);
    expect(px[1]).toBeCloseTo(64 / 255, 3);
    expect(px[2]).toBeCloseTo(32 / 255, 3);
    expect(px[0]).not.toBeCloseTo(srgbToLinear(128 / 255), 2);
  });

  it('stages external emissive and light-map images through role-correct transfer formats', () => {
    installGpuConstStubs();
    const external = { width: 1, height: 1 };
    const emissiveDevice = makeDevice();
    createMaterialTextureArray(
      emissiveDevice.device,
      [external],
      'rgba16float',
      [{
        layer: 0,
        uses: [{ materialIndex: 0, field: 'emissiveMap', colorSpace: 'srgb', texCoord: 0 }],
      }],
    );
    expect(emissiveDevice.createTexture).toHaveBeenCalledWith(expect.objectContaining({
      label: expect.stringContaining('radiance.srgbStage'),
      format: 'rgba8unorm-srgb',
    }));

    const lightMapDevice = makeDevice();
    createMaterialTextureArray(
      lightMapDevice.device,
      [external],
      'rgba16float',
      [{
        layer: 0,
        uses: [{ materialIndex: 0, field: 'lightMap', colorSpace: 'linear', texCoord: 0 }],
      }],
    );
    expect(lightMapDevice.createTexture).toHaveBeenCalledWith(expect.objectContaining({
      label: expect.stringContaining('radiance.linearStage'),
      format: 'rgba8unorm',
    }));
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

  it('validates every exact CPU-generated radiance mip before allocation', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    const f32Min = 1.401298464324817e-45;
    expect(() => createMaterialTextureArray(
      device,
      [{
        width: 2,
        height: 2,
        data: new Uint8Array([
          255, 0, 0, 255,
          0, 0, 0, 255,
          0, 0, 0, 255,
          0, 0, 0, 255,
        ]),
      }],
      'rgba16float',
      [{
        layer: 0,
        uses: [{ materialIndex: 0, field: 'lightMap', colorSpace: 'linear', texCoord: 0 }],
      }],
      new Set(),
      { lightMap: [[[f32Min, f32Min, f32Min]]] },
    )).toThrow(/lightMap layer 0 mip 1 texel 0 radiance.*underflow/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('uploads the exact validated CPU rgba16float mip bytes without GPU regeneration', () => {
    installGpuConstStubs();
    const { device, writeTexture, beginRenderPass } = makeDevice();
    createMaterialTextureArray(
      device,
      [{
        width: 2,
        height: 2,
        data: new Float32Array([
          4, 0, 0, 1,
          0, 0, 0, 1,
          0, 0, 0, 1,
          0, 0, 0, 1,
        ]),
      }],
      'rgba16float',
      [{
        layer: 0,
        uses: [{ materialIndex: 0, field: 'lightMap', colorSpace: 'linear', texCoord: 0 }],
      }],
      new Set(),
      { lightMap: [[[1, 1, 1]]] },
    );

    expect(writeTexture).toHaveBeenCalledTimes(2);
    expect(writeTexture.mock.calls[0]?.[0]).toEqual(expect.objectContaining({ mipLevel: 0 }));
    expect(writeTexture.mock.calls[1]?.[0]).toEqual(expect.objectContaining({ mipLevel: 1 }));
    const mipOne = writeTexture.mock.calls[1]?.[1] as Uint16Array;
    expect(Array.from(mipOne).map(halfToFloat)).toEqual([1, 0, 0, 1]);
    expect(beginRenderPass).not.toHaveBeenCalled();
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
    )).toThrow(/emissiveMap layer 0 mip 0 texel 0 radiance must remain finite/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects exact positive light-map texel products that disappear in Float32', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    const f32Min = 1.401298464324817e-45;
    expect(() => createMaterialTextureArray(
      device,
      [{ width: 1, height: 1, data: new Uint8Array([1, 0, 0, 255]) }],
      'rgba16float',
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
    )).toThrow(/positive .*lightMap layer 0 mip 0 texel 0 radiance.*underflow/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects opaque light-map sources without exact CPU-readable texels', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    const external = { width: 1, height: 1 };
    expect(() => createMaterialTextureArray(
      device,
      [external],
      'rgba16float',
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
    )).toThrow(/outgoing-radiance source 0 has no exact CPU-readable texels/);
    expect(createTexture).not.toHaveBeenCalled();
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

  it('snapshots the source list and image payload accessors exactly once', () => {
    installGpuConstStubs();
    const { device } = makeDevice();
    const reads = { length: 0, element: 0, image: 0, width: 0, height: 0, data: 0 };
    const image = Object.create(null) as Record<string, unknown>;
    Object.defineProperties(image, {
      width: { get: () => { reads.width += 1; return 1; } },
      height: { get: () => { reads.height += 1; return 1; } },
      data: { get: () => { reads.data += 1; return new Uint8Array([7, 8, 9, 255]); } },
    });
    const source = Object.create(null) as Record<string, unknown>;
    Object.defineProperty(source, 'image', {
      get: () => { reads.image += 1; return image; },
    });
    const sources = new Proxy({} as ReadonlyArray<unknown>, {
      get(_target, property) {
        if (property === 'length') { reads.length += 1; return 1; }
        if (property === '0') { reads.element += 1; return source; }
        return undefined;
      },
    });

    createMaterialTextureArray(device, sources, 'rgba8unorm');

    expect(reads).toEqual({ length: 1, element: 1, image: 1, width: 1, height: 1, data: 1 });
  });

  it('enforces non-monotonic authored-channel policies by material-map role', () => {
    installGpuConstStubs();
    const info = (field: string) => [{
      layer: 0,
      uses: [{ materialIndex: 0, field, colorSpace: 'linear' as const, texCoord: 0 }],
    }];
    const twoChannel = { width: 1, height: 1, data: new Uint8Array([128, 64]) };
    const threeChannel = { width: 1, height: 1, data: new Uint8Array([128, 64, 32]) };

    for (const field of ['normalMap', 'anisotropyMap', 'metallicMap']) {
      const { device, createTexture } = makeDevice();
      expect(() => createMaterialTextureArray(
        device, [twoChannel], 'rgba8unorm', info(field),
      )).toThrow(new RegExp(`${field} cannot consume an authored 2-channel source`));
      expect(createTexture).not.toHaveBeenCalled();
    }
    {
      const { device, createTexture } = makeDevice();
      expect(() => createMaterialTextureArray(
        device, [threeChannel], 'rgba8unorm', info('sheenRoughnessMap'),
      )).toThrow(/sheenRoughnessMap cannot consume an authored 3-channel source/);
      expect(createTexture).not.toHaveBeenCalled();
    }
    {
      const { device, writeTexture } = makeDevice();
      createMaterialTextureArray(
        device,
        [{ width: 1, height: 1, data: new Uint8Array([64]) }],
        'rgba8unorm',
        info('metallicMap'),
      );
      expect(Array.from(writeTexture.mock.calls[0]![1] as Uint8Array)).toEqual([64, 64, 64, 255]);
    }
    {
      const { device } = makeDevice();
      expect(() => createMaterialTextureArray(
        device, [threeChannel], 'rgba8unorm', info('normalMap'),
      )).not.toThrow();
    }
  });

  it('rejects ambiguous external images for authored-alpha scalar maps', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    expect(() => createMaterialTextureArray(
      device,
      [{ width: 1, height: 1 }],
      'rgba8unorm',
      [{
        layer: 0,
        uses: [{
          materialIndex: 0,
          field: 'specularIntensityMap',
          colorSpace: 'linear',
          texCoord: 0,
        }],
      }],
    )).toThrow(/requires an explicitly authored alpha channel/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('rejects a direct layer description that mixes incompatible source profiles', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    expect(() => createMaterialTextureArray(
      device,
      [rawImage(1, 1)],
      'rgba8unorm',
      [{
        layer: 0,
        uses: [
          { materialIndex: 0, field: 'normalMap', colorSpace: 'linear', texCoord: 0 },
          { materialIndex: 0, field: 'transmissionMap', colorSpace: 'linear', texCoord: 0 },
        ],
      }],
    )).toThrow(/mixes incompatible material-map source profiles/);
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('deep-owns each raw payload before observing a later source', () => {
    installGpuConstStubs();
    const { device, writeTexture } = makeDevice();
    const firstBytes = new Uint8Array([7, 8, 9, 255]);
    const first = { width: 1, height: 1, data: firstBytes };
    const second = Object.create(null) as Record<string, unknown>;
    Object.defineProperties(second, {
      width: { value: 1, enumerable: true },
      height: { value: 1, enumerable: true },
      data: {
        enumerable: true,
        get() {
          firstBytes.fill(0);
          return new Uint8Array([1, 2, 3, 255]);
        },
      },
    });

    createMaterialTextureArray(device, [first, second], 'rgba8unorm');

    expect(Array.from(writeTexture.mock.calls[0]![1] as Uint8Array)).toEqual([7, 8, 9, 255]);
  });

  it('rejects SharedArrayBuffer-backed raw Uint8, Uint16, and Float32 texels before allocation', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    const sharedViews = [
      new Uint8Array(new SharedArrayBuffer(4 * Uint8Array.BYTES_PER_ELEMENT)),
      new Uint16Array(new SharedArrayBuffer(4 * Uint16Array.BYTES_PER_ELEMENT)),
      new Float32Array(new SharedArrayBuffer(4 * Float32Array.BYTES_PER_ELEMENT)),
    ];
    for (const shared of sharedViews) {
      expect(() => createMaterialTextureArray(
        device,
        [{ width: 1, height: 1, data: shared }],
        'rgba8unorm',
      )).toThrow(/SharedArrayBuffer/);
    }
    expect(createTexture).not.toHaveBeenCalled();
  });

  it('never destroys a host GPU source returned as a target or native candidate', () => {
    installGpuConstStubs();
    for (const aliasPosition of ['target', 'native'] as const) {
      const { device, createTexture } = makeDevice();
      const sourceTexture = mockGpuTexture();
      const source = createPtWebgpuTextureSource(device, sourceTexture, {
        format: 'rgba8unorm',
        colorSpace: 'linear',
      });
      const target = mockGpuTexture();
      createTexture.mockReset();
      if (aliasPosition === 'target') {
        createTexture.mockReturnValueOnce(sourceTexture);
      } else {
        createTexture.mockReturnValueOnce(target).mockReturnValueOnce(sourceTexture);
      }
      const plan = stageMaterialTextureUploadPlan(device, [{
        sources: [source],
        format: 'rgba8unorm',
      }]);

      expect(() => createMaterialTextureArrayFromStaged(
        device,
        plan.arrays[0]!,
      )).toThrow(/aliased/);
      expect(sourceTexture.destroy).not.toHaveBeenCalled();
      if (aliasPosition === 'native') expect(target.destroy).toHaveBeenCalledTimes(1);
    }
  });

  it('shares target/native identities across staged array consumes without double destruction', () => {
    installGpuConstStubs();
    const request = { sources: [rawImage(1, 1)], format: 'rgba8unorm' as const };

    {
      const { device, createTexture } = makeDevice();
      const firstTarget = mockGpuTexture();
      const firstNative = mockGpuTexture();
      createTexture.mockReset()
        .mockReturnValueOnce(firstTarget)
        .mockReturnValueOnce(firstNative)
        .mockReturnValueOnce(firstTarget);
      const plan = stageMaterialTextureUploadPlan(device, [request, request]);
      createMaterialTextureArrayFromStaged(device, plan.arrays[0]!);
      expect(() => createMaterialTextureArrayFromStaged(device, plan.arrays[1]!))
        .toThrow(/aliased/);
      expect(firstTarget.destroy).not.toHaveBeenCalled();
      expect(firstNative.destroy).toHaveBeenCalledTimes(1);
    }

    {
      const { device, createTexture } = makeDevice();
      const firstTarget = mockGpuTexture();
      const retiredNative = mockGpuTexture();
      const secondTarget = mockGpuTexture();
      createTexture.mockReset()
        .mockReturnValueOnce(firstTarget)
        .mockReturnValueOnce(retiredNative)
        .mockReturnValueOnce(secondTarget)
        .mockReturnValueOnce(retiredNative);
      const plan = stageMaterialTextureUploadPlan(device, [request, request]);
      createMaterialTextureArrayFromStaged(device, plan.arrays[0]!);
      expect(() => createMaterialTextureArrayFromStaged(device, plan.arrays[1]!))
        .toThrow(/aliased/);
      expect(retiredNative.destroy).toHaveBeenCalledTimes(1);
      expect(secondTarget.destroy).toHaveBeenCalledTimes(1);
    }
  });

  it('shares dummy identities across consumes and makes each staged token single-use', () => {
    installGpuConstStubs();
    const { device, createTexture } = makeDevice();
    const sharedDummy = mockGpuTexture();
    createTexture.mockReset().mockReturnValue(sharedDummy);
    const plan = stageMaterialTextureUploadPlan(device, [
      { sources: [], format: 'rgba8unorm' },
      { sources: [], format: 'rgba8unorm' },
    ]);

    createMaterialTextureArrayFromStaged(device, plan.arrays[0]!);
    expect(() => createMaterialTextureArrayFromStaged(device, plan.arrays[0]!))
      .toThrow(/invalid or foreign/);
    expect(() => createMaterialTextureArrayFromStaged(device, plan.arrays[1]!))
      .toThrow(/aliased/);
    expect(sharedDummy.destroy).not.toHaveBeenCalled();
  });
});
