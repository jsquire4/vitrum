import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  _probeRadiometricAtlasValidationBudgetForTest,
  packMaterialTextureAtlas,
} from '../bvh/materialTextureAtlasPack.js';
import {
  unpackMaterialTextureAtlasPixels,
} from '../bvh/materialTextureAtlasCodec.js';
import {
  createWalkaroundWebGpuTextureSource,
} from '../materialTextureSource.js';
import { MATERIAL_ATLAS_WGSL } from '../shaders/materialAtlas.wgsl.js';

function material(partial: Partial<MaterialSpec>): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 0.5,
    metallic: 0,
    ...partial,
  };
}

function pack(single: MaterialSpec) {
  return packMaterialTextureAtlas(
    [single],
    new Uint32Array([0]),
    1,
  );
}

function linearFloatMap(
  rgba: readonly number[],
  width = 1,
  height = 1,
) {
  return {
    width,
    height,
    data: new Float32Array(rgba),
    __vitrum_hint__: {
      channels: 4,
      dataType: 'float32',
      colorSpace: 'linear',
    } as const,
  };
}

function gpuTexture(
  format: GPUTextureFormat,
  width = 1,
  height = 1,
): GPUTexture {
  return {
    width,
    height,
    depthOrArrayLayers: 1,
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: '2d',
    format,
    usage: 0x04,
    createView: () => ({} as GPUTextureView),
    destroy: () => undefined,
  } as unknown as GPUTexture;
}

describe('walkaround radiometric material-atlas preflight', () => {
  it('rejects validation staging before entering over-budget codec allocators', () => {
    let decodeCalls = 0;
    let mipCalls = 0;
    expect(() => _probeRadiometricAtlasValidationBudgetForTest({
      // A decoded rgba32float texel needs 16 bytes. One already-retained byte
      // makes this 17-byte peak exceed the synthetic 16-byte transaction cap.
      usedBytes: 1,
      aggregateLimitBytes: 16,
      mipLevelCount: 1,
      onDecode: () => { decodeCalls += 1; },
      onGenerateMip: () => { mipCalls += 1; },
    })).toThrow(/decoded RGBA staging requires 16 CPU bytes \(aggregate 17\)/);
    expect(decodeCalls).toBe(0);
    expect(mipCalls).toBe(0);

    expect(() => _probeRadiometricAtlasValidationBudgetForTest({
      usedBytes: 0,
      // Current decode (16) + target RGBA (16) fit. The returned rgba32float
      // packed mip needs 16 more bytes, so generation must not be entered.
      aggregateLimitBytes: 47,
      mipLevelCount: 2,
      onDecode: () => { decodeCalls += 1; },
      onGenerateMip: () => { mipCalls += 1; },
    })).toThrow(/generated packed codec planes requires 16 CPU bytes \(aggregate 48\)/);
    expect(decodeCalls).toBe(1);
    expect(mipCalls).toBe(0);

    // At the exact 48-byte peak both callbacks run; scoped reservations are
    // released without an internal budget underflow across the next level.
    expect(() => _probeRadiometricAtlasValidationBudgetForTest({
      usedBytes: 0,
      aggregateLimitBytes: 48,
      mipLevelCount: 2,
      onDecode: () => { decodeCalls += 1; },
      onGenerateMip: () => { mipCalls += 1; },
    })).not.toThrow();
    expect(decodeCalls).toBe(3);
    expect(mipCalls).toBe(1);
  });

  it('treats unhinted Float32 emissive maps as linear HDR atlas data', () => {
    const payload = pack(material({
      emissive: [1, 1, 1],
      emissiveMap: {
        handle: {
          width: 1,
          height: 1,
          data: new Float32Array([0.25, 0.5, 4, 1]),
        },
      },
    }));
    const layer = payload.atlasLayers[0]!;
    expect(layer.kind).toBe('cpu');
    if (layer.kind !== 'cpu') return;
    expect(layer.decodeSrgb).toBe(false);
    expect(Array.from(
      unpackMaterialTextureAtlasPixels(layer.data, layer.encoding).slice(0, 3),
    )).toEqual([0.25, 0.5, 4]);
  });

  it('checks every emissive texel rather than accepting a finite average', () => {
    const maxF32 = Math.fround(3.4028234663852886e38);
    const map = linearFloatMap([
      0, 0, 0, 1,
      0, 0, 0, 1,
      0, 0, 0, 1,
      maxF32, 0, 0, 1,
    ], 4, 1);
    expect(() => pack(material({
      emissive: [2, 0, 0],
      emissiveMap: { handle: map },
    }))).toThrow(/remain finite in Float32/);
  });

  it('rejects complete positive emissive-map multiplication collapse', () => {
    const small = 2 ** -80;
    expect(() => pack(material({
      emissive: [small, 0, 0],
      emissiveMap: {
        handle: linearFloatMap([small, 0, 0, 1]),
      },
    }))).toThrow(/underflow completely to zero/);
  });

  it('checks readable light-map spikes, products, and scalar publication', () => {
    const maxF32 = Math.fround(3.4028234663852886e38);
    expect(() => pack(material({
      lightMap: {
        handle: linearFloatMap([
          0, 0, 0, 1,
          0, 0, 0, 1,
          0, 0, 0, 1,
          maxF32, 0, 0, 1,
        ], 4, 1),
      },
      lightMapIntensity: 2,
    }))).toThrow(/remain finite in Float32/);

    const small = 2 ** -80;
    expect(() => pack(material({
      lightMap: { handle: linearFloatMap([small, 0, 0, 1]) },
      lightMapIntensity: small,
    }))).toThrow(/underflow completely to zero/);

    expect(() => pack(material({
      lightMapIntensity: Number.MIN_VALUE,
    }))).toThrow(/remain positive after Float32 packing/);
  });

  it('requires an exact CPU mirror for GPU emissive maps and scans that mirror', () => {
    const device = {} as GPUDevice;
    const withoutMirror = createWalkaroundWebGpuTextureSource(
      device,
      gpuTexture('rgba32float'),
      { format: 'rgba32float', colorSpace: 'linear' },
    );
    expect(() => pack(material({
      emissive: [1, 1, 1],
      emissiveMap: { handle: withoutMirror },
    }))).toThrow(/cpuMirror required/);

    const maxF32 = Math.fround(3.4028234663852886e38);
    const withMirror = createWalkaroundWebGpuTextureSource(
      device,
      gpuTexture('rgba32float'),
      {
        format: 'rgba32float',
        colorSpace: 'linear',
        cpuMirror: {
          width: 1,
          height: 1,
          channels: 4,
          dataType: 'float32',
          colorSpace: 'linear',
          data: new Float32Array([maxF32, 0, 0, 1]),
        },
      },
    );
    expect(() => pack(material({
      emissive: [2, 0, 0],
      emissiveMap: { handle: withMirror },
    }))).toThrow(/remain finite in Float32/);
  });

  it('uses canonical GPU-mirror RG channels and rejects codec collapse', () => {
    const device = {} as GPUDevice;
    const green = createWalkaroundWebGpuTextureSource(
      device,
      gpuTexture('rg32float'),
      {
        format: 'rg32float',
        colorSpace: 'linear',
        cpuMirror: {
          width: 1,
          height: 1,
          channels: 2,
          dataType: 'float32',
          colorSpace: 'linear',
          data: new Float32Array([0, 0.5]),
        },
      },
    );
    expect(() => pack(material({
      emissive: [0, 2, 0],
      emissiveMap: { handle: green },
    }))).not.toThrow();

    const halfCollapse = createWalkaroundWebGpuTextureSource(
      device,
      gpuTexture('r16float'),
      {
        format: 'r16float',
        colorSpace: 'linear',
        cpuMirror: {
          width: 1,
          height: 1,
          channels: 1,
          dataType: 'float32',
          colorSpace: 'linear',
          data: new Float32Array([2 ** -25]),
        },
      },
    );
    expect(() => pack(material({
      emissive: [1, 0, 0],
      emissiveMap: { handle: halfCollapse },
    }))).toThrow(/collapse completely through atlas codec/);
  });

  it('publishes a native-sRGB GPU emissive mirror as the exact canonical CPU atlas layer', () => {
    const source = createWalkaroundWebGpuTextureSource(
      {} as GPUDevice,
      gpuTexture('rgba8unorm-srgb', 2, 2),
      {
        format: 'rgba8unorm-srgb',
        colorSpace: 'srgb',
        cpuMirror: {
          width: 2,
          height: 2,
          channels: 4,
          dataType: 'uint8',
          colorSpace: 'srgb',
          data: new Uint8Array([
            128, 64, 32, 255,
            255, 128, 64, 255,
            32, 64, 128, 255,
            64, 128, 255, 255,
          ]),
        },
      },
    );
    const payload = pack(material({
      emissive: [1, 1, 1],
      emissiveMap: { handle: source },
    }));

    expect(payload.gpuSourceLayers).toHaveLength(0);
    expect(payload.atlasLayers).toHaveLength(1);
    const layer = payload.atlasLayers[0]!;
    expect(layer.kind).toBe('cpu');
    if (layer.kind !== 'cpu') return;
    expect(layer.decodeSrgb).toBe(true);
    expect(layer.mipLevelCount).toBe(2);
    expect(Array.from(
      unpackMaterialTextureAtlasPixels(layer.data, layer.encoding).slice(0, 4),
    )).toEqual([
      Math.fround(128 / 255),
      Math.fround(64 / 255),
      Math.fround(32 / 255),
      1,
    ]);
  });

  it('requires and publishes an exact CPU mirror for GPU light maps before shader use', () => {
    const device = {} as GPUDevice;
    const withoutMirror = createWalkaroundWebGpuTextureSource(
      device,
      gpuTexture('rgba32float'),
      { format: 'rgba32float', colorSpace: 'linear' },
    );
    expect(() => pack(material({
      lightMap: { handle: withoutMirror },
      lightMapIntensity: 2,
    }))).toThrow(/lightMap uses a GPU source without the exact cpuMirror/);

    const withMirror = createWalkaroundWebGpuTextureSource(
      device,
      gpuTexture('rgba32float', 2, 2),
      {
        format: 'rgba32float',
        colorSpace: 'linear',
        cpuMirror: {
          width: 2,
          height: 2,
          channels: 4,
          dataType: 'float32',
          colorSpace: 'linear',
          data: new Float32Array([
            1, 0.5, 0.25, 1,
            0.5, 1, 0.25, 1,
            0.25, 0.5, 1, 1,
            1, 1, 1, 1,
          ]),
        },
      },
    );
    const payload = pack(material({
      lightMap: { handle: withMirror },
      lightMapIntensity: 2,
    }));
    expect(payload.gpuSourceLayers).toHaveLength(0);
    expect(payload.atlasLayers).toEqual([
      expect.objectContaining({
        kind: 'cpu',
        width: 2,
        height: 2,
        mipLevelCount: 2,
        decodeSrgb: false,
      }),
    ]);

    // Runtime guards remain defense in depth after strict publication-time
    // validation; malformed radiance can never be published intentionally.
    expect(MATERIAL_ATLAS_WGSL)
      .toContain('fn materialAtlasFiniteNonNegativeRadianceOrBlack');
    expect(MATERIAL_ATLAS_WGSL)
      .toContain('all(abs(value) <= vec3f(maxFiniteF32))');
    expect(MATERIAL_ATLAS_WGSL)
      .toContain('texelColor.value.rgb * max(intensityMeta.x, 0.0)');
  });

  it('rejects a positive GPU light-map mirror that collapses in a generated mip', () => {
    const source = createWalkaroundWebGpuTextureSource(
      {} as GPUDevice,
      gpuTexture('rgba16float', 2, 2),
      {
        format: 'rgba16float',
        colorSpace: 'linear',
        cpuMirror: {
          width: 2,
          height: 2,
          channels: 4,
          dataType: 'float32',
          colorSpace: 'linear',
          data: new Float32Array([
            2 ** -24, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
          ]),
        },
      },
    );
    expect(() => pack(material({
      lightMap: { handle: source },
      lightMapIntensity: 1,
    }))).toThrow(/collapse completely through atlas codec/);
  });
});
