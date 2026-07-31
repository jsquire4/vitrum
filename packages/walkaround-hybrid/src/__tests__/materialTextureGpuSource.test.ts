import { describe, expect, it, vi } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import { createWalkaroundWebGpuTextureSource } from '../materialTextureSource.js';
import {
  MATERIAL_ATLAS_GENERATE_MIP_WGSL,
  MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL,
  materialTextureAtlasFingerprintParts,
  packMaterialTextureAtlas,
  planMaterialTextureAtlasLayout,
  uploadMaterialTextureAtlas,
} from '../pipeline/materialTextureAtlas.js';

const TEXTURE_BINDING = 0x04;

function makeSourceTexture(overrides: Partial<{
  width: number;
  height: number;
  depthOrArrayLayers: number;
  mipLevelCount: number;
  sampleCount: number;
  dimension: GPUTextureDimension;
  format: GPUTextureFormat;
  usage: GPUTextureUsageFlags;
}> = {}) {
  const createView = vi.fn(() => ({}));
  const destroy = vi.fn();
  const texture = {
    width: 4,
    height: 4,
    depthOrArrayLayers: 1,
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: '2d',
    format: 'rgba8unorm',
    usage: TEXTURE_BINDING,
    createView,
    destroy,
    ...overrides,
  } as unknown as GPUTexture;
  return { texture, createView, destroy };
}

function materialWithMaps(baseColorHandle: unknown, normalHandle?: unknown): MaterialSpec {
  return {
    baseColor: [1, 1, 1],
    roughness: 0.5,
    metallic: 0,
    baseColorMap: { handle: baseColorHandle },
    ...(normalHandle == null ? {} : { normalMap: { handle: normalHandle } }),
  };
}

function makeDestinationTexture(descriptor: GPUTextureDescriptor) {
  return {
    descriptor,
    createView: vi.fn(() => ({})),
    destroy: vi.fn(),
  };
}

function makeUploadHarness(failBeginComputePass = false) {
  const destinations: ReturnType<typeof makeDestinationTexture>[] = [];
  const mappedRanges: ArrayBuffer[] = [];
  const createTexture = vi.fn((descriptor: GPUTextureDescriptor) => {
    const destination = makeDestinationTexture(descriptor);
    destinations.push(destination);
    return destination;
  });
  const createShaderModule = vi.fn((descriptor: GPUShaderModuleDescriptor) => descriptor);
  const createBindGroupLayout = vi.fn((descriptor: GPUBindGroupLayoutDescriptor) => descriptor);
  const createPipelineLayout = vi.fn((descriptor: GPUPipelineLayoutDescriptor) => descriptor);
  const createComputePipeline = vi.fn(() => ({}));
  const createBindGroup = vi.fn(() => ({}));
  const bufferDestroy = vi.fn();
  const createBuffer = vi.fn((descriptor: GPUBufferDescriptor) => {
    const mappedRange = new ArrayBuffer(Number(descriptor.size));
    mappedRanges.push(mappedRange);
    return {
      getMappedRange: vi.fn(() => mappedRange),
      unmap: vi.fn(),
      destroy: bufferDestroy,
    };
  });
  const dispatchWorkgroups = vi.fn();
  const pass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups,
    end: vi.fn(),
  };
  const beginComputePass = vi.fn(() => {
    if (failBeginComputePass) throw new Error('compute pass construction failed');
    return pass;
  });
  const finish = vi.fn(() => ({}));
  const copyTextureToTexture = vi.fn();
  const writeTexture = vi.fn();
  const submit = vi.fn();
  const device = {
    limits: { maxTextureDimension2D: 4096, maxTextureArrayLayers: 256 },
    createTexture,
    createShaderModule,
    createBindGroupLayout,
    createPipelineLayout,
    createComputePipeline,
    createBindGroup,
    createBuffer,
    createCommandEncoder: vi.fn(() => ({
      beginComputePass,
      copyTextureToTexture,
      finish,
    })),
    queue: { writeTexture, submit },
  } as unknown as GPUDevice;
  return {
    device,
    destinations,
    createTexture,
    createShaderModule,
    createBindGroupLayout,
    createBuffer,
    bufferDestroy,
    dispatchWorkgroups,
    copyTextureToTexture,
    beginComputePass,
    writeTexture,
    submit,
    mappedRanges,
  };
}

describe('walkaround explicit WebGPU material texture sources', () => {
  it('preflights metadata clones, mip staging, and renderer replicas before GPU allocation', () => {
    const harness = makeUploadHarness();
    const payload = packMaterialTextureAtlas(
      [{
        baseColor: [1, 1, 1],
        roughness: 0.5,
        metallic: 0,
        baseColorMap: {
          handle: {
            width: 2,
            height: 2,
            data: new Uint8Array(16).fill(255),
            __vitrum_hint__: { channels: 4, dataType: 'uint8' },
          },
          mipFilter: 'linear',
        },
      }],
      new Uint32Array([0]),
      1,
    );
    expect(() => uploadMaterialTextureAtlas(
      harness.device,
      payload,
      { maxCpuTransactionBytes: 1 },
    )).toThrow(/Material atlas CPU transaction requires/);
    expect(harness.createTexture).not.toHaveBeenCalled();

    const plan = planMaterialTextureAtlasLayout(
      payload,
      harness.device.limits,
    );
    expect(() => uploadMaterialTextureAtlas(
      harness.device,
      payload,
      {
        replicatedResidentCopies: 2,
        maxTransactionBytes:
          plan.candidatePeakBytes + plan.allocatedBytes - 1,
      },
    )).toThrow(/peer replicas/);
    expect(harness.createTexture).not.toHaveBeenCalled();
  });

  it('preflights the transaction budget before allocating a GPU-source atlas', () => {
    const harness = makeUploadHarness();
    const device = harness.device;
    const makeSource = (colorSpace: 'srgb' | 'linear') => {
      const { texture } = makeSourceTexture({ width: 4096, height: 4096 });
      return createWalkaroundWebGpuTextureSource(device, texture, {
        format: 'rgba8unorm',
        colorSpace,
      });
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      baseColorMap: { handle: makeSource('srgb') },
      normalMap: { handle: makeSource('linear') },
    };

    const payload = packMaterialTextureAtlas(
      [material],
      new Uint32Array([0]),
      1,
    );
    expect(() => uploadMaterialTextureAtlas(
      device,
      payload,
      { maxTransactionBytes: 1 },
    )).toThrow(/Material atlas transaction requires .*above the 1-byte transaction budget/);
    expect(harness.createTexture).not.toHaveBeenCalled();
  });

  it('pins source format, transfer function, selected subresource, and identity', () => {
    const device = {} as GPUDevice;
    const { texture } = makeSourceTexture({
      width: 16,
      height: 8,
      depthOrArrayLayers: 3,
      mipLevelCount: 4,
    });
    const first = createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      baseMipLevel: 2,
      arrayLayer: 1,
    });
    const second = createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      baseMipLevel: 2,
      arrayLayer: 1,
    });

    expect(first).toMatchObject({
      device,
      texture,
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      dimension: '2d',
      ownership: 'host',
      baseMipLevel: 2,
      arrayLayer: 1,
      width: 4,
      height: 2,
    });
    expect(Object.isFrozen(first)).toBe(true);
    expect(second.sourceId).toBeGreaterThan(first.sourceId);
    expect([second.compatibilityKeyLo, second.compatibilityKeyHi]).not.toEqual(
      [first.compatibilityKeyLo, first.compatibilityKeyHi],
    );
  });

  it('uses stable content revisions for portable GI compatibility and session keys otherwise', () => {
    const device = {} as GPUDevice;
    const { texture } = makeSourceTexture();
    const make = (contentRevision?: string) =>
      createWalkaroundWebGpuTextureSource(device, texture, {
        format: 'rgba8unorm',
        colorSpace: 'srgb',
        ...(contentRevision == null ? {} : { contentRevision }),
      });
    const atlasFingerprint = (source: ReturnType<typeof make>): number[] => {
      const payload = packMaterialTextureAtlas(
        [{
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
          baseColorMap: { handle: source },
        }],
        new Uint32Array([0]),
        1,
      );
      return Array.from(
        materialTextureAtlasFingerprintParts(payload)[0] as Uint32Array,
      );
    };

    expect(atlasFingerprint(make())).not.toEqual(atlasFingerprint(make()));
    expect(atlasFingerprint(make('asset-a@sha256:123'))).toEqual(
      atlasFingerprint(make('asset-a@sha256:123')),
    );
    expect(atlasFingerprint(make('asset-a@sha256:123'))).not.toEqual(
      atlasFingerprint(make('asset-a@sha256:456')),
    );
    expect(() => make('')).toThrow(/contentRevision/);
  });

  it('snapshots an exact immutable CPU mirror without changing GPU-backed atlas identity', () => {
    const device = {} as GPUDevice;
    const { texture } = makeSourceTexture({ width: 2, height: 2 });
    const pixels = new Uint8Array([
      255, 0, 0, 255,
      0, 255, 0, 255,
      0, 0, 255, 255,
      128, 64, 32, 255,
    ]);
    const source = createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
      cpuMirror: {
        width: 2,
        height: 2,
        channels: 4,
        dataType: 'uint8',
        colorSpace: 'linear',
        data: pixels,
      },
    });

    pixels[0] = 0;
    expect(source.cpuMirror?.data[0]).toBe(255);
    expect(() => Reflect.set(source.cpuMirror!.data, '0', 0)).toThrow(/immutable/);

    const payload = packMaterialTextureAtlas(
      [materialWithMaps(source)],
      new Uint32Array([0]),
      1,
    );
    expect(payload.gpuSourceLayers).toEqual([
      expect.objectContaining({
        layer: 0,
        source,
        encoding: 0,
        mipLevelCount: 1,
        decodeSrgb: false,
      }),
    ]);
    expect(payload.atlasLayers).toEqual([
      expect.objectContaining({
        kind: 'gpu',
        layer: 0,
        width: 2,
        height: 2,
        source,
      }),
    ]);
  });

  it('rejects CPU mirrors that cannot exactly identify the selected GPU subresource', () => {
    const device = {} as GPUDevice;
    const { texture } = makeSourceTexture({ width: 2, height: 2 });
    const base = {
      width: 2,
      height: 2,
      channels: 4 as const,
      dataType: 'uint8' as const,
      colorSpace: 'linear' as const,
      data: new Uint8Array(16),
    };

    expect(() => createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
      cpuMirror: { ...base, width: 1 },
    })).toThrow(/dimensions must exactly match/);
    expect(() => createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
      cpuMirror: base,
    })).toThrow(/colorSpace must match/);
    expect(() => createWalkaroundWebGpuTextureSource(device, texture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
      cpuMirror: { ...base, data: new Uint8Array(15) },
    })).toThrow(/length must be exactly 16/);

    const singleChannel = makeSourceTexture({
      width: 2,
      height: 2,
      format: 'r8unorm',
    }).texture;
    expect(() => createWalkaroundWebGpuTextureSource(device, singleChannel, {
      format: 'r8unorm',
      colorSpace: 'linear',
      cpuMirror: base,
    })).toThrow(/cpuMirror\.channels must match.*r8unorm \(1\)/);
  });

  it('rejects guessed formats, incompatible native-sRGB declarations, and invalid subresources', () => {
    const device = {} as GPUDevice;
    const linear = makeSourceTexture();
    expect(() => createWalkaroundWebGpuTextureSource(device, linear.texture, {
      format: 'rgba16float',
      colorSpace: 'linear',
    })).toThrow(/does not match texture\.format/);

    const nativeSrgb = makeSourceTexture({ format: 'rgba8unorm-srgb' });
    expect(() => createWalkaroundWebGpuTextureSource(device, nativeSrgb.texture, {
      format: 'rgba8unorm-srgb',
      colorSpace: 'linear',
    })).toThrow(/cannot be declared as linear/);

    expect(() => createWalkaroundWebGpuTextureSource(device, linear.texture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
      baseMipLevel: 1,
    })).toThrow(/outside 1 texture mip levels/);
  });

  it('packs mixed CPU and GPU sources without CPU readback and records conversion work', () => {
    const harness = makeUploadHarness();
    const sourceTexture = makeSourceTexture();
    const source = createWalkaroundWebGpuTextureSource(harness.device, sourceTexture.texture, {
      format: 'rgba8unorm',
      colorSpace: 'linear',
    });
    const cpuHandle = {
      width: 2,
      height: 2,
      data: new Uint8Array(2 * 2 * 4).fill(255),
      __vitrum_hint__: { channels: 4 as const, dataType: 'uint8' as const },
    };

    const payload = packMaterialTextureAtlas(
      [materialWithMaps(cpuHandle, source)],
      new Uint32Array([0]),
      1,
    );

    expect(payload.atlasLayers).toEqual([
      expect.objectContaining({
        kind: 'cpu',
        layer: 0,
        width: 2,
        height: 2,
        mipLevelCount: 1,
      }),
      expect.objectContaining({
        kind: 'gpu',
        layer: 1,
        width: 4,
        height: 4,
        mipLevelCount: 1,
      }),
    ]);
    expect(payload.readableBaseColorLayerCount).toBe(1);
    expect(payload.readableNormalLayerCount).toBe(1);
    expect(payload.gpuSourceLayers).toEqual([
      expect.objectContaining({
        layer: 1,
        source,
        encoding: 0,
        mipLevelCount: 1,
        decodeSrgb: false,
      }),
    ]);
  });

  it('rejects raw GPUTexture handles and sRGB sources used for linear-data maps', () => {
    const device = {} as GPUDevice;
    const raw = makeSourceTexture();
    expect(() => packMaterialTextureAtlas(
      [materialWithMaps(raw.texture)],
      new Uint32Array([0]),
      1,
    )).toThrow(/Wrap it with createWalkaroundWebGpuTextureSource/);

    const encoded = createWalkaroundWebGpuTextureSource(device, raw.texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
    });
    expect(() => packMaterialTextureAtlas(
      [materialWithMaps({ width: 1, height: 1, data: new Uint8Array(4) }, encoded)],
      new Uint32Array([0]),
      1,
    )).toThrow(/normalMap is a linear-data map.*declares srgb/);
  });

  it('uploads GPU layers, generates every mip, and preserves host source ownership', () => {
    const harness = makeUploadHarness();
    const sourceTexture = makeSourceTexture();
    const source = createWalkaroundWebGpuTextureSource(harness.device, sourceTexture.texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
    });
    const payload = packMaterialTextureAtlas(
      [{
        ...materialWithMaps(source),
        baseColorMap: { handle: source, mipFilter: 'linear' },
      }],
      new Uint32Array([0]),
      1,
    );

    const uploaded = uploadMaterialTextureAtlas(harness.device, payload);

    expect(harness.destinations).toHaveLength(3);
    expect(harness.destinations[0]?.descriptor).toMatchObject({
      mipLevelCount: 1,
      format: 'r32uint',
    });
    expect(Number(harness.destinations[0]?.descriptor.usage) & 0x08).toBe(0x08);
    expect(harness.destinations[1]?.descriptor).toMatchObject({
      size: { width: 4, height: 4, depthOrArrayLayers: 1 },
      mipLevelCount: 3,
      format: 'r32uint',
    });
    expect(harness.writeTexture).toHaveBeenCalledTimes(1);
    expect(harness.createShaderModule.mock.calls.map((call) => call[0]?.code)).toEqual([
      MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL,
      MATERIAL_ATLAS_GENERATE_MIP_WGSL,
    ]);
    expect(harness.createBindGroupLayout.mock.calls.map((call) => call[0]?.entries)).toEqual([
      [
        expect.objectContaining({
          binding: 0,
          texture: expect.objectContaining({
            sampleType: 'unfilterable-float',
            viewDimension: '2d',
          }),
        }),
        expect.objectContaining({
          binding: 1,
          storageTexture: expect.objectContaining({
            access: 'write-only',
            format: 'r32uint',
            viewDimension: '2d-array',
          }),
        }),
        expect.objectContaining({ binding: 2, buffer: { type: 'uniform' } }),
      ],
      [
        expect.objectContaining({
          binding: 0,
          texture: expect.objectContaining({
            sampleType: 'uint',
            viewDimension: '2d-array',
          }),
        }),
        expect.objectContaining({
          binding: 1,
          storageTexture: expect.objectContaining({
            access: 'write-only',
            format: 'r32uint',
            viewDimension: '2d-array',
          }),
        }),
        expect.objectContaining({ binding: 2, buffer: { type: 'uniform' } }),
      ],
    ]);
    expect(sourceTexture.createView).toHaveBeenCalledWith(expect.objectContaining({
      baseMipLevel: 0,
      baseArrayLayer: 0,
      arrayLayerCount: 1,
    }));
    expect(harness.dispatchWorkgroups.mock.calls).toEqual([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ]);
    expect(harness.submit).toHaveBeenCalledTimes(1);
    expect(harness.copyTextureToTexture).toHaveBeenCalledTimes(3);
    expect(harness.bufferDestroy).toHaveBeenCalledTimes(3);
    expect(harness.destinations[1]?.destroy).toHaveBeenCalledTimes(1);
    expect(sourceTexture.destroy).not.toHaveBeenCalled();
    expect(uploaded.atlasWidth).toBeGreaterThanOrEqual(1);
    expect(uploaded.atlasHeight).toBeGreaterThanOrEqual(1);
    expect(uploaded.atlasArrayLayerCount).toBeGreaterThanOrEqual(1);
  });

  it('re-encodes native-sRGB textureLoad results before packed sRGB storage', () => {
    const harness = makeUploadHarness();
    const sourceTexture = makeSourceTexture({ format: 'rgba8unorm-srgb' });
    const source = createWalkaroundWebGpuTextureSource(
      harness.device,
      sourceTexture.texture,
      { format: 'rgba8unorm-srgb', colorSpace: 'srgb' },
    );
    const payload = packMaterialTextureAtlas(
      [materialWithMaps(source)],
      new Uint32Array([0]),
      1,
    );

    uploadMaterialTextureAtlas(harness.device, payload);

    expect(harness.mappedRanges).toHaveLength(1);
    expect(Array.from(new Uint32Array(harness.mappedRanges[0]!))).toEqual([
      4, 4, 0, 1, 1, 4, 0, 0,
    ]);
    expect(MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL)
      .toContain('if (params.encodeSrgb != 0u)');
    expect(MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL)
      .toContain('value = atlasEncodeSrgb(value);');
  });

  it.each([
    ['r8unorm', 1],
    ['rg8unorm', 2],
  ] as const)(
    'publishes %s channel count and normalizes it through the public raw-map rule',
    (format, expectedChannels) => {
      const harness = makeUploadHarness();
      const sourceTexture = makeSourceTexture({ format });
      const source = createWalkaroundWebGpuTextureSource(
        harness.device,
        sourceTexture.texture,
        { format, colorSpace: 'linear' },
      );
      const payload = packMaterialTextureAtlas(
        [materialWithMaps(source)],
        new Uint32Array([0]),
        1,
      );

      uploadMaterialTextureAtlas(harness.device, payload);

      expect(harness.mappedRanges).toHaveLength(1);
      expect(Array.from(new Uint32Array(harness.mappedRanges[0]!))).toEqual([
        4, 4, 0, 1, 0, expectedChannels, 0, 0,
      ]);
      expect(MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL)
        .toContain('value = vec4f(value.rrr, 1.0);');
      expect(MATERIAL_ATLAS_GPU_SOURCE_CONVERT_WGSL)
        .toContain('value = vec4f(value.rg, 0.0, 1.0);');
    },
  );

  it('reuses one scratch mip chain across five GPU-backed material maps', () => {
    const harness = makeUploadHarness();
    const source = (
      colorSpace: 'srgb' | 'linear',
      cpuMirror = false,
    ) => {
      const texture = makeSourceTexture();
      return createWalkaroundWebGpuTextureSource(
        harness.device,
        texture.texture,
        {
          format: 'rgba8unorm',
          colorSpace,
          ...(cpuMirror
            ? {
                cpuMirror: {
                  width: 4,
                  height: 4,
                  channels: 4 as const,
                  dataType: 'uint8' as const,
                  colorSpace,
                  data: new Uint8Array(4 * 4 * 4).fill(255),
                },
              }
            : {}),
        },
      );
    };
    const material: MaterialSpec = {
      baseColor: [1, 1, 1],
      roughness: 0.5,
      metallic: 0,
      baseColorMap: { handle: source('srgb') },
      emissiveMap: { handle: source('srgb', true) },
      specularColorMap: { handle: source('srgb') },
      sheenColorMap: { handle: source('srgb') },
      normalMap: { handle: source('linear') },
    };
    const payload = packMaterialTextureAtlas(
      [material],
      new Uint32Array([0]),
      1,
    );

    uploadMaterialTextureAtlas(harness.device, payload);

    expect(payload.gpuSourceLayers).toHaveLength(5);
    expect(harness.destinations).toHaveLength(3);
    expect(harness.destinations.filter(({ descriptor }) =>
      descriptor.label === 'vitrum.materialTextureAtlas.gpu-source.scratch',
    )).toHaveLength(1);
    expect(harness.dispatchWorkgroups).toHaveBeenCalledTimes(5);
    expect(harness.copyTextureToTexture).toHaveBeenCalledTimes(5);
  });

  it('fails preflight before destination allocation for device, format, and usage mismatches', () => {
    const sourceHarness = makeUploadHarness();
    const otherHarness = makeUploadHarness();
    const sourceTexture = makeSourceTexture();
    const foreign = createWalkaroundWebGpuTextureSource(sourceHarness.device, sourceTexture.texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
    });
    const foreignPayload = packMaterialTextureAtlas(
      [materialWithMaps(foreign)], new Uint32Array([0]), 1,
    );
    expect(() => uploadMaterialTextureAtlas(otherHarness.device, foreignPayload))
      .toThrow(/different GPUDevice/);
    expect(otherHarness.createTexture).not.toHaveBeenCalled();

    const unsupportedTexture = makeSourceTexture({ format: 'depth24plus' });
    const unsupported = createWalkaroundWebGpuTextureSource(otherHarness.device, unsupportedTexture.texture, {
      format: 'depth24plus',
      colorSpace: 'linear',
    });
    expect(() => packMaterialTextureAtlas(
      [materialWithMaps({ width: 1, height: 1, data: new Uint8Array(4) }, unsupported)],
      new Uint32Array([0]),
      1,
    )).toThrow(/Unsupported material texture GPU format depth24plus/);
    expect(otherHarness.createTexture).not.toHaveBeenCalled();

    const noBindingTexture = makeSourceTexture({ usage: 0 });
    const noBinding = createWalkaroundWebGpuTextureSource(otherHarness.device, noBindingTexture.texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
    });
    const noBindingPayload = packMaterialTextureAtlas(
      [materialWithMaps(noBinding)], new Uint32Array([0]), 1,
    );
    expect(() => uploadMaterialTextureAtlas(otherHarness.device, noBindingPayload))
      .toThrow(/lacks GPUTextureUsage\.TEXTURE_BINDING/);
    expect(otherHarness.createTexture).not.toHaveBeenCalled();
  });

  it('destroys destination resources and transient buffers after compute setup failure', () => {
    const harness = makeUploadHarness(true);
    const sourceTexture = makeSourceTexture();
    const source = createWalkaroundWebGpuTextureSource(harness.device, sourceTexture.texture, {
      format: 'rgba8unorm',
      colorSpace: 'srgb',
    });
    const payload = packMaterialTextureAtlas(
      [materialWithMaps(source)], new Uint32Array([0]), 1,
    );

    expect(() => uploadMaterialTextureAtlas(harness.device, payload))
      .toThrow('compute pass construction failed');
    expect(harness.destinations).toHaveLength(2);
    expect(harness.destinations[0]?.destroy).toHaveBeenCalledTimes(1);
    expect(harness.destinations[1]?.destroy).toHaveBeenCalledTimes(1);
    expect(harness.createBuffer).toHaveBeenCalledTimes(1);
    expect(harness.bufferDestroy).toHaveBeenCalledTimes(1);
    expect(sourceTexture.destroy).not.toHaveBeenCalled();
    expect(harness.submit).not.toHaveBeenCalled();
  });
});
