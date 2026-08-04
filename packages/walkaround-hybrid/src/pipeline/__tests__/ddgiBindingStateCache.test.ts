import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';

installWebGPUPolyfills();

import { OptionalSubsystemBindingState } from '../OptionalSubsystemBindingState.js';
import { PipelineResourceCache } from '../PipelineResourceCache.js';
import type { FrameResources } from '../resourceManager.js';

type StubBuffer = {
  label?: string;
  size: number;
  usage: number;
  destroy: () => void;
  getMappedRange: () => ArrayBuffer;
  unmap: () => void;
};

function makeStubBuffer(desc: GPUBufferDescriptor): StubBuffer {
  const mapped = new ArrayBuffer(desc.size);
  return {
    ...(desc.label !== undefined && { label: desc.label }),
    size: desc.size,
    usage: desc.usage,
    destroy: vi.fn(),
    getMappedRange: vi.fn(() => mapped),
    unmap: vi.fn(),
  };
}

function makeTexture(): GPUTexture {
  return { createView: vi.fn(() => ({} as GPUTextureView)) } as unknown as GPUTexture;
}

function makeDevice(): GPUDevice {
  return {
    createBuffer: vi.fn((desc: GPUBufferDescriptor) => makeStubBuffer(desc)),
    createBindGroupLayout: vi.fn(() => ({} as GPUBindGroupLayout)),
    createBindGroup: vi.fn(() => ({} as GPUBindGroup)),
    queue: { writeBuffer: vi.fn() },
  } as unknown as GPUDevice;
}

function makeFrameResources(): FrameResources {
  return {
    common: {
      nearestSampler: {} as GPUSampler,
    },
    ddgi: {
      ddgiPlaceholderRgba16f: makeTexture(),
      ddgiPlaceholderVisRgba16f: makeTexture(),
      ddgiUboBuffer: { size: 64, usage: GPUBufferUsage.UNIFORM } as GPUBuffer,
    },
    ppg: {},
  } as unknown as FrameResources;
}

describe('OptionalSubsystemBindingState resource cache and memory sections', () => {
  it.each([0, 63, 65])(
    'rejects a %i-byte DDGI grid UBO before changing texture identities',
    (byteLength) => {
      const device = makeDevice();
      const state = new OptionalSubsystemBindingState(device);
      const frameResources = makeFrameResources();
      const firstIrr = makeTexture();
      const firstVis = makeTexture();
      state.setInputs({
        irradianceTex: firstIrr,
        visibilityTex: firstVis,
        gridParams: new ArrayBuffer(64),
      }, frameResources);
      const internal = state as unknown as {
        _irrTex: GPUTexture | null;
        _visTex: GPUTexture | null;
      };

      expect(() => state.setInputs({
        irradianceTex: makeTexture(),
        visibilityTex: makeTexture(),
        gridParams: new ArrayBuffer(byteLength),
      }, frameResources)).toThrow('gridParams must be exactly 64 bytes');
      expect(internal._irrTex).toBe(firstIrr);
      expect(internal._visTex).toBe(firstVis);
      expect(device.queue.writeBuffer).toHaveBeenCalledTimes(1);
    },
  );

  it('keeps the accepted DDGI texture generation when a UBO write fails', () => {
    const device = makeDevice();
    const state = new OptionalSubsystemBindingState(device);
    const frameResources = makeFrameResources();
    const firstIrr = makeTexture();
    const firstVis = makeTexture();
    state.setInputs({
      irradianceTex: firstIrr,
      visibilityTex: firstVis,
      gridParams: new ArrayBuffer(64),
    }, frameResources);
    const internal = state as unknown as {
      _irrTex: GPUTexture | null;
      _visTex: GPUTexture | null;
    };
    vi.mocked(device.queue.writeBuffer).mockImplementationOnce(() => {
      throw new Error('queue write failed');
    });

    expect(() => state.setInputs({
      irradianceTex: makeTexture(),
      visibilityTex: makeTexture(),
      gridParams: new ArrayBuffer(64),
    }, frameResources)).toThrow('queue write failed');
    expect(internal._irrTex).toBe(firstIrr);
    expect(internal._visTex).toBe(firstVis);

    vi.mocked(device.queue.writeBuffer).mockImplementationOnce(() => {
      throw new Error('placeholder write failed');
    });
    expect(() => state.setInputs(null, frameResources))
      .toThrow('placeholder write failed');
    expect(internal._irrTex).toBe(firstIrr);
    expect(internal._visTex).toBe(firstVis);
  });

  it('reuses the hybrid-layers bind group while resource identities are unchanged', () => {
    const device = makeDevice();
    const state = new OptionalSubsystemBindingState(device);
    const frameResources = makeFrameResources();
    const resourceCache = new PipelineResourceCache();
    const bglCache = {};

    const first = state.buildBindGroup(device, bglCache, frameResources, resourceCache);
    const second = state.buildBindGroup(device, bglCache, frameResources, resourceCache);

    expect(second).toBe(first);
    expect(device.createBindGroup).toHaveBeenCalledTimes(1);
    expect(frameResources.ddgi.ddgiPlaceholderRgba16f.createView).toHaveBeenCalledTimes(1);
    expect(frameResources.ddgi.ddgiPlaceholderVisRgba16f.createView).toHaveBeenCalledTimes(1);
  });

  it('reports binding-state-owned placeholder and RC params buffers for GPU memory estimates', () => {
    const device = makeDevice();
    const state = new OptionalSubsystemBindingState(device);
    const externalCascade0 = { size: 1024, usage: GPUBufferUsage.STORAGE } as GPUBuffer;

    state.setRCInputs({ cascade0Buffer: externalCascade0, paramsBytes: new ArrayBuffer(64) });
    state.buildBindGroup(device, {}, makeFrameResources(), new PipelineResourceCache());

    const section = state.gpuMemorySections().hybridBindingState;
    if (section == null) throw new Error('expected hybridBindingState memory section');
    expect(section.rcCascade0Placeholder).toMatchObject({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    expect(section.rcParamsPlaceholder).toMatchObject({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    expect(section.rcParamsBuffer).toMatchObject({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    expect(section.ppgTreePlaceholder).toMatchObject({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    expect(section).not.toHaveProperty('rcCascade0Buffer');
  });
});
