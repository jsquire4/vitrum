import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';

installWebGPUPolyfills();

import { getNrcHybridLayersBindGroupLayout } from '../../neural/nrc/nrcBindGroupLayout.js';
import { OptionalSubsystemBindingState } from '../OptionalSubsystemBindingState.js';
import {
  getHybridLayersBindGroupLayout,
  getShadeHybridLayersBindGroupLayout,
  getTransparentOitBindGroupLayout,
} from '../bindGroupLayouts.js';
import { createDdgiFrameResources } from '../frameResources/createDdgiFrameResources.js';
import type { FrameResources } from '../resourceManager.js';

function makeTexture(): GPUTexture {
  return {
    createView: vi.fn(() => ({} as GPUTextureView)),
  } as unknown as GPUTexture;
}

function makeBuffer(desc: GPUBufferDescriptor): GPUBuffer {
  const mapped = new ArrayBuffer(desc.size);
  return {
    size: desc.size,
    usage: desc.usage,
    destroy: vi.fn(),
    getMappedRange: vi.fn(() => mapped),
    unmap: vi.fn(),
  } as unknown as GPUBuffer;
}

describe('DDGI receiver exact-load atlas ABI', () => {
  it('does not allocate a sampler with the DDGI resources', () => {
    const device = {
      createTexture: vi.fn(() => makeTexture()),
      createSampler: vi.fn(),
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => makeBuffer(desc)),
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;

    const resources = createDdgiFrameResources(device);

    expect(resources).not.toHaveProperty('ddgiSampler');
    expect(device.createSampler).not.toHaveBeenCalled();
  });

  it('declares the two DDGI textures with no sampler slot in every receiver layout', () => {
    const layouts: GPUBindGroupLayoutDescriptor[] = [];
    const device = {
      createBindGroupLayout: vi.fn((descriptor: GPUBindGroupLayoutDescriptor) => {
        layouts.push(descriptor);
        return {} as GPUBindGroupLayout;
      }),
    } as unknown as GPUDevice;

    getHybridLayersBindGroupLayout(device, {});
    getShadeHybridLayersBindGroupLayout(device, {});
    getNrcHybridLayersBindGroupLayout(device, {});
    getTransparentOitBindGroupLayout(device, {});

    expect(layouts).toHaveLength(4);
    for (const layout of layouts) {
      const entries = [...layout.entries];
      expect(entries.find((entry) => entry.binding === 0)?.texture?.sampleType)
        .toBe('float');
      expect(entries.find((entry) => entry.binding === 1)?.texture?.sampleType)
        .toBe('float');
      expect(entries.some((entry) => entry.binding === 2)).toBe(false);
    }
  });

  it('builds receiver groups without routing the common nearest sampler into DDGI', () => {
    const nearestSampler = { label: 'common-nearest' } as GPUSampler;
    const bindGroups: GPUBindGroupDescriptor[] = [];
    const device = {
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => makeBuffer(desc)),
      createBindGroupLayout: vi.fn(() => ({} as GPUBindGroupLayout)),
      createBindGroup: vi.fn((descriptor: GPUBindGroupDescriptor) => {
        bindGroups.push(descriptor);
        return {} as GPUBindGroup;
      }),
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;
    const frameResources = {
      common: { nearestSampler },
      ddgi: {
        ddgiPlaceholderRgba16f: makeTexture(),
        ddgiPlaceholderVisRgba16f: makeTexture(),
        ddgiUboBuffer: makeBuffer({
          size: 64,
          usage: GPUBufferUsage.UNIFORM,
        }),
      },
      ppg: {},
    } as unknown as FrameResources;
    const state = new OptionalSubsystemBindingState(device);

    state.buildBindGroup(device, {}, frameResources);
    state.buildShadeBindGroup(device, {}, frameResources);
    state.buildTransparentOitBindGroup(
      device,
      {},
      frameResources,
      makeTexture(),
      makeTexture(),
    );

    const receiverGroups = bindGroups.filter((group) =>
      group.label === 'hybrid-layers-bg' ||
      group.label === 'shade-hybrid-layers-bg' ||
      group.label === 'transparent-oit-bg'
    );
    expect(receiverGroups).toHaveLength(3);
    for (const group of receiverGroups) {
      expect([...group.entries].some((entry) => entry.binding === 2)).toBe(false);
      expect([...group.entries].some((entry) => entry.resource === nearestSampler)).toBe(false);
    }
    expect(frameResources.common.nearestSampler).toBe(nearestSampler);
  });
});
