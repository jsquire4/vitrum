import { describe, expect, it, vi } from 'vitest';
import type { SceneBVHBuffers } from '../../restir/bvhTypes.js';
import { EMITTER_TRI_STRIDE_BYTES } from '../../restir/emitterList.js';

vi.mock('../resourceManager.js', () => ({
  uploadBuffer: vi.fn((_device, data: ArrayBuffer, usage: number) => ({
    size: data.byteLength,
    usage,
    destroy: vi.fn(),
  })),
  uploadBufferPadded: vi.fn((_device, data: ArrayBuffer, extraBytes: number, usage: number) => ({
    size: data.byteLength + extraBytes,
    usage,
    destroy: vi.fn(),
  })),
  createDummyStorageBuffer: vi.fn(() => ({
    size: 16,
    usage: 0x80,
    destroy: vi.fn(),
  })),
}));

// WS1 — beer is a texture now; mock its host helper so the test stays
// device-free (createTexture/writeTexture aren't on the mock device).
vi.mock('../bvhBeerTexture.js', () => ({
  uploadBeerTexture: vi.fn(() => ({
    texture: { createView: vi.fn(() => ({})), destroy: vi.fn() },
    width: 4096,
    height: 1,
  })),
  refreshBeerTexture: vi.fn(),
}));

// Camera-visible emitters — emissive Le is also a texture; mock its host helper
// for the same device-free reason as beer.
vi.mock('../bvhEmissiveTexture.js', () => ({
  uploadEmissiveTexture: vi.fn(() => ({
    texture: { createView: vi.fn(() => ({})), destroy: vi.fn() },
    width: 4096,
    height: 1,
  })),
  refreshEmissiveTexture: vi.fn(),
}));

vi.mock('../analyticLightsTexture.js', () => ({
  uploadAnalyticLightsTexture: vi.fn(() => ({
    texture: { createView: vi.fn(() => ({})), destroy: vi.fn() },
    width: 4,
    height: 1,
  })),
}));

// B3 — directional IBL env resources create GPU textures/sampler/uniform; mock
// the host helper so the test stays device-free (same pattern as beer/emissive).
const mockEnv = () => ({
  map: { createView: vi.fn(() => ({})), destroy: vi.fn() },
  marginal: { createView: vi.fn(() => ({})), destroy: vi.fn() },
  conditional: { createView: vi.fn(() => ({})), destroy: vi.fn() },
  sampler: {},
  paramsBuffer: { destroy: vi.fn() },
});
vi.mock('../environmentTexture.js', () => ({
  createPlaceholderEnvironment: vi.fn(() => mockEnv()),
  uploadEnvironment: vi.fn(() => mockEnv()),
  clearEnvironment: vi.fn(() => mockEnv()),
  disposeEnvironment: vi.fn(),
}));

import { BvhBufferHost } from '../BvhBufferHost.js';

function mockDevice(): GPUDevice {
  return {
    queue: { writeBuffer: vi.fn() },
  } as unknown as GPUDevice;
}

function storageBuffer(byteLength: number, count = 1) {
  const cpuData = new ArrayBuffer(byteLength);
  return { cpuData, byteLength, count };
}

function makeSceneBvhBuffers(emitterCount = 1): SceneBVHBuffers {
  const buf = storageBuffer(64, 1);
  const emitters = storageBuffer(EMITTER_TRI_STRIDE_BYTES * emitterCount, emitterCount);
  const emitterCdf = storageBuffer(4 * emitterCount, emitterCount);
  return {
    bvhNodes: buf,
    bvhIndex: buf,
    bvhBeerColors: buf,
    bvhEmissiveLe: buf,
    bvhRoughMetal: buf,
    bvhNormals: buf,
    bvhPositions: buf,
    emitters,
    emitterCdf,
    emitterCount,
    totalEmissivePower: 0,
    lightTree: buf,
    lightTreeNodeCount: 0,
    lightTreeEnabled: false,
    bvhMode: 'merged',
  } as SceneBVHBuffers;
}

describe('BvhBufferHost', () => {
  it('uploadInitial exposes scene bind-group resources', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    host.uploadInitial(device, makeSceneBvhBuffers());
    const r = host.sceneBindGroupResources();
    const r2 = host.sceneBindGroupResources();
    expect(r.bvhNodesBuffer).toBeDefined();
    expect(r.tlasNodesBuffer).toBeDefined();
    expect(r2.bvhBeerTextureView).toBe(r.bvhBeerTextureView);
    expect(r2.bvhEmissiveTextureView).toBe(r.bvhEmissiveTextureView);
    expect(host.lightTreeBuffer()).toBeDefined();
    const mem = host.gpuMemorySections().staticScene;
    if (mem == null) throw new Error('expected staticScene memory section');
    expect(mem['bvhNodesBuffer']).toMatchObject({ size: 64, usage: 0x80 });
    expect(mem['lightTreeBuffer']).toMatchObject({ size: 64, usage: 0x80 });
    expect(mem['tlasNodesBuffer']).toMatchObject({ size: 16, usage: 0x80 });
    expect(mem['bvhBeerTexture']).toMatchObject({ width: 4096, height: 1, format: 'r32uint' });
    expect(mem['bvhEmissiveTexture']).toMatchObject({ width: 4096, height: 1, format: 'rgba32float' });
    host.dispose();
  });

  it('updateEmitters uses the canonical emitter payload count', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    host.uploadInitial(device, makeSceneBvhBuffers(1));

    const next = makeSceneBvhBuffers(2);
    host.updateEmitters(device, {
      emitters: next.emitters,
      emitterCdf: next.emitterCdf,
      lightTree: next.lightTree,
    });

    expect(host.emitterBufferAndCount()?.count).toBe(2);
    host.dispose();
  });

  it('updateEmitters rejects malformed emitter byte lengths before replacing live buffers', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    host.uploadInitial(device, makeSceneBvhBuffers(1));

    const next = makeSceneBvhBuffers(1);
    const malformed = {
      cpuData: new ArrayBuffer(EMITTER_TRI_STRIDE_BYTES + 4),
      byteLength: EMITTER_TRI_STRIDE_BYTES + 4,
      count: 1,
    };

    expect(() => host.updateEmitters(device, {
      emitters: malformed,
      emitterCdf: next.emitterCdf,
      lightTree: next.lightTree,
    })).toThrow(/not aligned to the 80-byte EmitterTri stride/);
    expect(host.emitterBufferAndCount()?.count).toBe(1);

    host.dispose();
  });

  it('sceneBindGroupResources throws before upload', () => {
    const host = new BvhBufferHost();
    expect(() => host.sceneBindGroupResources()).toThrow(/uploadInitial/);
  });
});
