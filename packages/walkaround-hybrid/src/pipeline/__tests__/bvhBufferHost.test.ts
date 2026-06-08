import { describe, expect, it, vi } from 'vitest';
import type { SceneBVHBuffers } from '../../restir/bvhTypes.js';

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

import { BvhBufferHost } from '../BvhBufferHost.js';

function mockDevice(): GPUDevice {
  return {
    queue: { writeBuffer: vi.fn() },
  } as unknown as GPUDevice;
}

describe('BvhBufferHost', () => {
  it('uploadInitial exposes scene bind-group resources', () => {
    const host = new BvhBufferHost();
    const device = mockDevice();
    const cpu = new ArrayBuffer(64);
    const buf = { cpuData: cpu, byteLength: 64, count: 1 };
    host.uploadInitial(device, {
      bvhNodes: buf,
      bvhIndex: buf,
      bvhBeerColors: buf,
      bvhEmissiveLe: buf,
      bvhNormals: buf,
      bvhPositions: buf,
      emitters: buf,
      emitterCdf: buf,
      emitterCount: 0,
      totalEmissivePower: 0,
      lightTree: buf,
      lightTreeNodeCount: 0,
      lightTreeEnabled: false,
      bvhMode: 'merged',
    } as SceneBVHBuffers);
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

  it('sceneBindGroupResources throws before upload', () => {
    const host = new BvhBufferHost();
    expect(() => host.sceneBindGroupResources()).toThrow(/uploadInitial/);
  });
});
