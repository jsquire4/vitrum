import { describe, expect, it, vi } from 'vitest';
import type { SceneBVHBuffers } from '../../restir/bvhCompute.js';

vi.mock('../resourceManager.js', () => ({
  uploadBuffer: vi.fn(() => ({ destroy: vi.fn() })),
  createDummyStorageBuffer: vi.fn(() => ({ destroy: vi.fn() })),
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
    expect(r.bvhNodesBuffer).toBeDefined();
    expect(r.tlasNodesBuffer).toBeDefined();
    expect(host.lightTreeBuffer()).toBeDefined();
    host.dispose();
  });

  it('sceneBindGroupResources throws before upload', () => {
    const host = new BvhBufferHost();
    expect(() => host.sceneBindGroupResources()).toThrow(/uploadInitial/);
  });
});
