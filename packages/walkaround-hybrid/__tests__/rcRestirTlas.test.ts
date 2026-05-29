import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';

installWebGPUPolyfills();
import { makeRestirBvhSnapshot } from '../src/restir/restirBvhSnapshot.js';
import type { SceneBVHBuffers } from '../src/restir/bvhCompute.js';
import * as THREE from 'three';

function tlasBuffers(): SceneBVHBuffers {
  const geo = new THREE.BufferGeometry();
  geo.boundingBox = new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 1, 1));
  return {
    bvhMode: 'tlas',
    bvhNodes: { cpuData: new ArrayBuffer(32), byteLength: 32, count: 1 },
    bvhIndex: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    bvhPositions: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    triangleMaterialIds: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
    bvhBeerColors: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
    bvhNormals: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    emitters: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 0 },
    emitterCdf: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 0 },
    emitterCount: 0,
    totalEmissivePower: 0,
    lightTree: { cpuData: new ArrayBuffer(48), byteLength: 48, count: 1 },
    lightTreeNodeCount: 0,
    lightTreeEnabled: false,
    mergedGeometry: geo,
    meshVertexRanges: [],
    bvhIndicesStride3: new Uint32Array(3),
    buildMaterials: [new THREE.MeshStandardMaterial()],
    emitterNormals: new Float32Array(12),
    primitiveTlasBindings: [],
    tlas: {
      nodes: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 2 },
      instanceIndices: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
      blasRoots: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
      worldToLocal: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
      localToWorld: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 1 },
      nodeCount: 2,
    },
  };
}

describe('RCSubsystem TLAS sync (C2)', () => {
  it('makeRestirBvhSnapshot exposes TLAS payload for RC upload', () => {
    const snap = makeRestirBvhSnapshot(tlasBuffers());
    expect(snap.bvhMode).toBe('tlas');
    expect(snap.tlasNodeCount).toBe(2);
    expect(snap.tlas?.nodes.byteLength).toBe(64);
  });

  it('syncRestirBvhBuffers skips merged mode', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const device = {
      createBuffer: vi.fn(() => ({
        getMappedRange: () => new ArrayBuffer(64),
        unmap: vi.fn(),
        destroy: vi.fn(),
      })),
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;
    const rc = new RCSubsystem(device);
    rc.syncRestirBvhBuffers(null);
    expect(device.createBuffer).not.toHaveBeenCalled();
  });

  it('TLAS-only version bump refits GPU buffers without recreating dispatcher', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const writeBuffer = vi.fn();
    const createBuffer = vi.fn(() => ({
      getMappedRange: () => new ArrayBuffer(256),
      unmap: vi.fn(),
      destroy: vi.fn(),
    }));
    const device = {
      createBuffer,
      queue: { writeBuffer },
    } as unknown as GPUDevice;

    const base = tlasBuffers();
    const rc = new RCSubsystem(device);
    rc.syncRestirBvhBuffers(base as SceneBVHBuffers);
    const createCallsAfterFirst = createBuffer.mock.calls.length;
    expect(createCallsAfterFirst).toBeGreaterThan(0);

    const snap = makeRestirBvhSnapshot(base as SceneBVHBuffers);
    const tlasNodes = new Uint32Array(snap.tlas!.nodes);
    if (tlasNodes.length > 0) tlasNodes[0]! ^= 0x1;
    const bumped: SceneBVHBuffers = {
      ...base,
      tlas: {
        ...base.tlas!,
        nodes: {
          cpuData: tlasNodes.buffer,
          byteLength: tlasNodes.byteLength,
          count: base.tlas!.nodeCount,
        },
      },
    };

    rc.syncRestirBvhBuffers(bumped);
    expect(createBuffer.mock.calls.length).toBe(createCallsAfterFirst);
    expect(writeBuffer).toHaveBeenCalled();
  });
});
