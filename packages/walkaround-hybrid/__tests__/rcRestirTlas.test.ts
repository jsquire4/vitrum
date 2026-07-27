import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';

installWebGPUPolyfills();
import { makeRestirBvhSnapshot } from '../src/restir/restirBvhSnapshot.js';
import type { RestirMergedGeometryLike, SceneBVHBuffers } from '../src/restir/bvhTypes.js';
import { packMaterialTextureAtlas } from '../src/pipeline/materialTextureAtlas.js';

function boundsGeometry(): RestirMergedGeometryLike {
  return {
    boundingBox: {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 1, y: 1, z: 1 },
    },
    computeBoundingBox: () => undefined,
    dispose: () => undefined,
  };
}

function tlasBuffers(): SceneBVHBuffers {
  return {
    bvhMode: 'tlas',
    bvhNodes: { cpuData: new ArrayBuffer(32), byteLength: 32, count: 1 },
    bvhIndex: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    bvhPositions: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    triangleMaterialIds: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
    bvhBeerColors: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
    bvhEmissiveLe: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    materialTextureAtlas: packMaterialTextureAtlas([], new Uint32Array([0]), 1),
    bvhRoughMetal: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 1 },
    bvhNormals: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    bvhTangents: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    bvhColors: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 1 },
    emitters: { cpuData: new ArrayBuffer(64), byteLength: 64, count: 0 },
    emitterCdf: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 0 },
    emitterAlias: { cpuData: new ArrayBuffer(16), byteLength: 16, count: 0 },
    emitterCount: 0,
    totalEmissivePower: 0,
    lightTree: { cpuData: new ArrayBuffer(48), byteLength: 48, count: 1 },
    lightTreeNodeCount: 0,
    lightTreeEnabled: false,
    mergedGeometry: boundsGeometry(),
    meshVertexRanges: [],
    bvhIndicesStride3: new Uint32Array(3),
    buildMaterials: [{ color: { r: 1, g: 1, b: 1 } }],
    coreMaterials: [],
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

  it('borrows main-arena BLAS ranges across initial sync and full BLAS mutation', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    const createBuffer = vi.fn((desc: { label?: string; size?: number }) => ({
      label: desc.label,
      size: desc.size ?? 16,
      usage: 7,
      getMappedRange: () => new ArrayBuffer(Math.max(256, desc.size ?? 16)),
      unmap: vi.fn(),
      destroy: vi.fn(),
    }));
    const device = {
      createBuffer,
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;
    const arena = {
      label: 'main-scene-arena',
      size: 2_048,
      usage: 7,
      destroy: vi.fn(),
    } as unknown as GPUBuffer;
    const sharedGeometry = {
      bvhNodes: { buffer: arena, offset: 0, size: 32 },
      bvhIndices: { buffer: arena, offset: 256, size: 16 },
      bvhPositions: { buffer: arena, offset: 512, size: 16 },
      bvhNormals: { buffer: arena, offset: 768, size: 16 },
    };
    const base = tlasBuffers();
    const rc = new RCSubsystem(device);

    rc.syncRestirBvhBuffers(base, sharedGeometry);
    const internal = rc as unknown as {
      _bvhBuffers: Record<string, GPUBuffer>;
    };
    expect(rc.sharesSceneGeometry).toBe(true);
    expect(internal._bvhBuffers.bvhNodesBuf).toBeUndefined();
    expect(internal._bvhBuffers.bvhIndicesBuf).toBeUndefined();
    expect(internal._bvhBuffers.bvhPositionsBuf).toBeUndefined();
    expect(internal._bvhBuffers.bvhNormalsBuf).toBeUndefined();

    const changedNodes = new Uint32Array(base.bvhNodes.cpuData.slice(0));
    changedNodes[0] = (changedNodes[0] ?? 0) ^ 1;
    const changed: SceneBVHBuffers = {
      ...base,
      bvhNodes: {
        cpuData: changedNodes.buffer,
        byteLength: changedNodes.byteLength,
        count: base.bvhNodes.count,
      },
    };
    const prepared = rc.prepareSceneMutation(changed, undefined, {
      geometryChanged: true,
      refreshMaterials: false,
      allowMergedRefit: false,
    });
    prepared.commit();
    prepared.finalize();

    const labels = createBuffer.mock.calls.map(([desc]) => desc.label ?? '');
    expect(labels).not.toContain('rc-restir-bvh-nodes');
    expect(labels).not.toContain('rc-restir-bvh-index');
    expect(labels).not.toContain('rc-restir-bvh-positions');
    expect(labels).not.toContain('rc-restir-bvh-normals');
    expect(arena.destroy).not.toHaveBeenCalled();
    rc.dispose();
    expect(arena.destroy).not.toHaveBeenCalled();
  });

  it('TLAS-only version bump transactionally replaces RC buffers without discarding the reusable arena', async () => {
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

    const dispatcherBefore = (rc as unknown as { _dispatcher: unknown })._dispatcher;
    const invalidateSpy = vi.spyOn(
      dispatcherBefore as { invalidateBindings: () => void },
      'invalidateBindings',
    );
    rc.syncRestirBvhBuffers(bumped);
    expect(createBuffer.mock.calls.length).toBe(createCallsAfterFirst + 5);
    expect(writeBuffer).not.toHaveBeenCalled();
    expect((rc as unknown as { _dispatcher: unknown })._dispatcher).toBe(dispatcherBefore);
    // The dispatcher refreshes same-sized scene-arena copy sources on its next
    // dispatch. Eager invalidation would unnecessarily discard the arena and
    // pipelines; replacement buffer identity is enough to mark the copies dirty.
    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it('preserves the prior TLAS state when any replacement allocation fails', async () => {
    const { RCSubsystem } = await import('../src/HybridEngineRC.js');
    let failAt = Number.POSITIVE_INFINITY;
    let allocation = 0;
    let created: Array<GPUBuffer & { destroy: ReturnType<typeof vi.fn> }> = [];
    const createBuffer = vi.fn((desc: { size?: number }) => {
      allocation += 1;
      if (allocation === failAt) throw new Error(`rc allocation fault ${failAt}`);
      const mapped = new ArrayBuffer(Math.max(256, desc.size ?? 16));
      const buffer = {
        getMappedRange: () => mapped,
        unmap: vi.fn(),
        destroy: vi.fn(),
      } as unknown as GPUBuffer & { destroy: ReturnType<typeof vi.fn> };
      created.push(buffer);
      return buffer;
    });
    const device = {
      createBuffer,
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;

    const base = tlasBuffers();
    const rc = new RCSubsystem(device);
    rc.syncRestirBvhBuffers(base);

    const internal = rc as unknown as {
      _bvhBuffers: Record<string, GPUBuffer> | null;
      _dispatcher: unknown;
      _restirSnapshot: unknown;
      _lastBvhVersion: number;
    };
    const previousBvh = internal._bvhBuffers;
    const previousDispatcher = internal._dispatcher;
    const previousSnapshot = internal._restirSnapshot;
    const previousVersion = internal._lastBvhVersion;
    const previousEntries = Object.entries(previousBvh ?? {});
    const previousTlasDestroySpies = previousEntries
      .filter(([key]) => key.startsWith('tlas'))
      .map(([, buffer]) => buffer.destroy as ReturnType<typeof vi.fn>);
    const previousBlasDestroySpies = previousEntries
      .filter(([key]) => !key.startsWith('tlas'))
      .map(([, buffer]) => buffer.destroy as ReturnType<typeof vi.fn>);
    const previousDestroySpies = [
      ...previousBlasDestroySpies,
      ...previousTlasDestroySpies,
    ];

    const snap = makeRestirBvhSnapshot(base);
    const tlasNodes = new Uint32Array(snap.tlas!.nodes);
    tlasNodes[0] = (tlasNodes[0] ?? 0) ^ 1;
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

    for (let stage = 1; stage <= 5; stage += 1) {
      allocation = 0;
      failAt = stage;
      created = [];
      expect(() => rc.syncRestirBvhBuffers(bumped))
        .toThrow(`rc allocation fault ${stage}`);
      expect(internal._bvhBuffers).toBe(previousBvh);
      expect(internal._dispatcher).toBe(previousDispatcher);
      expect(internal._restirSnapshot).toBe(previousSnapshot);
      expect(internal._lastBvhVersion).toBe(previousVersion);
      for (const destroy of previousDestroySpies) expect(destroy).not.toHaveBeenCalled();
      for (const buffer of created) expect(buffer.destroy).toHaveBeenCalledTimes(1);
    }

    allocation = 0;
    failAt = Number.POSITIVE_INFINITY;
    created = [];
    rc.syncRestirBvhBuffers(bumped);
    expect(internal._bvhBuffers).not.toBe(previousBvh);
    for (const destroy of previousTlasDestroySpies) expect(destroy).toHaveBeenCalledTimes(1);
    for (const destroy of previousBlasDestroySpies) expect(destroy).not.toHaveBeenCalled();

    const previousFullBvh = internal._bvhBuffers;
    const previousFullSnapshot = internal._restirSnapshot;
    const previousFullVersion = internal._lastBvhVersion;
    const previousFullDestroySpies = Object.values(previousFullBvh ?? {})
      .map((buffer) => buffer.destroy as ReturnType<typeof vi.fn>);
    const blasNodes = new Uint32Array(bumped.bvhNodes.cpuData.slice(0));
    blasNodes[0] = (blasNodes[0] ?? 0) ^ 1;
    const blasBumped: SceneBVHBuffers = {
      ...bumped,
      bvhNodes: {
        cpuData: blasNodes.buffer,
        byteLength: blasNodes.byteLength,
        count: bumped.bvhNodes.count,
      },
    };

    for (let stage = 1; stage <= 11; stage += 1) {
      allocation = 0;
      failAt = stage;
      created = [];
      expect(() => rc.syncRestirBvhBuffers(blasBumped))
        .toThrow(`rc allocation fault ${stage}`);
      expect(internal._bvhBuffers).toBe(previousFullBvh);
      expect(internal._dispatcher).toBe(previousDispatcher);
      expect(internal._restirSnapshot).toBe(previousFullSnapshot);
      expect(internal._lastBvhVersion).toBe(previousFullVersion);
      for (const destroy of previousFullDestroySpies) expect(destroy).not.toHaveBeenCalled();
      for (const buffer of created) expect(buffer.destroy).toHaveBeenCalledTimes(1);
    }

    allocation = 0;
    failAt = Number.POSITIVE_INFINITY;
    created = [];
    rc.syncRestirBvhBuffers(blasBumped);
    expect(internal._bvhBuffers).not.toBe(previousFullBvh);
    for (const destroy of previousFullDestroySpies) expect(destroy).toHaveBeenCalledTimes(1);

    expect(() => rc.dispose()).not.toThrow();
    expect(() => rc.dispose()).not.toThrow();
  });
});
