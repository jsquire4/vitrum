/**
 * GpuSkinningSubsystem — bindMatrix routing.
 *
 * The GPU skin kernel applies only `combineSkinMatrices(bones, boneInverses)`
 * and does NOT wrap by `bindMatrix` / `bindMatrixInverse` the way the CPU
 * `solveSkin` does. So a skinned mesh with a NON-identity bindMatrix would skin
 * WRONG on the GPU path. These tests pin that:
 *   - a non-identity-bind, morph-free mesh falls back to the CPU solver
 *     (observed via `host.updatePrimitive` receiving solved positions/normals,
 *     and `host.applyGpuSkinnedRefit` NOT being called for it),
 *   - an identity-bind (or absent-bind) mesh still takes the GPU fast path
 *     (observed via `host.applyGpuSkinnedRefit`, and NOT `updatePrimitive`).
 */

import { describe, it, expect, vi } from 'vitest';
import type { SkinnedMeshPrimitive, Scene, ScenePrimitive } from '@vitrum/core';
import { GpuSkinningSubsystem } from '../src/skin/GpuSkinningSubsystem.js';
import type { GpuSkinningHost } from '../src/skin/GpuSkinningSubsystem.js';

// ── Minimal fake GPUDevice that satisfies every call the GPU skin path makes ──
// (createCommandEncoder, createBuffer, createShaderModule, createComputePipeline,
//  createBindGroup, queue.writeBuffer/submit, and the compute-pass surface).
// It does no real work — its only job is to let the GPU path run to completion
// so the routing decision is observable via the host callbacks.
function makeFakeDevice(): GPUDevice {
  const fakePass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
    end: vi.fn(),
  };
  const fakeEncoder = {
    beginComputePass: vi.fn(() => fakePass),
    finish: vi.fn(() => ({}) as GPUCommandBuffer),
  };
  const fakePipeline = {
    getBindGroupLayout: vi.fn(() => ({}) as GPUBindGroupLayout),
  };
  return {
    createCommandEncoder: vi.fn(() => fakeEncoder),
    createBuffer: vi.fn(() => ({ destroy: vi.fn() }) as unknown as GPUBuffer),
    createShaderModule: vi.fn(() => ({}) as GPUShaderModule),
    createComputePipeline: vi.fn(() => fakePipeline),
    createBindGroup: vi.fn(() => ({}) as GPUBindGroup),
    queue: {
      writeBuffer: vi.fn(),
      submit: vi.fn(),
    },
  } as unknown as GPUDevice;
}

// ── Skinned-mesh fixture builder ──────────────────────────────────────────────
// A single-bone, 3-vertex mesh. Each vertex is fully weighted to bone 0.
function makeSkinnedMesh(
  id: string,
  bind?: { bindMatrix: Float32Array; bindMatrixInverse: Float32Array },
): SkinnedMeshPrimitive {
  const vertCount = 3;
  // Bone 0 = identity bone, identity inverse-bind (so the rest pose is the
  // deformed pose — keeps the CPU solver math trivial / finite).
  const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  return {
    kind: 'skinned-mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    skinIndices: new Uint32Array(vertCount * 4), // all bone 0
    skinWeights: new Float32Array([
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
    ]),
    bones: new Float32Array(identity),
    boneInverses: new Float32Array(identity),
    material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    ...(bind ?? {}),
  };
}

function sceneOf(...primitives: ScenePrimitive[]): Scene {
  return { primitives, emitters: [], environment: { kind: 'none' } };
}

// ── Host stub: provides the merged-BVH buffer + a matching vertex range so the
// GPU path is otherwise eligible; records which callback each primitive hit. ──
function makeHost(meshIds: string[]): {
  host: GpuSkinningHost;
  updatePrimitive: ReturnType<typeof vi.fn>;
  applyGpuSkinnedRefit: ReturnType<typeof vi.fn>;
  applySkinningBatch: ReturnType<typeof vi.fn>;
} {
  const updatePrimitive = vi.fn();
  const applyGpuSkinnedRefit = vi.fn();
  const applySkinningBatch = vi.fn(
    (updates: Parameters<GpuSkinningHost['applySkinningBatch']>[0]) => {
      for (const update of updates) {
        if (update.gpuWritten) applyGpuSkinnedRefit(update.id);
        else updatePrimitive(update.id, update.patch);
      }
    },
  );
  const host: GpuSkinningHost = {
    getGpuSkinningBvhBuffer: () => ({ destroy: vi.fn() }) as unknown as GPUBuffer,
    getGpuSkinningNormalBuffer: () => ({ destroy: vi.fn() }) as unknown as GPUBuffer,
    getMeshVertexRanges: () =>
      meshIds.map((name) => ({
        name,
        vertexStart: 0,
        vertexCount: 3,
        triStart: 0,
        triCount: 1,
        matrixWorldAtBuild: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      })),
    getBvhMode: () => 'merged',
    getPrimitiveTlasBindings: () => null,
    updatePrimitive,
    applySkinningBatch,
    applyGpuSkinnedRefit,
  };
  return {
    host,
    updatePrimitive,
    applyGpuSkinnedRefit,
    applySkinningBatch: applySkinningBatch as ReturnType<typeof vi.fn>,
  };
}

describe('GpuSkinningSubsystem — bindMatrix routing', () => {
  it.each([
    ['position buffer', { getGpuSkinningBvhBuffer: () => null }],
    ['normal buffer', { getGpuSkinningNormalBuffer: () => null }],
    ['mesh vertex ranges', { getMeshVertexRanges: () => null }],
  ] satisfies ReadonlyArray<readonly [string, Partial<GpuSkinningHost>]>)(
    'falls back to CPU skinning when the host %s is unavailable',
    (_label, overrides) => {
      const first = makeSkinnedMesh('skin-a');
      const second = makeSkinnedMesh('skin-b');
      const { host, updatePrimitive, applyGpuSkinnedRefit } = makeHost(['skin-a', 'skin-b']);
      const device = makeFakeDevice();
      const patchedHost = { ...host, ...overrides } as GpuSkinningHost;

      new GpuSkinningSubsystem(device, /* preferGpu */ true).run(patchedHost, sceneOf(first, second));

      expect(updatePrimitive).toHaveBeenCalledTimes(2);
      expect(updatePrimitive).toHaveBeenNthCalledWith(
        1,
        'skin-a',
        expect.objectContaining({
          positions: expect.any(Float32Array),
          normals: expect.any(Float32Array),
        }),
      );
      expect(updatePrimitive).toHaveBeenNthCalledWith(
        2,
        'skin-b',
        expect.objectContaining({
          positions: expect.any(Float32Array),
          normals: expect.any(Float32Array),
        }),
      );
      expect(applyGpuSkinnedRefit).not.toHaveBeenCalled();
      expect((device.createComputePipeline as ReturnType<typeof vi.fn>)).not.toHaveBeenCalled();
      expect((device.queue.submit as ReturnType<typeof vi.fn>)).not.toHaveBeenCalled();
    },
  );

  it('routes a NON-identity-bind, morph-free mesh to the CPU solver', () => {
    const id = 'bound-mesh';
    // bindMatrix with a translation → non-identity. bindMatrixInverse is its
    // inverse (negated translation) so solveSkin's bind round-trip is finite.
    const bind = {
      bindMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1]),
      bindMatrixInverse: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -5, 0, 0, 1]),
    };
    const prim = makeSkinnedMesh(id, bind);
    const { host, updatePrimitive, applyGpuSkinnedRefit } = makeHost([id]);
    const device = makeFakeDevice();

    new GpuSkinningSubsystem(device, /* preferGpu */ true).run(host, sceneOf(prim));

    // CPU path: updatePrimitive got the solved positions/normals…
    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(updatePrimitive).toHaveBeenCalledWith(
      id,
      expect.objectContaining({
        positions: expect.any(Float32Array),
        normals: expect.any(Float32Array),
      }),
    );
    // …and the GPU refit was NOT taken for it.
    expect(applyGpuSkinnedRefit).not.toHaveBeenCalled();
    // The GPU compute pipeline was never even built (no GPU dispatch occurred).
    expect((device.createComputePipeline as ReturnType<typeof vi.fn>)).not.toHaveBeenCalled();
  });

  it('keeps an IDENTITY-bind mesh on the GPU fast path', () => {
    const id = 'identity-bound-mesh';
    const bind = {
      bindMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      bindMatrixInverse: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    };
    const prim = makeSkinnedMesh(id, bind);
    const { host, updatePrimitive, applyGpuSkinnedRefit } = makeHost([id]);
    const device = makeFakeDevice();

    new GpuSkinningSubsystem(device, /* preferGpu */ true).run(host, sceneOf(prim));

    expect(applyGpuSkinnedRefit).toHaveBeenCalledTimes(1);
    expect(applyGpuSkinnedRefit).toHaveBeenCalledWith(id);
    expect(updatePrimitive).not.toHaveBeenCalled();
  });

  it('keeps an ABSENT-bind (glTF-typical) mesh on the GPU fast path', () => {
    const id = 'no-bind-mesh';
    const prim = makeSkinnedMesh(id); // no bindMatrix at all
    const { host, updatePrimitive, applyGpuSkinnedRefit } = makeHost([id]);
    const device = makeFakeDevice();

    new GpuSkinningSubsystem(device, /* preferGpu */ true).run(host, sceneOf(prim));

    expect(applyGpuSkinnedRefit).toHaveBeenCalledTimes(1);
    expect(applyGpuSkinnedRefit).toHaveBeenCalledWith(id);
    expect(updatePrimitive).not.toHaveBeenCalled();
  });

  it('routes only the bound mesh to CPU when both kinds are present', () => {
    const boundId = 'bound';
    const freeId = 'free';
    const bind = {
      bindMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1]),
      bindMatrixInverse: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -2, 0, 1]),
    };
    const bound = makeSkinnedMesh(boundId, bind);
    const free = makeSkinnedMesh(freeId); // identity / absent bind
    const { host, updatePrimitive, applyGpuSkinnedRefit } = makeHost([boundId, freeId]);
    const device = makeFakeDevice();

    new GpuSkinningSubsystem(device, /* preferGpu */ true).run(host, sceneOf(bound, free));

    // Bound → CPU.
    expect(updatePrimitive).toHaveBeenCalledTimes(1);
    expect(updatePrimitive).toHaveBeenCalledWith(boundId, expect.any(Object));
    // Free → GPU.
    expect(applyGpuSkinnedRefit).toHaveBeenCalledTimes(1);
    expect(applyGpuSkinnedRefit).toHaveBeenCalledWith(freeId);
  });

  it('publishes two CPU fallbacks plus one GPU job through one unsubmitted batch', () => {
    const nonIdentityBind = {
      bindMatrix: new Float32Array([
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1,
      ]),
      bindMatrixInverse: new Float32Array([
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -2, 0, 0, 1,
      ]),
    };
    const cpuA = makeSkinnedMesh('cpu-a', nonIdentityBind);
    const cpuB = makeSkinnedMesh('cpu-b', nonIdentityBind);
    const gpu = makeSkinnedMesh('gpu');
    const { host, applySkinningBatch } = makeHost(['cpu-a', 'cpu-b', 'gpu']);
    const device = makeFakeDevice();

    new GpuSkinningSubsystem(device, true).run(
      host,
      sceneOf(cpuA, cpuB, gpu),
    );

    expect(applySkinningBatch).toHaveBeenCalledTimes(1);
    const [updates, commandBuffer] = applySkinningBatch.mock.calls[0] as [
      Array<{ id: string; gpuWritten: boolean }>,
      GPUCommandBuffer,
    ];
    expect(updates.map(({ id, gpuWritten }) => ({ id, gpuWritten }))).toEqual([
      { id: 'cpu-a', gpuWritten: false },
      { id: 'cpu-b', gpuWritten: false },
      { id: 'gpu', gpuWritten: true },
    ]);
    expect(commandBuffer).toBeDefined();
    expect(device.queue.submit).not.toHaveBeenCalled();
    expect(device.createCommandEncoder).toHaveBeenCalledTimes(1);
  });
});
