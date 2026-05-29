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
  const fakeBuffer = { destroy: vi.fn() } as unknown as GPUBuffer;
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
    createBuffer: vi.fn(() => fakeBuffer),
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
} {
  const updatePrimitive = vi.fn();
  const applyGpuSkinnedRefit = vi.fn();
  const host: GpuSkinningHost = {
    getGpuSkinningBvhBuffer: () => ({ destroy: vi.fn() }) as unknown as GPUBuffer,
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
    applyGpuSkinnedRefit,
  };
  return { host, updatePrimitive, applyGpuSkinnedRefit };
}

describe('GpuSkinningSubsystem — bindMatrix routing', () => {
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
});
