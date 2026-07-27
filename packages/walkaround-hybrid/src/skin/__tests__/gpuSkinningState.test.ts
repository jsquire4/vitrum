import { describe, expect, it, vi } from 'vitest';
import type { Scene, SkinnedMeshPrimitive } from '@vitrum/core';
import { GpuSkinningSubsystem, type GpuSkinningHost } from '../GpuSkinningSubsystem.js';

const IDENTITY = new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
]);
function translated(x: number): Float32Array {
  const matrix = new Float32Array(IDENTITY); matrix[12] = x; return matrix;
}
function mesh(id: string): SkinnedMeshPrimitive {
  return {
    kind: 'skinned-mesh', id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    skinIndices: new Uint32Array(12),
    skinWeights: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
    bones: new Float32Array(IDENTITY), boneInverses: new Float32Array(IDENTITY),
    material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
  };
}
function sceneOf(primitive: SkinnedMeshPrimitive): Scene {
  return { primitives: [primitive], emitters: [], environment: { kind: 'none' } };
}
interface BufferRecord { readonly buffer: GPUBuffer; readonly destroy: ReturnType<typeof vi.fn>; }
function gpuHarness() {
  const buffers: BufferRecord[] = [];
  const bindGroups: GPUBindGroup[] = [];
  const bindGroupDescriptors: GPUBindGroupDescriptor[] = [];
  const dispatchedBindGroups: GPUBindGroup[] = [];
  let injectedBuffer: GPUBuffer | null = null;
  let bindFailure: (() => void) | null = null;
  const createBuffer = vi.fn((descriptor: GPUBufferDescriptor): GPUBuffer => {
    if (injectedBuffer != null) {
      const result = injectedBuffer; injectedBuffer = null; return result;
    }
    const destroy = vi.fn();
    const buffer = { label: descriptor.label, destroy } as unknown as GPUBuffer;
    buffers.push({ buffer, destroy });
    return buffer;
  });
  const createBindGroup = vi.fn((descriptor: GPUBindGroupDescriptor): GPUBindGroup => {
    if (bindFailure != null) {
      const beforeThrow = bindFailure; bindFailure = null; beforeThrow();
      throw new Error('injected bind-group failure');
    }
    const bindGroup = { descriptor } as unknown as GPUBindGroup;
    bindGroups.push(bindGroup); bindGroupDescriptors.push(descriptor); return bindGroup;
  });
  const writeBuffer = vi.fn();
  const pipeline = { getBindGroupLayout: vi.fn(() => ({} as GPUBindGroupLayout)) };
  const device = {
    createBuffer,
    createShaderModule: vi.fn(() => ({} as GPUShaderModule)),
    createComputePipeline: vi.fn(() => pipeline as unknown as GPUComputePipeline),
    createBindGroup,
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(),
        setBindGroup: vi.fn((_index: number, group: GPUBindGroup) => dispatchedBindGroups.push(group)),
        dispatchWorkgroups: vi.fn(), end: vi.fn(),
      })),
      finish: vi.fn(() => ({} as GPUCommandBuffer)),
    })),
    queue: { writeBuffer, submit: vi.fn() },
  } as unknown as GPUDevice;
  return {
    device, buffers, bindGroups, bindGroupDescriptors, dispatchedBindGroups,
    writeBuffer, createBuffer, createBindGroup,
    injectNextBuffer(buffer: GPUBuffer) { injectedBuffer = buffer; },
    failNextBindGroup(beforeThrow: () => void) { bindFailure = beforeThrow; },
  };
}
function hostHarness(id: string) {
  const sharedPositionDestroy = vi.fn();
  const sharedNormalDestroy = vi.fn();
  const positionBuffer = { destroy: sharedPositionDestroy } as unknown as GPUBuffer;
  const normalBuffer = { destroy: sharedNormalDestroy } as unknown as GPUBuffer;
  const applySkinningBatch = vi.fn();
  const range = {
    name: id, vertexStart: 0, vertexCount: 3, triStart: 0, triCount: 1,
    matrixWorldAtBuild: new Float32Array(IDENTITY),
  };
  const host: GpuSkinningHost = {
    getGpuSkinningBvhBuffer: () => positionBuffer,
    getGpuSkinningNormalBuffer: () => normalBuffer,
    getMeshVertexRanges: () => [range], getBvhMode: () => 'merged',
    getPrimitiveTlasBindings: () => null, updatePrimitive: vi.fn(),
    applyGpuSkinnedRefit: vi.fn(), applySkinningBatch,
  };
  return {
    host,
    positionBuffer,
    normalBuffer,
    sharedPositionDestroy,
    sharedNormalDestroy,
    applySkinningBatch,
  };
}

describe('GpuSkinningSubsystem cached mesh state', () => {
  it('routes wider-than-four influence sets through the exact CPU solver', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const base = mesh('skin');
    const twoBones = new Float32Array(32);
    twoBones.set(IDENTITY, 0);
    twoBones.set(translated(2), 16);
    const inverseBones = new Float32Array(32);
    inverseBones.set(IDENTITY, 0);
    inverseBones.set(IDENTITY, 16);
    const wide: SkinnedMeshPrimitive = {
      ...base,
      skinInfluencesPerVertex: 5,
      skinIndices: new Uint32Array([
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
      ]),
      skinWeights: new Float32Array([
        0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5,
      ]),
      bones: twoBones,
      boneInverses: inverseBones,
    };

    subsystem.run(host.host, sceneOf(wide));

    expect(gpu.createBuffer).not.toHaveBeenCalled();
    const [updates, commands] = host.applySkinningBatch.mock.calls[0]!;
    expect(commands).toBeNull();
    expect(updates[0].gpuWritten).toBe(false);
    expect(Array.from(updates[0].patch.positions)).toEqual([
      1, 0, 0, 2, 0, 0, 1, 1, 0,
    ]);
  });

  it('propagates morph-animated arbitrary UV lanes through the CPU batch', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const primitive: SkinnedMeshPrimitive = {
      ...mesh('skin'),
      uvSets: [undefined, undefined, new Float32Array([0, 0, 1, 0, 0, 1])],
      morphTargets: [new Float32Array(9)],
      morphTargetUvSets: [
        undefined,
        undefined,
        [new Float32Array([0.2, 0.4, -0.2, 0.4, 0.2, -0.4])],
      ],
      morphWeights: new Float32Array([0.5]),
    };

    subsystem.run(host.host, sceneOf(primitive));

    expect(gpu.createBuffer).not.toHaveBeenCalled();
    const [updates, commands] = host.applySkinningBatch.mock.calls[0]!;
    expect(commands).toBeNull();
    expect(updates[0].gpuWritten).toBe(false);
    expect(Array.from(updates[0].patch.uvSets[2])).toEqual(
      Array.from(new Float32Array([0.1, 0.2, 0.9, 0.2, 0.1, 0.8])),
    );
  });

  it('preserves the old state across hostile aliases/failure, cleans candidates, and retries', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const original = mesh('skin');
    subsystem.run(host.host, sceneOf(original));
    const old = gpu.buffers.slice(0, 6);
    const edited = {
      ...original,
      positions: new Float32Array([2, 0, 0, 3, 0, 0, 2, 1, 0]),
    };

    gpu.injectNextBuffer(host.positionBuffer);
    expect(() => subsystem.run(host.host, sceneOf(edited))).toThrow(/live\/shared buffer alias/);
    expect(host.sharedPositionDestroy).not.toHaveBeenCalled();
    gpu.injectNextBuffer(old[0]!.buffer);
    expect(() => subsystem.run(host.host, sceneOf(edited))).toThrow(/live\/shared buffer alias/);
    expect(old[0]!.destroy).not.toHaveBeenCalled();

    gpu.failNextBindGroup(() => {
      gpu.buffers[6]!.destroy.mockImplementationOnce(() => { throw new Error('hostile destroy'); });
    });
    expect(() => subsystem.run(host.host, sceneOf(edited))).toThrow(/bind-group failure/);
    const failed = gpu.buffers.slice(6, 12);
    expect(failed).toHaveLength(6);
    for (const candidate of failed) expect(candidate.destroy).toHaveBeenCalledTimes(1);
    for (const live of old) expect(live.destroy).not.toHaveBeenCalled();

    subsystem.run(host.host, sceneOf(original));
    expect(gpu.buffers).toHaveLength(12);
    old[5]!.destroy.mockImplementationOnce(() => { throw new Error('hostile old destroy'); });
    subsystem.run(host.host, sceneOf(edited));
    expect(gpu.buffers).toHaveLength(18);
    for (const retired of old) expect(retired.destroy).toHaveBeenCalledTimes(1);
    for (const accepted of gpu.buffers.slice(12)) expect(accepted.destroy).not.toHaveBeenCalled();
    expect(host.applySkinningBatch).toHaveBeenCalledTimes(3);
  });

  it('rebuilds for a same-count rest edit, then reuses its new rest buffer for two poses', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const original = mesh('skin');
    subsystem.run(host.host, sceneOf(original));
    const positions = new Float32Array([5, 0, 0, 6, 0, 0, 5, 1, 0]);
    const pose1 = { ...original, positions, bones: translated(1) };
    const pose2 = { ...pose1, bones: translated(2) };
    subsystem.run(host.host, sceneOf(pose1));
    subsystem.run(host.host, sceneOf(pose2));

    expect(gpu.createBuffer).toHaveBeenCalledTimes(12);
    expect(gpu.createBindGroup).toHaveBeenCalledTimes(2);
    const newRestBuffer = gpu.buffers[6]!.buffer;
    const restEntry = Array.from(gpu.bindGroupDescriptors[1]!.entries)
      .find((entry: GPUBindGroupEntry) => entry.binding === 1)!;
    expect((restEntry.resource as GPUBufferBinding).buffer).toBe(newRestBuffer);
    expect(gpu.dispatchedBindGroups).toEqual([
      gpu.bindGroups[0], gpu.bindGroups[1], gpu.bindGroups[1],
    ]);
    const writes = gpu.writeBuffer.mock.calls as unknown as Array<
      [GPUBuffer, number, Float32Array]
    >;
    const restWrites = writes.filter(([buffer]) => buffer === newRestBuffer);
    expect(restWrites).toHaveLength(1);
    expect([...restWrites[0]![2]]).toEqual([
      5, 0, 0, 1, 6, 0, 0, 1, 5, 1, 0, 1,
    ]);
  });

  it('rejects a private buffer alias owned by another cached mesh', () => {
    const gpu = gpuHarness();
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const otherHost = hostHarness('other');
    const other = mesh('other');
    subsystem.run(otherHost.host, sceneOf(other));
    const otherPrivateBuffer = gpu.buffers[0]!;

    const host = hostHarness('skin');
    const original = mesh('skin');
    subsystem.run(host.host, sceneOf(original));
    expect(gpu.buffers).toHaveLength(12);
    const edited = {
      ...original,
      positions: new Float32Array([2, 0, 0, 3, 0, 0, 2, 1, 0]),
    };

    gpu.injectNextBuffer(otherPrivateBuffer.buffer);
    expect(() => subsystem.run(host.host, sceneOf(edited)))
      .toThrow(/live\/shared buffer alias/);
    expect(otherPrivateBuffer.destroy).not.toHaveBeenCalled();
    expect(gpu.buffers).toHaveLength(12);

    subsystem.run(host.host, sceneOf(original));
    expect(gpu.buffers).toHaveLength(12);
    expect(host.applySkinningBatch).toHaveBeenCalledTimes(2);
  });
});
