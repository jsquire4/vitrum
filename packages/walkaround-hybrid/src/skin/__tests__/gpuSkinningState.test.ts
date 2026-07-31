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
  const createCommandEncoder = vi.fn(() => ({
    beginComputePass: vi.fn(() => ({
      setPipeline: vi.fn(),
      setBindGroup: vi.fn((_index: number, group: GPUBindGroup) => dispatchedBindGroups.push(group)),
      dispatchWorkgroups: vi.fn(), end: vi.fn(),
    })),
    finish: vi.fn(() => ({} as GPUCommandBuffer)),
  }));
  const device = {
    createBuffer,
    createShaderModule: vi.fn(() => ({} as GPUShaderModule)),
    createComputePipeline: vi.fn(() => pipeline as unknown as GPUComputePipeline),
    createBindGroup,
    createCommandEncoder,
    queue: { writeBuffer, submit: vi.fn() },
  } as unknown as GPUDevice;
  return {
    device, buffers, bindGroups, bindGroupDescriptors, dispatchedBindGroups,
    writeBuffer, createBuffer, createBindGroup, createCommandEncoder,
    injectNextBuffer(buffer: GPUBuffer) { injectedBuffer = buffer; },
    failNextBindGroup(beforeThrow: () => void) { bindFailure = beforeThrow; },
  };
}
function hostHarness(id: string | readonly string[]) {
  const ids = typeof id === 'string' ? [id] : id;
  const sharedPositionDestroy = vi.fn();
  const sharedNormalDestroy = vi.fn();
  const positionBuffer = { destroy: sharedPositionDestroy } as unknown as GPUBuffer;
  const normalBuffer = { destroy: sharedNormalDestroy } as unknown as GPUBuffer;
  const applySkinningBatch = vi.fn();
  const ranges = ids.map((primitiveId, index) => ({
    name: primitiveId,
    vertexStart: index * 3,
    vertexCount: 3,
    triStart: index,
    triCount: 1,
    matrixWorldAtBuild: new Float32Array(IDENTITY),
  }));
  const host: GpuSkinningHost = {
    getGpuSkinningBvhBuffer: () => positionBuffer,
    getGpuSkinningNormalBuffer: () => normalBuffer,
    getMeshVertexRanges: () => ranges, getBvhMode: () => 'merged',
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
  it('does not solve, encode, or publish a second frame when pose inputs are unchanged', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const primitive = mesh('skin');

    subsystem.run(host.host, sceneOf(primitive));
    subsystem.run(host.host, sceneOf(primitive));

    expect(gpu.createCommandEncoder).toHaveBeenCalledTimes(1);
    expect(host.applySkinningBatch).toHaveBeenCalledTimes(1);
  });

  it('detects an in-place bone-matrix animation edit after an unchanged frame', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const primitive = mesh('skin');

    subsystem.run(host.host, sceneOf(primitive));
    subsystem.run(host.host, sceneOf(primitive));
    primitive.bones[12] = 0.25;
    subsystem.run(host.host, sceneOf(primitive));

    expect(gpu.createCommandEncoder).toHaveBeenCalledTimes(2);
    expect(host.applySkinningBatch).toHaveBeenCalledTimes(2);
  });

  it('detects in-place morph-weight edits but skips a steady CPU-fallback pose', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, false);
    const primitive: SkinnedMeshPrimitive = {
      ...mesh('skin'),
      morphTargets: [new Float32Array(9)],
      morphWeights: new Float32Array([0]),
    };

    subsystem.run(host.host, sceneOf(primitive));
    subsystem.run(host.host, sceneOf(primitive));
    primitive.morphWeights![0] = 0.5;
    subsystem.run(host.host, sceneOf(primitive));

    expect(gpu.createCommandEncoder).not.toHaveBeenCalled();
    expect(host.applySkinningBatch).toHaveBeenCalledTimes(2);
  });

  it('re-applies an unchanged pose when the renderer replaces its target buffers', () => {
    const gpu = gpuHarness();
    const firstHost = hostHarness('skin');
    const replacementHost = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const primitive = mesh('skin');

    subsystem.run(firstHost.host, sceneOf(primitive));
    subsystem.run(replacementHost.host, sceneOf(primitive));

    expect(firstHost.applySkinningBatch).toHaveBeenCalledOnce();
    expect(replacementHost.applySkinningBatch).toHaveBeenCalledOnce();
    expect(gpu.createCommandEncoder).toHaveBeenCalledTimes(2);
  });

  it('does not accept a pose fingerprint when publication fails', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const primitive = mesh('skin');
    host.applySkinningBatch.mockImplementationOnce(() => {
      throw new Error('injected publication failure');
    });

    expect(() => subsystem.run(host.host, sceneOf(primitive)))
      .toThrow('injected publication failure');
    subsystem.run(host.host, sceneOf(primitive));

    expect(host.applySkinningBatch).toHaveBeenCalledTimes(2);
    expect(gpu.createCommandEncoder).toHaveBeenCalledTimes(2);
  });

  it('retires cached per-mesh GPU buffers when the skinned primitive disappears', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);

    subsystem.run(host.host, sceneOf(mesh('skin')));
    subsystem.run(host.host, {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' },
    });

    expect(gpu.buffers).toHaveLength(6);
    for (const record of gpu.buffers) {
      expect(record.destroy).toHaveBeenCalledOnce();
    }
    expect(host.applySkinningBatch).toHaveBeenCalledOnce();
  });

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

  it('routes any stored-f32 non-identity bind through the CPU solver', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const primitive = mesh('skin');
    const bindMatrix = new Float32Array(IDENTITY);
    bindMatrix[12] = 1e-7;
    const bindMatrixInverse = new Float32Array(IDENTITY);
    bindMatrixInverse[12] = -1e-7;

    subsystem.run(host.host, sceneOf({
      ...primitive,
      bindMatrix,
      bindMatrixInverse,
    }));

    expect(gpu.createCommandEncoder).not.toHaveBeenCalled();
    expect(host.applySkinningBatch).toHaveBeenCalledOnce();
    expect(host.applySkinningBatch.mock.calls[0]?.[0]?.[0]?.gpuWritten).toBe(false);
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

  it('restores a morph lane above the native array-index ceiling', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const semanticIndex = 4_294_967_295;
    const baseLane = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const colorSets: Array<Float32Array | undefined> = [];
    const morphTargetColorSets: Array<
      ReadonlyArray<Float32Array> | undefined
    > = [];
    Object.defineProperty(colorSets, String(semanticIndex), {
      value: baseLane,
      enumerable: true,
    });
    Object.defineProperty(morphTargetColorSets, String(semanticIndex), {
      value: [new Float32Array([
        -0.5, 0.5, 0,
        0.5, -0.5, 0,
        0, 0.5, -0.5,
      ])],
      enumerable: true,
    });
    const base: SkinnedMeshPrimitive = {
      ...mesh('skin'),
      colorSets,
      morphTargets: [new Float32Array(9)],
      morphTargetColorSets,
      morphWeights: new Float32Array([1]),
    };

    subsystem.run(host.host, sceneOf(base));
    subsystem.run(host.host, sceneOf({
      ...base,
      morphWeights: new Float32Array([0]),
    }));

    expect(gpu.createBuffer).not.toHaveBeenCalled();
    const [activeUpdates] = host.applySkinningBatch.mock.calls[0]!;
    expect(activeUpdates[0].patch.colorSets[semanticIndex]).not.toBe(baseLane);
    const [restoreUpdates, restoreCommands] =
      host.applySkinningBatch.mock.calls[1]!;
    expect(restoreCommands).toBeNull();
    expect(restoreUpdates[0].patch.colorSets).toBe(colorSets);
    expect(restoreUpdates[0].patch.colorSets[semanticIndex]).toBe(baseLane);
  });

  it('keeps a mixed color-morph and GPU-eligible batch on one CPU topology candidate', () => {
    const gpu = gpuHarness();
    const host = hostHarness(['color-morph', 'gpu-skin']);
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const colorMorph: SkinnedMeshPrimitive = {
      ...mesh('color-morph'),
      colors: new Float32Array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
      ]),
      morphTargets: [new Float32Array(9)],
      morphTargetColors: [new Float32Array([
        -0.5, 0.5, 0,
        0.5, -0.5, 0,
        0, 0.5, -0.5,
      ])],
      morphWeights: new Float32Array([0.5]),
    };
    const gpuEligible = mesh('gpu-skin');
    const scene: Scene = {
      primitives: [colorMorph, gpuEligible],
      emitters: [],
      environment: { kind: 'none' },
    };

    subsystem.run(host.host, scene);

    // A color stream is a topology payload in HybridEngine. Encoding the
    // second mesh against the current buffers before that replacement would
    // make applySkinningBatch reject the mixed transaction.
    expect(gpu.createBuffer).not.toHaveBeenCalled();
    expect(host.applySkinningBatch).toHaveBeenCalledOnce();
    const [updates, commands] = host.applySkinningBatch.mock.calls[0]!;
    expect(commands).toBeNull();
    expect(updates.map((update: { gpuWritten: boolean }) => update.gpuWritten))
      .toEqual([false, false]);
    expect(Array.from(updates[0].patch.colors)).toEqual(
      Array.from(new Float32Array([
        0.75, 0.25, 0,
        0.25, 0.75, 0,
        0, 0.25, 0.75,
      ])),
    );
  });

  it('restores dormant COLOR_0/TEXCOORD_0 once, then returns to the GPU fast path', () => {
    const gpu = gpuHarness();
    const host = hostHarness('skin');
    const subsystem = new GpuSkinningSubsystem(gpu.device, true);
    const baseUvs = new Float32Array([0, 0, 1, 0, 0, 1]);
    const baseColors = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const base: SkinnedMeshPrimitive = {
      ...mesh('skin'),
      uvs: baseUvs,
      colors: baseColors,
      morphTargets: [new Float32Array(9)],
      morphTargetUvs: [
        new Float32Array([0.2, 0.4, -0.2, 0.4, 0.2, -0.4]),
      ],
      morphTargetColors: [
        new Float32Array([
          -0.5, 0.5, 0,
          0.5, -0.5, 0,
          0, 0.5, -0.5,
        ]),
      ],
      morphWeights: new Float32Array([1]),
    };

    subsystem.run(host.host, sceneOf(base));
    subsystem.run(host.host, sceneOf({
      ...base,
      morphWeights: new Float32Array([0]),
    }));
    subsystem.run(host.host, sceneOf({
      ...base,
      morphWeights: new Float32Array([0]),
    }));

    expect(host.applySkinningBatch).toHaveBeenCalledTimes(3);
    const [activeUpdates, activeCommands] =
      host.applySkinningBatch.mock.calls[0]!;
    expect(activeCommands).toBeNull();
    expect([...activeUpdates[0].patch.uvs]).toEqual([
      ...new Float32Array([0.2, 0.4, 0.8, 0.4, 0.2, 0.6]),
    ]);
    expect([...activeUpdates[0].patch.colors]).toEqual([
      ...new Float32Array([
        0.5, 0.5, 0,
        0.5, 0.5, 0,
        0, 0.5, 0.5,
      ]),
    ]);

    const [restoreUpdates, restoreCommands] =
      host.applySkinningBatch.mock.calls[1]!;
    expect(restoreCommands).toBeNull();
    expect(restoreUpdates[0].patch.uvs).toBe(baseUvs);
    expect(restoreUpdates[0].patch.colors).toBe(baseColors);

    const [steadyUpdates, steadyCommands] =
      host.applySkinningBatch.mock.calls[2]!;
    expect(steadyCommands).not.toBeNull();
    expect(steadyUpdates[0].gpuWritten).toBe(true);
    expect(steadyUpdates[0].patch.uvs).toBeUndefined();
    expect(steadyUpdates[0].patch.colors).toBeUndefined();
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
    expect(host.applySkinningBatch).toHaveBeenCalledTimes(2);
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
    const host = hostHarness(['other', 'skin']);
    const other = mesh('other');
    const original = mesh('skin');
    const originalScene: Scene = {
      primitives: [other, original],
      emitters: [],
      environment: { kind: 'none' },
    };
    subsystem.run(host.host, originalScene);
    const otherPrivateBuffer = gpu.buffers[0]!;

    expect(gpu.buffers).toHaveLength(12);
    const edited = {
      ...original,
      positions: new Float32Array([2, 0, 0, 3, 0, 0, 2, 1, 0]),
    };
    const editedScene: Scene = {
      ...originalScene,
      primitives: [other, edited],
    };

    gpu.injectNextBuffer(otherPrivateBuffer.buffer);
    expect(() => subsystem.run(host.host, editedScene))
      .toThrow(/live\/shared buffer alias/);
    expect(otherPrivateBuffer.destroy).not.toHaveBeenCalled();
    expect(gpu.buffers).toHaveLength(12);

    subsystem.run(host.host, originalScene);
    expect(gpu.buffers).toHaveLength(12);
    expect(host.applySkinningBatch).toHaveBeenCalledOnce();
  });
});
