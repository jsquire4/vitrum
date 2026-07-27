import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Scene, type SkinnedMeshPrimitive } from '@vitrum/core';
import { HybridEngine, type HybridEngineOptions } from '../HybridEngine.js';
import { buildReSTIRSceneBVHForCoreScene, type SceneBVHBuffers } from '../restir/bvhCore.js';
import type { CollectedBvhMutation } from '../pipeline/CollectingBvhUpdateSink.js';

const IDENTITY = new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
]);
function deviceStub(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
    createBindGroup: vi.fn(),
    queue: { submit: vi.fn(), writeBuffer: vi.fn(), writeTexture: vi.fn() },
    addEventListener: vi.fn(), removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}
function options(): HybridEngineOptions {
  return {
    device: deviceStub(), width: 64, height: 64,
    primaryLightDir: [0, -1, 0], primaryLightIntensity: 2,
    skyTint: [1, 1, 1], skyIrradiance: 1,
  };
}
function skin(id: string, x: number): SkinnedMeshPrimitive {
  const transform = new Float32Array(IDENTITY); transform[12] = x;
  return {
    kind: 'skinned-mesh', id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    skinIndices: new Uint32Array(12),
    skinWeights: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
    bones: new Float32Array(IDENTITY), boneInverses: new Float32Array(IDENTITY),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    transform: asMat4(transform),
  };
}
function shifted(x: number): Float32Array {
  return new Float32Array([x, 0, 0, x + 1, 0, 0, x, 1, 0]);
}
interface EngineInternals {
  _state: string;
  _lastScene: Scene | null;
  _renderScene: Scene | null;
  _bvhBuffers: SceneBVHBuffers | null;
  _pipeline: unknown;
  _ddgi: unknown;
  _rc: unknown;
}

describe('HybridEngine mixed skinning transaction rollback', () => {
  it('restores all CPU ranges when the single final pipeline publication fails', () => {
    const scene: Scene = {
      primitives: [skin('cpu-a', 0), skin('cpu-b', 3), skin('gpu', 6)],
      emitters: [], environment: { kind: 'none' },
    };
    const engine = new HybridEngine(options());
    const bvh = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'merged' });
    const beforeNodes = bvh.bvhNodes.cpuData.slice(0);
    const beforePositions = bvh.bvhPositions.cpuData.slice(0);
    const beforeNormals = bvh.bvhNormals.cpuData.slice(0);
    let capturedMutation: CollectedBvhMutation | null = null;
    let capturedPrefixes: readonly GPUCommandBuffer[] = [];
    const pipelineRollback = vi.fn();
    const pipelineCommit = vi.fn(() => { throw new Error('injected submit failure'); });
    const prepareSceneMutation = vi.fn((
      mutation: CollectedBvhMutation,
      _nextBvh: SceneBVHBuffers,
      prefixes: readonly GPUCommandBuffer[],
    ) => {
      capturedMutation = mutation; capturedPrefixes = prefixes;
      return { commit: pipelineCommit, rollback: pipelineRollback, finalize: vi.fn() };
    });
    const ddgiRollback = vi.fn();
    const internals = engine as unknown as EngineInternals;
    internals._state = 'ready';
    internals._lastScene = scene;
    internals._renderScene = scene;
    internals._bvhBuffers = bvh;
    internals._pipeline = { prepareSceneMutation };
    internals._ddgi = {
      prepareSceneMutation: vi.fn(() => ({
        commit: vi.fn(), rollback: ddgiRollback, finalize: vi.fn(),
      })),
    };
    internals._rc = null;
    const skinCommand = {} as GPUCommandBuffer;

    expect(() => engine.applySkinningBatch([
      { id: 'cpu-a', patch: { positions: shifted(1), normals: (scene.primitives[0] as SkinnedMeshPrimitive).normals }, gpuWritten: false },
      { id: 'cpu-b', patch: { positions: shifted(2), normals: (scene.primitives[1] as SkinnedMeshPrimitive).normals }, gpuWritten: false },
      { id: 'gpu', patch: { positions: shifted(3), normals: (scene.primitives[2] as SkinnedMeshPrimitive).normals }, gpuWritten: true },
    ], skinCommand)).toThrow(/submit failure/);

    expect(prepareSceneMutation).toHaveBeenCalledTimes(1);
    expect(pipelineCommit).toHaveBeenCalledTimes(1);
    expect(pipelineRollback).toHaveBeenCalledTimes(1);
    expect(ddgiRollback).toHaveBeenCalledTimes(1);
    expect(capturedPrefixes).toEqual([skinCommand]);
    expect(capturedMutation!.positions).toHaveLength(2);
    expect(capturedMutation!.normals).toHaveLength(2);
    expect(capturedMutation!.learningPositions).toHaveLength(1);
    expect(capturedMutation!.nodes).toBeDefined();
    expect(internals._bvhBuffers).toBe(bvh);
    expect(internals._lastScene).toBe(scene);
    expect(internals._renderScene).toBe(scene);
    expect(new Uint8Array(bvh.bvhNodes.cpuData)).toEqual(new Uint8Array(beforeNodes));
    expect(new Uint8Array(bvh.bvhPositions.cpuData)).toEqual(new Uint8Array(beforePositions));
    expect(new Uint8Array(bvh.bvhNormals.cpuData)).toEqual(new Uint8Array(beforeNormals));
    for (const primitive of scene.primitives) {
      expect([...(primitive as SkinnedMeshPrimitive).positions]).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    }
  });
});
