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
function changedNormals(x: number): Float32Array {
  return new Float32Array([x, 1, 0, x, 1, 0, x, 1, 0]);
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

function replaySlices(
  baseline: ArrayBuffer,
  slices: ReadonlyArray<{ readonly byteOffset: number; readonly data: ArrayBuffer }>,
): ArrayBuffer {
  const replayed = baseline.slice(0);
  for (const slice of slices) {
    new Uint8Array(replayed, slice.byteOffset, slice.data.byteLength).set(
      new Uint8Array(slice.data),
    );
  }
  return replayed;
}

describe('HybridEngine mixed skinning transaction rollback', () => {
  it.each(['merged', 'tlas'] as const)(
    'restores all CPU ranges when the single final %s publication fails',
    (bvhMode) => {
    const scene: Scene = {
      primitives: [skin('cpu-a', 0), skin('cpu-b', 3), skin('gpu', 6)],
      emitters: [], environment: { kind: 'none' },
    };
    const engine = new HybridEngine(options());
    const bvh = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode });
    const beforeNodes = bvh.bvhNodes.cpuData.slice(0);
    const beforePositions = bvh.bvhPositions.cpuData.slice(0);
    const beforeNormals = bvh.bvhNormals.cpuData.slice(0);
    const beforeTlasNodes = bvh.tlas?.nodes.cpuData.slice(0);
    const beforeWorldToLocal = bvh.tlas?.worldToLocal.cpuData.slice(0);
    const beforeLocalToWorld = bvh.tlas?.localToWorld.cpuData.slice(0);
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
      { id: 'cpu-a', patch: { positions: shifted(1) }, gpuWritten: false },
      { id: 'cpu-b', patch: { positions: shifted(2), normals: changedNormals(0.25) }, gpuWritten: false },
      { id: 'gpu', patch: { positions: shifted(3), normals: changedNormals(0.5) }, gpuWritten: true },
    ], skinCommand)).toThrow(/submit failure/);

    expect(prepareSceneMutation).toHaveBeenCalledTimes(1);
    expect(pipelineCommit).toHaveBeenCalledTimes(1);
    expect(pipelineRollback).toHaveBeenCalledTimes(1);
    expect(ddgiRollback).toHaveBeenCalledTimes(1);
    expect(capturedPrefixes).toEqual([skinCommand]);
    expect(capturedMutation!.positions).toHaveLength(2);
    expect(capturedMutation!.normals).toHaveLength(1);
    expect(capturedMutation!.learningPositions).toHaveLength(1);
    expect(capturedMutation!.nodes).toBeDefined();
    if (bvhMode === 'tlas') {
      expect(capturedMutation!.tlas?.nodes.length).toBeGreaterThanOrEqual(3);
    }
    expect(internals._bvhBuffers).toBe(bvh);
    expect(internals._lastScene).toBe(scene);
    expect(internals._renderScene).toBe(scene);
    expect(new Uint8Array(bvh.bvhNodes.cpuData)).toEqual(new Uint8Array(beforeNodes));
    expect(new Uint8Array(bvh.bvhPositions.cpuData)).toEqual(new Uint8Array(beforePositions));
    expect(new Uint8Array(bvh.bvhNormals.cpuData)).toEqual(new Uint8Array(beforeNormals));
    if (
      bvh.tlas != null &&
      beforeTlasNodes != null &&
      beforeWorldToLocal != null &&
      beforeLocalToWorld != null
    ) {
      expect(new Uint8Array(bvh.tlas.nodes.cpuData)).toEqual(
        new Uint8Array(beforeTlasNodes),
      );
      expect(new Uint8Array(bvh.tlas.worldToLocal.cpuData)).toEqual(
        new Uint8Array(beforeWorldToLocal),
      );
      expect(new Uint8Array(bvh.tlas.localToWorld.cpuData)).toEqual(
        new Uint8Array(beforeLocalToWorld),
      );
    }
    for (const primitive of scene.primitives) {
      expect([...(primitive as SkinnedMeshPrimitive).positions]).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    }
    },
  );

  it('restores direct solved morph streams once and keeps steady zero-weight refits structural-free', () => {
    const baseUvs = new Float32Array([0, 0, 1, 0, 0, 1]);
    const baseColors = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const authored: SkinnedMeshPrimitive = {
      ...skin('skin', 0),
      uvs: baseUvs,
      colors: baseColors,
      morphTargets: [new Float32Array(9)],
      morphTargetUvs: [new Float32Array(6)],
      morphTargetColors: [new Float32Array(9)],
      morphWeights: new Float32Array([0]),
    };
    const authoredScene: Scene = {
      primitives: [authored],
      emitters: [],
      environment: { kind: 'none' },
    };
    const staleRenderScene: Scene = {
      ...authoredScene,
      primitives: [{
        ...authored,
        uvs: new Float32Array(baseUvs),
        colors: new Float32Array(baseColors),
      }],
    };
    const engine = new HybridEngine(options());
    const internals = engine as unknown as EngineInternals;
    internals._lastScene = authoredScene;
    internals._renderScene = staleRenderScene;
    const applySkinningBatch = vi.spyOn(engine, 'applySkinningBatch')
      .mockImplementation(() => undefined);

    engine.applyGpuSkinnedRefit('skin');
    const restorePatch = applySkinningBatch.mock.calls[0]![0][0]!.patch;
    expect(restorePatch.uvs).toBe(baseUvs);
    expect(restorePatch.colors).toBe(baseColors);

    internals._renderScene = authoredScene;
    engine.applyGpuSkinnedRefit('skin');
    const steadyPatch = applySkinningBatch.mock.calls[1]![0][0]!.patch;
    expect(steadyPatch.uvs).toBeUndefined();
    expect(steadyPatch.colors).toBeUndefined();
    expect(steadyPatch.uvSets).toBeUndefined();
    expect(steadyPatch.colorSets).toBeUndefined();
  });

  it('publishes every ordered TLAS slice when two primitives move in one batch', () => {
    const scene: Scene = {
      primitives: [skin('cpu-a', 0), skin('cpu-b', 3)],
      emitters: [],
      environment: { kind: 'none' },
    };
    const engine = new HybridEngine(options());
    const bvh = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    const tlas = bvh.tlas!;
    const beforeTlasNodes = tlas.nodes.cpuData.slice(0);
    let capturedMutation: CollectedBvhMutation | null = null;
    const pipelineCommit = vi.fn();
    const prepareSceneMutation = vi.fn((mutation: CollectedBvhMutation) => {
      capturedMutation = mutation;
      return {
        commit: pipelineCommit,
        rollback: vi.fn(),
        finalize: vi.fn(),
      };
    });
    const internals = engine as unknown as EngineInternals;
    internals._state = 'ready';
    internals._lastScene = scene;
    internals._renderScene = scene;
    internals._bvhBuffers = bvh;
    internals._pipeline = { prepareSceneMutation };
    internals._ddgi = {
      prepareSceneMutation: vi.fn(() => ({
        commit: vi.fn(),
        rollback: vi.fn(),
        finalize: vi.fn(),
      })),
    };
    internals._rc = null;

    engine.applySkinningBatch([
      {
        id: 'cpu-a',
        patch: {
          positions: shifted(2),
          normals: (scene.primitives[0] as SkinnedMeshPrimitive).normals,
        },
        gpuWritten: false,
      },
      {
        id: 'cpu-b',
        patch: {
          positions: shifted(-2),
          normals: (scene.primitives[1] as SkinnedMeshPrimitive).normals,
        },
        gpuWritten: false,
      },
    ], null);

    expect(pipelineCommit).toHaveBeenCalledTimes(1);
    const published = capturedMutation as CollectedBvhMutation | null;
    expect(published?.tlas).toBeDefined();
    expect(new Uint8Array(tlas.nodes.cpuData)).not.toEqual(
      new Uint8Array(beforeTlasNodes),
    );
    const replayed = replaySlices(
      beforeTlasNodes,
      published!.tlas!.nodes,
    );
    expect(new Uint8Array(replayed)).toEqual(
      new Uint8Array(tlas.nodes.cpuData),
    );
    expect(published!.tlas!.worldToLocal).toEqual([]);
    expect(published!.tlas!.localToWorld).toEqual([]);
  });
});
