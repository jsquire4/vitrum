import { describe, expect, it, vi } from 'vitest';
import {
  asMat4,
  type InstancedMeshPrimitive,
  type MeshPrimitive,
  type Scene,
  type ScenePrimitive,
  type ScenePrimitivePatch,
  type SkinnedMeshPrimitive,
} from '@vitrum/core';
import { tlasRefitNodeIndices } from '@vitrum/shared-bvh';
import { buildReSTIRSceneBVHForCoreScene } from '../restir/bvhCore.js';
import {
  capturePrimitiveMutationUndo,
  materialPatch,
  positionsRefit,
  refitSkinnedMeshAfterGpuWrite,
  skinnedPosePatch,
  transformRefit,
  type PrimitiveUpdateContext,
} from '../HybridEnginePrimitiveUpdates.js';
import { CollectingBvhUpdateSink } from '../pipeline/CollectingBvhUpdateSink.js';

const IDENTITY = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

function translation(x: number): Float32Array {
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, 0, 0, 1,
  ]);
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

function skin(
  id: string,
  options: { tangents?: boolean; transform?: Float32Array } = {},
): SkinnedMeshPrimitive {
  return {
    kind: 'skinned-mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([
      Math.SQRT1_2, Math.SQRT1_2, 0,
      Math.SQRT1_2, Math.SQRT1_2, 0,
      Math.SQRT1_2, Math.SQRT1_2, 0,
    ]),
    ...(options.tangents
      ? { tangents: new Float32Array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]) }
      : {}),
    skinIndices: new Uint32Array(12),
    skinWeights: new Float32Array([
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
    ]),
    bones: new Float32Array(IDENTITY),
    boneInverses: new Float32Array(IDENTITY),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    ...(options.transform ? { transform: asMat4(options.transform) } : {}),
  };
}

function instancedMesh(id: string): InstancedMeshPrimitive {
  return {
    kind: 'instanced-mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    instances: [
      asMat4(translation(10)),
      asMat4(translation(20)),
    ],
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
  };
}

function sceneOf(...primitives: ScenePrimitive[]): Scene {
  return { primitives, emitters: [], environment: { kind: 'none' } };
}

function triangleSoup(id: string, xBase: number, triangleCount = 8): MeshPrimitive {
  const positions = new Float32Array(triangleCount * 9);
  const normals = new Float32Array(triangleCount * 9);
  for (let tri = 0; tri < triangleCount; tri += 1) {
    const offset = tri * 9;
    const x = xBase + tri * 0.25;
    positions.set([x, 0, 0, x + 0.1, 0, 0, x, 0.1, 0], offset);
    normals.set([0, 0, 1, 0, 0, 1, 0, 0, 1], offset);
  }
  return {
    kind: 'mesh',
    id,
    positions,
    normals,
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
  };
}

function context(
  scene: Scene,
  renderScene: Scene,
  pipeline: CollectingBvhUpdateSink,
  bvhMode: 'merged' | 'tlas' = 'merged',
): PrimitiveUpdateContext {
  return {
    bvhBuffers: buildReSTIRSceneBVHForCoreScene(renderScene, { bvhMode }),
    pipeline,
    ddgi: {
      invalidateProbeCache: vi.fn(),
      markInstancesDirty: vi.fn(),
    } as never,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 1,
    lastScene: scene,
    renderScene,
    coreSceneSuppliesMeshes: true,
    deferSubsystemSideEffects: true,
  };
}

describe('primitive mutation journal and collector', () => {
  it('retains nodes plus the affected position slice, not unrelated normals', () => {
    const scene = sceneOf(skin('skin'));
    const bvh = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'merged' });
    const hugePositions = new ArrayBuffer(8 * 1024 * 1024);
    const hugeNormals = new ArrayBuffer(8 * 1024 * 1024);
    bvh.bvhPositions.cpuData = hugePositions;
    bvh.bvhPositions.byteLength = hugePositions.byteLength;
    bvh.bvhNormals.cpuData = hugeNormals;
    bvh.bvhNormals.byteLength = hugeNormals.byteLength;
    const range = bvh.meshVertexRanges.find((candidate) => candidate.name === 'skin')!;
    const byteOffset = range.vertexStart * 16;
    const byteLength = range.vertexCount * 16;
    new Uint8Array(hugePositions, byteOffset, byteLength).fill(7);
    new Uint8Array(hugeNormals, byteOffset, byteLength).fill(11);
    const originalNodes = bvh.bvhNodes.cpuData.slice(0);

    const undo = capturePrimitiveMutationUndo(
      bvh,
      'skin',
      { positions: new Float32Array(9) },
    );

    expect(undo.retainedByteLength).toBe(
      bvh.bvhNodes.cpuData.byteLength + byteLength,
    );
    expect(undo.retainedByteLength).toBeLessThan(hugePositions.byteLength / 100);
    new Uint8Array(bvh.bvhNodes.cpuData).fill(0xff);
    new Uint8Array(hugePositions, byteOffset, byteLength).fill(1);
    new Uint8Array(hugeNormals, byteOffset, byteLength).fill(2);
    new Uint8Array(hugePositions)[hugePositions.byteLength - 1] = 99;
    undo.restore();

    expect(new Uint8Array(bvh.bvhNodes.cpuData)).toEqual(new Uint8Array(originalNodes));
    expect([...new Uint8Array(hugePositions, byteOffset, byteLength)]).toEqual(
      new Array(byteLength).fill(7),
    );
    expect([...new Uint8Array(hugeNormals, byteOffset, byteLength)]).toEqual(
      new Array(byteLength).fill(2),
    );
    expect(new Uint8Array(hugePositions)[hugePositions.byteLength - 1]).toBe(99);
  });

  it('retains no rollback bytes for candidate-only material publication', () => {
    const scene = sceneOf(skin('skin'));
    const bvh = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    const undo = capturePrimitiveMutationUndo(
      bvh,
      'skin',
      { material: { roughness: 0.25 } },
    );
    expect(undo.retainedByteLength).toBe(0);
  });

  it('journals and uploads only the changed TLAS BLAS and ancestor slices', () => {
    const scene = sceneOf(skin('skin-a'), skin('skin-b'));
    const collector = new CollectingBvhUpdateSink();
    const ctx = context(scene, scene, collector, 'tlas');
    const bvh = ctx.bvhBuffers!;
    const binding = bvh.primitiveTlasBindings.find(
      (candidate) => candidate.primitiveId === 'skin-b',
    )!;
    const nextRoot = bvh.primitiveTlasBindings
      .map((candidate) => candidate.blasRoot)
      .filter((root) => root > binding.blasRoot)
      .sort((a, b) => a - b)[0] ??
      bvh.bvhNodes.cpuData.byteLength / 32;
    const nodeByteOffset = binding.blasRoot * 32;
    const nodeByteLength = (nextRoot - binding.blasRoot) * 32;
    const unrelatedBefore = bvh.bvhNodes.cpuData.slice(0, nodeByteOffset);
    const targetInstance = bvh.primitiveTlasBindings
      .slice(0, bvh.primitiveTlasBindings.indexOf(binding))
      .reduce((sum, candidate) => sum + candidate.instanceCount, 0);
    const affectedTlasNodes = tlasRefitNodeIndices(
      {
        nodes: new Uint32Array(bvh.tlas!.nodes.cpuData),
        nodeCount: bvh.tlas!.nodeCount,
        instanceIndices: new Uint32Array(bvh.tlas!.instanceIndices.cpuData),
        blasRoots: new Uint32Array(bvh.tlas!.blasRoots.cpuData),
        instanceTransforms: new Float32Array(bvh.tlas!.worldToLocal.cpuData),
      },
      [targetInstance],
    );

    const undo = capturePrimitiveMutationUndo(
      bvh,
      'skin-b',
      {
        positions: new Float32Array(9),
        normals: new Float32Array(9),
      },
    );
    expect(undo.retainedByteLength).toBe(
      nodeByteLength +
        affectedTlasNodes.length * 32 +
        binding.vertexCount * 16 * 2,
    );
    undo.accept();

    positionsRefit(
      'skin-b',
      {
        positions: new Float32Array([
          2, 0, 0,
          3, 0, 0,
          2, 1, 0,
        ]),
      },
      ctx,
    );
    const mutation = collector.snapshot();

    expect(mutation.nodes?.[0]?.byteOffset).toBe(nodeByteOffset);
    expect(mutation.nodes?.[0]?.data.byteLength).toBe(nodeByteLength);
    expect(new Uint8Array(bvh.bvhNodes.cpuData, 0, nodeByteOffset)).toEqual(
      new Uint8Array(unrelatedBefore),
    );
    expect(mutation.nodes?.[0]?.data.byteLength).toBeLessThan(
      bvh.bvhNodes.cpuData.byteLength,
    );
    expect(
      mutation.tlas?.nodes.reduce(
        (sum, slice) => sum + slice.data.byteLength,
        0,
      ),
    ).toBe(affectedTlasNodes.length * 32);
    expect(mutation.tlas?.worldToLocal).toEqual([]);
    expect(mutation.tlas?.localToWorld).toEqual([]);
  });

  it('journals, uploads, and restores one primitive TLAS transform exactly', () => {
    const primitives = Array.from({ length: 8 }, (_, index) =>
      skin(`skin-${index}`, { transform: translation(index * 10) }));
    const scene = sceneOf(...primitives);
    const collector = new CollectingBvhUpdateSink();
    const ctx = context(scene, scene, collector, 'tlas');
    const bvh = ctx.bvhBuffers!;
    const tlas = bvh.tlas!;
    const targetId = 'skin-5';
    const bindingIndex = bvh.primitiveTlasBindings.findIndex(
      (candidate) => candidate.primitiveId === targetId,
    );
    const targetInstance = bvh.primitiveTlasBindings
      .slice(0, bindingIndex)
      .reduce((sum, binding) => sum + binding.instanceCount, 0);
    const affectedNodes = tlasRefitNodeIndices(
      {
        nodes: new Uint32Array(tlas.nodes.cpuData),
        nodeCount: tlas.nodeCount,
        instanceIndices: new Uint32Array(tlas.instanceIndices.cpuData),
        blasRoots: new Uint32Array(tlas.blasRoots.cpuData),
        instanceTransforms: new Float32Array(tlas.worldToLocal.cpuData),
      },
      [targetInstance],
    );
    const nodesBefore = tlas.nodes.cpuData.slice(0);
    const worldToLocalBefore = tlas.worldToLocal.cpuData.slice(0);
    const localToWorldBefore = tlas.localToWorld.cpuData.slice(0);
    const matrixAtBuild = bvh.meshVertexRanges.find(
      (range) => range.name === targetId,
    )!.matrixWorldAtBuild;
    const matrixAtBuildBefore = new Float32Array(matrixAtBuild);
    const patch = {
      transform: asMat4(translation(55)),
    } as ScenePrimitivePatch;
    const undo = capturePrimitiveMutationUndo(bvh, targetId, patch);

    expect(undo.retainedByteLength).toBe(
      affectedNodes.length * 32 + 64 + 64 + matrixAtBuild.byteLength,
    );
    transformRefit(targetId, patch, ctx);
    const mutation = collector.snapshot().tlas!;

    expect(
      mutation.nodes.reduce(
        (sum, slice) => sum + slice.data.byteLength,
        0,
      ),
    ).toBe(affectedNodes.length * 32);
    expect(mutation.worldToLocal).toHaveLength(1);
    expect(mutation.localToWorld).toHaveLength(1);
    expect(mutation.worldToLocal[0]?.byteOffset).toBe(targetInstance * 64);
    expect(mutation.worldToLocal[0]?.data.byteLength).toBe(64);
    expect(mutation.localToWorld[0]?.byteOffset).toBe(targetInstance * 64);
    expect(mutation.localToWorld[0]?.data.byteLength).toBe(64);
    expect(
      new Float32Array(tlas.worldToLocal.cpuData)[targetInstance * 16 + 12],
    ).toBeCloseTo(-55);
    expect(
      new Float32Array(tlas.localToWorld.cpuData)[targetInstance * 16 + 12],
    ).toBeCloseTo(55);
    expect(new Uint8Array(tlas.nodes.cpuData)).not.toEqual(
      new Uint8Array(nodesBefore),
    );
    expect(new Uint8Array(replaySlices(nodesBefore, mutation.nodes))).toEqual(
      new Uint8Array(tlas.nodes.cpuData),
    );
    expect(
      new Uint8Array(replaySlices(worldToLocalBefore, mutation.worldToLocal)),
    ).toEqual(new Uint8Array(tlas.worldToLocal.cpuData));
    expect(
      new Uint8Array(replaySlices(localToWorldBefore, mutation.localToWorld)),
    ).toEqual(new Uint8Array(tlas.localToWorld.cpuData));
    const affectedSet = new Set(affectedNodes);
    for (let node = 0; node < tlas.nodeCount; node += 1) {
      if (affectedSet.has(node)) continue;
      expect(
        new Uint8Array(tlas.nodes.cpuData, node * 32, 32),
      ).toEqual(new Uint8Array(nodesBefore, node * 32, 32));
    }
    for (let instance = 0; instance < primitives.length; instance += 1) {
      if (instance === targetInstance) continue;
      expect(
        new Uint8Array(tlas.worldToLocal.cpuData, instance * 64, 64),
      ).toEqual(new Uint8Array(worldToLocalBefore, instance * 64, 64));
      expect(
        new Uint8Array(tlas.localToWorld.cpuData, instance * 64, 64),
      ).toEqual(new Uint8Array(localToWorldBefore, instance * 64, 64));
    }

    undo.restore();
    expect(new Uint8Array(tlas.nodes.cpuData)).toEqual(
      new Uint8Array(nodesBefore),
    );
    expect(new Uint8Array(tlas.worldToLocal.cpuData)).toEqual(
      new Uint8Array(worldToLocalBefore),
    );
    expect(new Uint8Array(tlas.localToWorld.cpuData)).toEqual(
      new Uint8Array(localToWorldBefore),
    );
    expect(matrixAtBuild).toEqual(matrixAtBuildBefore);
  });

  it('indexes merged leaves once and copies only affected node ranges', () => {
    const first = triangleSoup('first', 0);
    const second = triangleSoup('second', 100);
    const scene = sceneOf(first, second);
    const collector = new CollectingBvhUpdateSink();
    const ctx = context(scene, scene, collector, 'merged');
    const bvh = ctx.bvhBuffers!;
    const affected = bvh.primitiveRefitNodeIndices!.get('second')!;
    const totalNodes = bvh.bvhNodes.count;
    const unrelatedNode = Array.from(
      { length: totalNodes },
      (_, node) => node,
    ).find((node) => !affected.includes(node))!;
    const unrelatedBefore = bvh.bvhNodes.cpuData.slice(
      unrelatedNode * 32,
      unrelatedNode * 32 + 32,
    );
    const range = bvh.meshVertexRanges.find(
      (candidate) => candidate.name === 'second',
    )!;
    const moved = new Float32Array(second.positions);
    for (let lane = 0; lane < moved.length; lane += 3) {
      moved[lane] = moved[lane]! + 1;
    }

    expect(affected.length).toBeGreaterThan(0);
    expect(affected.length).toBeLessThan(totalNodes);
    const undo = capturePrimitiveMutationUndo(
      bvh,
      'second',
      { positions: moved },
    );
    expect(undo.retainedByteLength).toBe(
      affected.length * 32 + range.vertexCount * 16,
    );
    undo.accept();

    positionsRefit(
      'second',
      { positions: moved },
      ctx,
    );
    const mutation = collector.snapshot();
    const uploadedNodeBytes = mutation.nodes?.reduce(
      (sum, nodes) => sum + nodes.data.byteLength,
      0,
    ) ?? 0;

    expect(uploadedNodeBytes).toBe(affected.length * 32);
    expect(uploadedNodeBytes).toBeLessThan(bvh.bvhNodes.cpuData.byteLength);
    expect(
      new Uint8Array(bvh.bvhNodes.cpuData, unrelatedNode * 32, 32),
    ).toEqual(new Uint8Array(unrelatedBefore));
  });

  it('keeps the last node payload and every disjoint position/normal slice', () => {
    const collector = new CollectingBvhUpdateSink();
    collector.refreshBvhRefit(new Uint32Array([1]).buffer, {
      byteOffset: 0,
      data: new Uint32Array([10]).buffer,
    });
    collector.refreshBvhNormalsSlice({
      byteOffset: 0,
      data: new Uint32Array([20]).buffer,
    });
    collector.refreshBvhRefit(new Uint32Array([2]).buffer, {
      byteOffset: 64,
      data: new Uint32Array([30]).buffer,
    }, 128);
    collector.refreshBvhNormalsSlice({
      byteOffset: 64,
      data: new Uint32Array([40]).buffer,
    });
    const mutation = collector.snapshot();

    expect(mutation.nodes?.map((nodes) => nodes.byteOffset)).toEqual([0, 128]);
    expect([...new Uint32Array(mutation.nodes![1]!.data)]).toEqual([2]);
    expect(mutation.positions?.map((slice) => slice.byteOffset)).toEqual([0, 64]);
    expect(mutation.normals?.map((slice) => slice.byteOffset)).toEqual([0, 64]);
  });
});

describe('skinned render-state ownership', () => {
  it.each(['merged', 'tlas'] as const)(
    'publishes transform+bone pose atomically in %s BVH mode',
    (bvhMode) => {
      const authored = skin('skin');
      const restPositions = new Float32Array(authored.positions);
      const scene = sceneOf(authored);
      const collector = new CollectingBvhUpdateSink();
      const ctx: PrimitiveUpdateContext = {
        ...context(scene, scene, collector, bvhMode),
        restirBvhModeOverride: bvhMode,
      };
      const result = skinnedPosePatch('skin', {
        bones: translation(2),
        transform: asMat4(translation(5)),
      }, ctx);
      const authoredAfter = result.updatedScene.primitives[0] as SkinnedMeshPrimitive;
      const renderedAfter = result.updatedRenderScene!.primitives[0] as SkinnedMeshPrimitive;
      const range = result.bvhBuffers.meshVertexRanges.find(
        (candidate) => candidate.name === 'skin',
      )!;
      const packedPositions = new Float32Array(result.bvhBuffers.bvhPositions.cpuData);

      expect([...authoredAfter.positions]).toEqual([...restPositions]);
      expect(authoredAfter.bones[12]).toBeCloseTo(2, 6);
      expect(authoredAfter.transform?.[12]).toBeCloseTo(5, 6);
      expect([...renderedAfter.positions]).toEqual([2, 0, 0, 3, 0, 0, 2, 1, 0]);
      expect(renderedAfter.transform?.[12]).toBeCloseTo(5, 6);
      expect(range.matrixWorldAtBuild[12]).toBeCloseTo(5, 6);

      if (bvhMode === 'merged') {
        expect(packedPositions[range.vertexStart * 4]).toBeCloseTo(7, 6);
      } else {
        const binding = result.bvhBuffers.primitiveTlasBindings.find(
          (candidate) => candidate.primitiveId === 'skin',
        )!;
        expect(packedPositions[binding.vertexStart * 4]).toBeCloseTo(2, 6);
        expect(new Float32Array(result.bvhBuffers.tlas!.localToWorld.cpuData)[12])
          .toBeCloseTo(5, 6);
        expect(
          packedPositions[binding.vertexStart * 4]! +
            new Float32Array(result.bvhBuffers.tlas!.localToWorld.cpuData)[12]!,
        ).toBeCloseTo(7, 6);
      }
    },
  );

  it('solves two non-identity frames from immutable authored rest arrays', () => {
    const authored = skin('skin');
    const originalPositions = new Float32Array(authored.positions);
    const scene = sceneOf(authored);
    const collector = new CollectingBvhUpdateSink();
    const ctx = context(scene, scene, collector, 'merged');

    const frame1 = skinnedPosePatch('skin', { bones: translation(1) }, ctx);
    const frame2 = skinnedPosePatch('skin', { bones: translation(2) }, {
      ...ctx,
      bvhBuffers: frame1.bvhBuffers,
      lastScene: frame1.updatedScene,
      renderScene: frame1.updatedRenderScene!,
    });
    const authoredAfter = frame2.updatedScene.primitives[0] as SkinnedMeshPrimitive;
    const renderedAfter = frame2.updatedRenderScene!.primitives[0] as SkinnedMeshPrimitive;

    expect([...authoredAfter.positions]).toEqual([...originalPositions]);
    expect([...renderedAfter.positions]).toEqual([2, 0, 0, 3, 0, 0, 2, 1, 0]);
  });

  it('restores authored COLOR_0 and TEXCOORD_0 when morph weights return to zero', () => {
    const baseUvs = new Float32Array([0, 0, 1, 0, 0, 1]);
    const baseColors = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const authored: SkinnedMeshPrimitive = {
      ...skin('skin'),
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
      morphWeights: new Float32Array([0]),
    };
    const scene = sceneOf(authored);
    const ctx = context(scene, scene, new CollectingBvhUpdateSink(), 'merged');

    const active = skinnedPosePatch(
      'skin',
      { morphWeights: new Float32Array([1]) },
      ctx,
    );
    const activeRender =
      active.updatedRenderScene!.primitives[0] as SkinnedMeshPrimitive;
    expect([...activeRender.uvs!]).toEqual([
      ...new Float32Array([0.2, 0.4, 0.8, 0.4, 0.2, 0.6]),
    ]);
    expect([...activeRender.colors!]).toEqual([
      ...new Float32Array([
        0.5, 0.5, 0,
        0.5, 0.5, 0,
        0, 0.5, 0.5,
      ]),
    ]);

    const inactive = skinnedPosePatch(
      'skin',
      { morphWeights: new Float32Array([0]) },
      {
        ...ctx,
        bvhBuffers: active.bvhBuffers,
        lastScene: active.updatedScene,
        renderScene: active.updatedRenderScene!,
      },
    );
    const inactiveRender =
      inactive.updatedRenderScene!.primitives[0] as SkinnedMeshPrimitive;
    expect(inactiveRender.uvs).toBe(baseUvs);
    expect(inactiveRender.colors).toBe(baseColors);
    expect([...inactiveRender.uvs!]).toEqual([...baseUvs]);
    expect([...inactiveRender.colors!]).toEqual([...baseColors]);

    const steady = skinnedPosePatch(
      'skin',
      { morphWeights: new Float32Array([0]) },
      {
        ...ctx,
        bvhBuffers: inactive.bvhBuffers,
        lastScene: inactive.updatedScene,
        renderScene: inactive.updatedRenderScene!,
      },
    );
    const steadyRender =
      steady.updatedRenderScene!.primitives[0] as SkinnedMeshPrimitive;
    expect(steady.bvhBuffers).toBe(inactive.bvhBuffers);
    expect(steadyRender.uvs).toBe(baseUvs);
    expect(steadyRender.colors).toBe(baseColors);
  });

  it('restores authored streams when an active morph definition is cleared', () => {
    const baseUvs = new Float32Array([0, 0, 1, 0, 0, 1]);
    const baseColors = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const authored: SkinnedMeshPrimitive = {
      ...skin('skin'),
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
      morphWeights: new Float32Array([0]),
    };
    const scene = sceneOf(authored);
    const ctx = context(scene, scene, new CollectingBvhUpdateSink(), 'merged');
    const active = skinnedPosePatch(
      'skin',
      { morphWeights: new Float32Array([1]) },
      ctx,
    );
    const activeRender =
      active.updatedRenderScene!.primitives[0] as SkinnedMeshPrimitive;
    expect(activeRender.uvs).not.toBe(baseUvs);
    expect(activeRender.colors).not.toBe(baseColors);

    // `morphTargets: []` is rejected by the core contract. At the JavaScript
    // boundary, explicit undefined values clear the optional definition fields,
    // while the defined bones co-patch makes this reachable through routing.
    const clearDefinitionPatch = {
      bones: new Float32Array(IDENTITY),
      morphTargets: undefined,
      morphWeights: undefined,
      morphTargetUvs: undefined,
      morphTargetColors: undefined,
    } as unknown as ScenePrimitivePatch;
    const cleared = skinnedPosePatch(
      'skin',
      clearDefinitionPatch,
      {
        ...ctx,
        bvhBuffers: active.bvhBuffers,
        lastScene: active.updatedScene,
        renderScene: active.updatedRenderScene!,
      },
    );
    const clearedAuthored =
      cleared.updatedScene.primitives[0] as SkinnedMeshPrimitive;
    const clearedRender =
      cleared.updatedRenderScene!.primitives[0] as SkinnedMeshPrimitive;
    expect(clearedAuthored.morphTargets).toBeUndefined();
    expect(clearedAuthored.morphWeights).toBeUndefined();
    expect(clearedRender.uvs).toBe(baseUvs);
    expect(clearedRender.colors).toBe(baseColors);

    const steady = skinnedPosePatch(
      'skin',
      { bones: new Float32Array(IDENTITY) },
      {
        ...ctx,
        bvhBuffers: cleared.bvhBuffers,
        lastScene: cleared.updatedScene,
        renderScene: cleared.updatedRenderScene!,
      },
    );
    expect(steady.bvhBuffers).toBe(cleared.bvhBuffers);
    const steadyRender =
      steady.updatedRenderScene!.primitives[0] as SkinnedMeshPrimitive;
    expect(steadyRender.uvs).toBe(baseUvs);
    expect(steadyRender.colors).toBe(baseColors);
  });

  it('rebuilds away stale attributes when active morph definitions and bases are cleared', () => {
    const authored: SkinnedMeshPrimitive = {
      ...skin('skin'),
      uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
      colors: new Float32Array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
      ]),
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
      morphWeights: new Float32Array([0]),
    };
    const scene = sceneOf(authored);
    const ctx = context(scene, scene, new CollectingBvhUpdateSink(), 'merged');
    const active = skinnedPosePatch(
      'skin',
      { morphWeights: new Float32Array([1]) },
      ctx,
    );
    const activePackedPositions =
      new Float32Array(active.bvhBuffers.bvhPositions.cpuData);
    const activePackedColors =
      new Float32Array(active.bvhBuffers.bvhColors.cpuData);
    expect(activePackedPositions[3]).not.toBe(0);
    expect([...activePackedColors.slice(0, 4)]).toEqual([0.5, 0.5, 0, 1]);

    const clearDefinitionAndBasePatch = {
      bones: new Float32Array(IDENTITY),
      uvs: undefined,
      colors: undefined,
      morphTargets: undefined,
      morphWeights: undefined,
      morphTargetUvs: undefined,
      morphTargetColors: undefined,
    } as unknown as ScenePrimitivePatch;
    const cleared = skinnedPosePatch(
      'skin',
      clearDefinitionAndBasePatch,
      {
        ...ctx,
        bvhBuffers: active.bvhBuffers,
        lastScene: active.updatedScene,
        renderScene: active.updatedRenderScene!,
      },
    );
    const clearedAuthored =
      cleared.updatedScene.primitives[0] as SkinnedMeshPrimitive;
    const clearedRender =
      cleared.updatedRenderScene!.primitives[0] as SkinnedMeshPrimitive;
    expect(cleared.bvhBuffers).not.toBe(active.bvhBuffers);
    expect(clearedAuthored.uvs).toBeUndefined();
    expect(clearedAuthored.colors).toBeUndefined();
    expect(clearedRender.uvs).toBeUndefined();
    expect(clearedRender.colors).toBeUndefined();
    const clearedPackedPositions =
      new Float32Array(cleared.bvhBuffers.bvhPositions.cpuData);
    const clearedPackedColors =
      new Float32Array(cleared.bvhBuffers.bvhColors.cpuData);
    expect([
      clearedPackedPositions[3],
      clearedPackedPositions[7],
      clearedPackedPositions[11],
    ]).toEqual([0, 0, 0]);
    expect([...clearedPackedColors.slice(0, 4)]).toEqual([1, 1, 1, 1]);

    const steady = skinnedPosePatch(
      'skin',
      { bones: new Float32Array(IDENTITY) },
      {
        ...ctx,
        bvhBuffers: cleared.bvhBuffers,
        lastScene: cleared.updatedScene,
        renderScene: cleared.updatedRenderScene!,
      },
    );
    expect(steady.bvhBuffers).toBe(cleared.bvhBuffers);
  });

  it('preserves rest arrays across a GPU-refit to CPU topology-rebuild transition', () => {
    const authored = skin('skin', { tangents: true });
    const originalPositions = new Float32Array(authored.positions);
    const scene = sceneOf(authored);
    const collector = new CollectingBvhUpdateSink();
    const ctx = context(scene, scene, collector, 'merged');
    const gpuPose = new Float32Array([1, 0, 0, 2, 0, 0, 1, 1, 0]);

    const gpuFrame = refitSkinnedMeshAfterGpuWrite(
      'skin',
      gpuPose,
      authored.normals,
      ctx,
    );
    const rebuilt = skinnedPosePatch('skin', { bones: translation(2) }, {
      ...ctx,
      bvhBuffers: gpuFrame.bvhBuffers,
      lastScene: gpuFrame.updatedScene,
      renderScene: gpuFrame.updatedRenderScene!,
    });
    const authoredAfter = rebuilt.updatedScene.primitives[0] as SkinnedMeshPrimitive;
    const renderedAfter = rebuilt.updatedRenderScene!.primitives[0] as SkinnedMeshPrimitive;

    expect([...authoredAfter.positions]).toEqual([...originalPositions]);
    expect([...renderedAfter.positions]).toEqual([2, 0, 0, 3, 0, 0, 2, 1, 0]);
  });

  it('mirrors merged GPU-kernel normal bytes under nonuniform world scale', () => {
    const transform = new Float32Array([
      1, 0, 0, 0,
      1, 2, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
    const authored = skin('skin', { transform });
    const scene = sceneOf(authored);
    const collector = new CollectingBvhUpdateSink();
    const ctx = context(scene, scene, collector, 'merged');
    const result = refitSkinnedMeshAfterGpuWrite(
      'skin',
      authored.positions,
      authored.normals,
      ctx,
    );
    const range = result.bvhBuffers.meshVertexRanges[0]!;
    const actual = new Uint8Array(
      result.bvhBuffers.bvhNormals.cpuData,
      range.vertexStart * 16,
      16,
    );
    // For M=[[1,1,0],[0,2,0],[0,0,1]], M^-T maps normalized (1,1,0)
    // to (1,0,0). M^-1 would leave it on the (1,1,0) diagonal.
    const expected = new Float32Array([1, 0, 0, 0]);

    expect([...actual]).toEqual([...new Uint8Array(expected.buffer)]);
    expect(collector.snapshot().normals).toBeUndefined();
  });
});

describe('merged instanced geometry mutations', () => {
  it('rebuilds every flattened instance range for one shared positions patch', () => {
    const primitive = instancedMesh('instances');
    const scene = sceneOf(primitive);
    const collector = new CollectingBvhUpdateSink();
    const ctx: PrimitiveUpdateContext = {
      ...context(scene, scene, collector, 'merged'),
      restirBvhModeOverride: 'merged',
    };
    const moved = new Float32Array([2, 0, 0, 3, 0, 0, 2, 1, 0]);

    const result = positionsRefit('instances', { positions: moved }, ctx);
    const ranges = result.bvhBuffers.meshVertexRanges.filter(
      (candidate) => candidate.name === 'instances',
    );
    const packed = new Float32Array(result.bvhBuffers.bvhPositions.cpuData);

    expect(result.bvhBuffers.bvhMode).toBe('merged');
    expect(ranges).toHaveLength(2);
    expect(ranges.map((range) => packed[range.vertexStart * 4])).toEqual([12, 22]);
    expect(
      (result.updatedScene.primitives[0] as InstancedMeshPrimitive).positions,
    ).toEqual(moved);
    expect(collector.snapshot().replacement).not.toBeNull();
  });
});

describe('material candidate isolation', () => {
  it('fails closed before a direct material fast path can change optical topology', () => {
    const scene = sceneOf({
      kind: 'mesh',
      id: 'a',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: {
        baseColor: [1, 1, 1], roughness: 0, metallic: 0,
      },
    });
    const collector = new CollectingBvhUpdateSink();
    const ctx = context(scene, scene, collector, 'merged');
    const identityBefore = ctx.bvhBuffers!.opticalTriangleIdentity.cpuData;

    expect(() => materialPatch(
      'a',
      { material: { transmission: 1 } },
      ctx,
    )).toThrow(/must be routed through a topology rebuild/);
    expect(ctx.bvhBuffers!.opticalTriangleIdentity.cpuData).toBe(identityBefore);
    expect(collector.snapshot().material).toBeUndefined();
  });

  it('isolates two TLAS primitives authored with the same material object', () => {
    const shared = { baseColor: [0.4, 0.4, 0.4] as [number, number, number], roughness: 0.5, metallic: 0 };
    const mesh = (id: string, x: number): ScenePrimitive => ({
      kind: 'mesh',
      id,
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: shared,
      transform: asMat4(translation(x)),
    });
    const scene = sceneOf(mesh('a', 0), mesh('b', 3));
    const collector = new CollectingBvhUpdateSink();
    const ctx = context(scene, scene, collector, 'tlas');
    const live = ctx.bvhBuffers!;
    const indexIdentity = live.bvhIndex.cpuData;
    const beerIdentity = live.bvhBeerColors.cpuData;
    const roughIdentity = live.bvhRoughMetal.cpuData;
    const oldIndex = indexIdentity.slice(0);

    const result = materialPatch(
      'a',
      { material: { roughness: 0.25 } },
      ctx,
    );

    expect(result.bvhBuffers.coreMaterials).toHaveLength(2);
    expect(result.bvhBuffers.coreMaterials[0]?.roughness).toBe(0.25);
    expect(result.bvhBuffers.coreMaterials[1]?.roughness).toBe(0.5);
    expect([...new Uint32Array(result.bvhBuffers.triangleMaterialIds.cpuData)]).toEqual([0, 1]);
    expect(live.bvhIndex.cpuData).toBe(indexIdentity);
    expect(live.bvhBeerColors.cpuData).toBe(beerIdentity);
    expect(live.bvhRoughMetal.cpuData).toBe(roughIdentity);
    expect(new Uint8Array(live.bvhIndex.cpuData)).toEqual(new Uint8Array(oldIndex));
    expect(result.bvhBuffers.bvhIndex.cpuData).not.toBe(indexIdentity);
  });
});
