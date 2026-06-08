import { describe, expect, it, vi } from 'vitest';
import type { Mat4, MaterialSpec, MeshPrimitive, Scene } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import {
  buildReSTIRSceneBVHForCoreScene,
  rebuildReSTIRSceneBVHPrimitiveCore,
} from '../restir/bvhCore.js';
import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import {
  materialPatch,
  type PrimitiveUpdateContext,
} from '../HybridEnginePrimitiveUpdates.js';
import {
  packBVHBeerColorsFromCore,
  packBVHEmissiveLeFromCore,
  packBVHIndexWFromCore,
} from '../restir/packingHelpers.js';

vi.mock('@vitrum/three-bindings', () => ({
  applyVitrumMaterialToMesh: vi.fn(() => {
    throw new Error('material patch should not require the THREE material bridge');
  }),
  findMeshByPrimitiveId: vi.fn(() => {
    throw new Error('material patch should not require a synthesized THREE root');
  }),
}));

function translate(x: number, y = 0, z = 0): Mat4 {
  return asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]));
}

function tri(id: string, material: MaterialSpec, transform?: Mat4): MeshPrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2]),
    material,
    ...(transform !== undefined ? { transform } : {}),
  };
}

function sceneWithMaterials(
  materialA: MaterialSpec,
  materialB: MaterialSpec,
  transformB = translate(4, 0, 0),
): Scene {
  return {
    primitives: [
      tri('mesh-a', materialA),
      tri('mesh-b', materialB, transformB),
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function expectU32ArrayEquals(actual: Uint32Array, expected: Uint32Array): void {
  expect(Array.from(actual)).toEqual(Array.from(expected));
}

function expectF32ArrayEquals(actual: Float32Array, expected: Float32Array): void {
  expect(Array.from(actual)).toEqual(Array.from(expected));
}

function assertSceneBvhBuffers(
  value: SceneBVHBuffers | { ok: false; reason: string },
): asserts value is SceneBVHBuffers {
  if ('ok' in value && value.ok === false) {
    throw new Error(value.reason);
  }
}

describe('THREE-decouple seams for core Scene walkaround ingestion', () => {
  it('builds the ReSTIR TLAS pack from a core Scene without a synthesized THREE root', () => {
    const materialA: MaterialSpec = { baseColor: [0.2, 0.4, 0.6], roughness: 0.7, metallic: 0 };
    const materialB: MaterialSpec = { baseColor: [0.8, 0.1, 0.1], roughness: 0.3, metallic: 0 };
    const transformB = translate(4, 2, -1);
    const scene = sceneWithMaterials(materialA, materialB, transformB);

    const bvh = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });

    expect(bvh.bvhMode).toBe('tlas');
    expect(bvh.buildMaterials).toHaveLength(0);
    expect(bvh.coreMaterials).toEqual([materialA, materialB]);
    expect(bvh.primitiveTlasBindings.map((b) => b.primitiveId)).toEqual(['mesh-a', 'mesh-b']);
    expect(bvh.meshVertexRanges.map((r) => r.name)).toEqual(['mesh-a', 'mesh-b']);

    const meshBRange = bvh.meshVertexRanges.find((r) => r.name === 'mesh-b');
    expect(meshBRange).toBeDefined();
    expect(meshBRange!.matrixWorldAtBuild[12]).toBeCloseTo(transformB[12]!, 6);
    expect(meshBRange!.matrixWorldAtBuild[13]).toBeCloseTo(transformB[13]!, 6);
    expect(meshBRange!.matrixWorldAtBuild[14]).toBeCloseTo(transformB[14]!, 6);
  });

  it('rebuilds one primitive BLAS from core data without a THREE root or material list', () => {
    const materialA: MaterialSpec = { baseColor: [0.2, 0.4, 0.6], roughness: 0.7, metallic: 0 };
    const materialB: MaterialSpec = { baseColor: [0.8, 0.1, 0.1], roughness: 0.3, metallic: 0 };
    const nextMaterialB: MaterialSpec = { baseColor: [0.1, 0.8, 0.2], roughness: 0.45, metallic: 0 };
    const prevScene = sceneWithMaterials(materialA, materialB);
    const prev = buildReSTIRSceneBVHForCoreScene(prevScene, { bvhMode: 'tlas' });
    const nextScene = sceneWithMaterials(materialA, nextMaterialB, translate(6, 0, 0));

    const rebuilt = rebuildReSTIRSceneBVHPrimitiveCore(nextScene, 'mesh-b', prev);
    assertSceneBvhBuffers(rebuilt);

    expect(rebuilt.bvhMode).toBe('tlas');
    expect(rebuilt.buildMaterials).toHaveLength(0);
    expect(rebuilt.coreMaterials).toEqual([materialA, nextMaterialB]);
    expect(rebuilt.scenePack?.triangleCount).toBe(prev.scenePack?.triangleCount);
    expect(rebuilt.bvhPositions.byteLength).toBe(prev.bvhPositions.byteLength);
  });
});

describe('THREE-decouple seam for material-only primitive patches', () => {
  it('re-packs material slices from core materials when no THREE root is available', () => {
    const baseMaterial: MaterialSpec = { baseColor: [0.2, 0.4, 0.6], roughness: 0.7, metallic: 0 };
    const nextMaterial: MaterialSpec = {
      baseColor: [0.9, 0.2, 0.1],
      roughness: 0.35,
      metallic: 1,
      emissive: [1.5, 0.5, 0.25],
      emissiveIntensity: 6,
    };
    const scene: Scene = {
      primitives: [tri('mesh-a', baseMaterial)],
      emitters: [],
      environment: { kind: 'none' },
    };
    const bvh = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    expect(bvh.buildMaterials).toHaveLength(0);
    expect(bvh.coreMaterials).toEqual([baseMaterial]);

    const pipeline = {
      refreshBvhRefit: vi.fn(),
      refreshBvhNodesOnly: vi.fn(),
      refreshTlasRefit: vi.fn(),
      refreshBvhFullRebuild: vi.fn(),
      updateEmitters: vi.fn(),
      refreshBvhMaterialSlice: vi.fn(),
      requestAccumReset: vi.fn(),
    };
    const ddgi = {
      invalidateProbeCache: vi.fn(),
    };
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers: bvh,
      pipeline,
      ddgi: ddgi as unknown as PrimitiveUpdateContext['ddgi'],
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      lastScene: scene,
      renderScene: scene,
      coreSceneSuppliesMeshes: true,
      restirBvhModeOverride: 'tlas',
    };

    const result = materialPatch('mesh-a', { material: nextMaterial }, ctx);
    const patched = result.bvhBuffers;
    const triMaterialIds = new Uint32Array(patched.triangleMaterialIds.cpuData);
    const expectedCoreMaterials = [nextMaterial];
    const expectedIndex = packBVHIndexWFromCore(
      patched.bvhIndicesStride3,
      triMaterialIds,
      expectedCoreMaterials,
      patched.bvhBeerColors.count,
    );
    const expectedBeer = packBVHBeerColorsFromCore(
      triMaterialIds,
      expectedCoreMaterials,
      patched.bvhBeerColors.count,
    );
    const expectedEmissive = packBVHEmissiveLeFromCore(
      triMaterialIds,
      expectedCoreMaterials,
      patched.bvhBeerColors.count,
    );

    expect(patched.coreMaterials).toEqual(expectedCoreMaterials);
    expectU32ArrayEquals(new Uint32Array(patched.bvhIndex.cpuData), expectedIndex);
    expectU32ArrayEquals(new Uint32Array(patched.bvhBeerColors.cpuData), expectedBeer);
    expectF32ArrayEquals(new Float32Array(patched.bvhEmissiveLe.cpuData), expectedEmissive);
    expect(pipeline.refreshBvhMaterialSlice).toHaveBeenCalledTimes(1);
    expect(pipeline.refreshBvhFullRebuild).not.toHaveBeenCalled();
    expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
    expect(ddgi.invalidateProbeCache).toHaveBeenCalledTimes(1);
  });
});
