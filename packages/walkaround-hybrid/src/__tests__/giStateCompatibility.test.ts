import { describe, expect, it } from 'vitest';
import {
  GI_STATE_COMPATIBILITY_WORDS,
  giStateCompatibilityMatches,
  isValidGIStateCompatibility,
  makeGIStateCompatibility,
  type GIStateCompatibilityInputs,
} from '../giStateCompatibility.js';
import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import {
  MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT,
  packMaterialTextureAtlasPixels,
} from '../bvh/materialTextureAtlasCodec.js';

function storage(values: readonly number[]) {
  const cpuData = new Uint32Array(values).buffer;
  return { cpuData, byteLength: cpuData.byteLength, count: values.length };
}

function atlas(seed: number): SceneBVHBuffers['materialTextureAtlas'] {
  return {
    atlasLayers: [{
      kind: 'cpu',
      layer: 0,
      width: 1,
      height: 1,
      encoding: MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT,
      mipLevelCount: 1,
      decodeSrgb: false,
      data: packMaterialTextureAtlasPixels(
        new Float32Array([seed, 0.25, 0.5, 1]),
        MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT,
      ),
    }],
    gpuSourceLayers: [],
    baseColorMetaData: new Float32Array([seed, 0, 0, 0]),
    baseColorMetaWidth: 1,
    baseColorMetaHeight: 1,
    readableBaseColorLayerCount: 1,
    readableNormalLayerCount: 0,
    readableRoughnessLayerCount: 0,
    readableMetallicLayerCount: 0,
    readableAoLayerCount: 0,
    readableAlphaLayerCount: 0,
    readableEmissiveLayerCount: 0,
    readableTransmissionLayerCount: 0,
    readableLightLayerCount: 0,
    readableSpecularColorLayerCount: 0,
    readableSpecularIntensityLayerCount: 0,
    readableClearcoatLayerCount: 0,
    readableClearcoatRoughnessLayerCount: 0,
    readableClearcoatNormalLayerCount: 0,
    readableSheenColorLayerCount: 0,
    readableSheenRoughnessLayerCount: 0,
    readableAnisotropyLayerCount: 0,
    readableIridescenceLayerCount: 0,
    readableIridescenceThicknessLayerCount: 0,
    readableThicknessLayerCount: 0,
    readableBumpLayerCount: 0,
    diagnostics: [],
  };
}

function bvh(seed = 1): SceneBVHBuffers {
  const positions = new Uint8Array(70_001);
  positions[0] = seed;
  positions[70_000] = seed + 1;
  const positionData = positions.buffer;
  return {
    bvhMode: 'tlas',
    bvhNodes: storage([seed, 2]),
    bvhIndex: storage([0, 1, 2, seed]),
    bvhPositions: {
      cpuData: positionData,
      byteLength: positionData.byteLength,
      count: Math.floor(positionData.byteLength / 16),
    },
    opticalTriangleIdentity: storage([0, seed]),
    opticalInstanceBoundaryIdBasePlusOne: storage([1]),
    triangleMaterialIds: storage([seed]),
    bvhBeerColors: storage([seed + 1]),
    bvhEmissiveLe: storage([seed + 2]),
    materialTextureAtlas: atlas(seed),
    bvhRoughMetal: storage([seed + 3]),
    bvhNormals: storage([seed + 4]),
    bvhTangents: storage([seed + 5]),
    bvhColors: storage([seed + 6]),
    emitters: storage([seed + 7]),
    emitterCdf: storage([seed + 8]),
    emitterAlias: storage([seed + 9]),
    emitterCount: 1,
    totalEmissivePower: 1,
    lightTree: storage([seed + 10]),
    lightTreeNodeCount: 1,
    lightTreeEnabled: true,
    mergedGeometry: {
      boundingBox: null,
      computeBoundingBox: () => undefined,
    },
    meshVertexRanges: [],
    bvhIndicesStride3: new Uint32Array([0, 1, 2]),
    buildMaterials: [],
    coreMaterials: [],
    emitterNormals: new Float32Array([0, 1, 0]),
    tlas: {
      nodes: storage([seed + 11]),
      instanceIndices: storage([seed + 12]),
      blasRoots: storage([seed + 13]),
      worldToLocal: storage([seed + 14]),
      localToWorld: storage([seed + 15]),
      nodeCount: 1,
    },
    primitiveTlasBindings: [],
  };
}

function inputs(seed = 1): GIStateCompatibilityInputs {
  return {
    bvh: bvh(seed),
    directionalEnvironment: {
      width: 1,
      height: 1,
      map: new Float32Array([seed, 0.5, 0.25, 1]),
      pdf: new Float32Array([0.5]),
      marginal: new Float32Array([0.5, 0, 0, 0]),
      conditional: new Float32Array([0.5, 0, 0, 0]),
      totalWeight: 1,
    },
    environmentRotationY: 0.25,
    environmentIntensity: 2,
    primaryLightDirection: [0, -1, 0],
    primaryLightIntensity: 3,
    skyTint: [0.1, 0.2, 0.3],
    skyIrradiance: 0.5,
    estimatorConfiguration: new Uint32Array([7, 11, 13]),
  };
}

describe('GI-state compatibility key', () => {
  it('is fixed-width, deterministic, and validates its schema marker', () => {
    const a = makeGIStateCompatibility(inputs());
    const b = makeGIStateCompatibility(inputs());
    expect(a).toHaveLength(GI_STATE_COMPATIBILITY_WORDS);
    expect(isValidGIStateCompatibility(a)).toBe(true);
    expect(giStateCompatibilityMatches(a, b)).toBe(true);

    const corrupt = a.slice();
    corrupt[0] = 0;
    expect(isValidGIStateCompatibility(corrupt)).toBe(false);
    expect(giStateCompatibilityMatches(a, corrupt)).toBe(false);
  });

  it('changes independently for geometry, material, emitter, TLAS, environment, and sky inputs', () => {
    const base = makeGIStateCompatibility(inputs());
    const cases: GIStateCompatibilityInputs[] = [];

    const geometry = inputs();
    new Uint8Array(geometry.bvh.bvhPositions.cpuData)[35_003] = 91;
    cases.push(geometry);

    const material = inputs();
    const materialLayer = material.bvh.materialTextureAtlas.atlasLayers[0]!;
    if (materialLayer.kind !== 'cpu') throw new Error('expected CPU atlas layer');
    materialLayer.data[0] = 99;
    cases.push(material);

    const emitter = inputs();
    new Uint32Array(emitter.bvh.emitters.cpuData)[0] = 99;
    cases.push(emitter);

    const tlas = inputs();
    new Uint32Array(tlas.bvh.tlas!.worldToLocal.cpuData)[0] = 99;
    cases.push(tlas);

    cases.push({
      ...inputs(),
      directionalEnvironment: {
        ...inputs().directionalEnvironment!,
        map: new Float32Array([99, 0.5, 0.25, 1]),
      },
    });
    cases.push({ ...inputs(), environmentRotationY: 0.5 });
    cases.push({ ...inputs(), primaryLightIntensity: 3.5 });
    cases.push({ ...inputs(), skyIrradiance: 0.75 });
    cases.push({
      ...inputs(),
      estimatorConfiguration: new Uint32Array([7, 11, 17]),
    });

    for (const changed of cases) {
      expect(
        giStateCompatibilityMatches(
          base,
          makeGIStateCompatibility(changed),
        ),
      ).toBe(false);
    }
  });

  it('hashes every byte of large CPU mirrors, including an interior non-sampled offset', () => {
    const original = inputs();
    const changed = inputs();
    const bytes = new Uint8Array(changed.bvh.bvhPositions.cpuData);
    bytes[34_999] = (bytes[34_999] ?? 0) ^ 0xff;
    expect(
      giStateCompatibilityMatches(
        makeGIStateCompatibility(original),
        makeGIStateCompatibility(changed),
      ),
    ).toBe(false);
  });
});
