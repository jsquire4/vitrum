import { describe, expect, it } from 'vitest';
import {
  deserializeGIStateExtendedSections,
  serializeGIStateExtendedSections,
  type GIStateExtendedSections,
} from '../giStateExtendedSections.js';
import { makeGIStateCompatibility } from '../giStateCompatibility.js';
import { nrcStateShape, type NrcStateConfig } from '../neural/nrc/nrcStateSnapshot.js';
import type { SceneBVHBuffers } from '../restir/bvhTypes.js';

function emptyStorage() {
  return { cpuData: new ArrayBuffer(0), byteLength: 0, count: 0 };
}

function compatibility(): Uint32Array {
  const empty = emptyStorage();
  const bvh = {
    bvhMode: 'merged',
    bvhNodes: empty,
    bvhIndex: empty,
    bvhPositions: empty,
    triangleMaterialIds: empty,
    bvhBeerColors: empty,
    bvhEmissiveLe: empty,
    bvhRoughMetal: empty,
    bvhNormals: empty,
    bvhTangents: empty,
    bvhColors: empty,
    emitters: empty,
    emitterCdf: empty,
    emitterAlias: empty,
    emitterCount: 0,
    lightTree: empty,
    lightTreeNodeCount: 0,
    lightTreeEnabled: false,
    materialTextureAtlas: {
      atlasLayers: [],
      atlasData: new Float32Array(4),
      atlasDim: 1,
      atlasLayerCount: 1,
      atlasMipLevelCount: 1,
      gpuSourceLayers: [],
      baseColorMetaData: new Float32Array(4),
      baseColorMetaWidth: 1,
      baseColorMetaHeight: 1,
      readableBaseColorLayerCount: 0,
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
    },
  } as unknown as SceneBVHBuffers;
  return makeGIStateCompatibility({
    bvh,
    environmentRotationY: 0,
    environmentIntensity: 0,
    primaryLightDirection: [0, -1, 0],
    primaryLightIntensity: 1,
    skyTint: [1, 1, 1],
    skyIrradiance: 0,
    estimatorConfiguration: new Uint32Array([1]),
  });
}

const config: NrcStateConfig = {
  levels: 1,
  featuresPerEntry: 2,
  tableSize: 2,
  nMin: 2,
  growth: 2,
  oneBlobBins: 1,
  width: 12,
  hidden: 1,
  spreadC: 0.01,
  recordCap: 4,
  learningRate: 0.01,
  tableLearningRate: 0.1,
  useF16: false,
  tileB: 4,
  warmupSteps: 2,
};

function sections(): GIStateExtendedSections {
  const reservoir = new Uint32Array(64);
  const shape = nrcStateShape(config);
  return {
    compatibility: compatibility(),
    restirDI: {
      representationVersion: 1,
      width: 1,
      height: 1,
      strideU32: 8,
      current: reservoir.slice(),
      previous: reservoir.slice(),
      spatial: reservoir.slice(),
    },
    nrc: {
      config,
      sceneBoundsMin: [-1, -1, -1],
      sceneBoundsMax: [1, 1, 1],
      trainedSteps: 7,
      mlp: {
        weights: new Float32Array(shape.weightScalars).fill(0.1),
        biases: new Float32Array(shape.biasScalars).fill(0.2),
        firstMomentWeights: new Float32Array(shape.weightScalars).fill(0.3),
        secondMomentWeights: new Float32Array(shape.weightScalars).fill(0.4),
        firstMomentBiases: new Float32Array(shape.biasScalars).fill(0.5),
        secondMomentBiases: new Float32Array(shape.biasScalars).fill(0.6),
        adamT: 7,
      },
      hashGrid: {
        tables: new Float32Array(shape.tableScalars).fill(0.7),
        firstMoment: new Float32Array(shape.tableScalars).fill(0.8),
        secondMoment: new Float32Array(shape.tableScalars).fill(0.9),
        adamT: 6,
      },
    },
  };
}

describe('GI-state complete-state extension block', () => {
  it('round-trips compatibility, ReSTIR-DI, and complete NRC learned state', () => {
    const source = sections();
    const decoded = deserializeGIStateExtendedSections(
      serializeGIStateExtendedSections(source),
    );
    expect(Array.from(decoded.compatibility)).toEqual(
      Array.from(source.compatibility),
    );
    expect(Array.from(decoded.restirDI!.current)).toEqual(
      Array.from(source.restirDI!.current),
    );
    expect(decoded.nrc!.trainedSteps).toBe(7);
    expect(decoded.nrc!.mlp.adamT).toBe(7);
    expect(decoded.nrc!.hashGrid.adamT).toBe(6);
    expect(Array.from(decoded.nrc!.hashGrid.secondMoment)).toEqual(
      Array.from(source.nrc!.hashGrid.secondMoment),
    );
  });

  it('supports a compatibility-only block when DI and NRC are inactive', () => {
    const source = { compatibility: compatibility() };
    const decoded = deserializeGIStateExtendedSections(
      serializeGIStateExtendedSections(source),
    );
    expect(decoded.restirDI).toBeUndefined();
    expect(decoded.nrc).toBeUndefined();
  });

  it('cold-migrates version-1 linear DI reservoirs while retaining other state', () => {
    const source = sections();
    const encoded = serializeGIStateExtendedSections(source);
    const view = new DataView(encoded);
    view.setUint32(4, 1, true); // historical extension version
    const diStart = 32 + source.compatibility.byteLength;
    view.setUint32(diStart + 16, 0, true); // historical reserved word
    const current = diStart + 20;
    view.setUint32(current, 7, true);
    view.setUint32(current + 4, 1, true);
    view.setFloat32(current + 8, 2, true);
    view.setFloat32(current + 12, 0.5, true);
    view.setFloat32(current + 16, 0.25, true);
    view.setFloat32(current + 20, 0.75, true);
    view.setUint32(current + 24, 1, true);
    view.setUint32(current + 28, 0, true);

    const decoded = deserializeGIStateExtendedSections(encoded);
    expect(decoded.restirDI?.representationVersion).toBe(1);
    expect(decoded.restirDI?.current.every((word) => word === 0)).toBe(true);
    expect(decoded.nrc?.trainedSteps).toBe(source.nrc?.trainedSteps);
  });

  it('rejects a representation marker that disagrees with extension version', () => {
    const encoded = serializeGIStateExtendedSections(sections());
    const diStart = 32 + sections().compatibility.byteLength;
    new DataView(encoded).setUint32(diStart + 16, 0, true);
    expect(() => deserializeGIStateExtendedSections(encoded)).toThrow(
      /representation marker/,
    );
  });

  it('rejects unknown flags and inconsistent section lengths', () => {
    const encoded = serializeGIStateExtendedSections(sections());
    const badFlags = encoded.slice(0);
    new DataView(badFlags).setUint32(12, 0x8000_0000, true);
    expect(() => deserializeGIStateExtendedSections(badFlags)).toThrow(
      /header/,
    );

    const badLength = encoded.slice(0);
    new DataView(badLength).setUint32(20, 4, true);
    expect(() => deserializeGIStateExtendedSections(badLength)).toThrow(
      /section lengths|section length/,
    );
  });

  it('rejects semantic corruption inside nested DI and NRC payloads', () => {
    const encoded = serializeGIStateExtendedSections(sections());
    // Header (32) + compatibility (64) + DI sub-header (20), then DI current.
    const badDi = encoded.slice(0);
    new DataView(badDi).setUint32(96 + 20 + 4, 1, true);
    expect(() => deserializeGIStateExtendedSections(badDi)).toThrow(
      /support counts/,
    );

    const badNrc = encoded.slice(0);
    const diBytes = new DataView(badNrc).getUint32(20, true);
    const nrcStart = 32 + 64 + diBytes;
    new DataView(badNrc).setUint32(nrcStart, 0, true);
    expect(() => deserializeGIStateExtendedSections(badNrc)).toThrow(/magic/);
  });
});
