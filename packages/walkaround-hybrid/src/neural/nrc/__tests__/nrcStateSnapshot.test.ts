import { describe, expect, it } from 'vitest';
import {
  assertNrcLearnedStateSnapshot,
  deserializeNrcLearnedState,
  nrcStateBoundsMatch,
  nrcStateConfigMatches,
  nrcStateShape,
  serializeNrcLearnedState,
  type NrcLearnedStateSnapshot,
  type NrcStateConfig,
} from '../nrcStateSnapshot.js';

const config: NrcStateConfig = {
  levels: 2,
  featuresPerEntry: 2,
  tableSize: 4,
  nMin: 2,
  growth: 2,
  oneBlobBins: 2,
  width: 16,
  hidden: 2,
  spreadC: 0.01,
  recordCap: 8,
  learningRate: 0.01,
  tableLearningRate: 0.1,
  useF16: false,
  tileB: 4,
  warmupSteps: 3,
};

function snapshot(): NrcLearnedStateSnapshot {
  const shape = nrcStateShape(config);
  return {
    config,
    sceneBoundsMin: [-1, -2, -3],
    sceneBoundsMax: [4, 5, 6],
    trainedSteps: 9,
    mlp: {
      weights: new Float32Array(shape.weightScalars).fill(0.25),
      biases: new Float32Array(shape.biasScalars).fill(-0.5),
      firstMomentWeights: new Float32Array(shape.weightScalars).fill(-0.01),
      secondMomentWeights: new Float32Array(shape.weightScalars).fill(0.02),
      firstMomentBiases: new Float32Array(shape.biasScalars).fill(0.03),
      secondMomentBiases: new Float32Array(shape.biasScalars).fill(0.04),
      adamT: 9,
    },
    hashGrid: {
      tables: new Float32Array(shape.tableScalars).fill(0.001),
      firstMoment: new Float32Array(shape.tableScalars).fill(-0.002),
      secondMoment: new Float32Array(shape.tableScalars).fill(0.003),
      adamT: 9,
    },
  };
}

describe('NRC learned-state snapshot', () => {
  it('derives the exact network and hash-grid state shape', () => {
    expect(nrcStateShape(config)).toEqual({
      weightScalars: 2 * 16 * 16 + 3 * 16,
      biasScalars: 2 * 16 + 3,
      tableScalars: 2 * 4 * 2,
    });
    expect(() => assertNrcLearnedStateSnapshot(snapshot())).not.toThrow();
  });

  it('rejects non-finite parameters and negative Adam second moments', () => {
    const nonFinite = snapshot();
    nonFinite.mlp.weights[0] = Number.NaN;
    expect(() => assertNrcLearnedStateSnapshot(nonFinite)).toThrow(/finite/);

    const negativeVariance = snapshot();
    negativeVariance.hashGrid.secondMoment[1] = -1;
    expect(() => assertNrcLearnedStateSnapshot(negativeVariance))
      .toThrow(/non-negative/);
  });

  it('rejects shape and counter mismatches before any GPU import', () => {
    const badShape = snapshot();
    const malformed = {
      ...badShape,
      mlp: {
        ...badShape.mlp,
        weights: badShape.mlp.weights.slice(1),
      },
    };
    expect(() => assertNrcLearnedStateSnapshot(malformed)).toThrow(/weights/);

    const badCounter = { ...snapshot(), trainedSteps: 0x1_0000_0000 };
    expect(() => assertNrcLearnedStateSnapshot(badCounter)).toThrow(/trainedSteps/);
  });

  it('matches serialized f32 metadata but rejects a different training contract', () => {
    const roundTrippedConfig = {
      ...config,
      growth: Math.fround(config.growth),
      spreadC: Math.fround(config.spreadC),
      learningRate: Math.fround(config.learningRate),
      tableLearningRate: Math.fround(config.tableLearningRate),
    };
    expect(nrcStateConfigMatches(roundTrippedConfig, config)).toBe(true);
    expect(nrcStateConfigMatches(
      { ...roundTrippedConfig, warmupSteps: config.warmupSteps + 1 },
      config,
    )).toBe(false);

    const state = snapshot();
    expect(nrcStateBoundsMatch(state, [-1, -2, -3], [4, 5, 6])).toBe(true);
    expect(nrcStateBoundsMatch(state, [-1, -2, -3], [4, 5, 7])).toBe(false);
  });

  it('round-trips every learned value and Adam counter through an owning buffer', () => {
    const source = snapshot();
    const buffer = serializeNrcLearnedState(source);
    const restored = deserializeNrcLearnedState(buffer);

    expect(restored.config).toEqual({
      ...source.config,
      growth: Math.fround(source.config.growth),
      spreadC: Math.fround(source.config.spreadC),
      learningRate: Math.fround(source.config.learningRate),
      tableLearningRate: Math.fround(source.config.tableLearningRate),
    });
    expect(restored.sceneBoundsMin).toEqual(source.sceneBoundsMin);
    expect(restored.sceneBoundsMax).toEqual(source.sceneBoundsMax);
    expect(restored.trainedSteps).toBe(source.trainedSteps);
    expect(restored.mlp.adamT).toBe(source.mlp.adamT);
    expect(restored.hashGrid.adamT).toBe(source.hashGrid.adamT);
    for (const key of [
      'weights',
      'biases',
      'firstMomentWeights',
      'secondMomentWeights',
      'firstMomentBiases',
      'secondMomentBiases',
    ] as const) {
      expect(Array.from(restored.mlp[key])).toEqual(Array.from(source.mlp[key]));
      expect(restored.mlp[key].buffer).not.toBe(source.mlp[key].buffer);
    }
    for (const key of ['tables', 'firstMoment', 'secondMoment'] as const) {
      expect(Array.from(restored.hashGrid[key])).toEqual(
        Array.from(source.hashGrid[key]),
      );
      expect(restored.hashGrid[key].buffer).not.toBe(
        source.hashGrid[key].buffer,
      );
    }
  });

  it('rejects corrupted headers, architecture lengths, booleans, and tensor values', () => {
    const source = serializeNrcLearnedState(snapshot());

    const badMagic = source.slice(0);
    new DataView(badMagic).setUint32(0, 0, true);
    expect(() => deserializeNrcLearnedState(badMagic)).toThrow(/magic/);

    const truncated = source.slice(0, source.byteLength - 4);
    new DataView(truncated).setUint32(12, truncated.byteLength, true);
    expect(() => deserializeNrcLearnedState(truncated)).toThrow(
      /architecture/,
    );

    const badBool = source.slice(0);
    new DataView(badBool).setUint32(64, 2, true);
    expect(() => deserializeNrcLearnedState(badBool)).toThrow(/0 or 1/);

    const nonFinite = source.slice(0);
    new DataView(nonFinite).setFloat32(112, Number.NaN, true);
    expect(() => deserializeNrcLearnedState(nonFinite)).toThrow(/finite/);
  });

  it('rejects metadata that cannot survive the float32 wire boundary', () => {
    expect(() =>
      serializeNrcLearnedState({
        ...snapshot(),
        config: { ...config, growth: Number.MAX_VALUE },
      }),
    ).toThrow(/float32/);
    expect(() =>
      serializeNrcLearnedState({
        ...snapshot(),
        sceneBoundsMax: [4, 5, Number.MAX_VALUE],
      }),
    ).toThrow(/float32/);
  });
});
