import { describe, expect, it } from 'vitest';
import { assertAtrousVarianceWebGPUBufferShapes } from '../src/atrousVarianceWebGPU.js';

describe('assertAtrousVarianceWebGPUBufferShapes', () => {
  const minimal = {
    rgb: new Float32Array(12),
    width: 2,
    height: 2,
  };

  it('accepts rgb-only payloads', () => {
    expect(() => assertAtrousVarianceWebGPUBufferShapes(minimal)).not.toThrow();
  });

  it('throws when rgb is undersized', () => {
    expect(() => assertAtrousVarianceWebGPUBufferShapes({ rgb: new Float32Array(8), width: 2, height: 2 })).toThrow(/rgb/);
  });

  it('throws when prevRadianceRgb is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        prevRadianceRgb: new Float32Array(11),
      }),
    ).toThrow(/prevRadianceRgb/);
  });

  it('throws when gbufferNormalsRgb is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        gbufferNormalsRgb: new Float32Array(11),
      }),
    ).toThrow(/gbufferNormalsRgb/);
  });

  it('throws when linearDepth is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        linearDepth: new Float32Array(3),
      }),
    ).toThrow(/linearDepth/);
  });

  it('throws when motionRg is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        motionRg: new Float32Array(7),
      }),
    ).toThrow(/motionRg/);
  });

  it('throws when welfordMeanM2 is undersized', () => {
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        ...minimal,
        welfordMeanM2: new Float32Array(7),
      }),
    ).toThrow(/welfordMeanM2/);
  });

  it('accepts fully populated slices', () => {
    const px = 4;
    expect(() =>
      assertAtrousVarianceWebGPUBufferShapes({
        rgb: new Float32Array(px * 3),
        width: 2,
        height: 2,
        prevRadianceRgb: new Float32Array(px * 3),
        gbufferNormalsRgb: new Float32Array(px * 3),
        linearDepth: new Float32Array(px),
        motionRg: new Float32Array(px * 2),
        welfordMeanM2: new Float32Array(px * 2),
      }),
    ).not.toThrow();
  });
});
