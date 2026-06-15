import { describe, expect, it } from 'vitest';
import type { MaterialSpec } from '@vitrum/core';
import {
  classifyTriangleEmitterCore,
  materialSpecEmissiveLe,
} from '../emitterClassify.js';

function material(partial: Partial<MaterialSpec>): MaterialSpec {
  return {
    baseColor: [0.2, 0.2, 0.2],
    roughness: 0.8,
    metallic: 0,
    ...partial,
  };
}

describe('materialSpecEmissiveLe', () => {
  it('defaults missing emissiveIntensity to one', () => {
    expect(materialSpecEmissiveLe(material({
      emissive: [0.5, 0.25, 0.1],
    }))).toEqual([0.5, 0.25, 0.1]);
  });

  it('pre-multiplies authored emissiveIntensity into HDR radiance', () => {
    expect(materialSpecEmissiveLe(material({
      emissive: [0.5, 0.25, 0.1],
      emissiveIntensity: 4,
    }))).toEqual([2.0, 1.0, 0.4]);
  });

  it('rejects zero intensity as non-emissive', () => {
    expect(materialSpecEmissiveLe(material({
      emissive: [1, 1, 1],
      emissiveIntensity: 0,
    }))).toBeNull();
  });
});

describe('classifyTriangleEmitterCore', () => {
  it('classifies emissive core materials without an explicit intensity', () => {
    const emitter = classifyTriangleEmitterCore(
      material({ emissive: [0.25, 0.5, 1] }),
      { x: 0, y: 0, z: 1 },
      { x: 0, y: 0, z: -1 },
      10,
    );

    expect(emitter).toEqual({
      color: [0.25, 0.5, 1],
      intensity: 1,
    });
  });
});
