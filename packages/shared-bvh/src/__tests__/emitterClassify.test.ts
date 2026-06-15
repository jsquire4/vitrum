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

function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
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

  it('modulates Le by readable linear emissiveMap averages', () => {
    const handle = {
      width: 2,
      height: 1,
      data: new Float32Array([
        0.25, 0.5, 1, 1,
        0.75, 0.25, 0.5, 1,
      ]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };

    expect(materialSpecEmissiveLe(material({
      emissive: [2, 2, 2],
      emissiveIntensity: 3,
      emissiveMap: { handle },
    }))).toEqual([3, 2.25, 4.5]);
  });

  it('decodes readable sRGB emissiveMap handles before modulating Le', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Uint8Array([128, 64, 32, 255]),
      __vitrum_hint__: { channels: 4, dataType: 'uint8', colorSpace: 'srgb' },
    };

    const le = materialSpecEmissiveLe(material({
      emissive: [1, 1, 1],
      emissiveIntensity: 2,
      emissiveMap: { handle },
    }));

    expect(le?.[0]).toBeCloseTo(srgbToLinear(128 / 255) * 2, 6);
    expect(le?.[1]).toBeCloseTo(srgbToLinear(64 / 255) * 2, 6);
    expect(le?.[2]).toBeCloseTo(srgbToLinear(32 / 255) * 2, 6);
  });

  it('treats a readable black emissiveMap as non-emissive', () => {
    const handle = {
      width: 1,
      height: 1,
      data: new Float32Array([0, 0, 0, 1]),
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
    };

    expect(materialSpecEmissiveLe(material({
      emissive: [10, 10, 10],
      emissiveIntensity: 1,
      emissiveMap: { handle },
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
