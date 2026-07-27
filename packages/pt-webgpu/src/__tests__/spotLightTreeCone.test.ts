import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import {
  buildLightTreeInputForScene,
  packEmitterArrays,
} from '../scene/emitterPacking.js';
import { walkPositionalEmitters } from '../bdpt/flatEmitterWalk.js';

function spotScene(angle: number): Scene {
  return {
    primitives: [],
    environment: { kind: 'none' },
    emitters: [
      {
        id: 'spot',
        kind: 'spot',
        position: [1, 2, 3],
        direction: [0, -2, 0],
        angle,
        penumbra: 0.25,
        color: [1, 0.5, 0.25],
        intensity: 4,
      },
    ],
  };
}

describe('spotlight light-tree emission cone', () => {
  it.each([0.17, 0.63, 1.12])(
    'recovers the authored outer half-angle %f from the packed cosine',
    (authoredAngle) => {
      const scene = spotScene(authoredAngle);
      const packed = packEmitterArrays(scene);
      const walked = [...walkPositionalEmitters(packed)];
      expect(walked).toHaveLength(1);
      expect(walked[0]?.kind).toBe('spot');
      if (walked[0]?.kind !== 'spot') throw new Error('expected one walked spotlight');

      // Independent packing oracle: the packed record must carry cos(angle).
      expect(walked[0].cosOuter).toBeCloseTo(Math.cos(authoredAngle), 6);

      const input = buildLightTreeInputForScene(scene, { packed });
      const cone = input.cones?.[0];
      expect(cone).toBeDefined();
      expect(cone?.axis).toEqual([0, -1, 0]);
      expect(cone?.thetaO).toBe(0);
      expect(cone?.thetaE).toBeCloseTo(authoredAngle, 6);
    },
  );
});
