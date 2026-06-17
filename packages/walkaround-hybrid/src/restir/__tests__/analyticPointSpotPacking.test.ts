import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { packAnalyticPointSpotEmitters } from '../emitterHelpers.js';

describe('packAnalyticPointSpotEmitters', () => {
  it('packs authored point and spot distance/decay into the analytic-light texture lanes', () => {
    const scene: Scene = {
      primitives: [],
      environment: { kind: 'none' },
      emitters: [
        {
          kind: 'point',
          id: 'point-a',
          position: [1, 2, 3],
          color: [1, 0.5, 0.25],
          intensity: 4,
          distance: 7,
          decay: 0,
        },
        {
          kind: 'spot',
          id: 'spot-a',
          position: [4, 5, 6],
          direction: [0, -2, 0],
          angle: Math.PI / 4,
          penumbra: 0.25,
          color: [0.25, 0.5, 1],
          intensity: 8,
          distance: 9,
          decay: 1.5,
          castShadow: false,
        },
      ],
    };

    const packed = packAnalyticPointSpotEmitters(scene);

    expect(packed.count).toBe(2);
    expect(packed.data[14]).toBeCloseTo(7);
    expect(packed.data[15]).toBeCloseTo(0);
    const spotBase = 16;
    expect(packed.data[spotBase + 8]).toBeCloseTo(0);
    expect(packed.data[spotBase + 9]).toBeCloseTo(-1);
    expect(packed.data[spotBase + 10]).toBeCloseTo(0);
    expect(packed.data[spotBase + 13]).toBe(1);
    expect(packed.data[spotBase + 14]).toBeCloseTo(9);
    expect(packed.data[spotBase + 15]).toBeCloseTo(1.5);
  });
});
