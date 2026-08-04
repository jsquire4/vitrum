import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { buildRCSceneBVHFromCore } from '../src/rc/bvhCore.js';

describe('RC standalone optical identity', () => {
  it('lowers a bulk analytic and publishes exact identity for every generated triangle', () => {
    const scene: Scene = {
      primitives: [{
        kind: 'analytic',
        id: 'bulk-sphere',
        shape: 'sphere',
        params: new Float32Array([0, 0, 0, 1]),
        material: {
          baseColor: [1, 1, 1],
          roughness: 0,
          metallic: 0,
          transmission: 1,
          thickness: 1,
        },
      }],
      emitters: [],
      environment: { kind: 'none' },
    };

    const bvh = buildRCSceneBVHFromCore(scene);
    const triangleCount = bvh.indices.array.length / 3;
    expect(triangleCount).toBeGreaterThan(0);
    expect(bvh.opticalTriangleIdentity.itemSize).toBe(2);
    expect(bvh.opticalTriangleIdentity.array.length).toBe(triangleCount * 2);
    expect(new Set(
      Array.from(bvh.opticalTriangleIdentity.array).filter((_, index) => index % 2 === 0),
    )).toEqual(new Set([1]));
    expect(new Set(
      Array.from(bvh.opticalTriangleIdentity.array).filter((_, index) => index % 2 === 1),
    )).toEqual(new Set([1]));
    expect([...bvh.opticalInstanceBoundaryIdBasePlusOne.array]).toEqual([1]);
  });
});
