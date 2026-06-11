import { describe, expect, it } from 'vitest';
import type { Scene } from '@vitrum/core';
import { rebuildPrimitiveBlas } from '@vitrum/shared-bvh';
import type { SceneBVHBuffers } from '../src/restir/bvhTypes.js';
import {
  buildReSTIRSceneBVHForCoreScene,
  rebuildReSTIRSceneBVHPrimitiveCore,
} from '../src/restir/bvhCore.js';

function twoBoxScene(offsetB = 0): Scene {
  const positionsB = new Float32Array([
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
  ]);
  if (offsetB !== 0) {
    for (let i = 0; i < positionsB.length; i += 3) positionsB[i] = (positionsB[i] ?? 0) + offsetB;
  }
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'box-a',
        positions: new Float32Array([
          0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
        ]),
        normals: new Float32Array(24).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0)),
        indices: new Uint32Array([
          0, 1, 2, 4, 1, 2, 1, 5, 6, 5, 4, 6, 0, 2, 3, 2, 6, 7, 0, 1, 3, 1, 5, 3,
          3, 5, 7, 5, 6, 7, 0, 4, 3, 4, 6, 7,
        ]),
        material: { baseColor: [0.6, 0.6, 0.6], roughness: 0.5, metallic: 0 },
      },
      {
        kind: 'mesh',
        id: 'box-b',
        positions: positionsB,
        normals: new Float32Array(24).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0)),
        indices: new Uint32Array([
          0, 1, 2, 4, 1, 2, 1, 5, 6, 5, 4, 6, 0, 2, 3, 2, 6, 7, 0, 1, 3, 1, 5, 3,
          3, 5, 7, 5, 6, 7, 0, 4, 3, 4, 6, 7,
        ]),
        material: { baseColor: [0.4, 0.4, 0.8], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('rebuildReSTIRSceneBVHPrimitiveCore', () => {
  it('preserves buffer sizes when BLAS splice applies', () => {
    const scene = twoBoxScene();
    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    expect(buffers.scenePack).toBeDefined();

    const moved = twoBoxScene(0.05);
    const rebuilt = rebuildReSTIRSceneBVHPrimitiveCore(moved, 'box-b', buffers);
    if ('ok' in rebuilt && rebuilt.ok === false) {
      throw new Error(rebuilt.reason);
    }
    const nextBuffers = rebuilt as SceneBVHBuffers;
    expect(nextBuffers.bvhPositions.byteLength).toBe(buffers.bvhPositions.byteLength);
    expect(nextBuffers.scenePack?.triangleCount).toBe(buffers.scenePack?.triangleCount);

    const blas = rebuildPrimitiveBlas(moved, 'box-b', buffers.scenePack!, {
      tlas: true,
      resolveMaterialId: () => 0,
    });
    expect(blas.ok).toBe(true);
    if (blas.ok) expect(blas.strategy).toBe('splice');
  });
});
