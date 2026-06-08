import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Scene } from '@vitrum/core';
import { buildReSTIRSceneBVHForCoreScene } from '../src/restir/bvhCore.js';
import { positionsRefit } from '../src/HybridEnginePrimitiveUpdates.js';
import type { PrimitiveUpdateContext } from '../src/HybridEnginePrimitiveUpdates.js';

function twoBoxScene(): Scene {
  const box = (id: string, ox: number): Scene['primitives'][number] => ({
    kind: 'mesh',
    id,
    positions: new Float32Array([
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
    ]),
    normals: new Float32Array(24).fill(0).map((_, i) => (i % 3 === 2 ? 1 : 0)),
    indices: new Uint32Array([
      0, 1, 2, 4, 1, 2, 1, 5, 6, 5, 4, 6, 0, 2, 3, 2, 6, 7, 0, 1, 3, 1, 5, 3,
      3, 5, 7, 5, 6, 7, 0, 4, 3, 4, 6, 7,
    ]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    transform: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ox, 0, 0, 1])),
  });
  return {
    primitives: [box('box-a', 0), box('box-b', 2)],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('positionsRefit TLAS (C2)', () => {
  it('refits BLAS positions + TLAS bounds without full rebuild', () => {
    const scene = twoBoxScene();
    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    expect(buffers.bvhMode).toBe('tlas');

    const meshA = scene.primitives[0];
    if (meshA?.kind !== 'mesh') throw new Error('expected mesh');
    const shifted = meshA.positions.slice();
    for (let i = 0; i < shifted.length; i += 3) shifted[i] = (shifted[i] ?? 0) + 0.05;

    const pipeline = {
      refreshBvhRefit: vi.fn(),
      refreshTlasRefit: vi.fn(),
      requestAccumReset: vi.fn(),
    };
    const ddgi = { invalidateProbeCache: vi.fn(), markInstancesDirty: vi.fn() };

    const ctx: PrimitiveUpdateContext = {
      bvhBuffers: buffers,
      pipeline: pipeline as never,
      ddgi: ddgi as never,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      lastScene: scene,
      renderScene: scene,
    };

    const result = positionsRefit('box-a', { positions: shifted }, ctx);

    expect(result.rcRefitBounds).toBeDefined();
    expect(pipeline.refreshBvhRefit).toHaveBeenCalled();
    expect(pipeline.refreshTlasRefit).toHaveBeenCalled();
    expect(pipeline.requestAccumReset).toHaveBeenCalled();
    expect(ddgi.invalidateProbeCache).toHaveBeenCalled();
  });
});
