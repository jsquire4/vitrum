import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Scene } from '@vitrum/core';
import { buildReSTIRSceneBVHForCoreScene } from '../src/restir/bvhCore.js';
import { topologyRebuild } from '../src/HybridEnginePrimitiveUpdates.js';
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

describe('topologyRebuild TLAS (C2)', () => {
  it('returns rcRefitBounds for multi-mesh TLAS topology rebuild', () => {
    const scene = twoBoxScene();
    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    expect(buffers.bvhMode).toBe('tlas');
    expect(buffers.primitiveTlasBindings.length).toBeGreaterThan(0);

    const ddgi = {
      invalidateProbeCache: vi.fn(),
      markInstancesDirty: vi.fn(),
    };
    const pipeline = {
      replaceBvhAndEmitters: vi.fn(),
      updateEmitters: vi.fn(),
      requestAccumReset: vi.fn(),
    };

    const ctx: PrimitiveUpdateContext = {
      bvhBuffers: buffers,
      pipeline: pipeline as never,
      ddgi: ddgi as never,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      lastScene: scene,
      renderScene: scene,
      restirBvhModeOverride: 'tlas',
    };

    const meshA = scene.primitives[0];
    if (meshA?.kind !== 'mesh') throw new Error('expected mesh');
    const flipped = meshA.indices!.slice();
    flipped[1] = flipped[2]!;

    const result = topologyRebuild('box-a', { indices: flipped }, ctx);

    expect(result.rcRefitBounds).toBeDefined();
    expect(result.rcRefitBounds!.min[0]).toBeLessThanOrEqual(result.rcRefitBounds!.max[0]);
    expect(pipeline.replaceBvhAndEmitters).toHaveBeenCalledTimes(1);
    expect(ddgi.invalidateProbeCache).toHaveBeenCalled();
  });

  it('keeps the prior CPU BVH renderable when GPU replacement fails', () => {
    const scene = twoBoxScene();
    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    const previousBounds = buffers.mergedGeometry.boundingBox;
    const pipeline = {
      replaceBvhAndEmitters: vi.fn(() => {
        throw new Error('gpu replacement failed');
      }),
      requestAccumReset: vi.fn(),
    };
    const ddgi = {
      invalidateProbeCache: vi.fn(),
      markInstancesDirty: vi.fn(),
    };
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers: buffers,
      pipeline: pipeline as never,
      ddgi: ddgi as never,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      lastScene: scene,
      renderScene: scene,
      restirBvhModeOverride: 'tlas',
    };

    const meshA = scene.primitives[0];
    if (meshA?.kind !== 'mesh') throw new Error('expected mesh');
    const flipped = meshA.indices!.slice();
    flipped[1] = flipped[2]!;

    expect(() => topologyRebuild('box-a', { indices: flipped }, ctx))
      .toThrow('gpu replacement failed');
    const restoredBuffers = ctx.bvhBuffers;
    expect(restoredBuffers).toBe(buffers);
    if (restoredBuffers == null) {
      throw new Error('expected the previous BVH generation to be restored');
    }
    expect(restoredBuffers.mergedGeometry.boundingBox).toBe(previousBounds);
    expect(pipeline.requestAccumReset).not.toHaveBeenCalled();
    expect(ddgi.invalidateProbeCache).not.toHaveBeenCalled();
  });
});
