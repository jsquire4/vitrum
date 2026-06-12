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
      refreshBvhNormalsSlice: vi.fn(),
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

  it('uploads a TLAS normals slice for count-preserving positions+normals patches', () => {
    const scene = twoBoxScene();
    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    expect(buffers.bvhMode).toBe('tlas');

    const meshA = scene.primitives[0];
    if (meshA?.kind !== 'mesh') throw new Error('expected mesh');
    if (meshA.normals == null) throw new Error('expected mesh normals');
    const shifted = meshA.positions.slice();
    for (let i = 0; i < shifted.length; i += 3) shifted[i] = (shifted[i] ?? 0) + 0.05;

    const replacementNormals = new Float32Array(meshA.normals.length);
    for (let i = 0; i < replacementNormals.length; i += 3) {
      replacementNormals[i] = 1;
      replacementNormals[i + 1] = 0;
      replacementNormals[i + 2] = 0;
    }

    const pipeline = {
      refreshBvhRefit: vi.fn(),
      refreshTlasRefit: vi.fn(),
      refreshBvhNormalsSlice: vi.fn(),
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

    const result = positionsRefit('box-a', { positions: shifted, normals: replacementNormals }, ctx);

    expect(result.rcRefitBounds).toBeDefined();
    expect(pipeline.refreshBvhNormalsSlice).toHaveBeenCalledTimes(1);
    const [normalsSlice] = pipeline.refreshBvhNormalsSlice.mock.calls[0] as [
      { byteOffset: number; data: ArrayBuffer },
    ];
    expect(normalsSlice.byteOffset).toBe(0);

    const f32 = new Float32Array(normalsSlice.data);
    expect(f32.length).toBe(meshA.normals.length / 3 * 4);
    for (let v = 0; v < f32.length / 4; v += 1) {
      expect(f32[v * 4]).toBeCloseTo(1, 5);
      expect(f32[v * 4 + 1]).toBeCloseTo(0, 5);
      expect(f32[v * 4 + 2]).toBeCloseTo(0, 5);
      expect(f32[v * 4 + 3]).toBeCloseTo(0, 5);
    }
  });
});
