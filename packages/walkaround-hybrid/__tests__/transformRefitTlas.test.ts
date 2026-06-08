import { describe, expect, it, vi } from 'vitest';
import { analyticPrimitiveToMesh, asMat4, type Scene, type ScenePrimitive } from '@vitrum/core';
import { buildReSTIRSceneBVHForCoreScene } from '../src/restir/bvhCore.js';
import { transformRefit } from '../src/HybridEnginePrimitiveUpdates.js';
import type { PrimitiveUpdateContext } from '../src/HybridEnginePrimitiveUpdates.js';

function twoOffsetMeshes(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'wall-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.8, 0.2, 0.2], roughness: 0.5, metallic: 0 },
        transform: asMat4(new Float32Array([
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1,
        ])),
      },
      {
        kind: 'mesh',
        id: 'wall-b',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.2, 0.2, 0.8], roughness: 0.5, metallic: 0 },
        transform: asMat4(new Float32Array([
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
        ])),
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function analyticSphereScenes(): { authored: Scene; render: Scene } {
  const sphere: ScenePrimitive = {
    kind: 'analytic',
    id: 'sphere-a',
    shape: 'sphere',
    params: new Float32Array([0, 0, 0, 1]),
    material: { baseColor: [0.8, 0.7, 0.6], roughness: 0.35, metallic: 0 },
  };
  return {
    authored: {
      primitives: [sphere],
      emitters: [],
      environment: { kind: 'none' },
    },
    render: {
      primitives: [analyticPrimitiveToMesh(sphere)],
      emitters: [],
      environment: { kind: 'none' },
    },
  };
}

describe('transformRefit TLAS (C2)', () => {
  it('TLAS-only path: refreshTlasRefit + markInstancesDirty + rcRefitBounds', () => {
    const scene = twoOffsetMeshes();
    const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
    expect(buffers.bvhMode).toBe('tlas');

    const pipeline = {
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

    const movedTransform = asMat4(new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0.5, 0, 1,
    ]));
    const result = transformRefit('wall-b', { transform: movedTransform }, ctx);

    expect(result.rcRefitBounds).toBeDefined();
    expect(pipeline.refreshTlasRefit).toHaveBeenCalled();
    expect(pipeline.requestAccumReset).toHaveBeenCalled();
    expect(ddgi.markInstancesDirty).toHaveBeenCalled();
    expect(ddgi.invalidateProbeCache).not.toHaveBeenCalled();
  });

  it('uses render-scene mesh fallbacks for authored analytic transform refits', () => {
    const { authored, render } = analyticSphereScenes();
    const buffers = buildReSTIRSceneBVHForCoreScene(render, { bvhMode: 'tlas' });
    expect(buffers.bvhMode).toBe('tlas');

    const pipeline = {
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
      lastScene: authored,
      renderScene: render,
    };

    const movedTransform = asMat4(new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0.25, 0, 1,
    ]));
    const result = transformRefit('sphere-a', { transform: movedTransform }, ctx);

    expect(result.rcRefitBounds).toBeDefined();
    expect(pipeline.refreshTlasRefit).toHaveBeenCalled();
    expect(ddgi.markInstancesDirty).toHaveBeenCalled();
    const updated = result.updatedScene.primitives[0];
    expect(updated?.kind).toBe('analytic');
    if (updated?.kind !== 'analytic') throw new Error('expected authored analytic to stay analytic');
    expect(Array.from(updated.transform ?? [])).toEqual(Array.from(movedTransform));
  });
});
