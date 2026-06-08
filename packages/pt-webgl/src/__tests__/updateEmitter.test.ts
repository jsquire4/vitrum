import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { isEmitterOnlyPatch } from '../legacy/three/scenePatch.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

const updateLights = vi.fn();
const setScene = vi.fn();
const reset = vi.fn();

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

vi.mock('@vitrum/three-bindings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@vitrum/three-bindings')>();
  return {
    ...actual,
    vitrumSceneToThree: vi.fn(() => ({
      updateMatrixWorld: vi.fn(),
      traverse: () => undefined,
    })),
  };
});

vi.mock('three-gpu-pathtracer', () => {
  class WebGLPathTracer {
    readonly target = { texture: {} };
    samples = 0;
    bounces = 0;
    transmissiveBounces = 0;
    filterGlossyFactor = 0;
    renderDelay = 0;
    minSamples = 0;
    dynamicLowRes = false;
    multipleImportanceSampling = false;
    tileRepeatFactors: Uint8Array | null = null;
    configureAdditiveAccumulation = vi.fn();
    readonly tiles = { set: vi.fn() };
    readonly _pathTracer = { material: { uniforms: {} } };

    setScene = setScene;
    setCamera(): void {}
    renderSample(): void {
      this.samples += 1;
    }
    reset = reset;
    dispose(): void {}
    updateEnvironment(): void {}
    updateLights = updateLights;
  }

  return { WebGLPathTracer };
});

/** Scene with a mesh-area emitter so `meshId` is a valid patch field. */
function makeScene(): Scene {
  return {
    primitives: [
      {
        id: 'panel-a',
        kind: 'mesh',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0, emissive: [1, 1, 1] },
      },
      {
        id: 'panel-b',
        kind: 'mesh',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0, emissive: [1, 1, 1] },
      },
    ],
    emitters: [
      { id: 'cell-light', kind: 'mesh-area', color: [1, 1, 1], intensity: 2, meshId: 'panel-a' },
    ],
    environment: { kind: 'none' },
  };
}

describe('scenePatch.isEmitterOnlyPatch (A4 — meshId vs phantom meshPrimitiveId)', () => {
  it('classifies an intensity-only patch as light-only (fast path eligible)', () => {
    expect(isEmitterOnlyPatch({ intensity: 5 })).toBe(true);
  });

  it('classifies a meshId repoint as NOT light-only (must rebuild BVH)', () => {
    expect(isEmitterOnlyPatch({ meshId: 'panel-b' })).toBe(false);
  });

  it('classifies a kind change as NOT light-only', () => {
    // `kind` change is rejected by patchEmitterInScene downstream; the
    // classifier still must not route it to the light-only path.
    expect(isEmitterOnlyPatch({ kind: 'point' } as never)).toBe(false);
  });
});

describe('PTEngineWebGL2.updateEmitter (A4 fast-path routing)', () => {
  let teardownGlobalStub: (() => void) | null = null;

  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  it('takes the light-only fast path for an intensity-only change (updateLights, no setScene)', async () => {
    setScene.mockClear();
    updateLights.mockClear();
    reset.mockClear();

    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene(makeScene());

    setScene.mockClear();
    updateLights.mockClear();
    engine.updateEmitter!('cell-light', { intensity: 7 });

    expect(updateLights).toHaveBeenCalledTimes(1);
    expect(setScene).not.toHaveBeenCalled();
  });

  it('does NOT take the light-only fast path for a meshId repoint — triggers a full setScene/BVH rebuild', async () => {
    setScene.mockClear();
    updateLights.mockClear();
    reset.mockClear();

    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene(makeScene());

    setScene.mockClear();
    updateLights.mockClear();
    // Repoint the mesh-area emitter from panel-a to panel-b. This changes which
    // geometry feeds the area emitter, so the cheap updateLights() path must be
    // skipped in favor of a full setScene (BVH rebuild).
    engine.updateEmitter!('cell-light', { meshId: 'panel-b' });

    expect(setScene).toHaveBeenCalledTimes(1);
    expect(updateLights).not.toHaveBeenCalled();
  });
});
