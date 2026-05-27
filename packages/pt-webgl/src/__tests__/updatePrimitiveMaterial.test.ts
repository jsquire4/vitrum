import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

const updateMaterials = vi.fn();
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
      traverse: () => undefined,
    })),
    findMeshByPrimitiveId: vi.fn(() => ({ material: {} })),
    applyVitrumMaterialToMesh: vi.fn(),
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
    updateMaterials = updateMaterials;
  }

  return { WebGLPathTracer };
});

describe('PTEngineWebGL2.updatePrimitive material-only (PR-8)', () => {
  let teardownGlobalStub: (() => void) | null = null;

  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  it('calls updateMaterials without setScene for material-only patches', async () => {
    setScene.mockClear();
    updateMaterials.mockClear();
    reset.mockClear();

    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({
      primitives: [
        {
          id: 'floor',
          kind: 'mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
          material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metalness: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    });

    setScene.mockClear();
    engine.updatePrimitive('floor', {
      material: { baseColor: [0.9, 0.1, 0.1], roughness: 0.2, metalness: 0 },
    });

    expect(updateMaterials).toHaveBeenCalledTimes(1);
    expect(setScene).not.toHaveBeenCalled();
    expect(reset).not.toHaveBeenCalled();
    expect(engine.capabilities.incrementalPatchSupport?.material).toBe(true);
  });
});
