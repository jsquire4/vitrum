import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

const setScene = vi.fn();
const reset = vi.fn();
const generate = vi.fn(() => ({ bvhChanged: true, bvh: { mock: true } }));

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
    findMeshByPrimitiveId: vi.fn(() => {
      const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
      return {
        geometry: {
          getAttribute: () => ({ count: 3 }),
          setAttribute: vi.fn(),
        },
        material: {},
      };
    }),
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
    readonly _pathTracer = {
      material: {
        bvh: { updateFrom: vi.fn() },
        attributesArray: { updateFrom: vi.fn() },
        materialIndexAttribute: { updateFrom: vi.fn() },
        uniforms: {},
      },
    };
    readonly _generator = { initialized: true, generate };

    setScene = setScene;
    setCamera(): void {}
    renderSample(): void {
      this.samples += 1;
    }
    reset = reset;
    dispose(): void {}
    updateEnvironment(): void {}
    updateMaterials = vi.fn();
  }

  return { WebGLPathTracer };
});

describe('PTEngineWebGL2.updatePrimitive positions-only (PR-8c)', () => {
  let teardownGlobalStub: (() => void) | null = null;

  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  it('regenerates geometry without setScene when vertex count is unchanged', async () => {
    setScene.mockClear();
    generate.mockClear();
    reset.mockClear();

    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({
      primitives: [
        {
          id: 'mesh-a',
          kind: 'mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
          material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    });

    setScene.mockClear();
    engine.updatePrimitive!('mesh-a', {
      positions: new Float32Array([0, 0.1, 0, 1, 0.1, 0, 0, 1.1, 0]),
    });

    expect(generate).toHaveBeenCalledTimes(1);
    expect(setScene).not.toHaveBeenCalled();
    expect(reset).toHaveBeenCalled();
    expect(engine.capabilities.incrementalPatchSupport?.positions).toBe(true);
  });
});
