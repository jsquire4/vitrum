/**
 * updateEnvironment.test.ts
 *
 * Verifies PTEngineWebGL2.updateEnvironment(env) — the env-only fast path
 * that avoids a full setScene() / BVH rebuild on host-driven timeOfDay
 * scrubs. Covers:
 *
 *  - calling updateEnvironment does NOT call WebGLPathTracer.setScene
 *    (no BVH rebuild)
 *  - calling updateEnvironment DOES call WebGLPathTracer.updateEnvironment
 *    (env uniforms refreshed)
 *  - the engine's internal THREE.Scene env / intensity / rotation fields are
 *    mutated in place to match the supplied SceneEnvironment
 *  - calling before setScene() throws (engine has no scene to mutate)
 *  - null env clears the environment
 *
 * The setup mocks both `three-gpu-pathtracer` (WebGLPathTracer) and
 * `@vitrum/three-bindings` so we can intercept calls and assert on them.
 * `vitrumSceneToThree` is replaced with a stub that returns a real THREE.Scene
 * (so `applyEnvironment` can mutate environmentIntensity etc. on it).
 */

import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { Scene as ThreeScene, Texture } from 'three';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

// We need a real applyEnvironment to mutate the THREE.Scene, so we DO NOT
// mock @vitrum/three-bindings entirely. We only override `vitrumSceneToThree`
// to return a fresh THREE.Scene without triggering vitrum scene conversion.
vi.mock('@vitrum/three-bindings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@vitrum/three-bindings')>();
  return {
    ...actual,
    vitrumSceneToThree: vi.fn(() => new ThreeScene()),
  };
});

// Instance tracker shared between the mock and the tests. We push every
// constructed WebGLPathTracer here so tests can introspect the spied calls
// without relying on vitest auto-spy of class constructors. The tracker is
// typed as `unknown` and tests narrow at the use site to avoid vi.fn variance
// pitfalls when typing across a class implements clause.
const pathTracerInstances: unknown[] = [];

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
    scene: unknown = null;
    configureAdditiveAccumulation = vi.fn();
    readonly tiles = { set: vi.fn() };
    readonly _pathTracer = {
      material: {
        uniforms: {
          uCausticStrategy: { value: -1 },
          uMneeMaxIterations: { value: 0 },
          uMneeMaxChainLength: { value: 0 },
          uCmfX: { value: null },
          uCmfY: { value: null },
          uCmfZ: { value: null },
          uYCmfCdf: { value: null },
          uYCmfIntegral: { value: 0 },
          uSpectralRendering: { value: -1 },
          uRadianceClamp: { value: -1 },
        },
      },
    };

    setScene = vi.fn((scene: unknown) => {
      this.scene = scene;
    });
    setCamera = vi.fn();
    renderSample = vi.fn(() => {
      this.samples += 1;
    });
    reset = vi.fn();
    dispose = vi.fn();
    updateEnvironment = vi.fn();

    constructor() {
      pathTracerInstances.push(this);
    }
  }

  return { WebGLPathTracer };
});

interface PathTracerStubView {
  readonly scene: unknown;
  readonly setScene: { mock: { calls: unknown[][] } };
  readonly updateEnvironment: { mock: { calls: unknown[][] } };
}

/** Return the most-recently constructed path-tracer stub instance. */
function lastPathTracer(): PathTracerStubView {
  const last = pathTracerInstances[pathTracerInstances.length - 1];
  if (last == null) throw new Error('no WebGLPathTracer instance constructed yet');
  return last as PathTracerStubView;
}

describe('PTEngineWebGL2.updateEnvironment', () => {
  let teardownGlobalStub: (() => void) | null = null;
  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  it('throws before setScene() is called', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    expect(() => {
      (engine as unknown as { updateEnvironment: (e: unknown) => void }).updateEnvironment({
        kind: 'none',
      });
    }).toThrow(/setScene/);
  });

  it('does NOT trigger setScene/BVH rebuild', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });

    const instance = lastPathTracer();
    const setSceneCallsBefore = instance.setScene.mock.calls.length;
    const updateEnvCallsBefore = instance.updateEnvironment.mock.calls.length;

    // The hot path under test:
    (engine as unknown as { updateEnvironment: (e: unknown) => void }).updateEnvironment({
      kind: 'none',
    });

    // setScene must NOT be called again (no BVH rebuild)
    expect(instance.setScene.mock.calls.length).toBe(setSceneCallsBefore);
    // updateEnvironment must be called (env uniforms refreshed)
    expect(instance.updateEnvironment.mock.calls.length).toBe(updateEnvCallsBefore + 1);
  });

  it('mutates internal THREE.Scene environment + intensity for HDRI env', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });

    const instance = lastPathTracer();
    const internalScene = instance.scene as ThreeScene;

    // Sanity: pre-state is empty environment (the stub vitrumSceneToThree
    // mock returns a fresh THREE.Scene which has env=null by default and
    // applyEnvironment for kind:'none' clears intensity to 1).
    expect(internalScene.environment).toBeNull();

    // Apply HDRI env with custom intensity + rotation
    const hdri = new Texture();
    (engine as unknown as { updateEnvironment: (e: unknown) => void }).updateEnvironment({
      kind: 'hdri',
      hdri,
      intensity: 2.5,
      rotationY: Math.PI / 4,
    });

    expect(internalScene.environment).toBe(hdri);
    expect(internalScene.environmentIntensity).toBeCloseTo(2.5);
    expect(internalScene.backgroundIntensity).toBeCloseTo(2.5);
    expect(internalScene.environmentRotation.y).toBeCloseTo(Math.PI / 4);
    expect(internalScene.backgroundRotation.y).toBeCloseTo(Math.PI / 4);
  });

  it('clears environment when called with null', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
    const internalScene = lastPathTracer().scene as ThreeScene;

    // First put an HDRI env in place
    const hdri = new Texture();
    (engine as unknown as { updateEnvironment: (e: unknown) => void }).updateEnvironment({
      kind: 'hdri',
      hdri,
      intensity: 2,
    });
    expect(internalScene.environment).toBe(hdri);

    // Now clear with null
    (engine as unknown as { updateEnvironment: (e: unknown) => void }).updateEnvironment(null);
    expect(internalScene.environment).toBeNull();
    expect(internalScene.environmentIntensity).toBe(1);
    expect(internalScene.environmentRotation.y).toBe(0);
  });

  it('clears environment when called with kind: "none"', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
    const internalScene = lastPathTracer().scene as ThreeScene;

    const hdri = new Texture();
    (engine as unknown as { updateEnvironment: (e: unknown) => void }).updateEnvironment({
      kind: 'hdri',
      hdri,
      intensity: 3,
    });
    (engine as unknown as { updateEnvironment: (e: unknown) => void }).updateEnvironment({
      kind: 'none',
    });
    expect(internalScene.environment).toBeNull();
  });

  it('throws after dispose', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
    engine.dispose();
    expect(() => {
      (engine as unknown as { updateEnvironment: (e: unknown) => void }).updateEnvironment({
        kind: 'none',
      });
    }).toThrow(/disposed/);
  });
});
