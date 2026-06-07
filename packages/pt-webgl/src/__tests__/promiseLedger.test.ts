import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

vi.mock('@vitrum/three-bindings', () => ({
  vitrumSceneToThree: () => ({
    traverse: () => undefined,
  }),
}));

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
        uniforms: {},
      },
    };

    setScene(): void {}
    setCamera(): void {}
    renderSample(): void {
      this.samples += 1;
    }
    reset(): void {}
    dispose(): void {}
    updateEnvironment(): void {}
  }

  return { WebGLPathTracer };
});

function sorted(values: Iterable<string>): string[] {
  return Array.from(values).sort();
}

describe('pt-webgl promise ledger compliance', () => {
  let teardownGlobalStub: (() => void) | null = null;
  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  it('matches declared capability and optional-method promises', async () => {
    const expected = BACKEND_PROMISE_LEDGER['pt-webgl'];
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    const caps = engine.capabilities;

    expect(caps.supportsIncrementalScene).toBe(expected.supportsIncrementalScene);
    expect(caps.supportsAddRemovePrimitive).toBe(expected.supportsAddRemovePrimitive);
    expect(caps.supportsAuxBuffers).toBe(expected.supportsAuxBuffers);
    expect(caps.accumulates).toBe(expected.accumulates);
    expect(caps.presentationMode).toBe(expected.presentationMode);

    expect(caps.incrementalPatchSupport).toEqual(expected.incrementalPatchSupport);
    expect(sorted(caps.supportedPrimitiveKinds ?? [])).toEqual(sorted(expected.supportedPrimitiveKinds));
    expect(sorted(caps.supportedEmitterKinds)).toEqual(sorted(expected.supportedEmitterKinds));
    expect(sorted(caps.supportedEnvironmentKinds ?? [])).toEqual(sorted(expected.supportedEnvironmentKinds));
    expect(sorted(caps.supportedAnalyticShapes)).toEqual(sorted(expected.supportedAnalyticShapes));
    expect(caps.supportDetails).toEqual(expected.supportDetails);

    expect(typeof engine.updatePrimitive === 'function').toBe(expected.methodPromises.updatePrimitive);
    expect(typeof engine.updateEmitter === 'function').toBe(expected.methodPromises.updateEmitter);
    expect(typeof engine.updateEnvironment === 'function').toBe(expected.methodPromises.updateEnvironment);
    expect(typeof engine.addPrimitive === 'function').toBe(expected.methodPromises.addPrimitive);
    expect(typeof engine.removePrimitive === 'function').toBe(expected.methodPromises.removePrimitive);
    expect(typeof engine.setSize === 'function').toBe(expected.methodPromises.setSize);
    expect(typeof engine.updateLighting === 'function').toBe(expected.methodPromises.updateLighting);
    expect(typeof engine.onFrame === 'function').toBe(expected.methodPromises.onFrame);
    expect(typeof engine.onProgress === 'function').toBe(expected.methodPromises.onProgress);
    expect(typeof engine.debug === 'object').toBe(expected.methodPromises.debug);
    expect(typeof engine.getScene === 'function').toBe(expected.methodPromises.getScene);
  });
});
