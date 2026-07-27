import { describe, expect, it } from 'vitest';
import type { Engine } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { HybridEngine } from '../src/HybridEngine.js';
import { WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT } from '../src/neural/shapeContract.js';

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: () => ({ finish: () => ({}) }),
    queue: {
      writeBuffer: () => {},
      submit: () => {},
    },
    addEventListener: () => {},
    removeEventListener: () => {},
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function sorted(values: Iterable<string>): string[] {
  return Array.from(values).sort();
}

function hasFunctionProperty(target: object, key: PropertyKey): boolean {
  return typeof (target as Record<PropertyKey, unknown>)[key] === 'function';
}

describe('walkaround-hybrid promise ledger compliance', () => {
  it('matches declared capability and optional-method promises', () => {
    const expected = BACKEND_PROMISE_LEDGER['walkaround-hybrid'];
    const engine = new HybridEngine({
      device: makeMockDevice(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
    });
    const engineView = engine as unknown as Engine;
    const caps = engine.capabilities;

    expect(caps.supportsIncrementalScene).toBe(expected.supportsIncrementalScene);
    expect(caps.supportsAuxBuffers).toBe(expected.supportsAuxBuffers);
    expect(caps.accumulates).toBe(expected.accumulates);
    expect(caps.presentationMode).toBe(expected.presentationMode);

    expect(caps.incrementalPatchSupport).toEqual(expected.incrementalPatchSupport);
    expect(sorted(caps.supportedPrimitiveKinds ?? [])).toEqual(sorted(expected.supportedPrimitiveKinds));
    expect(sorted(caps.supportedEmitterKinds)).toEqual(sorted(expected.supportedEmitterKinds));
    expect(sorted(caps.supportedEnvironmentKinds ?? [])).toEqual(sorted(expected.supportedEnvironmentKinds));
    expect(sorted(caps.supportedAnalyticShapes)).toEqual(sorted(expected.supportedAnalyticShapes));
    expect(caps.supportDetails).toEqual({
      ...expected.supportDetails,
      denoiserSpatialShapeRequirements: {
        ...expected.supportDetails.denoiserSpatialShapeRequirements,
        neural: WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT,
      },
      denoisers: {
        ...expected.supportDetails.denoisers,
        neural: 'unsupported',
      },
    });

    expect(typeof engineView.updatePrimitive === 'function').toBe(expected.methodPromises.updatePrimitive);
    expect(typeof engineView.updateEmitter === 'function').toBe(expected.methodPromises.updateEmitter);
    expect(typeof engineView.updateEnvironment === 'function').toBe(expected.methodPromises.updateEnvironment);
    expect(typeof engineView.addPrimitive === 'function').toBe(expected.methodPromises.addPrimitive);
    expect(typeof engineView.removePrimitive === 'function').toBe(expected.methodPromises.removePrimitive);
    expect(typeof engineView.setSize === 'function').toBe(expected.methodPromises.setSize);
    expect(typeof engineView.updateLighting === 'function').toBe(expected.methodPromises.updateLighting);
    expect(typeof engineView.onFrame === 'function').toBe(expected.methodPromises.onFrame);
    expect(typeof engineView.onProgress === 'function').toBe(expected.methodPromises.onProgress);
    expect(typeof engineView.debug === 'object').toBe(expected.methodPromises.debug);
    expect(typeof engineView.getScene === 'function').toBe(expected.methodPromises.getScene);
    expect(typeof engineView.onError === 'function').toBe(expected.methodPromises.onError);
    expect(typeof engineView.onWarning === 'function').toBe(expected.methodPromises.onWarning);
    expect(typeof engineView.captureFrame === 'function').toBe(expected.methodPromises.captureFrame);
    expect(typeof engineView.createInverseSession === 'function').toBe(expected.methodPromises.createInverseSession);
    expect(typeof engineView.getRestirPtResultBuffer === 'function').toBe(
      expected.methodPromises.getRestirPtResultBuffer,
    );
    expect(typeof engineView.getPresentationSource === 'function').toBe(
      expected.methodPromises.getPresentationSource,
    );
    expect(typeof engineView.getProgressiveSeedTexture === 'function').toBe(
      expected.methodPromises.getProgressiveSeedTexture,
    );
    expect(typeof engineView.seedAccumulator === 'function').toBe(expected.methodPromises.seedAccumulator);
    expect(hasFunctionProperty(engineView, 'exportGIState') || hasFunctionProperty(engineView, 'importGIState')).toBe(
      expected.methodPromises.giStatePersistence,
    );
  });
});
